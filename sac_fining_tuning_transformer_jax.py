#!/usr/bin/env python3
"""
SAC算法 + FlowMLP + Trainable Transformer Decoder
- FlowMLP作为预训练的velocity预测网络（保持原有逻辑）
- Transformer Decoder初始化为恒等映射，但可通过SAC训练
- 支持SDE/ODE分离采样
- 分离的FlowMLP/Decoder网络优化
- Poly-Tanh变换
- 在Decoder中加入适合SAC的随机性
- 类似第三份代码的初始化策略，确保初始性能与预训练模型一致
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import time
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch  # 仅用于加载PyTorch检查点
import tyro
from torch.utils.tensorboard import SummaryWriter

# JAX相关导入
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import gymnasium as gym

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# 导入make_async和cleanrl buffer
try:
    from env.gym_utils import make_async
except ImportError:
    log.error("无法导入make_async函数，请确保env.gym_utils模块可用")
    raise

try:
    from cleanrl_utils.buffers import ReplayBuffer
except ImportError:
    log.error("无法导入cleanrl_utils，请安装cleanrl")
    raise


@dataclass
class Args:
    exp_name: str = "sac_flowmlp_trainable_decoder"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "sac-flowmlp-trainable-decoder"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Environment
    env_id: str = "Walker2d-v2"
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    max_episode_steps: int = 1000
    """maximum steps per episode"""
    
    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    flowmlp_lr: float = 1e-5  # FlowMLP学习率，很小因为是预训练网络
    """the learning rate of the FlowMLP network optimizer"""
    decoder_lr: float = 3e-4
    """the learning rate of the decoder network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    learning_starts: int = 50000
    """timestep to start learning"""
    
    # 网络冻结和学习率调度参数
    flowmlp_freeze_steps: int = 1000000
    """number of steps to freeze FlowMLP training (0 = no freezing)"""
    decoder_freeze_steps: int = 0
    """number of steps to freeze decoder training (0 = no freezing)"""
    flowmlp_lr_schedule: str = "constant"
    """learning rate schedule for FlowMLP: constant, linear_decay, cosine_decay"""
    decoder_lr_schedule: str = "constant"
    """learning rate schedule for decoder: constant, linear_decay, cosine_decay"""
    warmup_steps_flowmlp: int = 1000000

    warmup_steps_decoder: int = 1
    
    # FlowMLP参数
    load_pretrained: bool = True
    """whether to load pretrained weights for FlowMLP"""
    checkpoint_path: str = "state_80.pt"
    """path to the pre-trained FlowMLP checkpoint"""
    normalization_path: str = "/home/yixian/ReinFlow/data/gym/walker2d-medium-v2/normalization.npz"
    """path to normalization file for wrapper"""
    inference_steps: int = 4
    """number of inference steps for flow matching"""
    horizon_steps: int = 4
    """number of horizon steps the network outputs"""
    cond_steps: int = 1
    """number of observation steps"""
    act_steps: int = 4
    """number of action steps"""
    denoised_clip_value: float = 100
    """clip intermediate actions during inference"""
    
    # FlowMLP架构参数
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    """MLP dimensions for FlowMLP"""
    time_dim: int = 16
    """time embedding dimension"""
    residual_style: bool = True
    """whether to use residual connections in FlowMLP"""
    activation_type: str = "ReLU"
    """activation function type"""
    use_layernorm: bool = False
    """whether to use layer normalization"""
    
    # Transformer Decoder参数
    use_decoder: bool = True
    """whether to use trainable transformer decoder"""
    decoder_num_layers: int = 6
    """number of decoder layers"""
    decoder_num_heads: int = 8
    """number of attention heads in decoder"""
    decoder_d_ff: int = 512
    """feedforward dimension in decoder"""
    decoder_dropout: float = 0.0
    """dropout rate in decoder"""
    decoder_stochastic: bool = True
    """whether decoder should be stochastic (required for SAC)"""
    decoder_log_std_min: float = -5
    """minimum log std for decoder stochastic output"""
    decoder_log_std_max: float = 2
    """maximum log std for decoder stochastic output"""
    
    # SDE采样参数
    sde_sigma: float = 0.5
    """noise strength for SDE sampling during training"""
    
    # Poly-Tanh变换参数
    use_poly_squash: bool = True
    """是否使用 tanh(poly(x)) 作为最终的动作压缩函数"""
    poly_order: int = 5
    """多项式的阶数"""

    def __post_init__(self):
        if self.mlp_dims is None:
            self.mlp_dims = [512, 512, 512]


# ==================== Poly-Tanh变换函数 ====================

def poly_squash_transform(x, order):
    """应用 tanh(poly(x)) 变换"""
    x = jnp.clip(x, -5.0, 5.0) 
    
    poly_x = jnp.zeros_like(x)
    for i in range(1, order + 1, 2):
        poly_x += (x**i) / i
    
    return jnp.tanh(poly_x)

def poly_tanh_log_prob_correction(x, order):
    """计算 tanh(poly(x)) 变换的log概率修正项"""
    x = jnp.clip(x, -5.0, 5.0)
    
    # 计算 poly(x)
    poly_x = jnp.zeros_like(x)
    for i in range(1, order + 1, 2):
        poly_x += (x**i) / i
    
    # 计算 poly'(x)
    poly_deriv = jnp.zeros_like(x)
    for i in range(1, order + 1, 2):
        poly_deriv += x**(i-1)
    
    # 计算 tanh(poly(x))
    tanh_poly_x = jnp.tanh(poly_x)
    
    # 计算雅可比行列式的绝对值
    jacobian = (1 - tanh_poly_x**2) * poly_deriv
    
    return jnp.log(jnp.abs(jacobian) + 1e-6)

def create_poly_squash_jit(order):
    @jax.jit
    def _poly_squash_jit(x):
        return poly_squash_transform(x, order)
    return _poly_squash_jit

def create_poly_log_prob_correction_jit(order):
    @jax.jit
    def _poly_log_prob_correction_jit(x):
        return poly_tanh_log_prob_correction(x, order)
    return _poly_log_prob_correction_jit


# ==================== FlowMLP 实现（保持原有结构） ====================

class SinusoidalPosEmbFlax(nn.Module):
    dim: int
    
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

class MLPFlax(nn.Module):
    dim_list: List[int]
    activation_type: str = "ReLU"
    out_activation_type: str = "Identity"
    use_layernorm: bool = False
    
    def get_activation(self, activation_type):
        if activation_type == "Tanh":
            return nn.tanh
        elif activation_type == "ReLU":
            return nn.relu
        elif activation_type == "ELU":
            return nn.elu
        elif activation_type == "Mish":
            return lambda x: x * jnp.tanh(nn.softplus(x))
        elif activation_type == "GELU":
            return nn.gelu
        elif activation_type == "Sigmoid":
            return nn.sigmoid
        elif activation_type == "SiLU":
            return nn.swish
        elif activation_type == "Softplus":
            return nn.softplus
        else:  # Identity
            return lambda x: x
    
    @nn.compact
    def __call__(self, x, training=True):
        num_layer = len(self.dim_list) - 1
        for idx in range(num_layer):
            x = nn.Dense(self.dim_list[idx + 1])(x)
            
            if self.use_layernorm and idx < num_layer - 1:
                x = nn.LayerNorm(epsilon=1e-06)(x)
            
            activation_fn = (
                self.get_activation(self.activation_type)
                if idx != num_layer - 1
                else self.get_activation(self.out_activation_type)
            )
            x = activation_fn(x)
        
        return x

class TwoLayerPreActivationResNetLinearFlax(nn.Module):
    hidden_dim: int
    activation_type: str = "ReLU"
    use_layernorm: bool = False
    
    def get_activation(self, activation_type):
        if activation_type == "Tanh":
            return nn.tanh
        elif activation_type == "ReLU":
            return nn.relu
        elif activation_type == "ELU":
            return nn.elu
        elif activation_type == "Mish":
            return lambda x: x * jnp.tanh(nn.softplus(x))
        elif activation_type == "GELU":
            return nn.gelu
        elif activation_type == "Sigmoid":
            return nn.sigmoid
        elif activation_type == "SiLU":
            return nn.swish
        elif activation_type == "Softplus":
            return nn.softplus
        else:  # Identity
            return lambda x: x
    
    @nn.compact
    def __call__(self, x, training=True):
        x_input = x
        activation_fn = self.get_activation(self.activation_type)
        
        if self.use_layernorm:
            x = nn.LayerNorm(epsilon=1e-06)(x)
        x = activation_fn(x)
        x = nn.Dense(self.hidden_dim, name='l1')(x)
        
        if self.use_layernorm:
            x = nn.LayerNorm(epsilon=1e-06)(x)
        x = activation_fn(x)
        x = nn.Dense(self.hidden_dim, name='l2')(x)
        
        return x + x_input

class ResidualMLPFlax(nn.Module):
    dim_list: List[int]
    activation_type: str = "ReLU"
    out_activation_type: str = "Identity"
    use_layernorm: bool = False
    
    def get_activation(self, activation_type):
        if activation_type == "Tanh":
            return nn.tanh
        elif activation_type == "ReLU":
            return nn.relu
        elif activation_type == "ELU":
            return nn.elu
        elif activation_type == "Mish":
            return lambda x: x * jnp.tanh(nn.softplus(x))
        elif activation_type == "GELU":
            return nn.gelu
        elif activation_type == "Sigmoid":
            return nn.sigmoid
        elif activation_type == "SiLU":
            return nn.swish
        elif activation_type == "Softplus":
            return nn.softplus
        else:  # Identity
            return lambda x: x
    
    @nn.compact
    def __call__(self, x, training=True):
        hidden_dim = self.dim_list[1]
        num_hidden_layers = len(self.dim_list) - 3
        assert num_hidden_layers % 2 == 0, f"num_hidden_layers must be even, got {num_hidden_layers}"
        
        x = nn.Dense(hidden_dim)(x)
        
        num_residual_blocks = num_hidden_layers // 2
        for i in range(num_residual_blocks):
            x = TwoLayerPreActivationResNetLinearFlax(
                hidden_dim=hidden_dim,
                activation_type=self.activation_type,
                use_layernorm=self.use_layernorm,
                name=f'residual_block_{i}'
            )(x, training=training)
        
        x = nn.Dense(self.dim_list[-1])(x)
        
        activation_fn = self.get_activation(self.out_activation_type)
        x = activation_fn(x)
        
        return x

class FlowMLPFlax(nn.Module):
    """原始FlowMLP网络 - 保持与预训练模型完全一致"""
    horizon_steps: int
    action_dim: int
    cond_dim: int
    time_dim: int = 16
    mlp_dims: List[int] = None
    cond_mlp_dims: Optional[List[int]] = None
    activation_type: str = "ReLU"
    out_activation_type: str = "Identity"
    use_layernorm: bool = False
    residual_style: bool = False
    
    def setup(self):
        mlp_dims = list(self.mlp_dims) if self.mlp_dims is not None else [256, 256]
        act_dim_total = self.action_dim * self.horizon_steps
        
        # Time embedding layers
        self.sinusoidal_emb = SinusoidalPosEmbFlax(self.time_dim)
        self.time_dense1 = nn.Dense(self.time_dim * 2)
        self.time_dense2 = nn.Dense(self.time_dim)
        
        # Condition encoder
        if self.cond_mlp_dims:
            cond_mlp_dims_list = list(self.cond_mlp_dims)
            self.cond_mlp = MLPFlax(
                dim_list=[self.cond_dim] + cond_mlp_dims_list,
                activation_type=self.activation_type,
                out_activation_type="Identity",
            )
            cond_enc_dim = self.cond_mlp_dims[-1]
        else:
            cond_enc_dim = self.cond_dim
            
        input_dim = self.time_dim + self.action_dim * self.horizon_steps + cond_enc_dim
        
        # Main MLP - 输出velocity（与预训练模型一致）
        if self.residual_style:
            self.mlp_mean = ResidualMLPFlax(
                dim_list=[input_dim] + mlp_dims + [act_dim_total],
                activation_type=self.activation_type,
                out_activation_type=self.out_activation_type,
                use_layernorm=self.use_layernorm,
            )
        else:
            self.mlp_mean = MLPFlax(
                dim_list=[input_dim] + mlp_dims + [act_dim_total],
                activation_type=self.activation_type,
                out_activation_type=self.out_activation_type,
                use_layernorm=self.use_layernorm,
            )
    
    def __call__(self, action, time, cond, training=True):
        """
        FlowMLP forward pass - 与预训练模型完全一致
        Args:
            action: (B, Ta, Da) - current trajectory x_t
            time: (B,) or scalar - diffusion step
            cond: dict with key state - observation
        Returns:
            velocity: (B, Ta, Da) - predicted velocity
        """
        B, Ta, Da = action.shape

        # flatten action chunk
        action = action.reshape(B, -1)

        # flatten obs history
        state = cond["state"].reshape(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            cond_emb = self.cond_mlp(state, training=training)
        else:
            cond_emb = state
        
        # time encoder
        if jnp.ndim(time) == 0:  # scalar
            time = jnp.ones((B, 1)) * time
        elif time.shape == (B,):
            time = time.reshape(B, 1)
        
        time_emb = self.sinusoidal_emb(time)
        time_emb = self.time_dense1(time_emb)
        time_emb = time_emb * jnp.tanh(nn.softplus(time_emb))  # Mish activation
        time_emb = self.time_dense2(time_emb)
        time_emb = time_emb.reshape(B, self.time_dim)
        
        # velocity prediction
        vel_feature = jnp.concatenate([action, time_emb, cond_emb], axis=-1)
        vel = self.mlp_mean(vel_feature, training=training)
        
        return vel.reshape(B, Ta, Da)


# ==================== Trainable Transformer Decoder（严格恒等映射初始化）====================

class IdentityMultiHeadAttentionFlax(nn.Module):
    """
    修改后：初始化为零输出的Multi-head attention，但可训练
    """
    d_model: int
    num_heads: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, query, key, value, mask=None, training=True):
        batch_size, seq_len = query.shape[:2]
        head_dim = self.d_model // self.num_heads
        
        # 关键修改：所有投影层权重和偏置初始化为0，确保注意力模块输出为0
        q = nn.Dense(
            self.d_model, 
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='q_proj'
        )(query)
        k = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='k_proj'
        )(key)
        v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='v_proj'
        )(value)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention (后续计算结果仍为0)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attention_weights = nn.softmax(scores, axis=-1)
        
        if self.dropout > 0.0 and training:
            attention_weights = nn.Dropout(self.dropout, deterministic=not training)(attention_weights)
            
        attention_output = jnp.matmul(attention_weights, v)
        
        # Reshape and output projection
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # 输出投影层也初始化为0
        output = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='out_proj'
        )(attention_output)
        
        return output

class IdentityTransformerDecoderLayerFlax(nn.Module):
    """
    修改后：采用Pre-LayerNorm架构，并初始化为严格恒等映射
    """
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, tgt, memory, tgt_mask=None, memory_mask=None, training=True, rng_key=None):
        # 关键修改：改为Pre-LayerNorm架构，更容易实现恒等映射
        
        # Self-attention: x = x + Sublayer(LayerNorm(x))
        norm_tgt = nn.LayerNorm(name='norm1')(tgt)
        tgt2 = IdentityMultiHeadAttentionFlax(
            self.d_model, self.num_heads, self.dropout, name='self_attn'
        )(norm_tgt, norm_tgt, norm_tgt, mask=tgt_mask, training=training)
        tgt = tgt + tgt2  # tgt + 0 = tgt
        
        # Cross-attention
        if memory is not None:
            norm_tgt = nn.LayerNorm(name='norm2')(tgt)
            tgt2 = IdentityMultiHeadAttentionFlax(
                self.d_model, self.num_heads, self.dropout, name='cross_attn'
            )(norm_tgt, memory, memory, mask=memory_mask, training=training)
            tgt = tgt + tgt2  # tgt + 0 = tgt
        
        # Feed-forward
        norm_tgt = nn.LayerNorm(name='norm3')(tgt)
        
        # FFN第一层
        tgt2 = nn.Dense(
            self.d_ff,
            kernel_init=nn.initializers.xavier_uniform(), # 正常初始化
            bias_init=nn.initializers.zeros,
            name='ffn_linear1'
        )(norm_tgt)
        
        tgt2 = nn.gelu(tgt2)
        
        if self.dropout > 0.0 and training:
            tgt2 = nn.Dropout(self.dropout, deterministic=not training)(tgt2)
            
        # 关键修改：FFN第二层权重和偏置初始化为0，确保FFN输出为0
        tgt2 = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='ffn_linear2'
        )(tgt2)
        
        tgt = tgt + tgt2  # tgt + 0 = tgt
        
        return tgt

class TrainableTransformerDecoderFlax(nn.Module):
    """修改后：Trainable Transformer Decoder，严格初始化为恒等映射"""
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float = 0.0
    action_dim: int = 17
    horizon_steps: int = 4
    log_std_min: float = -5
    log_std_max: float = 2
    
    def setup(self):
        self.layers = [
            IdentityTransformerDecoderLayerFlax(
                d_model=self.d_model,
                num_heads=self.num_heads, 
                d_ff=self.d_ff,
                dropout=self.dropout,
                name=f'layer_{i}'
            )
            for i in range(self.num_layers)
        ]
        
        final_action_dim = self.action_dim * self.horizon_steps
        
        # 关键修改：action_mean_head的权重初始化为0，偏置也为0。
        # 这样，最终输出 mean = input + 0 = input
        self.action_mean_residual_head = nn.Dense(
            final_action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='action_mean_residual'
        )
        
        # log_std头初始化为较小值，以获得较小的初始探索噪声
        self.action_log_std_head = nn.Dense(
            final_action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-2.0), # std ≈ 0.135
            name='action_log_std'
        )
    
    def __call__(self, velocity_input, encoder_output=None, 
                 self_mask=None, cross_mask=None, training=True, rng_key=None):
        """
        Decoder forward pass - 接收velocity作为输入
        初始时，action_mean应该等于velocity_input
        """
        # Add sequence dimension for transformer processing
        x = jnp.expand_dims(velocity_input, axis=1)  # (B, 1, d_model)
        
        # Pass through decoder layers - 初始时是恒等映射
        if rng_key is not None:
            layer_keys = jax.random.split(rng_key, self.num_layers)
        else:
            layer_keys = [None] * self.num_layers

        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_output, self_mask, cross_mask, training, layer_keys[i])
        
        # 关键修改：移除final_norm，它会破坏恒等映射
        # x = self.final_norm(x)
        
        # Remove sequence dimension
        x = x.squeeze(axis=1)  # (B, d_model)
        
        # 关键修改：使用残差连接来计算action_mean
        # 初始时：residual = 0, action_mean = x + 0 = x
        # 这里的x就是velocity_input，因为前面的层都是恒等映射
        residual = self.action_mean_residual_head(x)
        action_mean = x + residual
        
        action_log_std = self.action_log_std_head(x)
        
        # Clamp log_std to reasonable range
        action_log_std = jnp.tanh(action_log_std)
        action_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (action_log_std + 1)
        
        return action_mean, action_log_std


# ==================== FlowMLP + Trainable Decoder Actor ====================

class FlowMLPWithTrainableDecoderActor(nn.Module):
    """FlowMLP + Trainable Transformer Decoder Actor"""
    obs_dim: int
    action_dim: int
    cond_steps: int = 1
    inference_steps: int = 4
    horizon_steps: int = 4
    denoised_clip_value: float = 3.0
    
    # FlowMLP parameters（保持与预训练模型一致）
    mlp_dims: List[int] = None
    time_dim: int = 16
    residual_style: bool = True
    activation_type: str = "ReLU"
    use_layernorm: bool = False
    
    # Decoder parameters  
    use_decoder: bool = True
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_d_ff: int = 512
    decoder_dropout: float = 0.0
    decoder_stochastic: bool = True # For SAC, decoder must be stochastic during training
    decoder_log_std_min: float = -5
    decoder_log_std_max: float = 2
    
    # SDE parameters
    sde_sigma: float = 0.01
    use_poly_squash: bool = True
    poly_order: int = 5
    
    def setup(self):
        # 计算条件维度
        cond_dim = self.obs_dim * self.cond_steps
        act_dim_total = self.action_dim * self.horizon_steps
        
        # FlowMLP参数
        flowmlp_params = {
            'horizon_steps': self.horizon_steps,
            'action_dim': self.action_dim,
            'cond_dim': cond_dim,
            'time_dim': self.time_dim,
            'mlp_dims': self.mlp_dims or [512, 512, 512],
            'cond_mlp_dims': None,
            'residual_style': self.residual_style,
            'activation_type': self.activation_type,
            'use_layernorm': self.use_layernorm,
            'out_activation_type': "Identity",
        }
        
        # 创建FlowMLP网络
        self.flowmlp = FlowMLPFlax(**flowmlp_params)
        
        # 创建Trainable Decoder
        if self.use_decoder:
            decoder_d_model = act_dim_total
            
            self.decoder = TrainableTransformerDecoderFlax(
                num_layers=self.decoder_num_layers,
                d_model=decoder_d_model,
                num_heads=self.decoder_num_heads,
                d_ff=self.decoder_d_ff,
                dropout=self.decoder_dropout,
                action_dim=self.action_dim,
                horizon_steps=self.horizon_steps,
                log_std_min=self.decoder_log_std_min,
                log_std_max=self.decoder_log_std_max
            )
        
        # Poly-tanh变换函数
        if self.use_poly_squash:
            self.poly_squash_jit = create_poly_squash_jit(self.poly_order)
            self.poly_log_prob_correction_jit = create_poly_log_prob_correction_jit(self.poly_order)
    
    def sample_first_point(self, B: int, key):
        """采样初始点并计算log probability"""
        xt = jax.random.normal(key, (B, self.horizon_steps * self.action_dim))
        log_prob = jax.scipy.stats.norm.logpdf(xt, 0, 1).sum(axis=-1)
        xt = xt.reshape(B, self.horizon_steps, self.action_dim)
        return xt, log_prob
    
    def __call__(self, obs, key, training=True, use_sde=True):
        """
        Forward pass: FlowMLP -> (Optional Decoder) -> Action
        """
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)
            single_obs = True
        else:
            single_obs = False
            
        B = obs.shape[0]
        
        # 构造条件字典
        if obs.ndim == 2:
            cond = {"state": jnp.expand_dims(obs, axis=1)}
        else:
            cond = {"state": obs}
        
        # 采样初始点
        key, sample_key = jax.random.split(key)
        xt, log_prob = self.sample_first_point(B, sample_key)
        
        # Flow matching采样循环
        dt = 1.0 / self.inference_steps
        time_steps = jnp.linspace(0, 1 - dt, self.inference_steps)
        
        for i in range(self.inference_steps):
            t_scalar = time_steps[i]
            t_tensor = jnp.full((B,), t_scalar)
            
            # 使用FlowMLP预测velocity
            velocity = self.flowmlp(xt, t_tensor, cond, training=training)
            
            # 如果使用Decoder，处理velocity
            if self.use_decoder:
                velocity_flat = velocity.reshape(B, -1)
                key, decoder_key = jax.random.split(key)
                
                if training and self.decoder_stochastic:
                    # 训练时：Decoder输出mean和log_std
                    velocity_mean, velocity_log_std = self.decoder(
                        velocity_flat, training=True, rng_key=decoder_key
                    )
                    velocity_std = jnp.exp(velocity_log_std)
                    
                    # 重参数化采样
                    epsilon = jax.random.normal(decoder_key, velocity_mean.shape)
                    processed_velocity = velocity_mean + velocity_std * epsilon
                    
                    # 计算log probability
                    decoder_log_prob = jax.scipy.stats.norm.logpdf(epsilon, 0, 1).sum(axis=-1)
                    log_prob = log_prob + decoder_log_prob
                    
                    velocity = processed_velocity.reshape(B, self.horizon_steps, self.action_dim)
                else:
                    # 推理时：使用mean
                    velocity_mean, _ = self.decoder(
                        velocity_flat, training=False, rng_key=None
                    )
                    velocity = velocity_mean.reshape(B, self.horizon_steps, self.action_dim)
            
            # Flow matching更新
            if use_sde and training:
                # SDE模式
                key, noise_key = jax.random.split(key)
                noise = jax.random.normal(noise_key, xt.shape)
                diffusion_coef = self.sde_sigma * jnp.sqrt(dt) * noise
                xt = xt + velocity * dt + diffusion_coef
                noise_log_prob = jax.scipy.stats.norm.logpdf(
                    noise.reshape(B, -1), 0, self.sde_sigma * jnp.sqrt(dt)
                ).sum(axis=-1)
                log_prob = log_prob + noise_log_prob
            else:
                # ODE模式
                xt = xt + velocity * dt
            
            # 中间裁剪
            if i < self.inference_steps - 1:
                xt = jnp.clip(xt, -self.denoised_clip_value, self.denoised_clip_value)
            else:
                # 最后一步：应用变换
                xt_flat = xt.reshape(B, -1)
                
                if self.use_poly_squash:
                    xt_squashed = self.poly_squash_jit(xt_flat)
                    poly_log_prob_correction = self.poly_log_prob_correction_jit(xt_flat)
                    log_prob = log_prob - poly_log_prob_correction.sum(axis=-1)
                else:
                    xt_squashed = jnp.tanh(xt_flat)
                    log_prob = log_prob - jnp.log(1 - xt_squashed**2 + 1e-6).sum(axis=-1)
                
                xt = xt_squashed.reshape(B, self.horizon_steps, self.action_dim)
        
        # 返回第0步动作
        action = xt[:, 0, :]
        
        if single_obs:
            return action.squeeze(0), log_prob.squeeze(0)
        else:
            return action, log_prob


# ==================== Parameter Conversion Functions ====================

def extract_torch_mlp_params(state_dict, prefix):
    """Extract MLP parameters from PyTorch state dict"""
    params = {}
    layer_idx = 0
    
    # Find all module indices
    module_indices = set()
    for key in state_dict.keys():
        if key.startswith(f"{prefix}.moduleList."):
            parts = key.split(".")
            if len(parts) >= 3:
                try:
                    module_idx = int(parts[2])
                    module_indices.add(module_idx)
                except ValueError:
                    continue
    
    for module_idx in sorted(module_indices):
        # Extract linear layer parameters
        weight_key = f"{prefix}.moduleList.{module_idx}.linear_1.weight"
        bias_key = f"{prefix}.moduleList.{module_idx}.linear_1.bias"
        
        if weight_key in state_dict and bias_key in state_dict:
            dense_name = f"Dense_{layer_idx}"
            params[dense_name] = {
                'kernel': jnp.array(state_dict[weight_key].detach().cpu().numpy().T),
                'bias': jnp.array(state_dict[bias_key].detach().cpu().numpy())
            }
            layer_idx += 1
    
    return params

def extract_torch_residual_mlp_params(state_dict, prefix):
    """Extract ResidualMLP parameters from PyTorch state dict"""
    params = {}
    dense_idx = 0
    
    # Find all layer indices
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith(f"{prefix}.layers."):
            parts = key.split(".")
            if len(parts) >= 3:
                try:
                    layer_idx = int(parts[2])
                    layer_indices.add(layer_idx)
                except ValueError:
                    continue
    
    residual_block_idx = 0
    for layer_idx in sorted(layer_indices):
        # Check if it's a Linear layer
        weight_key = f"{prefix}.layers.{layer_idx}.weight"
        bias_key = f"{prefix}.layers.{layer_idx}.bias"
        
        if weight_key in state_dict and bias_key in state_dict:
            dense_name = f"Dense_{dense_idx}"
            params[dense_name] = {
                'kernel': jnp.array(state_dict[weight_key].detach().cpu().numpy().T),
                'bias': jnp.array(state_dict[bias_key].detach().cpu().numpy())
            }
            dense_idx += 1
        else:
            # Check if it's a TwoLayerPreActivationResNetLinear
            l1_weight_key = f"{prefix}.layers.{layer_idx}.l1.weight"
            l1_bias_key = f"{prefix}.layers.{layer_idx}.l1.bias"
            l2_weight_key = f"{prefix}.layers.{layer_idx}.l2.weight"
            l2_bias_key = f"{prefix}.layers.{layer_idx}.l2.bias"
            
            if (l1_weight_key in state_dict and l1_bias_key in state_dict and 
                l2_weight_key in state_dict and l2_bias_key in state_dict):
                
                # Create residual block parameters
                residual_block_name = f"residual_block_{residual_block_idx}"
                params[residual_block_name] = {
                    'l1': {
                        'kernel': jnp.array(state_dict[l1_weight_key].detach().cpu().numpy().T),
                        'bias': jnp.array(state_dict[l1_bias_key].detach().cpu().numpy())
                    },
                    'l2': {
                        'kernel': jnp.array(state_dict[l2_weight_key].detach().cpu().numpy().T),
                        'bias': jnp.array(state_dict[l2_bias_key].detach().cpu().numpy())
                    }
                }
                residual_block_idx += 1
    
    return params

def torch_to_jax_flowmlp_params(torch_state_dict, jax_flowmlp_model, sample_input):
    """Convert PyTorch FlowMLP parameters to JAX FlowMLP format"""
    
    # Initialize JAX model to get parameter structure
    key = jax.random.PRNGKey(0)
    action, time, cond = sample_input
    jax_params = jax_flowmlp_model.init(key, action, time, cond)
    
    log.info("Converting PyTorch FlowMLP parameters to JAX...")
    
    # Create a new parameter dictionary
    new_params = {}
    
    # Time embedding parameters
    new_params['time_dense1'] = {
        'kernel': jnp.array(torch_state_dict['time_embedding.1.weight'].detach().cpu().numpy().T),
        'bias': jnp.array(torch_state_dict['time_embedding.1.bias'].detach().cpu().numpy())
    }
    new_params['time_dense2'] = {
        'kernel': jnp.array(torch_state_dict['time_embedding.3.weight'].detach().cpu().numpy().T),
        'bias': jnp.array(torch_state_dict['time_embedding.3.bias'].detach().cpu().numpy())
    }
    
    # Condition MLP parameters (if exists)
    cond_mlp_keys = [k for k in torch_state_dict.keys() if k.startswith('cond_mlp')]
    if cond_mlp_keys:
        new_params['cond_mlp'] = extract_torch_mlp_params(torch_state_dict, 'cond_mlp')
    
    # Main MLP parameters - check if it's residual style
    residual_keys = [k for k in torch_state_dict.keys() if 'mlp_mean.layers' in k and '.l1.' in k]
    if residual_keys:
        new_params['mlp_mean'] = extract_torch_residual_mlp_params(torch_state_dict, 'mlp_mean')
    else:
        new_params['mlp_mean'] = extract_torch_mlp_params(torch_state_dict, 'mlp_mean')
    
    return new_params

def initialize_decoder_as_identity(decoder_params):
    """将Decoder参数初始化为真正的恒等映射"""
    def process_layer_params(layer_params):
        """处理单层参数"""
        if isinstance(layer_params, dict):
            processed = {}
            for key, value in layer_params.items():
                if key == 'action_mean_head':
                    # action_mean头应该是恒等映射
                    if isinstance(value, dict) and 'kernel' in value:
                        kernel_shape = value['kernel'].shape
                        if len(kernel_shape) == 2:
                            min_dim = min(kernel_shape[0], kernel_shape[1])
                            # 创建一个接近恒等的矩阵
                            if kernel_shape[0] == kernel_shape[1]:
                                # 方阵：使用恒等矩阵
                                kernel = jnp.eye(kernel_shape[0])
                            else:
                                # 非方阵：创建尽可能接近恒等的矩阵
                                kernel = jnp.zeros(kernel_shape)
                                # 在对角线位置设置1
                                for i in range(min_dim):
                                    kernel = kernel.at[i, i].set(1.0)
                            
                            processed[key] = {
                                'kernel': kernel,
                                'bias': jnp.zeros_like(value['bias'])
                            }
                        else:
                            # 不是2D矩阵，使用小权重
                            processed[key] = {
                                'kernel': value['kernel'] * 0.001,
                                'bias': jnp.zeros_like(value['bias'])
                            }
                    else:
                        processed[key] = value
                elif key == 'action_log_std_head':
                    # log_std头初始化为较小的负值
                    if isinstance(value, dict) and 'kernel' in value:
                        processed[key] = {
                            'kernel': jnp.zeros_like(value['kernel']),
                            'bias': jnp.full_like(value['bias'], -2.0)
                        }
                    else:
                        processed[key] = value
                elif isinstance(value, dict):
                    # 递归处理嵌套字典
                    processed[key] = process_layer_params(value)
                elif hasattr(value, 'shape') and 'kernel' in key:
                    # 其他权重矩阵：小权重
                    processed[key] = value * 0.001
                elif hasattr(value, 'shape') and 'bias' in key:
                    # 偏置：零初始化
                    processed[key] = jnp.zeros_like(value)
                else:
                    processed[key] = value
            return processed
        else:
            # 如果是数组，小幅缩放
            if hasattr(value, 'shape'):
                return value * 0.001
            else:
                return value
    
    return process_layer_params(decoder_params)

def load_pretrained_flowmlp_params(checkpoint_path, flowmlp_config, sample_input):
    """加载预训练FlowMLP参数"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    log.info(f"Loading pretrained FlowMLP from: {checkpoint_path}")
    
    # 加载PyTorch检查点
    device = torch.device("cpu")
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        raise
    
    # 获取state_dict
    if 'model' in checkpoint_data:
        state_dict = checkpoint_data['model']
        log.info("Using 'model' key from checkpoint")
    elif 'ema' in checkpoint_data:
        state_dict = checkpoint_data['ema']
        log.info("Using 'ema' key from checkpoint")
    else:
        state_dict = checkpoint_data
        log.info("Using direct state_dict from checkpoint")
    
    log.info(f"Found {len(state_dict)} parameter keys in checkpoint")
    
    # 修正state_dict键名（移除'network.'前缀）
    corrected_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('network.', '', 1) if k.startswith('network.') else k
        corrected_state_dict[new_key] = v
    
    log.info(f"Key examples: {list(corrected_state_dict.keys())[:5]}...")
    
    # 创建独立的FlowMLP实例用于参数转换
    try:
        flowmlp_module = FlowMLPFlax(**flowmlp_config)
        log.info(f"Created FlowMLP with config: {flowmlp_config}")
    except Exception as e:
        log.error(f"Failed to create FlowMLP module: {e}")
        raise
    
    # 转换参数从PyTorch到JAX
    try:
        jax_params = torch_to_jax_flowmlp_params(corrected_state_dict, flowmlp_module, sample_input)
        log.info("Pretrained FlowMLP parameters converted to JAX successfully!")
        return jax_params
    except Exception as e:
        log.error(f"Failed to convert parameters: {e}")
        raise


# ==================== Learning Rate Schedulers ====================

def create_lr_schedule(base_lr: float, schedule_type: str, total_steps: int, 
                      warmup_steps: int = 0, decay_factor: float = 0.1):
    """创建JAX兼容的学习率调度器"""
    
    def warmup_schedule(step):
        warmup_factor = jnp.where(
            (warmup_steps > 0) & (step < warmup_steps),
            (step + 1) / warmup_steps,
            1.0
        )
        return base_lr * warmup_factor
    
    def constant_schedule(step):
        return warmup_schedule(step)
    
    def linear_decay_schedule(step):
        base_rate = warmup_schedule(step)
        decay_steps = total_steps - warmup_steps
        progress = jnp.maximum(0.0, (step - warmup_steps) / decay_steps)
        progress = jnp.minimum(progress, 1.0)
        decay_multiplier = 1.0 - progress * (1.0 - decay_factor)
        return base_rate * decay_multiplier
    
    def cosine_decay_schedule(step):
        base_rate = warmup_schedule(step)
        decay_steps = total_steps - warmup_steps
        progress = jnp.maximum(0.0, (step - warmup_steps) / decay_steps)
        progress = jnp.minimum(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay_multiplier = decay_factor + (1.0 - decay_factor) * cosine_decay
        return base_rate * decay_multiplier
    
    if schedule_type == "constant":
        return constant_schedule
    elif schedule_type == "linear_decay":
        return linear_decay_schedule
    elif schedule_type == "cosine_decay":
        return cosine_decay_schedule
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ==================== Critic Network ====================

class QNetwork(nn.Module):
    """SAC Critic网络 - JAX版本"""
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


# ==================== Entropy Coefficient ====================

class EntropyCoef(nn.Module):
    """Learnable entropy coefficient for SAC"""
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


# ==================== Training State with FlowMLP and Decoder ====================

class FlowMLPDecoderTrainState:
    """FlowMLP + Decoder训练状态"""
    def __init__(self, flowmlp_state: TrainState, decoder_state: Optional[TrainState] = None):
        self.flowmlp_state = flowmlp_state
        self.decoder_state = decoder_state
    
    @property
    def params(self):
        """组合参数"""
        if self.decoder_state is not None:
            return {
                'params': {
                    'flowmlp': self.flowmlp_state.params,
                    'decoder': self.decoder_state.params
                }
            }
        else:
            return {
                'params': {
                    'flowmlp': self.flowmlp_state.params
                }
            }
    
    def replace(self, flowmlp_state=None, decoder_state=None):
        """替换状态"""
        return FlowMLPDecoderTrainState(
            flowmlp_state=flowmlp_state if flowmlp_state is not None else self.flowmlp_state,
            decoder_state=decoder_state if decoder_state is not None else self.decoder_state
        )

# 注册FlowMLPDecoderTrainState为JAX pytree
def _flowmlp_decoder_train_state_tree_flatten(state):
    if state.decoder_state is not None:
        children = (state.flowmlp_state, state.decoder_state)
        aux_data = True
    else:
        children = (state.flowmlp_state,)
        aux_data = False
    return children, aux_data

def _flowmlp_decoder_train_state_tree_unflatten(aux_data, children):
    if aux_data:  # has decoder
        flowmlp_state, decoder_state = children
        return FlowMLPDecoderTrainState(flowmlp_state, decoder_state)
    else:  # no decoder
        flowmlp_state = children[0]
        return FlowMLPDecoderTrainState(flowmlp_state, None)

jax.tree_util.register_pytree_node(
    FlowMLPDecoderTrainState,
    _flowmlp_decoder_train_state_tree_flatten,
    _flowmlp_decoder_train_state_tree_unflatten
)


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


# ==================== Utility Functions ====================

def reset_env_all(venv, num_envs, verbose=False, options_venv=None, **kwargs):
    """重置所有环境"""
    if options_venv is None:
        options_venv = [
            {k: v for k, v in kwargs.items()} for _ in range(num_envs)
        ]
    
    obs_venv = venv.reset_arg(options_list=options_venv)
    
    if isinstance(obs_venv, list):
        obs_venv = {
            key: np.stack([obs_venv[i][key] for i in range(num_envs)])
            for key in obs_venv[0].keys()
        }
    
    return obs_venv

def process_obs(obs_venv, main_obs_key=None):
    """处理观察值，确保形状正确"""
    if isinstance(obs_venv, dict):
        obs = obs_venv[main_obs_key]
    else:
        obs = obs_venv
    
    # 如果观察值有多于2个维度，需要展平除第一个维度外的所有维度
    if obs.ndim > 2:
        obs = obs.reshape(obs.shape[0], -1)
    
    return obs


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}_flowmlp_lr{args.flowmlp_lr}_decoder_lr{args.decoder_lr}_poly{args.poly_order if args.use_poly_squash else 'off'}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, qf2_key, alpha_key, action_key = jax.random.split(key, 6)

    # 环境设置
    log.info(f"创建 {args.num_envs} 个并行环境: {args.env_id}")
    log.info(f"FlowMLP + Trainable Decoder架构配置:")
    log.info(f"  - FlowMLP: lr = {args.flowmlp_lr}, 冻结步数 = {args.flowmlp_freeze_steps}")
    log.info(f"  - Decoder: lr = {args.decoder_lr}, layers = {args.decoder_num_layers}")
    log.info(f"  - SDE参数: sigma = {args.sde_sigma}")
    if args.use_poly_squash:
        log.info(f"  - Poly-Tanh变换: 启用，阶数 = {args.poly_order}")
    
    # 环境wrapper配置
    wrappers_config = {
        "mujoco_locomotion_lowdim": {
            "normalization_path": args.normalization_path
        },
        "multi_step": {
            "n_obs_steps": args.cond_steps,
            "n_action_steps": args.act_steps,
            "max_episode_steps": args.max_episode_steps,
            "reset_within_step": True
        }
    }
    
    envs = make_async(
        args.env_id,
        env_type=None,
        num_envs=args.num_envs,
        asynchronous=True,
        max_episode_steps=args.max_episode_steps,
        wrappers=wrappers_config,
        robomimic_env_cfg_path=None,
        shape_meta=None,
        use_image_obs=False,
        render=args.capture_video,
        render_offscreen=args.capture_video,
        obs_dim=11,
        action_dim=3,
    )
    
    envs.seed([args.seed + i for i in range(args.num_envs)])
    
    # 获取环境规格
    dummy_obs = reset_env_all(envs, args.num_envs, verbose=False)
    if isinstance(dummy_obs, dict):
        main_obs_key = list(dummy_obs.keys())[0]
        obs_shape = dummy_obs[main_obs_key].shape
        if len(obs_shape) > 2:
            obs_flat_dim = np.prod(obs_shape[1:])
        else:
            obs_flat_dim = obs_shape[-1]
    else:
        obs_shape = dummy_obs.shape
        if len(obs_shape) > 2:
            obs_flat_dim = np.prod(obs_shape[1:])
        else:
            obs_flat_dim = obs_shape[-1]
        main_obs_key = None
    
    action_dim = envs.action_space.shape[-1]
    
    # 创建observation_space和action_space对象用于replay buffer
    observation_space = gym.spaces.Box(
        low=np.full(obs_flat_dim, -np.inf, dtype=np.float32),
        high=np.full(obs_flat_dim, np.inf, dtype=np.float32),
        shape=(obs_flat_dim,), 
        dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1, high=1, shape=(action_dim,), dtype=np.float32
    )
    
    # 获取实际的obs_dim
    obs_dim = obs_flat_dim // args.cond_steps if obs_flat_dim % args.cond_steps == 0 else obs_flat_dim
    
    # 创建FlowMLP + Trainable Decoder Actor
    log.info("创建FlowMLP + Trainable Decoder Actor（类似第三份代码的架构）")
    
    actor = FlowMLPWithTrainableDecoderActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cond_steps=args.cond_steps,
        inference_steps=args.inference_steps,
        horizon_steps=args.horizon_steps,
        denoised_clip_value=args.denoised_clip_value,
        mlp_dims=args.mlp_dims,
        time_dim=args.time_dim,
        residual_style=args.residual_style,
        activation_type=args.activation_type,
        use_layernorm=args.use_layernorm,
        use_decoder=args.use_decoder,
        decoder_num_layers=args.decoder_num_layers,
        decoder_num_heads=args.decoder_num_heads,
        decoder_d_ff=args.decoder_d_ff,
        decoder_dropout=args.decoder_dropout,
        decoder_stochastic=args.decoder_stochastic,
        decoder_log_std_min=args.decoder_log_std_min,
        decoder_log_std_max=args.decoder_log_std_max,
        sde_sigma=args.sde_sigma,
        use_poly_squash=args.use_poly_squash,
        poly_order=args.poly_order
    )
    
    # 创建初始观测用于网络初始化
    obs_venv = reset_env_all(envs, args.num_envs)
    obs = process_obs(obs_venv, main_obs_key)
    
    # 创建学习率调度器
    flowmlp_lr_schedule = create_lr_schedule(
        base_lr=args.flowmlp_lr,
        schedule_type=args.flowmlp_lr_schedule,
        total_steps=args.total_timesteps,
        warmup_steps=args.warmup_steps_flowmlp,
        decay_factor=0.1
    )
    
    decoder_lr_schedule = create_lr_schedule(
        base_lr=args.decoder_lr,
        schedule_type=args.decoder_lr_schedule,
        total_steps=args.total_timesteps,
        warmup_steps=args.warmup_steps_decoder,
        decay_factor=0.1
    )
    
    # 创建优化器
    flowmlp_tx = optax.adam(learning_rate=flowmlp_lr_schedule)
    decoder_tx = optax.adam(learning_rate=decoder_lr_schedule)
    
    # 初始化actor参数
    initial_params = actor.init(actor_key, obs, action_key)
    
    if args.load_pretrained and os.path.exists(args.checkpoint_path):
        log.info("准备加载预训练FlowMLP权重...")
        
        # 创建样例输入用于参数转换
        batch_size = 1
        sample_action = jnp.zeros((batch_size, args.horizon_steps, action_dim))
        sample_time = jnp.zeros((batch_size,))
        sample_cond = {"state": jnp.zeros((batch_size, args.cond_steps, obs_dim))}
        sample_input = (sample_action, sample_time, sample_cond)
        
        # 准备FlowMLP配置
        flowmlp_config = {
            'horizon_steps': args.horizon_steps,
            'action_dim': action_dim,
            'cond_dim': obs_dim * args.cond_steps,
            'time_dim': args.time_dim,
            'mlp_dims': args.mlp_dims,
            'cond_mlp_dims': None,
            'residual_style': args.residual_style,
            'activation_type': args.activation_type,
            'use_layernorm': args.use_layernorm,
            'out_activation_type': "Identity",
        }
        
        try:
            # 加载预训练FlowMLP参数
            pretrained_flowmlp_params = load_pretrained_flowmlp_params(
                args.checkpoint_path, flowmlp_config, sample_input
            )
            
            # 使用预训练的FlowMLP参数，恒等映射初始化Decoder参数
            flowmlp_params = pretrained_flowmlp_params
            decoder_params = initial_params['params']['decoder']
            
            # 将Decoder参数初始化为真正的恒等映射
            decoder_params = initialize_decoder_as_identity(decoder_params)
            
            log.info("✅ 预训练FlowMLP权重已加载，Decoder初始化为恒等映射")
            
        except Exception as e:
            log.error(f"预训练权重加载失败: {e}")
            log.info("回退到随机初始化...")
            
            flowmlp_params = initial_params['params']['flowmlp']
            decoder_params = initial_params['params']['decoder']
            decoder_params = initialize_decoder_as_identity(decoder_params)
    else:
        # 随机初始化
        flowmlp_params = initial_params['params']['flowmlp']
        decoder_params = initial_params['params']['decoder']
        decoder_params = initialize_decoder_as_identity(decoder_params)
        log.info("使用随机初始化的FlowMLP + Decoder网络")
    
    # 创建训练状态
    flowmlp_state = TrainState.create(
        apply_fn=lambda params, *args, **kwargs: actor.flowmlp.apply(params, *args, **kwargs),
        params=flowmlp_params,
        target_params=flowmlp_params,
        tx=flowmlp_tx,
    )
    
    decoder_state = TrainState.create(
        apply_fn=lambda params, *args, **kwargs: actor.decoder.apply(params, *args, **kwargs),
        params=decoder_params,
        target_params=decoder_params,
        tx=decoder_tx,
    )
    
    actor_state = FlowMLPDecoderTrainState(flowmlp_state, decoder_state)
    
    # 创建Critic网络
    qf = QNetwork()
    dummy_action = jnp.zeros((args.num_envs, action_dim))
    
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, dummy_action),
        target_params=qf.init(qf1_key, obs, dummy_action),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, dummy_action),
        target_params=qf.init(qf2_key, obs, dummy_action),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    
    # Entropy coefficient
    if args.autotune:
        target_entropy = -np.prod((action_dim,)).astype(np.float32)
        entropy_coef = EntropyCoef(args.alpha)
        alpha_state = TrainState.create(
            apply_fn=entropy_coef.apply,
            params=entropy_coef.init(alpha_key),
            target_params=entropy_coef.init(alpha_key),
            tx=optax.adam(learning_rate=args.q_lr),
        )
    else:
        target_entropy = 0.0
        alpha_state = None
    
    # JIT编译
    actor.apply = jax.jit(actor.apply, static_argnames=['training', 'use_sde'])
    qf.apply = jax.jit(qf.apply)

    # 创建replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device="cpu",
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    
    # 定义训练函数
    @jax.jit
    def update_critic(
        actor_state: FlowMLPDecoderTrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        alpha_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        terminations: jnp.ndarray,
        key: jnp.ndarray,
    ):
        key, sample_key = jax.random.split(key, 2)
        
        # 使用FlowMLP + Decoder Actor采样下一步动作（ODE模式）
        next_actions, next_log_prob = actor.apply(
            actor_state.params, next_observations, sample_key, training=False, use_sde=False
        )
        
        # Get current alpha value
        if alpha_state is not None:
            alpha_value = entropy_coef.apply(alpha_state.params)
        else:
            alpha_value = args.alpha
        
        # Compute target Q values
        qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_actions).reshape(-1)
        qf2_next_target = qf.apply(qf2_state.target_params, next_observations, next_actions).reshape(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = (rewards + (1 - terminations) * args.gamma * 
                       (min_qf_next_target - alpha_value * next_log_prob)).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key

    def update_actor_and_alpha(
        actor_state: FlowMLPDecoderTrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        alpha_state: TrainState,
        observations: jnp.ndarray,
        key: jnp.ndarray,
        flowmlp_frozen: bool,
        decoder_frozen: bool,
    ):
        key, sample_key = jax.random.split(key, 2)
        
        # Actor损失函数
        def actor_loss_fn(flowmlp_params, decoder_params):
            # 组合参数
            combined_params = {
                'params': {
                    'flowmlp': flowmlp_params,
                    'decoder': decoder_params
                }
            }
            
            # 使用SDE模式进行训练
            actions, log_prob = actor.apply(
                combined_params, observations, sample_key, training=True, use_sde=True
            )
            
            qf1_pi = qf.apply(qf1_state.params, observations, actions)
            qf2_pi = qf.apply(qf2_state.params, observations, actions)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            
            if alpha_state is not None:
                alpha_value = entropy_coef.apply(alpha_state.params)
            else:
                alpha_value = args.alpha
            
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            return actor_loss, log_prob.mean()

        # 分别计算FlowMLP和Decoder的梯度
        def flowmlp_loss_fn(flowmlp_params):
            loss, entropy = actor_loss_fn(flowmlp_params, actor_state.decoder_state.params)
            return loss, entropy
        
        def decoder_loss_fn(decoder_params):
            loss, entropy = actor_loss_fn(actor_state.flowmlp_state.params, decoder_params)
            return loss, entropy
        
        # 计算梯度
        (actor_loss_value, entropy), flowmlp_grads = jax.value_and_grad(flowmlp_loss_fn, has_aux=True)(actor_state.flowmlp_state.params)
        _, decoder_grads = jax.value_and_grad(decoder_loss_fn, has_aux=True)(actor_state.decoder_state.params)
        
        # 使用JAX条件操作来处理冻结状态
        def apply_flowmlp_grads(state):
            return state.apply_gradients(grads=flowmlp_grads)
        
        def keep_flowmlp_state(state):
            return state
        
        def apply_decoder_grads(state):
            return state.apply_gradients(grads=decoder_grads)
        
        def keep_decoder_state(state):
            return state
        
        # 应用梯度
        flowmlp_state_new = jax.lax.cond(
            flowmlp_frozen,
            keep_flowmlp_state,
            apply_flowmlp_grads,
            actor_state.flowmlp_state
        )
        
        decoder_state_new = jax.lax.cond(
            decoder_frozen,
            keep_decoder_state,
            apply_decoder_grads,
            actor_state.decoder_state
        )
        
        # 更新Actor状态
        actor_state = actor_state.replace(
            flowmlp_state=flowmlp_state_new,
            decoder_state=decoder_state_new
        )
        
        # Update alpha if autotune
        alpha_loss_value = 0.0
        if alpha_state is not None:
            def alpha_loss_fn(alpha_params):
                alpha_value = entropy_coef.apply(alpha_params)
                alpha_loss = (alpha_value * (-entropy - target_entropy)).mean()
                return alpha_loss
            
            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
            alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        # Update target networks
        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
        )
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, args.tau)
        )

        return actor_state, alpha_state, (qf1_state, qf2_state), actor_loss_value, alpha_loss_value, key
    
    # JIT编译更新函数
    update_actor_and_alpha = jax.jit(update_actor_and_alpha, static_argnames=['flowmlp_frozen', 'decoder_frozen'])

    start_time = time.time()
    
    # 跟踪episode统计
    episode_returns = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    completed_episodes = 0
    all_episode_returns = []
    
    log.info("开始SAC训练（FlowMLP + Trainable Transformer Decoder）")
    log.info("训练策略: 预训练FlowMLP作为velocity预测器，可训练Transformer作为Decoder，支持分离优化")
    if args.load_pretrained and os.path.exists(args.checkpoint_path):
        log.info("初始化状态: FlowMLP使用预训练权重，Decoder初始化为恒等映射")
        log.info("期望行为: 初始episode return应该接近预训练模型的性能")
    else:
        log.info("初始化状态: FlowMLP和Decoder都使用随机初始化（Decoder仍初始化为恒等映射）")
    
    for global_step in range(args.total_timesteps):
        # 动作选择
        if global_step < args.learning_starts:
            if args.load_pretrained and os.path.exists(args.checkpoint_path):
                # 使用FlowMLP + Decoder进行确定性推理（类似第三份代码的逻辑）
                key, action_key = jax.random.split(key, 2)
                actions, _ = actor.apply(
                    actor_state.params, obs, action_key, training=False, use_sde=False
                )
                actions = jax.device_get(actions)
                actions = np.array(actions, copy=True)
            else:
                # 随机动作
                actions = np.array([envs.action_space.sample() for _ in range(args.num_envs)])
                if actions.ndim > 2:
                    actions = actions.squeeze(1)
                if actions.ndim == 3:
                    actions = actions[:, 0, :]
        else:
            # 正常训练阶段：使用ODE模式进行环境交互
            key, action_key = jax.random.split(key, 2)
            actions, _ = actor.apply(
                actor_state.params, obs, action_key, training=False, use_sde=False
            )
            actions = jax.device_get(actions)
            actions = np.array(actions, copy=True)

        # 执行动作
        next_obs_venv, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs = process_obs(next_obs_venv, main_obs_key)

        # 更新episode统计
        episode_returns += rewards
        episode_lengths += 1

        # 记录完成的episodes
        for env_idx in range(args.num_envs):
            if terminations[env_idx] or truncations[env_idx]:
                all_episode_returns.append(episode_returns[env_idx])
                writer.add_scalar("charts/episodic_return", episode_returns[env_idx], global_step)
                writer.add_scalar("charts/episodic_length", episode_lengths[env_idx], global_step)
                completed_episodes += 1
                
                # 每20个episode打印一次进度
                if completed_episodes % 20 == 0:
                    recent_returns = np.array(all_episode_returns[-20:])
                    log.info(f"Episodes: {completed_episodes}, Recent 20 mean return: {np.mean(recent_returns):.2f} (FlowMLP+Decoder)")
                
                episode_returns[env_idx] = 0
                episode_lengths[env_idx] = 0

        # 保存数据到replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and hasattr(infos, '__getitem__') and 'final_observation' in infos:
                final_obs_processed = process_obs(infos["final_observation"], main_obs_key)
                real_next_obs[idx] = final_obs_processed[idx]
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # 训练
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # 转换为JAX数组
            observations = jnp.array(data.observations.numpy())
            actions = jnp.array(data.actions.numpy())
            next_observations = jnp.array(data.next_observations.numpy())
            rewards = jnp.array(data.rewards.flatten().numpy())
            terminations = jnp.array(data.dones.flatten().numpy())
            
            # 更新critic
            (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                alpha_state,
                observations,
                actions,
                next_observations,
                rewards,
                terminations,
                key,
            )

            # 更新actor（分离优化FlowMLP和Decoder）
            key, actor_update_key = jax.random.split(key)
            
            # 计算冻结状态
            flowmlp_frozen = global_step < args.flowmlp_freeze_steps
            decoder_frozen = global_step < args.decoder_freeze_steps
            
            actor_state, alpha_state, (qf1_state, qf2_state), actor_loss_value, alpha_loss_value, key = update_actor_and_alpha(
                actor_state,
                qf1_state,
                qf2_state,
                alpha_state,
                observations,
                actor_update_key,
                flowmlp_frozen,
                decoder_frozen,
            )

            # 记录损失
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", float(qf1_loss_value), global_step)
                writer.add_scalar("losses/qf2_loss", float(qf2_loss_value), global_step)
                writer.add_scalar("losses/qf1_values", float(qf1_a_values), global_step)
                writer.add_scalar("losses/qf2_values", float(qf2_a_values), global_step)
                writer.add_scalar("losses/actor_loss", float(actor_loss_value), global_step)
                
                # 记录学习率
                current_flowmlp_lr = flowmlp_lr_schedule(global_step)
                current_decoder_lr = decoder_lr_schedule(global_step)
                writer.add_scalar("learning_rates/flowmlp_lr", float(jax.device_get(current_flowmlp_lr)), global_step)
                writer.add_scalar("learning_rates/decoder_lr", float(jax.device_get(current_decoder_lr)), global_step)
                
                # 记录冻结状态
                writer.add_scalar("training_status/flowmlp_frozen", 1 if flowmlp_frozen else 0, global_step)
                writer.add_scalar("training_status/decoder_frozen", 1 if decoder_frozen else 0, global_step)
                
                # 记录架构信息
                writer.add_scalar("architecture/use_decoder", 1 if args.use_decoder else 0, global_step)
                writer.add_scalar("architecture/decoder_layers", args.decoder_num_layers, global_step)
                
                if args.use_poly_squash:
                    writer.add_scalar("training_status/poly_order", args.poly_order, global_step)
                
                if args.autotune:
                    current_alpha = entropy_coef.apply(alpha_state.params)
                    writer.add_scalar("losses/alpha", float(current_alpha), global_step)
                    writer.add_scalar("losses/alpha_loss", float(alpha_loss_value), global_step)
                else:
                    writer.add_scalar("losses/alpha", args.alpha, global_step)
                
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                
                if global_step % 5000 == 0:
                    print(f"Step {global_step}/{args.total_timesteps}, SPS: {sps}, Episodes: {completed_episodes} (FlowMLP+Decoder)")

    # 保存模型
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes([
                    actor_state.flowmlp_state.params,
                    actor_state.decoder_state.params,
                    qf1_state.params,
                    qf2_state.params,
                    alpha_state.params if alpha_state is not None else None,
                ])
            )
        print(f"Model saved to {model_path}")

    # 输出最终统计
    if len(all_episode_returns) > 0:
        final_returns = np.array(all_episode_returns)
        log.info(f"训练完成！总episodes: {len(final_returns)}, "
               f"平均回报: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")

    envs.close()
    writer.close()
    
    log.info("SAC FlowMLP + Trainable Transformer Decoder 训练完成！")
    log.info("核心架构特性:")
    log.info("  1. FlowMLP网络 - 保持与预训练模型完全一致的结构和逻辑")
    log.info("  2. Trainable Transformer Decoder - 从恒等映射开始，逐渐学习改进")
    log.info("  3. 类似第三份代码的初始化策略 - 确保初始性能与预训练模型一致")
    log.info("  4. SAC兼容的随机性 - Decoder中集成随机采样，支持探索-利用平衡")
    log.info("  5. SDE/ODE混合模式 - 训练时使用SDE增强探索，推理时使用ODE稳定执行")
    log.info("  6. 分离优化策略 - FlowMLP和Decoder可使用不同学习率和冻结策略")
    log.info("  7. 端到端可微分 - 整个流程支持梯度反向传播")
    
    log.info(f"\n训练配置总结:")
    log.info(f"  - 环境: {args.env_id}")
    log.info(f"  - FlowMLP学习率: {args.flowmlp_lr} (冻结步数: {args.flowmlp_freeze_steps})")
    log.info(f"  - Decoder学习率: {args.decoder_lr} (层数: {args.decoder_num_layers})")
    log.info(f"  - SDE强度: {args.sde_sigma}")
    log.info(f"  - Poly-Tanh阶数: {args.poly_order if args.use_poly_squash else 'Disabled'}")
    log.info(f"  - 预训练模型: {'Loaded' if args.load_pretrained else 'Random Init'}")
    
    log.info(f"\n关键改进:")
    log.info(f"  - 保持FlowMLP的原始velocity输出逻辑")
    log.info(f"  - Decoder初始化为真正的恒等映射（eye matrix + zero bias）")
    log.info(f"  - 在未训练时，整个网络行为与原始预训练FlowMLP完全一致")
    log.info(f"  - 训练过程中Decoder逐步学习改进velocity，提升策略性能")
    log.info(f"  - 支持FlowMLP冻结训练，专注于优化Decoder参数")
    
    if args.load_pretrained:
        log.info(f"\n预训练集成效果:")
        log.info(f"  - 初始性能应该与第三份代码的评估结果相近")
        log.info(f"  - 通过SAC训练，Decoder将学习进一步优化动作生成")
        log.info(f"  - 预期训练过程中性能会稳步提升")
    
    log.info(f"\n架构验证建议:")
    log.info(f"  1. 检查初始episode return是否接近预训练模型的性能")
    log.info(f"  2. 观察训练过程中return的提升趋势")
    log.info(f"  3. 监控FlowMLP和Decoder的梯度情况")
    log.info(f"  4. 可以尝试先冻结FlowMLP，只训练Decoder验证架构正确性")
    
    log.info(f"\n🎯 成功指标:")
    if args.load_pretrained:
        log.info(f"  - 前几个episodes的return > 1000 (说明预训练权重加载成功)")
        log.info(f"  - 训练过程中return逐步提升 (说明Decoder在学习改进)")
        log.info(f"  - 最终性能优于纯预训练模型 (说明SAC+Decoder架构有效)")
    else:
        log.info(f"  - 训练过程中return从低值稳步提升")
        log.info(f"  - Decoder学会从恒等映射逐步改进velocity预测")
    
    log.info(f"\n🔧 调试提示:")
    log.info(f"  - 如果初始性能很差，检查Decoder初始化是否为恒等映射")
    log.info(f"  - 如果训练无提升，尝试增大decoder_lr或减少flowmlp_freeze_steps")
    log.info(f"  - 如果性能不稳定，调整sde_sigma或decoder随机性参数")
    log.info(f"  - 监控TensorBoard中的learning rates和frozen status")
    
    log.info(f"\n✅ 代码现在应该能够:")
    log.info(f"  1. 像第三份代码一样保持预训练模型的初始性能")
    log.info(f"  2. 通过可训练的Decoder逐步改进动作生成")
    log.info(f"  3. 支持灵活的训练策略(冻结/解冻不同网络组件)")
    log.info(f"  4. 提供详细的训练监控和调试信息")