#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import time
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import torch 
import tyro
from torch.utils.tensorboard import SummaryWriter
# test
# Input the homework
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

try:
    from env.gym_utils import make_async
except ImportError:
    raise

try:
    from cleanrl_utils.buffers import ReplayBuffer
except ImportError:
    raise


@dataclass
class Args:
    exp_name: str = "sac_flowmlp_jax_separated_opt_poly_tanh"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "sac-flowmlp-jax-separated-opt-poly-tanh"
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
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer (deprecated, use flow_lr)"""
    flow_lr: float = 3e-5 
    """the learning rate of the flow network optimizer"""
    gate_lr: float = 1e-3
    """the learning rate of the gate network optimizer"""
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
    
    flow_freeze_steps: int = 250000
    """number of steps to freeze flow network training (0 = no freezing)"""
    gate_freeze_steps: int = 0
    """number of steps to freeze gate network training (0 = no freezing)"""
    flow_lr_schedule: str = "constant"
    """learning rate schedule for flow network: constant, linear_decay, cosine_decay"""
    gate_lr_schedule: str = "constant"
    """learning rate schedule for gate network: constant, linear_decay, cosine_decay"""
    flow_lr_warmup_steps: int = 400000
    """number of warmup steps for flow network learning rate"""
    gate_lr_warmup_steps: int = 100000
    """number of warmup steps for gate network learning rate"""
    flow_lr_decay_factor: float = 0.1
    """final learning rate factor for flow network decay schedules"""
    gate_lr_decay_factor: float = 0.1
    """final learning rate factor for gate network decay schedules"""
    
    load_pretrained: bool = True
    """whether to load pretrained weights for FlowMLP"""
    checkpoint_path: str = "pretrained_network/state_80_Walker2d.pt"
    """path to the pre-trained FlowMLP checkpoint (only used if load_pretrained=True)"""
    normalization_path: str = "data/gym/walker2d-medium-v2/normalization.npz"
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
    
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    """MLP dimensions for FlowMLP"""
    time_dim: int = 16
    """time embedding dimension"""
    residual_style: bool = True
    """whether to use residual connections"""
    activation_type: str = "ReLU"
    """activation function type"""
    use_layernorm: bool = False
    """whether to use layer normalization"""
    
    sde_sigma: float = 0.5
    """noise strength for SDE sampling during training"""
    gate_hidden_dim: int = 128
    """hidden dimension for gate network"""

    use_poly_squash: bool = True
    """ tanh(poly(x))"""
    poly_order: int = 5
    """ use_poly_squash=True()"""

    def __post_init__(self):
        if self.mlp_dims is None:
            self.mlp_dims = [512, 512, 512]
        if hasattr(self, 'policy_lr') and self.flow_lr == 3e-4:
            self.flow_lr = self.policy_lr


# ==================== Poly-Tanh ====================

def poly_squash_transform(x, order):
    """
     tanh(poly(x)) 
    poly(x) = x + x^3/3 + x^5/5 + ...
    """
    # xpoly(x)
    x = jnp.clip(x, -5.0, 5.0) 
    
    poly_x = jnp.zeros_like(x)
    for i in range(1, order + 1, 2):
        poly_x += (x**i) / i
    
    return jnp.tanh(poly_x)

def poly_derivative(x, order):
    """
    : d/dx [x + x^3/3 + x^5/5 + ...]
    = 1 + x^2 + x^4 + ...
    """
    x = jnp.clip(x, -5.0, 5.0)
    
    poly_deriv = jnp.zeros_like(x)
    for i in range(1, order + 1, 2):
        poly_deriv += x**(i-1)
    
    return poly_deriv

def poly_tanh_log_prob_correction(x, order):
    """
     tanh(poly(x)) log
    log|∂y/∂x| = log|∂tanh(poly(x))/∂x|
    = log|(1 - tanh²(poly(x))) * poly'(x)|
    """
    x = jnp.clip(x, -5.0, 5.0)
    
    #  poly(x)
    poly_x = jnp.zeros_like(x)
    for i in range(1, order + 1, 2):
        poly_x += (x**i) / i
    
    #  poly'(x)
    poly_deriv = poly_derivative(x, order)
    
    #  tanh(poly(x))
    tanh_poly_x = jnp.tanh(poly_x)
    
    # 
    jacobian = (1 - tanh_poly_x**2) * poly_deriv
    
    # logepsilonlog(0)
    return jnp.log(jnp.abs(jacobian) + 1e-6)

# JIT
def create_poly_squash_jit(order):
    """JIT"""
    @jax.jit
    def _poly_squash_jit(x):
        return poly_squash_transform(x, order)
    return _poly_squash_jit

def create_poly_log_prob_correction_jit(order):
    """logJIT"""
    @jax.jit
    def _poly_log_prob_correction_jit(x):
        return poly_tanh_log_prob_correction(x, order)
    return _poly_log_prob_correction_jit


# ==================== JAX/Flax FlowMLP Implementation ====================

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
    activation_type: str = "Tanh"
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
        
        # First layer with pre-activation
        if self.use_layernorm:
            x = nn.LayerNorm(epsilon=1e-06)(x)
        x = activation_fn(x)
        x = nn.Dense(self.hidden_dim, name='l1')(x)
        
        # Second layer with pre-activation
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
        
        # First linear layer
        x = nn.Dense(hidden_dim)(x)
        
        # Residual blocks
        num_residual_blocks = num_hidden_layers // 2
        for i in range(num_residual_blocks):
            x = TwoLayerPreActivationResNetLinearFlax(
                hidden_dim=hidden_dim,
                activation_type=self.activation_type,
                use_layernorm=self.use_layernorm,
                name=f'residual_block_{i}'
            )(x, training=training)
        
        # Final linear layer
        x = nn.Dense(self.dim_list[-1])(x)
        
        # Final activation
        activation_fn = self.get_activation(self.out_activation_type)
        x = activation_fn(x)
        
        return x


class FlowMLPFlax(nn.Module):
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
        
        # Main MLP
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
        **Args**:
            action: (B, Ta, Da)
            time: (B,) or scalar, diffusion step
            cond: dict with key state; more recent obs at the end
                    state: (B, To, Do)
        **Outputs**:
            velocity: (B, Ta, Da)
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
        
        # velocity head
        vel_feature = jnp.concatenate([action, time_emb, cond_emb], axis=-1)
        vel = self.mlp_mean(vel_feature, training=training)
        
        return vel.reshape(B, Ta, Da)


# ====================  ====================

class GateNetworkFlax(nn.Module):
    """ - SDE"""
    action_dim: int
    horizon_steps: int
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, flow_input):
        # hidden layer
        x = nn.Dense(self.hidden_dim)(flow_input)
        x = nn.swish(x)
        
        # 
        x = nn.Dense(
            self.action_dim * self.horizon_steps,
            kernel_init=nn.initializers.zeros,  # 0
            bias_init=nn.initializers.constant(5.0)  # 5.0
        )(x)
        
        # Sigmoid
        x = nn.sigmoid(x)
        
        #  (B, horizon_steps, action_dim)
        return x.reshape(-1, self.horizon_steps, self.action_dim)


# ==================== SDE Actor () ====================

class FlowMLPActorSDE(nn.Module):
    """JAX FlowMLP Actor - SDEPoly-Tanh"""
    obs_dim: int
    action_dim: int
    cond_steps: int = 1
    inference_steps: int = 4
    horizon_steps: int = 4
    denoised_clip_value: float = 3.0
    mlp_dims: List[int] = None
    time_dim: int = 16
    residual_style: bool = True
    activation_type: str = "ReLU"
    use_layernorm: bool = False
    sde_sigma: float = 0.01
    gate_hidden_dim: int = 128
    use_poly_squash: bool = True  # 
    poly_order: int = 5  # 
    
    def setup(self):
        # 
        cond_dim = self.obs_dim * self.cond_steps
        
        # FlowMLP
        model_params = {
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
        
        # FlowMLP
        self.flow_network = FlowMLPFlax(**model_params)
        
        # （SDE）
        self.gate_network = GateNetworkFlax(
            action_dim=self.action_dim,
            horizon_steps=self.horizon_steps,
            hidden_dim=self.gate_hidden_dim
        )
        
        # JITpoly
        if self.use_poly_squash:
            self.poly_squash_jit = create_poly_squash_jit(self.poly_order)
            self.poly_log_prob_correction_jit = create_poly_log_prob_correction_jit(self.poly_order)
    
    def sample_first_point(self, B: int, key):
        """log probability"""
        xt = jax.random.normal(key, (B, self.horizon_steps * self.action_dim))
        log_prob = jax.scipy.stats.norm.logpdf(xt, 0, 1).sum(axis=-1)
        xt = xt.reshape(B, self.horizon_steps, self.action_dim)
        return xt, log_prob
    
    def forward_step(self, xt, t, cond):
        """"""
        obs = cond["state"]
        obs_adjusted = obs
        if obs_adjusted.ndim == 2:
            obs_adjusted = jnp.expand_dims(obs_adjusted, axis=1)
        
        cond_adjusted = {"state": obs_adjusted}
        
        if t.ndim == 2:
            t_1d = t.squeeze(-1)
        else:
            t_1d = t
        
        velocity_output = self.flow_network(xt, t_1d, cond_adjusted, training=True)
        return velocity_output
    
    def compute_gate_values(self, obs_flat, xt_flat):
        """ z"""
        gate_input = jnp.concatenate([obs_flat, xt_flat], axis=-1)
        z = self.gate_network(gate_input)
        return z
    
    def __call__(self, obs, key):
        """SDE - log probabilityPoly-Tanh"""
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)
            single_obs = True
        else:
            single_obs = False
            
        B = obs.shape[0]
        
        # 
        if obs.ndim == 2:
            cond = {"state": jnp.expand_dims(obs, axis=1)}
        else:
            cond = {"state": obs}
        
        # log probability
        key, sample_key = jax.random.split(key)
        xt, log_prob = self.sample_first_point(B, sample_key)
        
        # SDE
        dt = 1.0 / self.inference_steps
        time_steps = jnp.linspace(0, 1 - dt, self.inference_steps)
        
        for i in range(self.inference_steps):
            t_scalar = time_steps[i]
            t_tensor = jnp.full((B,), t_scalar)
            
            # 
            key, noise_key = jax.random.split(key)
            
            # 
            u_t = self.forward_step(xt, t_tensor, cond)
            
            # 
            epsilon = 1e-8
            t_safe = jnp.maximum(t_scalar, epsilon)
            
            # （）
            sde_weight = nn.sigmoid((t_scalar - 0.01) * 100)
            
            # 
            s_t = (t_safe * u_t - xt) / jnp.maximum(1 - t_safe, epsilon)
            
            # SDE
            drift_coef = ((1/t_safe - 1 + (self.sde_sigma**2)/2) * s_t + 
                         (1/t_safe) * xt)
            
            # 
            obs_flat = cond["state"].reshape(B, -1)
            xt_flat = xt.reshape(B, -1)
            z = self.compute_gate_values(obs_flat, xt_flat)
            drift_coef = z * drift_coef
            
            # 
            noise = jax.random.normal(noise_key, xt.shape)
            diffusion_coef = self.sde_sigma * jnp.sqrt(dt) * noise
            
            # 
            deterministic_update = xt + u_t * dt
            stochastic_update = xt + drift_coef * dt + diffusion_coef
            
            # 
            xt = (1 - sde_weight) * deterministic_update + sde_weight * stochastic_update
            
            # log probability
            noise_log_prob = sde_weight * jax.scipy.stats.norm.logpdf(
                noise.reshape(B, -1), 0, self.sde_sigma * jnp.sqrt(dt)
            ).sum(axis=-1)
            log_prob = log_prob + noise_log_prob
            
            # 
            if i < self.inference_steps - 1:
                xt = jnp.clip(xt, -self.denoised_clip_value, self.denoised_clip_value)
            else:
                # Poly-Tanhtanh
                xt_flat = xt.reshape(B, -1)
                
                if self.use_poly_squash:
                    # Poly-Tanh
                    xt_squashed = self.poly_squash_jit(xt_flat)
                    
                    # Poly-Tanhlog
                    poly_log_prob_correction = self.poly_log_prob_correction_jit(xt_flat)
                    log_prob = log_prob - poly_log_prob_correction.sum(axis=-1)
                else:
                    # tanh
                    xt_squashed = jnp.tanh(xt_flat)
                    
                    # tanhlog
                    log_prob = log_prob - jnp.log(1 - xt_squashed**2 + 1e-6).sum(axis=-1)
                
                # 
                xt = xt_squashed.reshape(B, self.horizon_steps, self.action_dim)
        
        # 0
        action = xt[:, 0, :]
        
        if single_obs:
            return action.squeeze(0), log_prob.squeeze(0)
        else:
            return action, log_prob


# ==================== ODE Actor () ====================

class FlowMLPActorODE(nn.Module):
    """JAX FlowMLP Actor - ODEPoly-Tanh"""
    obs_dim: int
    action_dim: int
    cond_steps: int = 1
    inference_steps: int = 4
    horizon_steps: int = 4
    denoised_clip_value: float = 3.0
    mlp_dims: List[int] = None
    time_dim: int = 16
    residual_style: bool = True
    activation_type: str = "ReLU"
    use_layernorm: bool = False
    gate_hidden_dim: int = 128
    use_poly_squash: bool = True  # 
    poly_order: int = 5  # 
    
    def setup(self):
        # 
        cond_dim = self.obs_dim * self.cond_steps
        
        # FlowMLP
        model_params = {
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
        
        # FlowMLP
        self.flow_network = FlowMLPFlax(**model_params)
        
        # （ODE）
        self.gate_network = GateNetworkFlax(
            action_dim=self.action_dim,
            horizon_steps=self.horizon_steps,
            hidden_dim=self.gate_hidden_dim
        )
        
        # JITpoly
        if self.use_poly_squash:
            self.poly_squash_jit = create_poly_squash_jit(self.poly_order)
    
    def sample_first_point(self, B: int, key):
        """"""
        xt = jax.random.normal(key, (B, self.horizon_steps * self.action_dim))
        xt = xt.reshape(B, self.horizon_steps, self.action_dim)
        return xt
    
    def forward_step(self, xt, t, cond):
        """"""
        obs = cond["state"]
        obs_adjusted = obs
        if obs_adjusted.ndim == 2:
            obs_adjusted = jnp.expand_dims(obs_adjusted, axis=1)
        
        cond_adjusted = {"state": obs_adjusted}
        
        if t.ndim == 2:
            t_1d = t.squeeze(-1)
        else:
            t_1d = t
        
        velocity_output = self.flow_network(xt, t_1d, cond_adjusted, training=False)
        return velocity_output
    
    def compute_gate_values(self, obs_flat, xt_flat):
        """ z"""
        gate_input = jnp.concatenate([obs_flat, xt_flat], axis=-1)
        z = self.gate_network(gate_input)
        return z
    
    def __call__(self, obs, key):
        """ODE - Poly-Tanh"""
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)
            single_obs = True
        else:
            single_obs = False
            
        B = obs.shape[0]
        
        # 
        if obs.ndim == 2:
            cond = {"state": jnp.expand_dims(obs, axis=1)}
        else:
            cond = {"state": obs}
        
        # 
        key, sample_key = jax.random.split(key)
        xt = self.sample_first_point(B, sample_key)
        
        # ODE
        dt = 1.0 / self.inference_steps
        time_steps = jnp.linspace(0, 1 - dt, self.inference_steps)
        
        for i in range(self.inference_steps):
            t_scalar = time_steps[i]
            t_tensor = jnp.full((B,), t_scalar)
            
            # 
            u_t = self.forward_step(xt, t_tensor, cond)
            
            # ODE
            drift_coef = u_t
            
            # 
            obs_flat = cond["state"].reshape(B, -1)
            xt_flat = xt.reshape(B, -1)
            z = self.compute_gate_values(obs_flat, xt_flat)
            drift_coef = z * drift_coef
            
            # ODE
            xt = xt + drift_coef * dt
            
            # 
            if i < self.inference_steps - 1:
                xt = jnp.clip(xt, -self.denoised_clip_value, self.denoised_clip_value)
            else:
                # Poly-Tanhtanh
                xt_flat = xt.reshape(B, -1)
                
                if self.use_poly_squash:
                    # Poly-Tanh
                    xt_squashed = self.poly_squash_jit(xt_flat)
                else:
                    # tanh
                    xt_squashed = jnp.tanh(xt_flat)
                
                xt = xt_squashed.reshape(B, self.horizon_steps, self.action_dim)
        
        # 0
        action = xt[:, 0, :]
        
        return action.squeeze(0) if single_obs else action


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


def torch_to_jax_params(torch_state_dict, jax_model, sample_input):
    """Convert PyTorch model parameters to JAX/Flax format"""
    
    # Initialize JAX model to get parameter structure
    key = jax.random.PRNGKey(0)
    action, time, cond = sample_input
    jax_params = jax_model.init(key, action, time, cond)
    
    log.info("Converting PyTorch parameters to JAX...")
    
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
    
    # Replace the parameters in the JAX model
    jax_params['params'] = new_params
    
    return new_params  # FlowMLP


# ==================== Learning Rate Schedulers ====================

def create_lr_schedule(base_lr: float, schedule_type: str, total_steps: int, 
                      warmup_steps: int = 0, decay_factor: float = 0.1):
    """JAX"""
    
    def warmup_schedule(step):
        # JAXPython if
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
        
        # JAX
        decay_steps = total_steps - warmup_steps
        progress = jnp.maximum(0.0, (step - warmup_steps) / decay_steps)
        progress = jnp.minimum(progress, 1.0)
        
        # step < warmup_stepsprogressclamp01
        decay_multiplier = 1.0 - progress * (1.0 - decay_factor)
        
        return base_rate * decay_multiplier
    
    def cosine_decay_schedule(step):
        base_rate = warmup_schedule(step)
        
        # JAX
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
    """SAC Critic - JAX"""
    
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


# ==================== Training State with Separate Networks ====================

class SeparatedTrainState:
    """FlowGate"""
    def __init__(self, flow_state: TrainState, gate_state: TrainState):
        self.flow_state = flow_state
        self.gate_state = gate_state
    
    @property
    def params(self):
        """"""
        return {
            'params': {
                'flow_network': self.flow_state.params,
                'gate_network': self.gate_state.params
            }
        }
    
    def replace(self, flow_state=None, gate_state=None):
        """"""
        return SeparatedTrainState(
            flow_state=flow_state if flow_state is not None else self.flow_state,
            gate_state=gate_state if gate_state is not None else self.gate_state
        )

# SeparatedTrainStateJAX pytree
def _separated_train_state_tree_flatten(state):
    """SeparatedTrainStateJAX"""
    children = (state.flow_state, state.gate_state)
    aux_data = None
    return children, aux_data

def _separated_train_state_tree_unflatten(aux_data, children):
    """SeparatedTrainState"""
    flow_state, gate_state = children
    return SeparatedTrainState(flow_state, gate_state)

# pytree
jax.tree_util.register_pytree_node(
    SeparatedTrainState,
    _separated_train_state_tree_flatten,
    _separated_train_state_tree_unflatten
)


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


# ==================== Utility Functions ====================

def reset_env_all(venv, num_envs, verbose=False, options_venv=None, **kwargs):
    """"""
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
    """"""
    if isinstance(obs_venv, dict):
        obs = obs_venv[main_obs_key]
    else:
        obs = obs_venv
    
    # 2
    if obs.ndim > 2:
        obs = obs.reshape(obs.shape[0], -1)
    
    return obs


def load_pretrained_flowmlp_params(checkpoint_path, flowmlp_config, sample_input):
    """FlowMLPJAX"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    log.info(f"Loading pretrained FlowMLP from: {checkpoint_path}")
    
    # PyTorch
    device = torch.device("cpu")
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        raise
    
    # state_dict
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
    
    # state_dict（'network.'）
    corrected_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('network.', '', 1) if k.startswith('network.') else k
        corrected_state_dict[new_key] = v
    
    log.info(f"Key examples: {list(corrected_state_dict.keys())[:5]}...")
    
    # FlowMLP
    try:
        flowmlp_module = FlowMLPFlax(**flowmlp_config)
        log.info(f"Created FlowMLP with config: {flowmlp_config}")
    except Exception as e:
        log.error(f"Failed to create FlowMLP module: {e}")
        raise
    
    # PyTorchJAX
    try:
        jax_params = torch_to_jax_params(corrected_state_dict, flowmlp_module, sample_input)
        log.info("Pretrained FlowMLP parameters converted to JAX successfully!")
        return jax_params
    except Exception as e:
        log.error(f"Failed to convert parameters: {e}")
        raise


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}_sde{args.sde_sigma}_flow_lr{args.flow_lr}_gate_lr{args.gate_lr}_poly{args.poly_order if args.use_poly_squash else 'off'}__{int(time.time())}"
    
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

    # 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 
    key = jax.random.PRNGKey(args.seed)
    key, actor_sde_key, actor_ode_key, qf1_key, qf2_key, alpha_key, action_key = jax.random.split(key, 7)

    # 
    log.info(f" {args.num_envs} : {args.env_id}")
    log.info(f"SDE: sigma = {args.sde_sigma}")
    log.info(f": hidden_dim = {args.gate_hidden_dim}")
    log.info(f"Flow: {args.flow_lr}, : {args.flow_lr_schedule}, : {args.flow_freeze_steps}")
    log.info(f"Gate: {args.gate_lr}, : {args.gate_lr_schedule}, : {args.gate_freeze_steps}")
    # Poly-Tanh
    if args.use_poly_squash:
        log.info(f"🚀 Poly-Tanh:  = {args.poly_order}")
    else:
        log.info(f"❌ Poly-Tanh: tanh")
    
    # wrapper
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
    
    # 
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
    
    # observation_spaceaction_spacereplay buffer
    observation_space = gym.spaces.Box(
        low=np.full(obs_flat_dim, -np.inf, dtype=np.float32),
        high=np.full(obs_flat_dim, np.inf, dtype=np.float32),
        shape=(obs_flat_dim,), 
        dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1, high=1, shape=(action_dim,), dtype=np.float32
    )
    
    # obs_dim
    obs_dim = obs_flat_dim // args.cond_steps if obs_flat_dim % args.cond_steps == 0 else obs_flat_dim
    
    # SDE Actor（）
    log.info("SDE Actor（Poly-Tanh）")
    actor_sde = FlowMLPActorSDE(
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
        sde_sigma=args.sde_sigma,
        gate_hidden_dim=args.gate_hidden_dim,
        use_poly_squash=args.use_poly_squash,  # 
        poly_order=args.poly_order  # 
    )
    
    # ODE Actor（）
    log.info("ODE Actor（Poly-Tanh）")
    actor_ode = FlowMLPActorODE(
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
        gate_hidden_dim=args.gate_hidden_dim,
        use_poly_squash=args.use_poly_squash,  # 
        poly_order=args.poly_order  # 
    )
    
    # 
    obs_venv = reset_env_all(envs, args.num_envs)
    obs = process_obs(obs_venv, main_obs_key)
    
    # 
    flow_lr_schedule = create_lr_schedule(
        base_lr=args.flow_lr,
        schedule_type=args.flow_lr_schedule,
        total_steps=args.total_timesteps,
        warmup_steps=args.flow_lr_warmup_steps,
        decay_factor=args.flow_lr_decay_factor
    )
    
    gate_lr_schedule = create_lr_schedule(
        base_lr=args.gate_lr,
        schedule_type=args.gate_lr_schedule,
        total_steps=args.total_timesteps,
        warmup_steps=args.gate_lr_warmup_steps,
        decay_factor=args.gate_lr_decay_factor
    )
    
    # 
    flow_tx = optax.adam(learning_rate=flow_lr_schedule)
    gate_tx = optax.adam(learning_rate=gate_lr_schedule)
    
    # actor
    if args.load_pretrained and os.path.exists(args.checkpoint_path):
        log.info("FlowMLP...")
        
        # 
        batch_size = 1
        sample_action = jnp.zeros((batch_size, args.horizon_steps, action_dim))
        sample_time = jnp.zeros((batch_size,))
        sample_cond = {"state": jnp.zeros((batch_size, args.cond_steps, obs_dim))}
        sample_input = (sample_action, sample_time, sample_cond)
        
        # FlowMLP
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
            # 
            pretrained_flowmlp_params = load_pretrained_flowmlp_params(
                args.checkpoint_path, flowmlp_config, sample_input
            )
            
            # SDE ActorFlowGate
            initial_params_sde = actor_sde.init(actor_sde_key, obs, action_key)
            
            # 
            flow_params = pretrained_flowmlp_params  # 
            gate_params = initial_params_sde['params']['gate_network']  # 
            
            flow_state_sde = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_sde.flow_network.apply(params, *args, **kwargs),
                params=flow_params,
                target_params=flow_params,
                tx=flow_tx,
            )
            
            gate_state_sde = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_sde.gate_network.apply(params, *args, **kwargs),
                params=gate_params,
                target_params=gate_params,
                tx=gate_tx,
            )
            
            actor_sde_state = SeparatedTrainState(flow_state_sde, gate_state_sde)
            
            # ODE Actor
            initial_params_ode = actor_ode.init(actor_ode_key, obs, action_key)
            
            flow_state_ode = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_ode.flow_network.apply(params, *args, **kwargs),
                params=flow_params,
                target_params=flow_params,
                tx=flow_tx,
            )
            
            gate_state_ode = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_ode.gate_network.apply(params, *args, **kwargs),
                params=gate_params,
                target_params=gate_params,
                tx=gate_tx,
            )
            
            actor_ode_state = SeparatedTrainState(flow_state_ode, gate_state_ode)
            
            log.info("SDEODE ActorFlowMLP（）")
            
        except Exception as e:
            log.error(f": {e}")
            log.info("...")
            
            # 
            initial_params_sde = actor_sde.init(actor_sde_key, obs, action_key)
            initial_params_ode = actor_ode.init(actor_ode_key, obs, action_key)
            
            # SDE Actor
            flow_state_sde = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_sde.flow_network.apply(params, *args, **kwargs),
                params=initial_params_sde['params']['flow_network'],
                target_params=initial_params_sde['params']['flow_network'],
                tx=flow_tx,
            )
            
            gate_state_sde = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_sde.gate_network.apply(params, *args, **kwargs),
                params=initial_params_sde['params']['gate_network'],
                target_params=initial_params_sde['params']['gate_network'],
                tx=gate_tx,
            )
            
            actor_sde_state = SeparatedTrainState(flow_state_sde, gate_state_sde)
            
            # ODE Actor
            flow_state_ode = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_ode.flow_network.apply(params, *args, **kwargs),
                params=initial_params_ode['params']['flow_network'],
                target_params=initial_params_ode['params']['flow_network'],
                tx=flow_tx,
            )
            
            gate_state_ode = TrainState.create(
                apply_fn=lambda params, *args, **kwargs: actor_ode.gate_network.apply(params, *args, **kwargs),
                params=initial_params_ode['params']['gate_network'],
                target_params=initial_params_ode['params']['gate_network'],
                tx=gate_tx,
            )
            
            actor_ode_state = SeparatedTrainState(flow_state_ode, gate_state_ode)
            
            log.info("Actor")
    else:
        # 
        initial_params_sde = actor_sde.init(actor_sde_key, obs, action_key)
        initial_params_ode = actor_ode.init(actor_ode_key, obs, action_key)
        
        # SDE Actor
        flow_state_sde = TrainState.create(
            apply_fn=lambda params, *args, **kwargs: actor_sde.flow_network.apply(params, *args, **kwargs),
            params=initial_params_sde['params']['flow_network'],
            target_params=initial_params_sde['params']['flow_network'],
            tx=flow_tx,
        )
        
        gate_state_sde = TrainState.create(
            apply_fn=lambda params, *args, **kwargs: actor_sde.gate_network.apply(params, *args, **kwargs),
            params=initial_params_sde['params']['gate_network'],
            target_params=initial_params_sde['params']['gate_network'],
            tx=gate_tx,
        )
        
        actor_sde_state = SeparatedTrainState(flow_state_sde, gate_state_sde)
        
        # ODE Actor
        flow_state_ode = TrainState.create(
            apply_fn=lambda params, *args, **kwargs: actor_ode.flow_network.apply(params, *args, **kwargs),
            params=initial_params_ode['params']['flow_network'],
            target_params=initial_params_ode['params']['flow_network'],
            tx=flow_tx,
        )
        
        gate_state_ode = TrainState.create(
            apply_fn=lambda params, *args, **kwargs: actor_ode.gate_network.apply(params, *args, **kwargs),
            params=initial_params_ode['params']['gate_network'],
            target_params=initial_params_ode['params']['gate_network'],
            tx=gate_tx,
        )
        
        actor_ode_state = SeparatedTrainState(flow_state_ode, gate_state_ode)
        
        log.info("Actor")
    
    # Critic
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
    
    # JIT
    actor_sde.apply = jax.jit(actor_sde.apply)
    actor_ode.apply = jax.jit(actor_ode.apply)
    qf.apply = jax.jit(qf.apply)

    # replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device="cpu",
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    
    # 
    @jax.jit
    def update_critic(
        actor_sde_state: SeparatedTrainState,
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
        
        # SDE Actor（）
        next_actions, next_log_prob = actor_sde.apply(actor_sde_state.params, next_observations, sample_key)
        
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
        actor_sde_state: SeparatedTrainState,
        actor_ode_state: SeparatedTrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        alpha_state: TrainState,
        observations: jnp.ndarray,
        key: jnp.ndarray,
        flow_frozen: bool,
        gate_frozen: bool,
    ):
        key, sample_key = jax.random.split(key, 2)
        
        # Actor
        def actor_loss_fn(flow_params, gate_params):
            # 
            combined_params = {
                'params': {
                    'flow_network': flow_params,
                    'gate_network': gate_params
                }
            }
            
            # SDE Actor
            actions, log_prob = actor_sde.apply(combined_params, observations, sample_key)
            
            qf1_pi = qf.apply(qf1_state.params, observations, actions)
            qf2_pi = qf.apply(qf2_state.params, observations, actions)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            
            if alpha_state is not None:
                alpha_value = entropy_coef.apply(alpha_state.params)
            else:
                alpha_value = args.alpha
            
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            return actor_loss, log_prob.mean()

        # FlowGate
        def flow_loss_fn(flow_params):
            loss, entropy = actor_loss_fn(flow_params, actor_sde_state.gate_state.params)
            return loss, entropy
        
        def gate_loss_fn(gate_params):
            loss, entropy = actor_loss_fn(actor_sde_state.flow_state.params, gate_params)
            return loss, entropy
        
        # 
        (actor_loss_value, entropy), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(actor_sde_state.flow_state.params)
        _, gate_grads = jax.value_and_grad(gate_loss_fn, has_aux=True)(actor_sde_state.gate_state.params)
        
        # JAX
        def apply_flow_grads(state):
            return state.apply_gradients(grads=flow_grads)
        
        def keep_flow_state(state):
            return state
        
        def apply_gate_grads(state):
            return state.apply_gradients(grads=gate_grads)
        
        def keep_gate_state(state):
            return state
        
        # （JAX）
        flow_state_sde_new = jax.lax.cond(
            flow_frozen,
            keep_flow_state,
            apply_flow_grads,
            actor_sde_state.flow_state
        )
        
        gate_state_sde_new = jax.lax.cond(
            gate_frozen,
            keep_gate_state,
            apply_gate_grads,
            actor_sde_state.gate_state
        )
        
        # SDE Actor
        actor_sde_state = actor_sde_state.replace(
            flow_state=flow_state_sde_new,
            gate_state=gate_state_sde_new
        )
        
        # ODE Actor
        flow_state_ode_new = actor_ode_state.flow_state.replace(params=flow_state_sde_new.params)
        gate_state_ode_new = actor_ode_state.gate_state.replace(params=gate_state_sde_new.params)
        
        actor_ode_state = actor_ode_state.replace(
            flow_state=flow_state_ode_new,
            gate_state=gate_state_ode_new
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

        return actor_sde_state, actor_ode_state, alpha_state, (qf1_state, qf2_state), actor_loss_value, alpha_loss_value, key
    
    # JIT
    update_actor_and_alpha = jax.jit(update_actor_and_alpha, static_argnames=['flow_frozen', 'gate_frozen'])

    start_time = time.time()
    
    # episode
    episode_returns = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    completed_episodes = 0
    all_episode_returns = []
    
    log.info("SAC（Flow/Gate + Poly-Tanh）")
    log.info(": SDE ActorODE ActorFlowGate")
    
    for global_step in range(args.total_timesteps):
        # 
        if global_step < args.learning_starts:
            if args.load_pretrained and os.path.exists(args.checkpoint_path):
                # ODE Actor
                key, action_key = jax.random.split(key, 2)
                actions = actor_ode.apply(actor_ode_state.params, obs, action_key)
                actions = jax.device_get(actions)
                actions = np.array(actions, copy=True)
            else:
                # 
                actions = np.array([envs.action_space.sample() for _ in range(args.num_envs)])
                # 
                if actions.ndim > 2:
                    actions = actions.squeeze(1)
                if actions.ndim == 3:
                    actions = actions[:, 0, :]  # 
        else:
            # ODE Actor（）
            key, action_key = jax.random.split(key, 2)
            actions = actor_ode.apply(actor_ode_state.params, obs, action_key)
            actions = jax.device_get(actions)
            actions = np.array(actions, copy=True)

        # 
        next_obs_venv, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs = process_obs(next_obs_venv, main_obs_key)

        # episode
        episode_returns += rewards
        episode_lengths += 1

        # episodes
        for env_idx in range(args.num_envs):
            if terminations[env_idx] or truncations[env_idx]:
                all_episode_returns.append(episode_returns[env_idx])
                writer.add_scalar("charts/episodic_return", episode_returns[env_idx], global_step)
                writer.add_scalar("charts/episodic_length", episode_lengths[env_idx], global_step)
                completed_episodes += 1
                
                # 20episode
                if completed_episodes % 20 == 0:
                    recent_returns = np.array(all_episode_returns[-20:])
                    poly_status = f"Poly{args.poly_order}" if args.use_poly_squash else "Tanh"
                    log.info(f"Episodes: {completed_episodes}, Recent 20 mean return: {np.mean(recent_returns):.2f} ({poly_status})")
                
                episode_returns[env_idx] = 0
                episode_lengths[env_idx] = 0

        # replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and hasattr(infos, '__getitem__') and 'final_observation' in infos:
                final_obs_processed = process_obs(infos["final_observation"], main_obs_key)
                real_next_obs[idx] = final_obs_processed[idx]
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # 
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # JAX
            observations = jnp.array(data.observations.numpy())
            actions = jnp.array(data.actions.numpy())
            next_observations = jnp.array(data.next_observations.numpy())
            rewards = jnp.array(data.rewards.flatten().numpy())
            terminations = jnp.array(data.dones.flatten().numpy())
            
            # critic（SDE Actor）
            (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
                actor_sde_state,
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

            # actoralpha（FlowGate）
            key, actor_update_key = jax.random.split(key)
            
            # JIT
            flow_frozen = global_step < args.flow_freeze_steps
            gate_frozen = global_step < args.gate_freeze_steps
            
            actor_sde_state, actor_ode_state, alpha_state, (qf1_state, qf2_state), actor_loss_value, alpha_loss_value, key = update_actor_and_alpha(
                actor_sde_state,
                actor_ode_state,
                qf1_state,
                qf2_state,
                alpha_state,
                observations,
                actor_update_key,
                flow_frozen,
                gate_frozen,
            )

            # 
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", float(qf1_loss_value), global_step)
                writer.add_scalar("losses/qf2_loss", float(qf2_loss_value), global_step)
                writer.add_scalar("losses/qf1_values", float(qf1_a_values), global_step)
                writer.add_scalar("losses/qf2_values", float(qf2_a_values), global_step)
                writer.add_scalar("losses/actor_loss", float(actor_loss_value), global_step)
                
                # 
                current_flow_lr = flow_lr_schedule(global_step)
                current_gate_lr = gate_lr_schedule(global_step)
                writer.add_scalar("learning_rates/flow_lr", float(jax.device_get(current_flow_lr)), global_step)
                writer.add_scalar("learning_rates/gate_lr", float(jax.device_get(current_gate_lr)), global_step)
                
                # 
                writer.add_scalar("training_status/flow_frozen", 1 if global_step < args.flow_freeze_steps else 0, global_step)
                writer.add_scalar("training_status/gate_frozen", 1 if global_step < args.gate_freeze_steps else 0, global_step)
                
                # Poly-Tanh
                writer.add_scalar("training_status/use_poly_squash", 1 if args.use_poly_squash else 0, global_step)
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
                    poly_status = f"Poly{args.poly_order}" if args.use_poly_squash else "Tanh"
                    print(f"Step {global_step}/{args.total_timesteps}, SPS: {sps}, Episodes: {completed_episodes} ({poly_status})")

    # 
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes([
                    actor_sde_state.flow_state.params,
                    actor_sde_state.gate_state.params,
                    actor_ode_state.flow_state.params,
                    actor_ode_state.gate_state.params,
                    qf1_state.params,
                    qf2_state.params,
                    alpha_state.params if alpha_state is not None else None,
                ])
            )
        print(f"Model saved to {model_path}")

    # 
    if len(all_episode_returns) > 0:
        final_returns = np.array(all_episode_returns)
        poly_status = f"Poly{args.poly_order}" if args.use_poly_squash else "Tanh"
        log.info(f"episodes: {len(final_returns)}, "
               f": {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f} ({poly_status})")

    envs.close()
    writer.close()
    
    log.info("SAC FlowMLP JAX（Flow/Gate + Poly-Tanh）")
    log.info(":")
    log.info("  1. SDE Actor（）")
    log.info("  2. ODE Actor（）")
    log.info("  3. FlowGate")
    log.info("  4. FlowMLP")
    log.info("  5. JAX")
    if args.load_pretrained:
        log.info("  6. FlowMLPfine-tuning")
    else:
        log.info("  6. FlowMLP")
    log.info(f"  7. SDE: sigma = {args.sde_sigma}")
    log.info(f"  8. : hidden_dim = {args.gate_hidden_dim}")
    log.info(f"  9. Flow: lr = {args.flow_lr},  = {args.flow_lr_schedule},  = {args.flow_freeze_steps}")
    log.info(f"  10. Gate: lr = {args.gate_lr},  = {args.gate_lr_schedule},  = {args.gate_freeze_steps}")
    # Poly-Tanh
    if args.use_poly_squash:
        log.info(f"  11. ✅ Poly-Tanh:  = {args.poly_order}")
        log.info(f"      - ")
        log.info(f"      - log")
        log.info(f"      -  ([-5,5])")
        log.info(f"      - JIT")
    else:
        log.info(f"  11. ❌ Poly-Tanh: tanh")
    
    log.info("\n🔧 Poly-Tanh:")
    if args.use_poly_squash:
        log.info(f"  - : poly(x) = x + x³/3 + x⁵/5 + ... (={args.poly_order})")
        log.info(f"  - : tanh(poly(x))tanh(x)")
        log.info(f"  - log: ")
        log.info(f"  - : SDEODE")
        log.info(f"  - :  + epsilon")
    else:
        log.info(f"  - tanh(x)")
        log.info(f"  - ")
    
    log.info(f"\n📊 :")
    log.info(f"  - : {args.env_id}")
    log.info(f"  - : {args.total_timesteps}")
    log.info(f"  - : {args.batch_size}")
    log.info(f"  - : {args.inference_steps}")
    log.info(f"  - : {args.horizon_steps}")
    log.info(f"  - : {args.cond_steps}")
    log.info(f"  - MLP: {args.mlp_dims}")
    log.info(f"  - : {args.activation_type}")
    log.info(f"  - : {args.residual_style}")
    
    if args.use_poly_squash:
        log.info(f"\n🎯 Poly-Tanh:")
        log.info(f"  1. : (3,5,7)")
        log.info(f"  2.  {args.poly_order}: {'' if 3 <= args.poly_order <= 7 else ''}")
        log.info(f"  3. ")
        log.info(f"  4. ")
        log.info(f"  5. SDE sigma-")