# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import time
from dataclasses import dataclass
from typing import Tuple

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    denoising_steps: int = 4
    """number of denoising steps for transformer actor"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation in JAX/Flax"""
    d_model: int
    num_heads: int
    
    @nn.compact
    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]
        
        # Linear projections
        q = nn.Dense(self.d_model, name='q_proj')(query)
        k = nn.Dense(self.d_model, name='k_proj')(key) 
        v = nn.Dense(self.d_model, name='v_proj')(value)
        
        # Reshape for multi-head attention
        head_dim = self.d_model // self.num_heads
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        
        if mask is not None:
            scores = scores + mask
            
        attention_weights = nn.softmax(scores, axis=-1)
        attention_output = jnp.matmul(attention_weights, v)
        
        # Reshape and output projection
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = nn.Dense(self.d_model, name='out_proj')(attention_output)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention"""
    d_model: int
    num_heads: int
    
    @nn.compact
    def __call__(self, tgt, memory, tgt_mask=None):
        # Self-attention
        tgt2 = MultiHeadAttention(self.d_model, self.num_heads, name='self_attn')(
            tgt, tgt, tgt, mask=tgt_mask
        )
        tgt = tgt + tgt2  # Remove Dropout for deterministic behavior
        tgt = nn.LayerNorm()(tgt)
        
        # Cross-attention  
        tgt2 = MultiHeadAttention(self.d_model, self.num_heads, name='cross_attn')(
            tgt, memory, memory
        )
        tgt = tgt + tgt2  # Remove Dropout for deterministic behavior
        tgt = nn.LayerNorm()(tgt)
        
        # Feed-forward
        tgt2 = nn.Sequential([
            nn.Dense(self.d_model * 4),
            nn.gelu,
            nn.Dense(self.d_model),
        ])(tgt)
        tgt = tgt + tgt2  # Remove Dropout for deterministic behavior
        tgt = nn.LayerNorm()(tgt)
        
        return tgt


class FlowMatchingTransformerActor(nn.Module):
    """
    Flow Matching Velocity-based Transformer Actor for SAC
    Key changes:
    1. Uses diagonal mask (each position only sees itself)
    2. Predicts velocities instead of direct actions
    3. Accumulates velocities with DELTA_T to get final action
    4. Maintains stochastic sampling for SAC entropy regularization
    """
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    denoising_steps: int = 4
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2
    log_std_min: float = -5
    log_std_max: float = 2

    @nn.compact
    def __call__(self, obs, key, training=True):
        """
        Flow Matching Transformer forward pass with velocity prediction
        Args:
            obs: observations [batch_size, obs_dim]
            key: random key for sampling
            training: whether in training mode
        Returns:
            action: [batch_size, action_dim] 
            log_prob: [batch_size, 1]
            new_key: updated random key
        """
        batch_size = obs.shape[0]
        
        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps
        
        # Observation encoding (memory/context)
        obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        obs_emb = obs_encoder(obs)  # [batch_size, d_model]
        obs_emb = jnp.expand_dims(obs_emb, axis=1)  # [batch_size, 1, d_model]
        
        # Action input projection
        action_proj = nn.Dense(self.d_model, name='action_proj')
        
        # Time embedding
        time_embedding = nn.Sequential([
            nn.Dense(self.d_model // 4),
            nn.silu,
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Transformer decoder layers
        transformer_layers = []
        for i in range(self.n_layers):
            transformer_layers.append(
                TransformerDecoderLayer(self.d_model, self.n_head, name=f'layer_{i}')
            )
        
        # Velocity output heads (changed from action prediction to velocity prediction)
        velocity_mean_head = nn.Dense(self.action_dim, name='velocity_mean_head')
        velocity_log_std_head = nn.Dense(self.action_dim, name='velocity_log_std_head')
        
        # Generate initial random action x0 ~ N(0, I) (stochastic for SAC)
        key, init_key = jax.random.split(key)
        x_current = jax.random.normal(init_key, (batch_size, self.action_dim))
        
        # Calculate x0 log probability under N(0, I) 
        total_log_prob = jax.scipy.stats.norm.logpdf(x_current, 0, 1).sum(axis=-1, keepdims=True)
        
        # Flow Matching iterative velocity-based refinement
        for step in range(self.denoising_steps):
            # Project current action to embedding space
            x_input = jnp.expand_dims(x_current, axis=1)  # [batch_size, 1, action_dim]
            action_emb = action_proj(x_input)  # [batch_size, 1, d_model]
            
            # Add time embedding for current step
            time_value = step / self.denoising_steps
            time_value = jnp.full((batch_size, 1, 1), time_value)
            time_emb = time_embedding(time_value)
            
            # Combine action and time embeddings
            input_emb = action_emb + time_emb
            
            # Flow Matching uses diagonal mask (each position only sees itself)
            # For single position, no mask needed, but keeping for consistency
            diagonal_mask = jnp.full((1, 1), 0.0)  # No masking for single position
            diagonal_mask = jnp.expand_dims(diagonal_mask, axis=(0, 1))  # [1, 1, 1, 1]
            
            # Transformer forward pass
            output = input_emb
            for layer in transformer_layers:
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)
            
            # Predict velocity mean and log_std (key change from original)
            velocity_mean = velocity_mean_head(output[:, 0, :])  # [batch_size, action_dim]
            velocity_log_std = velocity_log_std_head(output[:, 0, :])  # [batch_size, action_dim]
            
            # Clamp log_std to reasonable range
            velocity_log_std = jnp.tanh(velocity_log_std)
            velocity_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (velocity_log_std + 1)
            velocity_std = jnp.exp(velocity_log_std)
            
            # Sample velocity with noise injection (stochastic for SAC)
            key, sample_key = jax.random.split(key)
            velocity_noise = jax.random.normal(sample_key, velocity_mean.shape)
            predicted_velocity = velocity_mean + velocity_std * velocity_noise
            
            # Calculate log probability for this velocity sample
            velocity_log_prob = jax.scipy.stats.norm.logpdf(velocity_noise, 0, 1).sum(axis=-1, keepdims=True)
            total_log_prob += velocity_log_prob
            
            # Flow Matching update: x_{t+1} = x_t + v_t * Δt
            x_current = x_current + predicted_velocity * DELTA_T
        
        # Apply tanh transformation and scaling to final action
        y_t = jnp.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias
        
        # Add Jacobian correction for tanh transformation
        tanh_correction = jnp.log(self.action_scale * (1 - y_t**2) + 1e-6).sum(axis=-1, keepdims=True)
        total_log_prob -= tanh_correction
        
        return action, total_log_prob, key


# Alternative implementation that builds full sequence (like original structure)
class FlowMatchingTransformerActorSequential(nn.Module):
    """
    Alternative Flow Matching implementation that builds full sequence with velocity prediction
    This maintains the original autoregressive structure but uses Flow Matching velocity principles
    """
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    denoising_steps: int = 4
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2
    log_std_min: float = -5
    log_std_max: float = 2

    @nn.compact
    def __call__(self, obs, key, training=True):
        batch_size = obs.shape[0]
        
        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps
        
        # Observation encoding (memory/context)
        obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        obs_emb = obs_encoder(obs)  # [batch_size, d_model]
        obs_emb = jnp.expand_dims(obs_emb, axis=1)  # [batch_size, 1, d_model]
        
        # Action input projection
        action_proj = nn.Dense(self.d_model, name='action_proj')
        
        # Time embedding
        time_embedding = nn.Sequential([
            nn.Dense(self.d_model // 4),
            nn.silu,
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Transformer decoder layers
        transformer_layers = []
        for i in range(self.n_layers):
            transformer_layers.append(
                TransformerDecoderLayer(self.d_model, self.n_head, name=f'layer_{i}')
            )
        
        # Velocity output heads
        velocity_mean_head = nn.Dense(self.action_dim, name='velocity_mean_head')
        velocity_log_std_head = nn.Dense(self.action_dim, name='velocity_log_std_head')
        
        # Generate initial random action x0 ~ N(0, I)
        key, init_key = jax.random.split(key)
        x_current = jax.random.normal(init_key, (batch_size, self.action_dim))
        
        # Initialize action sequence with x0
        action_sequence = jnp.expand_dims(x_current, axis=1)  # [batch_size, 1, action_dim]
        
        # Calculate x0 log probability under N(0, I)
        total_log_prob = jax.scipy.stats.norm.logpdf(x_current, 0, 1).sum(axis=-1, keepdims=True)
        
        # Flow Matching iterative generation with velocity prediction
        for step in range(self.denoising_steps):
            seq_len = action_sequence.shape[1]
            
            # Project action sequence
            action_emb = action_proj(action_sequence)  # [batch_size, seq_len, d_model]
            
            # Add time embedding - Flow Matching style
            time_values = jnp.arange(seq_len, dtype=jnp.float32) / self.denoising_steps
            time_values = jnp.expand_dims(time_values, axis=0)  # [1, seq_len]
            time_values = jnp.expand_dims(time_values, axis=-1)  # [1, seq_len, 1]
            time_values = jnp.broadcast_to(time_values, (batch_size, seq_len, 1))
            
            time_emb = time_embedding(time_values)
            
            # Combine action and time embeddings
            input_emb = action_emb + time_emb
            
            # Flow Matching uses diagonal mask (each position only sees itself)
            diagonal_mask = jnp.full((seq_len, seq_len), -jnp.inf)
            diagonal_mask = diagonal_mask.at[jnp.diag_indices(seq_len)].set(0.0)
            diagonal_mask = jnp.expand_dims(diagonal_mask, axis=(0, 1))  # [1, 1, seq_len, seq_len]
            
            # Transformer forward pass
            output = input_emb
            for layer in transformer_layers:
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)
            
            # Predict velocity for the last position
            velocity_mean = velocity_mean_head(output[:, -1, :])  # [batch_size, action_dim]
            velocity_log_std = velocity_log_std_head(output[:, -1, :])  # [batch_size, action_dim]
            
            # Clamp log_std to reasonable range
            velocity_log_std = jnp.tanh(velocity_log_std)
            velocity_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (velocity_log_std + 1)
            velocity_std = jnp.exp(velocity_log_std)
            
            # Sample velocity with noise injection (stochastic for SAC)
            key, sample_key = jax.random.split(key)
            velocity_noise = jax.random.normal(sample_key, velocity_mean.shape)
            predicted_velocity = velocity_mean + velocity_std * velocity_noise
            
            # Calculate log probability for this velocity sample
            velocity_log_prob = jax.scipy.stats.norm.logpdf(velocity_noise, 0, 1).sum(axis=-1, keepdims=True)
            total_log_prob += velocity_log_prob
            
            # Flow Matching update: x_{t+1} = x_t + v_t * Δt
            x_next = x_current + predicted_velocity * DELTA_T
            
            # Append to sequence for next iteration
            action_sequence = jnp.concatenate([
                action_sequence, 
                jnp.expand_dims(x_next, axis=1)
            ], axis=1)
            
            # Update current action
            x_current = x_next
        
        # Final action is the last generated action
        final_action_raw = x_current
        
        # Apply tanh transformation and scaling
        y_t = jnp.tanh(final_action_raw)
        action = y_t * self.action_scale + self.action_bias
        
        # Add Jacobian correction for tanh transformation
        tanh_correction = jnp.log(self.action_scale * (1 - y_t**2) + 1e-6).sum(axis=-1, keepdims=True)
        total_log_prob -= tanh_correction
        
        return action, total_log_prob, key


# Use the simpler version as the main actor
TransformerActor = FlowMatchingTransformerActor


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}_flow_matching_steps{args.denoising_steps}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, qf2_key, alpha_key = jax.random.split(key, 5)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    # Initialize Flow Matching Transformer Actor
    actor = TransformerActor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
        denoising_steps=args.denoising_steps,
    )
    
    # Initialize actor with obs only (like original SAC)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs, key, training=True),
        target_params=actor.init(actor_key, obs, key, training=True),
        tx=optax.adam(learning_rate=args.policy_lr),
    )
    
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    # Entropy coefficient
    if args.autotune:
        target_entropy = -np.prod(envs.single_action_space.shape).astype(np.float32)
        # Set target entropy to 0 for transformer (as in the pytorch version)
        # target_entropy = target_entropy * 0
        # target_entropy = -target_entropy
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

    actor.apply = jax.jit(actor.apply, static_argnames=['training'])
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        alpha_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        # Sample next actions from current policy using Flow Matching transformer
        next_actions, next_log_prob, key = actor.apply(
            actor_state.params, next_observations, key, training=False
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
                       (min_qf_next_target - alpha_value * next_log_prob.reshape(-1))).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key

    @jax.jit
    def update_actor_and_alpha(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        alpha_state: TrainState,
        observations: np.ndarray,
        key: jnp.ndarray,
    ):
        def actor_loss_fn(actor_params):
            actions, log_prob, new_key = actor.apply(actor_params, observations, key, training=False)
            
            qf1_pi = qf.apply(qf1_state.params, observations, actions)
            qf2_pi = qf.apply(qf2_state.params, observations, actions)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            
            if alpha_state is not None:
                alpha_value = entropy_coef.apply(alpha_state.params)
            else:
                alpha_value = args.alpha
            
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            return actor_loss, (log_prob.mean(), new_key)

        (actor_loss_value, (entropy, key)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        
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

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, key = actor.apply(actor_state.params, obs, key, training=False)
            actions = np.array(jax.device_get(actions))

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                alpha_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                key,
            )

            actor_state, alpha_state, (qf1_state, qf2_state), actor_loss_value, alpha_loss_value, key = update_actor_and_alpha(
                actor_state,
                qf1_state,
                qf2_state,
                alpha_state,
                data.observations.numpy(),
                key,
            )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                if args.autotune:
                    current_alpha = entropy_coef.apply(alpha_state.params)
                    writer.add_scalar("losses/alpha", current_alpha.item(), global_step)
                    writer.add_scalar("losses/alpha_loss", alpha_loss_value.item(), global_step)
                else:
                    writer.add_scalar("losses/alpha", args.alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                        qf2_state.params,
                        alpha_state.params if alpha_state is not None else None,
                    ]
                )
            )
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()