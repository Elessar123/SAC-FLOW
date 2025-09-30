# CrossQ implementation with Flow-based Actor - Optimized Version
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import random
import time
import functools
from dataclasses import dataclass
from collections import deque
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from typing import Any, Callable, Optional
from flax.linen.module import Module, compact, merge_param
from jax.nn import initializers
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
from flax.linen.initializers import zeros, constant
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = "large-param"
    """the name of this experiment"""
    seed: int = 3407
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Humanoid-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0  # CrossQ: use complete target network replacement
    """target smoothing coefficient (CrossQ uses 1.0)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 50000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    log_freq: int = 5000
    """how often to log scores"""
    
    # CrossQ specific parameters
    policy_delay: int = 3
    """policy is updated after this many critic updates (CrossQ default)"""
    use_batch_norm: bool = True
    """whether to use batch norm in networks (CrossQ default)"""
    batch_norm_momentum: float = 0.99
    """batch norm momentum (CrossQ default)"""
    n_critics: int = 2
    """number of critics to use"""
    crossq_style: bool = True
    """use CrossQ joint forward pass"""

    # Flow specific arguments
    denoising_steps: int = 4
    """number of denoising steps for flow matching"""

    # wandb
    wandb_project_name: str = "sacflow-fromscratch-" + env_id
    """the wandb's project name"""
    wandb_entity: str = "yushuang20010911"
    """the entity (team) of wandb's project"""


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

class BatchRenorm(nn.Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation: 
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Any = None
    param_dtype: Any = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = initializers.zeros
    scale_init: Callable = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable('batch_stats', 'r_max', lambda s: s, 3)
        d_max = self.variable('batch_stats', 'd_max', lambda s: s, 5)
        steps = self.variable('batch_stats', 'steps', lambda s: s, 0)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray, training: bool = True):
        x = jnp.concatenate([x, a], -1)
        
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
        
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
            
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
            
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    n_critics: int = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, training: bool = True):
        vmap_critic = nn.vmap(
            QNetwork,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
        )(obs, action, training)
        return q_values

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for time steps."""
    dim: int
    
    @nn.compact
    def __call__(self, t):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

# MODIFIED: Using the GRU-style Flow Actor from the first script
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    obs_dim: int
    denoising_steps: int = 4
    log_std_min: float = -5
    log_std_max: float = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    time_emb_dim: int = 32
    hidden_dim: int = 128

    def setup(self):
        self.time_mlp = nn.Sequential([
            SinusoidalPosEmb(self.time_emb_dim),
            nn.Dense(self.time_emb_dim * 2),
            nn.swish,
            nn.Dense(self.time_emb_dim),
        ])
        
        self.gate_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.swish,
            nn.Dense(self.action_dim, 
                     kernel_init=zeros,
                     bias_init=constant(5.0)),
        ])
        
        self.candidate_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.swish,
            nn.Dense(self.action_dim),
        ])
        
        self.fc_logstd = nn.Sequential([
            nn.Dense(512),
            nn.relu,
            nn.Dense(512), 
            nn.relu,
            nn.Dense(self.action_dim),
        ])
        
        if self.use_batch_norm:
            self.bn = BatchRenorm(momentum=self.batch_norm_momentum)

    def flow_step(self, obs, x_prev, t, training: bool):
        if t.ndim == 1: 
            t = jnp.expand_dims(t, -1)
        
        time_emb = self.time_mlp(t)
        
        net_input = jnp.concatenate([obs, x_prev], axis=-1)
        if self.use_batch_norm:
            net_input = self.bn(net_input, use_running_average=not training)
        
        net_input = jnp.concatenate([net_input, time_emb], axis=-1)

        z = nn.sigmoid(self.gate_net(net_input))
        h_tilde = self.candidate_net(net_input)
        vector_field = z * (h_tilde - x_prev)
        
        log_std = self.fc_logstd(obs)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        return vector_field, log_std

    @nn.compact
    def __call__(self, obs, key, training: bool = True):
        batch_size = obs.shape[0]
        
        dt = 1.0 / self.denoising_steps
        time_steps = jnp.linspace(0, 1 - dt, self.denoising_steps)
        
        key, subkey = jax.random.split(key)
        x_current = jax.random.normal(subkey, (batch_size, self.action_dim))
        
        total_log_prob = jnp.sum(-0.5 * x_current**2 - 0.5 * jnp.log(2 * jnp.pi), axis=1, keepdims=True)
        
        for step in range(self.denoising_steps):
            t = jnp.full((batch_size, 1), time_steps[step])
            
            u, log_std = self.flow_step(obs, x_current, t, training=training)
            std = jnp.exp(log_std)
            
            mean_next = x_current + u * dt
            
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, mean_next.shape)
            x_current = mean_next + std * noise
            
            step_log_prob = jnp.sum(-0.5 * ((x_current - mean_next) / std)**2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std), axis=1, keepdims=True)
            total_log_prob += step_log_prob
        
        y_t = jnp.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias
        
        tanh_correction = jnp.sum(jnp.log(self.action_scale * (1 - y_t**2) + 1e-6), axis=1, keepdims=True)
        total_log_prob -= tanh_correction
        
        return action, total_log_prob


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__crossq_gru_flow__{args.exp_name}__{args.seed}__steps{args.denoising_steps}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            group="crossq_gru_flow"
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf_key, alpha_key = jax.random.split(key, 4)

    # Environment setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )
    obs, _ = envs.reset(seed=args.seed)

    # Actor instantiation
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
        obs_dim=np.prod(envs.single_observation_space.shape),
        denoising_steps=args.denoising_steps,
        use_batch_norm=args.use_batch_norm,
        batch_norm_momentum=args.batch_norm_momentum,
        time_emb_dim=32,
        hidden_dim=512
    )
    
    # Actor initialization
    dummy_obs = obs
    dummy_key = jax.random.PRNGKey(0)
    if args.use_batch_norm:
        actor_variables = actor.init(actor_key, dummy_obs, key=dummy_key, training=True)
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_variables['params'],
            target_params=actor_variables['params'],
            batch_stats=actor_variables.get('batch_stats', {}),
            target_batch_stats=actor_variables.get('batch_stats', {}),
            tx=optax.adam(learning_rate=args.policy_lr, b1=0.5),
        )
    else:
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, dummy_obs, key=dummy_key, training=False)['params'],
            target_params=actor.init(actor_key, dummy_obs, key=dummy_key, training=False)['params'],
            batch_stats={},
            target_batch_stats={},
            tx=optax.adam(learning_rate=args.policy_lr, b1=0.5),
        )
    
    # Critic instantiation and initialization
    qf = VectorCritic(
        n_critics=args.n_critics,
        use_batch_norm=args.use_batch_norm, 
        batch_norm_momentum=args.batch_norm_momentum
    )
    dummy_action = envs.action_space.sample()
    if args.use_batch_norm:
        qf_variables = qf.init(qf_key, dummy_obs, dummy_action, training=True)
        qf_state = TrainState.create(
            apply_fn=qf.apply,
            params=qf_variables['params'],
            target_params=qf_variables['params'],
            batch_stats=qf_variables.get('batch_stats', {}),
            target_batch_stats=qf_variables.get('batch_stats', {}),
            tx=optax.adam(learning_rate=args.q_lr, b1=0.5),
        )
    else:
        qf_state = TrainState.create(
            apply_fn=qf.apply,
            params=qf.init(qf_key, dummy_obs, dummy_action, training=False)['params'],
            target_params=qf.init(qf_key, dummy_obs, dummy_action, training=False)['params'],
            batch_stats={},
            target_batch_stats={},
            tx=optax.adam(learning_rate=args.q_lr, b1=0.5),
        )

    # Entropy coefficient setup
    if args.autotune:
        target_entropy = -np.prod(envs.single_action_space.shape).astype(np.float32)
        target_entropy = target_entropy * 0
        entropy_coef = EntropyCoef(args.alpha)
        alpha_state = TrainState.create(
            apply_fn=entropy_coef.apply,
            params=entropy_coef.init(alpha_key)['params'],
            target_params=entropy_coef.init(alpha_key)['params'],
            batch_stats={},
            target_batch_stats={},
            tx=optax.adam(learning_rate=args.q_lr, b1=0.5),
        )
    else:
        alpha_state = None

    # 打印参数数量
    actor_params = sum(x.size for x in jax.tree_util.tree_leaves(actor_state.params))
    qf_params = sum(x.size for x in jax.tree_util.tree_leaves(qf_state.params))
    print("!!================================================")
    print(f"Actor parameters: {actor_params:,}, Critic parameters: {qf_params:,}")
    print("!!================================================")

    n_updates = 0

    # JIT compiled functions for actor
    @jax.jit
    def actor_apply_train(params, batch_stats, obs, key):
        if args.use_batch_norm:
            return actor.apply(
                {'params': params, 'batch_stats': batch_stats}, 
                obs, key=key, training=True, mutable=['batch_stats']
            )
        else:
            return actor.apply({'params': params}, obs, key=key, training=True), {}

    @jax.jit   
    def actor_apply_inference(params, batch_stats, obs, key):
        if args.use_batch_norm:
            return actor.apply(
                {'params': params, 'batch_stats': batch_stats}, 
                obs, key=key, training=False
            )
        else:
            return actor.apply(params, obs, key=key, training=False)

    # JIT compiled functions for critic
    @jax.jit
    def qf_apply_train(params, batch_stats, obs, action):
        if args.use_batch_norm:
            return qf.apply(
                {'params': params, 'batch_stats': batch_stats}, 
                obs, action, training=True, mutable=['batch_stats']
            )
        else:
            return qf.apply({'params': params}, obs, action, training=True), {}

    @jax.jit
    def qf_apply_inference(params, batch_stats, obs, action):
        if args.use_batch_norm:
            return qf.apply(
                {'params': params, 'batch_stats': batch_stats}, 
                obs, action, training=False
            )
        else:
            return qf.apply(params, obs, action, training=False)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf_state: TrainState,
        alpha_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, sample_key = jax.random.split(key, 2)
        
        next_actions, next_log_prob = actor_apply_inference(
            actor_state.params, actor_state.batch_stats, 
            next_observations, sample_key
        )
            
        if alpha_state is not None:
            alpha_value = entropy_coef.apply({'params': alpha_state.params})
        else:
            alpha_value = args.alpha

        def mse_loss(params, batch_stats):
            if not args.crossq_style:
                # Standard SAC: separate forward passes
                next_q_values = qf_apply_inference(
                    qf_state.target_params, qf_state.target_batch_stats,
                    next_observations, next_actions
                )
                current_q_values, new_batch_stats_dict = qf_apply_train(
                    params, batch_stats, observations, actions
                )
                new_batch_stats = new_batch_stats_dict.get('batch_stats', {})
            else:
                # CrossQ: Joint forward pass
                cat_observations = jnp.concatenate([observations, next_observations], axis=0)
                cat_actions = jnp.concatenate([actions, next_actions], axis=0)
                
                catted_q_values, new_batch_stats_dict = qf_apply_train(
                    params, batch_stats, cat_observations, cat_actions
                )
                new_batch_stats = new_batch_stats_dict.get('batch_stats', {})
                
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
            
            next_q_values = jnp.min(next_q_values, axis=0)
            next_q_values = next_q_values - alpha_value * next_log_prob
            target_q_values = rewards.reshape(-1, 1) + (1 - terminations.reshape(-1, 1)) * args.gamma * next_q_values
            
            loss = 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum()
            
            return loss, (current_q_values, next_q_values, new_batch_stats)
        
        (qf_loss_value, (current_q_values, next_q_values, new_batch_stats)), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, qf_state.batch_stats)
        
        qf_state = qf_state.apply_gradients(grads=grads)
        if args.use_batch_norm:
            qf_state = qf_state.replace(batch_stats=new_batch_stats)

        return qf_state, qf_loss_value, current_q_values.mean(), next_q_values.mean(), key

    @jax.jit
    def update_actor_and_alpha(
        actor_state: TrainState,
        qf_state: TrainState,
        alpha_state: TrainState,
        observations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, sample_key = jax.random.split(key, 2)
        
        def actor_loss_fn(actor_params, actor_batch_stats):
            (actions, log_prob), new_actor_batch_stats_dict = actor_apply_train(
                actor_params, actor_batch_stats, observations, sample_key
            )
            
            qf_pi = qf_apply_inference(
                qf_state.params, qf_state.batch_stats, observations, actions
            )
                
            min_qf_pi = jnp.min(qf_pi, axis=0)
            
            if alpha_state is not None:
                alpha_value = entropy_coef.apply({'params': alpha_state.params})
            else:
                alpha_value = args.alpha
            
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            new_batch_stats = new_actor_batch_stats_dict.get('batch_stats', {})
            return actor_loss, (log_prob.mean(), new_batch_stats)

        (actor_loss_value, (entropy, new_actor_batch_stats)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, actor_state.batch_stats)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        if args.use_batch_norm and new_actor_batch_stats:
            actor_state = actor_state.replace(batch_stats=new_actor_batch_stats)
        
        alpha_loss_value = 0.0
        if alpha_state is not None:
            def alpha_loss_fn(alpha_params):
                alpha_value = entropy_coef.apply({'params': alpha_params})
                alpha_loss = (alpha_value * (-entropy - target_entropy)).mean()
                return alpha_loss
            
            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
            alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return actor_state, alpha_state, actor_loss_value, alpha_loss_value, key

    @jax.jit
    def update_target(qf_state: TrainState, tau: float):
        qf_state = qf_state.replace(
            target_params=jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                qf_state.target_params, qf_state.params
            ),
            target_batch_stats=jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                qf_state.target_batch_stats, qf_state.batch_stats
            )
        )
        return qf_state

    start_time = time.time()
    
    completed_episodes = 0
    all_episode_returns = []
    log_buffer = {}
    
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            key, sample_key = jax.random.split(key, 2)
            actions, _ = actor_apply_inference(
                actor_state.params, actor_state.batch_stats, obs, sample_key
            )
            actions = np.array(jax.device_get(actions))

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info: continue
                episode_return = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                all_episode_returns.append(episode_return)
                completed_episodes += 1
                
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                
                if args.track:
                    log_buffer.setdefault("charts/episodic_return", deque(maxlen=20)).append(episode_return)
                    log_buffer.setdefault("charts/episodic_length", deque(maxlen=20)).append(episode_length)
                
                if completed_episodes % 20 == 0:
                    recent_returns = np.array(all_episode_returns[-20:])
                    recent_mean = np.mean(recent_returns)
                    print(f"Episodes: {completed_episodes}, Recent 20 mean return: {recent_mean:.2f}")
                    
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            qf_state, qf_loss_value, qf_a_values, next_q_values, key = update_critic(
                actor_state,
                qf_state,
                alpha_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                key,
            )

            qf_state = update_target(qf_state, args.tau)
            n_updates += 1

            actor_loss_value = 0.0
            alpha_loss_value = 0.0
            if n_updates % args.policy_delay == 0:
                actor_state, alpha_state, actor_loss_value, alpha_loss_value, key = update_actor_and_alpha(
                    actor_state,
                    qf_state,
                    alpha_state,
                    data.observations.numpy(),
                    key,
                )
            
            if args.track:
                log_buffer.setdefault("losses/qf_loss", deque(maxlen=20)).append(qf_loss_value.item())
                log_buffer.setdefault("losses/qf_values", deque(maxlen=20)).append(qf_a_values.item())
                log_buffer.setdefault("losses/next_q_values", deque(maxlen=20)).append(next_q_values.item())
                log_buffer.setdefault("losses/actor_loss", deque(maxlen=20)).append(actor_loss_value.item() if isinstance(actor_loss_value, jnp.ndarray) else actor_loss_value)
                alpha_value = args.alpha
                if args.autotune and alpha_state is not None:
                    current_alpha = entropy_coef.apply({'params': alpha_state.params})
                    alpha_value = current_alpha.item()
                    log_buffer.setdefault("losses/alpha_loss", deque(maxlen=20)).append(alpha_loss_value.item() if isinstance(alpha_loss_value, jnp.ndarray) else alpha_loss_value)
                log_buffer.setdefault("losses/alpha", deque(maxlen=20)).append(alpha_value)
                
            if global_step % args.log_freq == 0:
                writer.add_scalar("losses/qf_loss", qf_loss_value.item(), global_step)
                writer.add_scalar("losses/qf_values", qf_a_values.item(), global_step)
                writer.add_scalar("losses/next_q_values", next_q_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item() if isinstance(actor_loss_value, jnp.ndarray) else actor_loss_value, global_step)
                
                alpha_value = args.alpha
                if args.autotune and alpha_state is not None:
                    current_alpha = entropy_coef.apply({'params': alpha_state.params})
                    alpha_value = current_alpha.item()
                    writer.add_scalar("losses/alpha", alpha_value, global_step)
                    writer.add_scalar("losses/alpha_loss", alpha_loss_value.item() if isinstance(alpha_loss_value, jnp.ndarray) else alpha_loss_value, global_step)
                else:
                    writer.add_scalar("losses/alpha", alpha_value, global_step)
                
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)
                
                # 同时记录到wandb
                if args.track:
                    avg_logs = {key: np.mean(log_buffer[key]) for key in log_buffer.keys()}
                    avg_logs["charts/SPS"] = sps
                    wandb.log(avg_logs, step=global_step)
    
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf_state.params,
                        alpha_state.params if alpha_state is not None else None,
                    ]
                )
            )
        print(f"model saved to {model_path}")

    # 输出最终统计
    if len(all_episode_returns) > 0:
        final_returns = np.array(all_episode_returns)
        
        # 计算最终统计
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        max_return = np.max(final_returns)
        min_return = np.min(final_returns)
        
        print(f"训练完成！总episodes: {len(final_returns)}, 平均回报: {mean_return:.2f} ± {std_return:.2f}")
        
        # 记录最终统计到wandb
        if args.track:
            wandb.log({
                "final_stats/total_episodes": len(final_returns),
                "final_stats/mean_return": mean_return,
                "final_stats/std_return": std_return,
                "final_stats/max_return": max_return,
                "final_stats/min_return": min_return,
                "final_stats/total_timesteps": args.total_timesteps,
                "final_stats/training_time_hours": (time.time() - start_time) / 3600,
                "final_stats/denoising_steps": args.denoising_steps
            }, step=args.total_timesteps)

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()
