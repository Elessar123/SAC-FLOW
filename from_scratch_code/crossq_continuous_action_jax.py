# CrossQ implementation based on CleanRL SAC - rewritten to match SBX logic exactly
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import random
import time
from dataclasses import dataclass
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from flax.linen.module import Module, compact, merge_param
from jax.nn import initializers
from jax import lax
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
from collections import deque
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = "crossq"
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    log_freq: int = 5000

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
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

    wandb_project_name: str = "sacflow-fromscratch-" + env_id

    wandb_entity: str = " "

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
    dtype = None
    param_dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = initializers.zeros
    scale_init: Callable = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """
        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

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

        r_max = self.variable(
            'batch_stats',
            'r_max',
            lambda s: s,
            3,
        )
        d_max = self.variable(
            'batch_stats',
            'd_max',
            lambda s: s,
            5,
        )
        steps = self.variable(
            'batch_stats',
            'steps',
            lambda s: s,
            0,
        )

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
                # The code below is implemented following the Batch Renormalization paper
                r = 1
                d = 0
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
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
        
        # CrossQ: Use batch norm if enabled
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
        
        # CrossQ: Wider networks [2048, 2048] instead of [256, 256]
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
    """Vectorized critic network matching SBX implementation"""
    n_critics: int = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, training: bool = True):
        # Vectorized critics using vmap, exactly like SBX
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


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99

    @nn.compact
    def __call__(self, x, training: bool = True):
        # CrossQ: Use batch norm for actor too
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
            
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
            
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        if self.use_batch_norm:
            x = BatchRenorm(use_running_average=not training, momentum=self.batch_norm_momentum)(x)
            
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


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


def sample_action_and_log_prob(mean, log_std, action_scale, action_bias, key):
    """Sample action using reparameterization trick and compute log probability"""
    std = jnp.exp(log_std)
    normal_sample = jax.random.normal(key, mean.shape)
    x_t = mean + std * normal_sample
    y_t = jnp.tanh(x_t)
    action = y_t * action_scale + action_bias
    
    # Compute log probability
    log_prob = jax.scipy.stats.norm.logpdf(normal_sample, 0, 1).sum(axis=-1, keepdims=True)
    # Correct for tanh transformation
    log_prob -= jnp.log(action_scale * (1 - y_t**2) + 1e-6).sum(axis=-1, keepdims=True)
    
    return action, log_prob


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__CrossQ__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            group="crossq_gaussian"  
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
    key, actor_key, qf_key, alpha_key = jax.random.split(key, 4)

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

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
        use_batch_norm=args.use_batch_norm,
        batch_norm_momentum=args.batch_norm_momentum,
    )
    
    # Initialize with batch stats support
    dummy_obs = obs
    if args.use_batch_norm:
        actor_variables = actor.init(actor_key, dummy_obs, training=False)
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_variables['params'],
            target_params=actor_variables['params'],
            batch_stats=actor_variables.get('batch_stats', {}),
            target_batch_stats=actor_variables.get('batch_stats', {}),
            # CrossQ: Adam with b1=0.5 instead of default 0.9
            tx=optax.adam(learning_rate=args.policy_lr, b1=0.5),
        )
    else:
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, dummy_obs, training=False),
            target_params=actor.init(actor_key, dummy_obs, training=False),
            batch_stats={},
            target_batch_stats={},
            tx=optax.adam(learning_rate=args.policy_lr, b1=0.5),
        )
    
    # Use vectorized critic exactly like SBX
    qf = VectorCritic(
        n_critics=args.n_critics,
        use_batch_norm=args.use_batch_norm, 
        batch_norm_momentum=args.batch_norm_momentum
    )
    
    # Initialize vectorized Q network
    dummy_action = envs.action_space.sample()
    if args.use_batch_norm:
        qf_variables = qf.init(qf_key, dummy_obs, dummy_action, training=False)
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
            params=qf.init(qf_key, dummy_obs, dummy_action, training=False),
            target_params=qf.init(qf_key, dummy_obs, dummy_action, training=False),
            batch_stats={},
            target_batch_stats={},
            tx=optax.adam(learning_rate=args.q_lr, b1=0.5),
        )

    # print parameters
    actor_params = sum(x.size for x in jax.tree_util.tree_leaves(actor_state.params))
    qf_params = sum(x.size for x in jax.tree_util.tree_leaves(qf_state.params))
    print("!!================================================")
    print(f"Actor parameters: {actor_params:,}, Critic parameters: {qf_params:,}")
    print("!!================================================")

    # Entropy coefficient
    if args.autotune:
        target_entropy = -np.prod(envs.single_action_space.shape).astype(np.float32)
        entropy_coef = EntropyCoef(args.alpha)
        alpha_state = TrainState.create(
            apply_fn=entropy_coef.apply,
            params=entropy_coef.init(alpha_key),
            target_params=entropy_coef.init(alpha_key),
            batch_stats={},
            target_batch_stats={},
            tx=optax.adam(learning_rate=args.q_lr, b1=0.5),
        )
    else:
        target_entropy = 0.0
        alpha_state = None

    # CrossQ: Track updates for policy delay
    n_updates = 0

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
        
        # Sample next actions from current policy
        if args.use_batch_norm:
            next_mean, next_log_std = actor.apply(
                {'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, 
                next_observations, training=False
            )
        else:
            next_mean, next_log_std = actor.apply(actor_state.params, next_observations, training=False)
            
        next_actions, next_log_prob = sample_action_and_log_prob(
            next_mean, next_log_std, actor.action_scale, actor.action_bias, sample_key
        )
        
        # Get current alpha value
        if alpha_state is not None:
            alpha_value = entropy_coef.apply(alpha_state.params)
        else:
            alpha_value = args.alpha

        def mse_loss(params, batch_stats):
            if not args.crossq_style:
                # Standard SAC: separate forward passes
                # Next Q values from target networks
                if args.use_batch_norm:
                    next_q_values = qf.apply(
                        {'params': qf_state.target_params, 'batch_stats': qf_state.target_batch_stats},
                        next_observations, next_actions, training=False
                    )  # shape: (n_critics, batch_size, 1)
                    # Current Q values from live networks
                    current_q_values, new_batch_stats = qf.apply(
                        {'params': params, 'batch_stats': batch_stats},
                        observations, actions, training=True, mutable=['batch_stats']
                    )
                else:
                    next_q_values = qf.apply(qf_state.target_params, next_observations, next_actions, training=False)
                    current_q_values = qf.apply(params, observations, actions, training=True)
                    new_batch_stats = {}
            else:
                # CrossQ: Joint forward pass - exactly like SBX
                batch_size = observations.shape[0]
                cat_observations = jnp.concatenate([observations, next_observations], axis=0)
                cat_actions = jnp.concatenate([actions, next_actions], axis=0)
                
                if args.use_batch_norm:
                    # Joint forward pass on concatenated batch
                    catted_q_values, new_batch_stats = qf.apply(
                        {'params': params, 'batch_stats': batch_stats}, 
                        cat_observations, cat_actions, training=True, mutable=['batch_stats']
                    )  # shape: (n_critics, 2*batch_size, 1)
                    new_batch_stats = new_batch_stats['batch_stats']
                else:
                    catted_q_values = qf.apply(params, cat_observations, cat_actions, training=True)
                    new_batch_stats = {}
                
                # Split back into current and next
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
            
            # Take minimum across critics for next Q values (Double Q-learning)
            next_q_values = jnp.min(next_q_values, axis=0)  # shape: (batch_size, 1)
            next_q_values = next_q_values - alpha_value * next_log_prob
            target_q_values = rewards.reshape(-1, 1) + (1 - terminations.reshape(-1, 1)) * args.gamma * next_q_values
            
            # Compute MSE loss for all critics
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
            if args.use_batch_norm:
                (mean, log_std), new_batch_stats = actor.apply(
                    {'params': actor_params, 'batch_stats': actor_batch_stats}, 
                    observations, training=True, mutable=['batch_stats']
                )
                new_batch_stats = new_batch_stats['batch_stats']
            else:
                mean, log_std = actor.apply(actor_params, observations, training=True)
                new_batch_stats = {}
                
            actions, log_prob = sample_action_and_log_prob(
                mean, log_std, actor.action_scale, actor.action_bias, sample_key
            )
            
            if args.use_batch_norm:
                qf_pi = qf.apply(
                    {'params': qf_state.params, 'batch_stats': qf_state.batch_stats}, 
                    observations, actions, training=False
                )  # shape: (n_critics, batch_size, 1)
            else:
                qf_pi = qf.apply(qf_state.params, observations, actions, training=False)
                
            min_qf_pi = jnp.min(qf_pi, axis=0)  # Take minimum across critics
            
            if alpha_state is not None:
                alpha_value = entropy_coef.apply(alpha_state.params)
            else:
                alpha_value = args.alpha
            
            actor_loss = (alpha_value * log_prob - min_qf_pi).mean()
            return actor_loss, (log_prob.mean(), new_batch_stats)

        (actor_loss_value, (entropy, new_actor_batch_stats)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, actor_state.batch_stats)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        if args.use_batch_norm:
            actor_state = actor_state.replace(batch_stats=new_actor_batch_stats)
        
        # Update alpha if autotune
        alpha_loss_value = 0.0
        if alpha_state is not None:
            def alpha_loss_fn(alpha_params):
                alpha_value = entropy_coef.apply(alpha_params)
                alpha_loss = (alpha_value * (-entropy - target_entropy)).mean()
                return alpha_loss
            
            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
            alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return actor_state, alpha_state, actor_loss_value, alpha_loss_value, key

    @jax.jit
    def update_target(qf_state: TrainState, tau: float):
        # Use JAX-compatible soft update (when tau=1.0, this becomes hard update)
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
    # 
    completed_episodes = 0
    all_episode_returns = []
    log_buffer = {}
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            key, sample_key = jax.random.split(key, 2)
            if args.use_batch_norm:
                mean, log_std = actor.apply(
                    {'params': actor_state.params, 'batch_stats': actor_state.batch_stats}, 
                    obs, training=False
                )
            else:
                mean, log_std = actor.apply(actor_state.params, obs, training=False)
            actions, _ = sample_action_and_log_prob(
                mean, log_std, actor.action_scale, actor.action_bias, sample_key
            )
            actions = np.array(jax.device_get(actions))

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                episode_return = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                # 
                all_episode_returns.append(episode_return)
                completed_episodes += 1
                
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                
                # wandb buffer
                if args.track:
                    log_buffer.setdefault("charts/episodic_return", deque(maxlen=20)).append(episode_return)
                    log_buffer.setdefault("charts/episodic_length", deque(maxlen=20)).append(episode_length)
                
                # 20episode
                if completed_episodes % 20 == 0:
                    recent_returns = np.array(all_episode_returns[-20:])
                    recent_mean = np.mean(recent_returns)
                    print(f"Episodes: {completed_episodes}, Recent 20 mean return: {recent_mean:.2f}")
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

            # Always update critics
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

            # Update target networks
            qf_state = update_target(qf_state, args.tau)

            n_updates += 1

            # CrossQ: Update actor and alpha only every policy_delay steps
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
                log_buffer.setdefault("charts/n_updates", deque(maxlen=20)).append(n_updates)
                
                alpha_value = args.alpha
                if args.autotune:
                    current_alpha = entropy_coef.apply(alpha_state.params)
                    alpha_value = current_alpha.item()
                    log_buffer.setdefault("losses/alpha_loss", deque(maxlen=20)).append(alpha_loss_value.item() if isinstance(alpha_loss_value, jnp.ndarray) else alpha_loss_value)
                log_buffer.setdefault("losses/alpha", deque(maxlen=20)).append(alpha_value)

            if global_step % args.log_freq == 0:
                writer.add_scalar("losses/qf_loss", qf_loss_value.item(), global_step)
                writer.add_scalar("losses/qf_values", qf_a_values.item(), global_step)
                writer.add_scalar("losses/next_q_values", next_q_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item() if isinstance(actor_loss_value, jnp.ndarray) else actor_loss_value, global_step)
                writer.add_scalar("charts/n_updates", n_updates, global_step)
                if args.autotune:
                    current_alpha = entropy_coef.apply(alpha_state.params)
                    writer.add_scalar("losses/alpha", current_alpha.item(), global_step)
                    writer.add_scalar("losses/alpha_loss", alpha_loss_value.item() if isinstance(alpha_loss_value, jnp.ndarray) else alpha_loss_value, global_step)
                else:
                    writer.add_scalar("losses/alpha", args.alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                if args.track:
                    avg_logs = {key: np.mean(log_buffer[key]) for key in log_buffer.keys()}
                    avg_logs["charts/SPS"] = int(global_step / (time.time() - start_time))
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
                        actor_state.batch_stats if args.use_batch_norm else None,
                        qf_state.batch_stats if args.use_batch_norm else None,
                    ]
                )
            )
        print(f"model saved to {model_path}")
    # 
    if len(all_episode_returns) > 0:
        final_returns = np.array(all_episode_returns)
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        max_return = np.max(final_returns)
        min_return = np.min(final_returns)
        
        print(f"episodes: {len(final_returns)}, : {mean_return:.2f} ± {std_return:.2f}")
        
        if args.track:
            wandb.log({
                "final_stats/total_episodes": len(final_returns),
                "final_stats/mean_return": mean_return,
                "final_stats/std_return": std_return,
                "final_stats/max_return": max_return,
                "final_stats/min_return": min_return,
                "final_stats/total_timesteps": args.total_timesteps,
                "final_stats/training_time_hours": (time.time() - start_time) / 3600,
                "final_stats/policy_delay": args.policy_delay
            }, step=args.total_timesteps)
    envs.close()
    writer.close()