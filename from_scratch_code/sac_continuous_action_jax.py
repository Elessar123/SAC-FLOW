# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import random
import time
from dataclasses import dataclass

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

import collections


@dataclass
class Args:
    exp_name: str = "large-param"
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Humanoid-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
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
    
    # wandb
    wandb_project_name: str = "sacflow-fromscratch-" + env_id
    """the wandb's project name"""
    wandb_entity: str = " "
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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
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
    run_name = f"{args.env_id}__gaussian__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            group="gaussian"
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

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
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

    # print parameters
    actor_params = sum(x.size for x in jax.tree_util.tree_leaves(actor_state.params))
    qf_params = sum(x.size for x in jax.tree_util.tree_leaves(qf1_state.params)) + sum(x.size for x in jax.tree_util.tree_leaves(qf2_state.params))
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
            tx=optax.adam(learning_rate=args.q_lr),
        )
    else:
        target_entropy = 0.0
        alpha_state = None

    actor.apply = jax.jit(actor.apply)
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
        key, sample_key = jax.random.split(key, 2)
        
        # Sample next actions from current policy
        next_mean, next_log_std = actor.apply(actor_state.params, next_observations)
        next_actions, next_log_prob = sample_action_and_log_prob(
            next_mean, next_log_std, actor.action_scale, actor.action_bias, sample_key
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
        key, sample_key = jax.random.split(key, 2)
        
        def actor_loss_fn(actor_params):
            mean, log_std = actor.apply(actor_params, observations)
            actions, log_prob = sample_action_and_log_prob(
                mean, log_std, actor.action_scale, actor.action_bias, sample_key
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

        (actor_loss_value, entropy), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
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
    
    # episode
    completed_episodes = 0
    all_episode_returns = []
    log_buffer = {}
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            key, sample_key = jax.random.split(key, 2)
            mean, log_std = actor.apply(actor_state.params, obs)
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
                
                # print(f"global_step={global_step}, episodic_return={episode_return}")
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                
                # wandb
                if args.track:
                    log_buffer.setdefault("charts/episodic_return", collections.deque(maxlen=20)).append(episode_return)
                    log_buffer.setdefault("charts/episodic_length", collections.deque(maxlen=20)).append(episode_length)
                
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

            if args.track:
                log_buffer.setdefault("losses/qf1_loss", collections.deque(maxlen=20)).append(qf1_loss_value.item())
                log_buffer.setdefault("losses/qf2_loss", collections.deque(maxlen=20)).append(qf2_loss_value.item())
                log_buffer.setdefault("losses/qf1_values", collections.deque(maxlen=20)).append(qf1_a_values.item())
                log_buffer.setdefault("losses/qf2_values", collections.deque(maxlen=20)).append(qf2_a_values.item())
                log_buffer.setdefault("losses/actor_loss", collections.deque(maxlen=20)).append(actor_loss_value.item())
                alpha_value = args.alpha
                if args.autotune:
                    current_alpha = entropy_coef.apply(alpha_state.params)
                    alpha_value = current_alpha.item()
                    log_buffer.setdefault("losses/alpha_loss", collections.deque(maxlen=20)).append(alpha_loss_value.item())
                log_buffer.setdefault("losses/alpha", collections.deque(maxlen=20)).append(alpha_value)

            if global_step % args.log_freq == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                
                alpha_value = args.alpha
                if args.autotune:
                    current_alpha = entropy_coef.apply(alpha_state.params)
                    alpha_value = current_alpha.item()
                    writer.add_scalar("losses/alpha", alpha_value, global_step)
                    writer.add_scalar("losses/alpha_loss", alpha_loss_value.item(), global_step)
                else:
                    writer.add_scalar("losses/alpha", alpha_value, global_step)
                
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)
                
                # wandb
                if args.track:
                    avg_logs = {key: np.mean(log_buffer[key]) for key in log_buffer.keys()}
                    avg_logs["charts/SPS"] = sps
                    wandb.log(avg_logs, step=global_step)
                    # log_buffer = {}

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

    # 
    if len(all_episode_returns) > 0:
        final_returns = np.array(all_episode_returns)
        
        # 
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        max_return = np.max(final_returns)
        min_return = np.min(final_returns)
        
        print(f"episodes: {len(final_returns)}, : {mean_return:.2f} ± {std_return:.2f}")
        
        # wandb
        if args.track:
            wandb.log({
                "final_stats/total_episodes": len(final_returns),
                "final_stats/mean_return": mean_return,
                "final_stats/std_return": std_return,
                "final_stats/max_return": max_return,
                "final_stats/min_return": min_return,
                "final_stats/total_timesteps": args.total_timesteps,
                "final_stats/training_time_hours": (time.time() - start_time) / 3600,
            }, step=args.total_timesteps)

    envs.close()
    writer.close()
    
    # wandb
    if args.track:
        wandb.finish()