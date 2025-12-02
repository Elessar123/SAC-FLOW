# PPO2 (Proximal Policy Optimization) implementation with continuous actions using JAX
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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



@dataclass
class Args:
    exp_name: str = "ppo-large-param"
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
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 40000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 12.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
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


# Actor Network (optimized for PPO)
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    log_std_min: float = -5
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x):
        # Smaller network for more stable learning
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        mean = nn.Dense(self.action_dim)(x)
        
        # Use a separate parameter for log_std instead of network output
        # This is more common in PPO implementations
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        # log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        log_std = jnp.broadcast_to(log_std, mean.shape)
        
        return mean, log_std


# Critic Network (optimized for PPO)
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Smaller network matching actor architecture
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


# Efficient JAX-based functions for GAE computation
@jax.jit
def compute_gae_and_returns(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute GAE advantages and returns using JAX for parallel computation.
    
    Args:
        rewards: Array of shape (num_steps, num_envs)
        values: Array of shape (num_steps + 1, num_envs) - includes bootstrap value
        dones: Array of shape (num_steps, num_envs) - terminal flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Array of shape (num_steps, num_envs)
        returns: Array of shape (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    
    # Initialize arrays
    advantages = jnp.zeros((num_steps, num_envs))
    returns = jnp.zeros((num_steps, num_envs))
    
    # Compute TD errors (deltas)
    deltas = rewards + gamma * values[1:] * (1.0 - dones) - values[:-1]
    
    # Compute GAE advantages using scan for efficiency
    def gae_step(carry, inputs):
        gae = carry
        delta, done = inputs
        gae = delta + gamma * gae_lambda * gae * (1.0 - done)
        return gae, gae
    
    # Scan backwards through time
    _, advantages = jax.lax.scan(
        gae_step, 
        jnp.zeros(num_envs), 
        (deltas, dones), 
        reverse=True
    )
    
    # Compute returns
    returns = advantages + values[:-1]
    
    return advantages, returns

@jax.jit  
def normalize_advantages(advantages):
    """Normalize advantages across all environments and steps."""
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages)
    return (advantages - adv_mean) / (adv_std + 1e-8)

# Parallel PPO buffer using JAX operations
class ParallelPPOBuffer:
    def __init__(self, obs_dim, act_dim, num_steps, num_envs, gamma=0.99, gae_lambda=0.95):
        """
        Parallel PPO buffer that stores trajectories for multiple environments.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension  
            num_steps: Number of steps per rollout
            num_envs: Number of parallel environments
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Initialize buffers - shape (num_steps, num_envs, ...)
        self.obs_buf = jnp.zeros((num_steps, num_envs, obs_dim), dtype=jnp.float32)
        self.act_buf = jnp.zeros((num_steps, num_envs, act_dim), dtype=jnp.float32)  
        self.rew_buf = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        self.val_buf = jnp.zeros((num_steps + 1, num_envs), dtype=jnp.float32)  # +1 for bootstrap
        self.logp_buf = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        self.done_buf = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        
        # Computed buffers
        self.adv_buf = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        self.ret_buf = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        
        self.step = 0
        
    def store_step(self, step, obs, acts, rews, vals, logps, dones):
        """
        Store data for all environments at a given step.
        
        Args:
            step: Current step index
            obs: Observations (num_envs, obs_dim)
            acts: Actions (num_envs, act_dim)  
            rews: Rewards (num_envs,)
            vals: Values (num_envs,)
            logps: Log probabilities (num_envs,)
            dones: Done flags (num_envs,)
        """
        self.obs_buf = self.obs_buf.at[step].set(obs)
        self.act_buf = self.act_buf.at[step].set(acts)
        self.rew_buf = self.rew_buf.at[step].set(rews)
        self.val_buf = self.val_buf.at[step].set(vals)
        self.logp_buf = self.logp_buf.at[step].set(logps)
        self.done_buf = self.done_buf.at[step].set(dones)
        
    def store_final_values(self, final_vals):
        """Store bootstrap values for the final step."""
        self.val_buf = self.val_buf.at[self.num_steps].set(final_vals)
        
    def compute_advantages_and_returns(self):
        """Compute GAE advantages and returns for all stored trajectories."""
        self.adv_buf, self.ret_buf = compute_gae_and_returns(
            self.rew_buf, self.val_buf, self.done_buf, 
            self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        self.adv_buf = normalize_advantages(self.adv_buf)
        
    def get_batch(self):
        """
        Get flattened batch data for training.
        
        Returns:
            Dictionary with flattened arrays suitable for minibatch training
        """
        batch_size = self.num_steps * self.num_envs
        
        return {
            'obs': self.obs_buf.reshape(batch_size, self.obs_dim),
            'act': self.act_buf.reshape(batch_size, self.act_dim),
            'ret': self.ret_buf.reshape(batch_size),
            'adv': self.adv_buf.reshape(batch_size),
            'logp': self.logp_buf.reshape(batch_size),
            'val': self.val_buf[:-1].reshape(batch_size)  # Exclude bootstrap value
        }
        
    def reset(self):
        """Reset buffer for next rollout."""
        self.step = 0


def sample_action_and_log_prob(mean, log_std, action_scale, action_bias, key):
    """Sample action using reparameterization trick and compute log probability"""
    std = jnp.exp(log_std)
    normal_sample = jax.random.normal(key, mean.shape)
    x_t = mean + std * normal_sample
    y_t = jnp.tanh(x_t)
    action = y_t * action_scale + action_bias
    
    # Compute log probability of the pre-tanh action under the Gaussian distribution
    log_prob = -0.5 * jnp.log(2.0 * jnp.pi) - log_std - 0.5 * ((x_t - mean) / std) ** 2
    log_prob = log_prob.sum(axis=-1, keepdims=True)
    
    # Correct for tanh transformation: log|det(dy/dx)| = log(1 - tanh^2(x))
    log_prob -= jnp.log(1 - y_t**2 + 1e-6).sum(axis=-1, keepdims=True)
    
    return action, log_prob


def compute_log_prob(mean, log_std, action, action_scale, action_bias):
    """Compute log probability of given action under current policy"""
    # Inverse tanh transformation
    y_t = (action - action_bias) / action_scale
    # Use more stable clipping
    y_t = jnp.clip(y_t, -0.9999, 0.9999)  # Avoid numerical issues
    x_t = jnp.arctanh(y_t)
    
    std = jnp.exp(log_std)
    
    # Compute log probability of the pre-tanh action under the Gaussian distribution
    log_prob = -0.5 * jnp.log(2.0 * jnp.pi) - log_std - 0.5 * ((x_t - mean) / std) ** 2
    log_prob = log_prob.sum(axis=-1, keepdims=True)
    
    # Correct for tanh transformation: log|det(dy/dx)| = log(1 - tanh^2(x))
    log_prob -= jnp.log(1 - y_t**2 + 1e-6).sum(axis=-1, keepdims=True)
    
    return log_prob


if __name__ == "__main__":
    args = tyro.cli(Args)
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    
    run_name = f"{args.env_id}__ppo2__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            group="ppo"
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
    key, actor_key, critic_key = jax.random.split(key, 3)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) 
                                    for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Initialize networks
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    
    actor = Actor(
        action_dim=act_dim,
        action_scale=jnp.array((envs.single_action_space.high - envs.single_action_space.low) / 2.0),
        action_bias=jnp.array((envs.single_action_space.high + envs.single_action_space.low) / 2.0),
    )
    critic = Critic()

    # Initialize network states
    obs, _ = envs.reset(seed=args.seed)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # print parameters
    actor_params = sum(x.size for x in jax.tree_util.tree_leaves(actor_state.params))
    critic_params = sum(x.size for x in jax.tree_util.tree_leaves(critic_state.params))
    print("!!================================================")
    print(f"Actor parameters: {actor_params:,}, Critic parameters: {critic_params:,}")
    print("!!================================================")

    # Initialize parallel buffer
    buffer = ParallelPPOBuffer(obs_dim, act_dim, args.num_steps, args.num_envs, args.gamma, args.gae_lambda)

    # JIT compile functions
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def update_ppo(
        actor_state: TrainState,
        critic_state: TrainState,
        batch_obs: jnp.ndarray,
        batch_actions: jnp.ndarray,
        batch_logprobs: jnp.ndarray,
        batch_advantages: jnp.ndarray,
        batch_returns: jnp.ndarray,
        batch_values: jnp.ndarray,
        key: jnp.ndarray,
    ):
        def ppo_loss_fn(actor_params, critic_params):
            # Get current policy outputs
            mean, log_std = actor.apply(actor_params, batch_obs)
            
            # Compute current log probabilities
            newlogprob = compute_log_prob(mean, log_std, batch_actions, actor.action_scale, actor.action_bias)
            newlogprob = newlogprob.squeeze()
            
            # Compute entropy
            entropy = (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1)
            
            # PPO policy loss
            logratio = newlogprob - batch_logprobs
            ratio = jnp.exp(logratio)
            
            # Clipped surrogate objective
            pg_loss1 = -batch_advantages * ratio
            pg_loss2 = -batch_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            
            # Value loss
            newvalue = critic.apply(critic_params, batch_obs).squeeze()
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - batch_returns) ** 2
                v_clipped = batch_values + jnp.clip(
                    newvalue - batch_values,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - batch_returns) ** 2).mean()
            
            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            
            return loss, (pg_loss, v_loss, entropy_loss, ratio, newlogprob)

        (loss, (pg_loss, v_loss, entropy_loss, ratio, newlogprob)), grads = jax.value_and_grad(
            ppo_loss_fn, argnums=(0, 1), has_aux=True
        )(actor_state.params, critic_state.params)
        
        actor_grads, critic_grads = grads
        
        # Apply gradient clipping using direct computation
        actor_grad_norm = optax.global_norm(actor_grads)
        critic_grad_norm = optax.global_norm(critic_grads)
        
        # Clip gradients if norm exceeds max_grad_norm
        actor_grads = jax.tree.map(
            lambda g: g * jnp.minimum(1.0, args.max_grad_norm / jnp.maximum(actor_grad_norm, 1e-8)),
            actor_grads
        )
        critic_grads = jax.tree.map(
            lambda g: g * jnp.minimum(1.0, args.max_grad_norm / jnp.maximum(critic_grad_norm, 1e-8)),
            critic_grads
        )
        
        # Apply gradients
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        
        return actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, ratio

    start_time = time.time()
    
    completed_episodes = 0
    all_episode_returns = []
    episode_return_buffer = []  
    
    num_updates = args.total_timesteps // batch_size
    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            actor_state = actor_state.replace(tx=optax.adam(learning_rate=lrnow))
            critic_state = critic_state.replace(tx=optax.adam(learning_rate=lrnow))

        # Collect trajectories
        for step in range(args.num_steps):
            global_step = (update - 1) * args.num_steps * args.num_envs + step * args.num_envs
            
            # Sample actions
            key, sample_key = jax.random.split(key, 2)
            mean, log_std = actor.apply(actor_state.params, obs)
            actions, log_probs = sample_action_and_log_prob(
                mean, log_std, actor.action_scale, actor.action_bias, sample_key
            )
            
            # Get value estimates
            values = critic.apply(critic_state.params, obs)
            values = values.squeeze()
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = envs.step(np.array(jax.device_get(actions)))
            
            # Convert to JAX arrays and store in parallel buffer
            obs_jax = jnp.array(obs)
            actions_jax = actions
            rewards_jax = jnp.array(rewards)
            values_jax = values  
            log_probs_jax = log_probs.squeeze()
            dones_jax = jnp.array(terminations.astype(np.float32))
            
            # Store step data for all environments at once
            buffer.store_step(step, obs_jax, actions_jax, rewards_jax, values_jax, log_probs_jax, dones_jax)
            
            obs = next_obs
            
            # Handle episode endings
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        episode_return = info["episode"]["r"]
                        episode_length = info["episode"]["l"]
                        
                        all_episode_returns.append(episode_return)
                        episode_return_buffer.append(episode_return)  
                        completed_episodes += 1
                        
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        
                        
                        if completed_episodes % 20 == 0:
                            recent_returns = np.array(all_episode_returns[-20:])
                            recent_mean = np.mean(recent_returns)
                            print(f"Episodes: {completed_episodes}, Recent 20 mean return: {recent_mean:.2f}")

        # Bootstrap value for last observation
        next_values = critic.apply(critic_state.params, obs)
        next_values = next_values.squeeze()
        
        # Store final values and compute advantages/returns
        buffer.store_final_values(next_values)
        buffer.compute_advantages_and_returns()
        
        # Get batch data
        batch_data = buffer.get_batch()
        
        update_losses = []
        update_pg_losses = []
        update_v_losses = []
        update_entropy_losses = []
        update_ratios = []
        
        # Update policy and value networks
        b_inds = np.arange(batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, ratio = update_ppo(
                    actor_state,
                    critic_state,
                    batch_data["obs"][mb_inds],
                    batch_data["act"][mb_inds],
                    batch_data["logp"][mb_inds],
                    batch_data["adv"][mb_inds],
                    batch_data["ret"][mb_inds],
                    batch_data["val"][mb_inds],
                    key,
                )
                
                update_losses.append(float(loss))
                update_pg_losses.append(float(pg_loss))
                update_v_losses.append(float(v_loss))
                update_entropy_losses.append(float(entropy_loss))
                update_ratios.append(float(ratio.mean()))
                
                # Early stopping based on KL divergence
                approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                print("approx_kl:", approx_kl)
                if args.target_kl is not None:
                    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                    print("approx_kl:", approx_kl)
                    if approx_kl > args.target_kl:
                        break

        global_step = update * args.num_steps * args.num_envs
        
        avg_loss = np.mean(update_losses)
        avg_pg_loss = np.mean(update_pg_losses)
        avg_v_loss = np.mean(update_v_losses)
        avg_entropy_loss = np.mean(update_entropy_losses)
        avg_ratio = np.mean(update_ratios)
        
        writer.add_scalar("losses/policy_loss", avg_pg_loss, global_step)
        writer.add_scalar("losses/value_loss", avg_v_loss, global_step)
        writer.add_scalar("losses/entropy", avg_entropy_loss, global_step)
        writer.add_scalar("losses/total_loss", avg_loss, global_step)
        writer.add_scalar("debug/avg_ratio", avg_ratio, global_step)
        
        avg_episodic_return = None
        if len(episode_return_buffer) > 0:
            avg_episodic_return = np.mean(episode_return_buffer)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        
        if args.track:
            wandb_logs = {
                "losses/policy_loss": avg_pg_loss,
                "losses/value_loss": avg_v_loss,
                "losses/entropy": avg_entropy_loss,
                "losses/total_loss": avg_loss,
                "debug/avg_ratio": avg_ratio,
                "charts/SPS": sps,
            }
            
            if avg_episodic_return is not None:
                wandb_logs["charts/episodic_return"] = avg_episodic_return
            
            wandb.log(wandb_logs, step=global_step)
        
        avg_return_str = f"{avg_episodic_return:.2f}" if avg_episodic_return is not None else "N/A"
        episodes_in_buffer = len(episode_return_buffer)
        print(f"Update {update}/{num_updates}, SPS: {sps}, Avg Loss: {avg_loss:.4f}, Avg Return: {avg_return_str} ({episodes_in_buffer} episodes)")
        
        episode_return_buffer.clear()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        critic_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")

    if len(all_episode_returns) > 0:
        final_returns = np.array(all_episode_returns)
        
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        max_return = np.max(final_returns)
        min_return = np.min(final_returns)
        
        
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
    
    if args.track:
        wandb.finish()
