# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    # env_id: str = "Hopper-v4"
    # env_id: str = "HalfCheetah-v4"
    # env_id: str = "Humanoid-v4"
    env_id: str = "Ant-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    denoising_steps: int = 4
    """number of denoising steps for transformer-based actor"""
    d_model: int = 64
    """transformer model dimension"""
    n_head: int = 4
    """number of attention heads"""
    n_layers: int = 2
    """number of transformer layers"""


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
        tgt = tgt + tgt2  # Remove Dropout for deterministic TD3
        tgt = nn.LayerNorm()(tgt)
        
        # Cross-attention  
        tgt2 = MultiHeadAttention(self.d_model, self.num_heads, name='cross_attn')(
            tgt, memory, memory
        )
        tgt = tgt + tgt2  # Remove Dropout for deterministic TD3
        tgt = nn.LayerNorm()(tgt)
        
        # Feed-forward
        tgt2 = nn.Sequential([
            nn.Dense(self.d_model * 4),
            nn.gelu,
            nn.Dense(self.d_model),
        ])(tgt)
        tgt = tgt + tgt2  # Remove Dropout for deterministic TD3
        tgt = nn.LayerNorm()(tgt)
        
        return tgt


class TransformerActor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    obs_dim: int
    denoising_steps: int = 4
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2

    @nn.compact
    def __call__(self, obs):
        batch_size = obs.shape[0]
        
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
        
        # Output heads
        mean_head = nn.Dense(self.action_dim, name='mean_head')
        
        # Generate initial random action (deterministic for TD3)
        key = jax.random.PRNGKey(42)  # Fixed seed for deterministic behavior
        x_current = jax.random.normal(key, (batch_size, self.action_dim))
        
        # Initialize action sequence with x0
        action_sequence = jnp.expand_dims(x_current, axis=1)  # [batch_size, 1, action_dim]
        
        # Autoregressive generation for denoising_steps (deterministic)
        for step in range(self.denoising_steps):
            seq_len = action_sequence.shape[1]
            
            # Project action sequence
            action_emb = action_proj(action_sequence)  # [batch_size, seq_len, d_model]
            
            # Add time embedding
            time_values = jnp.arange(seq_len, dtype=jnp.float32) / self.denoising_steps
            time_values = jnp.expand_dims(time_values, axis=0)  # [1, seq_len]
            time_values = jnp.expand_dims(time_values, axis=-1)  # [1, seq_len, 1]
            time_values = jnp.broadcast_to(time_values, (batch_size, seq_len, 1))
            
            time_emb = time_embedding(time_values)
            
            # Combine action and time embeddings
            input_emb = action_emb + time_emb
            
            # Create causal mask
            causal_mask = jnp.triu(jnp.full((seq_len, seq_len), -jnp.inf), k=1)
            causal_mask = jnp.expand_dims(causal_mask, axis=(0, 1))  # [1, 1, seq_len, seq_len]
            
            # Transformer forward pass
            output = input_emb
            for layer in transformer_layers:
                output = layer(output, obs_emb, tgt_mask=causal_mask)
            
            # Get mean for the next step (last position in sequence)
            mean_next = mean_head(output[:, -1:, :])  # [batch_size, 1, action_dim]
            
            # Deterministic update (no sampling for TD3)
            x_next = mean_next.squeeze(1)  # [batch_size, action_dim]
            
            # Append to sequence for next iteration
            action_sequence = jnp.concatenate([
                action_sequence, 
                jnp.expand_dims(x_next, axis=1)
            ], axis=1)
        
        # Final action is the last generated action
        final_action_raw = action_sequence[:, -1, :]  # [batch_size, action_dim]
        
        # Apply tanh transformation and scaling
        final_action = jnp.tanh(final_action_raw)
        action = final_action * self.action_scale + self.action_bias
        
        return action


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}_transformer_steps{args.denoising_steps}_d{args.d_model}_h{args.n_head}_l{args.n_layers}__{int(time.time())}"
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
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)

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

    actor = TransformerActor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
        obs_dim=np.array(envs.single_observation_space.shape).prod(),
        denoising_steps=args.denoising_steps,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key = jax.random.split(key, 2)
        clipped_noise = (
            jnp.clip(
                (jax.random.normal(noise_key, actions.shape) * args.policy_noise),
                -args.noise_clip,
                args.noise_clip,
            )
            * actor.action_scale
        )
        next_state_actions = jnp.clip(
            actor.apply(actor_state.target_params, next_observations) + clipped_noise,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )
        qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
        qf2_next_target = qf.apply(qf2_state.target_params, next_observations, next_state_actions).reshape(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = (rewards + (1 - terminations) * args.gamma * (min_qf_next_target)).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
        )
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, args.tau)
        )
        return actor_state, (qf1_state, qf2_state), actor_loss_value

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(0, max_action * args.exploration_noise, size=envs.single_action_space.shape)
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

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
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                key,
            )

            if global_step % args.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
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
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(TransformerActor, QNetwork),
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "TD3", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()