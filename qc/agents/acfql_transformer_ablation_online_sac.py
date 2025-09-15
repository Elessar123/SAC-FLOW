import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation in JAX/Flax"""
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, query, key, value, mask=None):
        batch_size, query_seq_len = query.shape[:2]
        key_seq_len = key.shape[1]

        # Linear projections
        q = nn.Dense(self.d_model, name='q_proj')(query)
        k = nn.Dense(self.d_model, name='k_proj')(key)
        v = nn.Dense(self.d_model, name='v_proj')(value)

        # Reshape for multi-head attention
        head_dim = self.d_model // self.num_heads

        q = q.reshape(batch_size, query_seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, key_seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, key_seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

        if mask is not None:
            scores = scores + mask

        attention_weights = nn.softmax(scores, axis=-1)
        attention_output = jnp.matmul(attention_weights, v)

        # Reshape and output projection
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, query_seq_len, self.d_model)
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
        tgt = tgt + tgt2
        tgt = nn.LayerNorm()(tgt)

        # Cross-attention
        tgt2 = MultiHeadAttention(self.d_model, self.num_heads, name='cross_attn')(
            tgt, memory, memory
        )
        tgt = tgt + tgt2
        tgt = nn.LayerNorm()(tgt)

        # Feed-forward
        tgt2 = nn.Sequential([
            nn.Dense(self.d_model * 4),
            nn.gelu,
            nn.Dense(self.d_model),
        ])(tgt)
        tgt = tgt + tgt2
        tgt = nn.LayerNorm()(tgt)

        return tgt


class FlowMatchingTransformerActor(nn.Module):
    """
    Unified Flow Matching Transformer Actor that can handle both:
    1. BC training: (observation, action, time) -> velocity
    2. Stochastic sampling: observation -> action (with log_prob)
    """
    action_dim: int
    denoising_steps: int = 4
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2
    encoder: nn.Module = None
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, actions=None, time=None, is_encoded=False, 
                 rng_key=None, noise_std=0.1, deterministic=False):
        """
        Unified forward pass that handles both BC training and stochastic sampling.
        
        For BC training:
            observations: [batch_size, obs_dim]
            actions: [batch_size, action_dim] 
            time: [batch_size, 1]
            Returns: velocity [batch_size, action_dim]
            
        For stochastic sampling:
            observations: [batch_size, obs_dim]
            actions, time: None
            rng_key: JAX random key
            Returns: (actions, log_probs) if not deterministic else actions
        """
        batch_size = observations.shape[0]

        # Encode observations if not already encoded
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        # Observation encoding (memory/context)
        obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        obs_emb = obs_encoder(observations)  # [batch_size, d_model]
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

        # Velocity output head
        velocity_head = nn.Dense(self.action_dim, name='velocity_head')

        # BC training mode: compute velocity given (observation, action, time)
        if actions is not None and time is not None:
            # Handle input shapes
            if len(actions.shape) == 2:
                actions = jnp.expand_dims(actions, axis=1)  # [batch_size, 1, action_dim]
            if len(time.shape) == 2:
                time = jnp.expand_dims(time, axis=1)  # [batch_size, 1, 1]

            # Project action and add time embedding
            action_emb = action_proj(actions)  # [batch_size, 1, d_model]
            time_emb = time_embedding(time)  # [batch_size, 1, d_model]
            input_emb = action_emb + time_emb

            # Use diagonal mask for flow matching
            seq_len = input_emb.shape[1]
            diagonal_mask = jnp.full((seq_len, seq_len), -jnp.inf)
            diagonal_mask = diagonal_mask.at[jnp.diag_indices(seq_len)].set(0.0)
            diagonal_mask = jnp.expand_dims(diagonal_mask, axis=(0, 1))

            # Transformer forward pass
            output = input_emb
            for layer in transformer_layers:
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)

            # Output velocity
            velocity = velocity_head(output)  # [batch_size, seq_len, action_dim]
            if velocity.shape[1] == 1:
                velocity = velocity.squeeze(1)  # [batch_size, action_dim]
            else:
                velocity = velocity[:, -1, :]  # [batch_size, action_dim]
            
            return velocity

        # Stochastic sampling mode: generate actions from observations
        else:
            # Flow Matching time step size
            DELTA_T = 1.0 / self.denoising_steps

            # Generate initial random action 
            if rng_key is not None:
                key, init_key = jax.random.split(rng_key)
                x_current = jax.random.normal(init_key, (batch_size, self.action_dim))
            else:
                # Use deterministic seed for reproducibility
                deterministic_key = jax.random.PRNGKey(42)
                x_current = jax.random.normal(deterministic_key, (batch_size, self.action_dim))

            # Initialize log probability for stochastic case
            if not deterministic:
                total_log_prob = jnp.sum(-0.5 * x_current**2 - 0.5 * jnp.log(2 * jnp.pi), axis=-1, keepdims=True)
            else:
                total_log_prob = None

            # Flow Matching iterative generation
            for step in range(self.denoising_steps):
                # Project current action to embedding space
                x_input = jnp.expand_dims(x_current, axis=1)  # [batch_size, 1, action_dim]
                action_emb = action_proj(x_input)  # [batch_size, 1, d_model]

                # Add time embedding
                time_value = step / self.denoising_steps
                time_value = jnp.full((batch_size, 1, 1), time_value)
                time_emb = time_embedding(time_value)

                # Combine action and time embeddings
                input_emb = action_emb + time_emb

                # Flow Matching uses diagonal mask
                diagonal_mask = jnp.full((1, 1), 0.0)
                diagonal_mask = jnp.expand_dims(diagonal_mask, axis=(0, 1))

                # Transformer forward pass
                output = input_emb
                for layer in transformer_layers:
                    output = layer(output, obs_emb, tgt_mask=diagonal_mask)

                # Deterministic velocity prediction
                predicted_velocity = velocity_head(output[:, 0, :])  # [batch_size, action_dim]
                
                if deterministic:
                    # Deterministic update
                    x_current = x_current + predicted_velocity * DELTA_T
                else:
                    # Stochastic update with fixed noise_std
                    mean_next = x_current + predicted_velocity * DELTA_T
                    
                    # Add Gaussian noise with fixed std
                    if rng_key is not None:
                        key, noise_key = jax.random.split(key)
                        noise = jax.random.normal(noise_key, mean_next.shape)
                    else:
                        # Fallback to deterministic noise for reproducibility
                        noise_key = jax.random.PRNGKey(42 + step)
                        noise = jax.random.normal(noise_key, mean_next.shape)
                    
                    x_current = mean_next + noise_std * noise
                    
                    # Update log probability with fixed noise_std
                    step_log_prob = jnp.sum(
                        -0.5 * ((x_current - mean_next) / noise_std)**2 - 
                        0.5 * jnp.log(2 * jnp.pi) - jnp.log(noise_std),
                        axis=-1, keepdims=True
                    )
                    total_log_prob += step_log_prob

            # Apply tanh transformation
            y_t = jnp.tanh(x_current)

            # Add tanh Jacobian correction for stochastic case
            if not deterministic and total_log_prob is not None:
                tanh_correction = jnp.sum(jnp.log(1 - y_t**2 + 1e-6), axis=-1, keepdims=True)
                total_log_prob -= tanh_correction

            if deterministic:
                return y_t
            else:
                return y_t, total_log_prob


class EntropyCoef(nn.Module):
    """Learnable entropy coefficient for SAC."""
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ACFQLAgent_TransformerAblationOnlineSAC(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with Transformer and SAC-style training.
    
    Unified update logic:
    - Offline training (actor_loss): only BC flow loss
    - Online training (online_actor_loss): only Q loss and distill loss with SAC entropy regularization
    """

    rng: Any
    network: Any
    alpha_state: Any  # Separate TrainState for alpha
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss with SAC-style targets."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first action

        # TD loss with SAC-style next action sampling
        rng, sample_rng = jax.random.split(rng)
        next_actions, next_log_probs = self.sample_actions_with_log_prob(
            batch['next_observations'][..., -1, :], rng=sample_rng
        )

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        # Get current alpha value
        if self.config['autotune_alpha']:
            alpha_value = self.alpha_state.apply_fn(self.alpha_state.params)
        else:
            alpha_value = self.config['sac_alpha']

        # SAC-style target with entropy regularization
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * \
            (next_q - alpha_value * next_log_probs.reshape(-1))

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)

        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss - OFFLINE TRAINING: only BC flow loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss - only this is used for offline training
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_transformer')(
            batch['observations'], actions=x_t, time=t, is_encoded=False, params=grad_params
        )

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        # Set other losses to zero for offline training
        distill_loss = jnp.zeros(())
        q_loss = jnp.zeros(())
        actor_log_probs = jnp.zeros((batch_size, 1))

        # Total loss - only BC flow loss for offline training
        actor_loss = bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'actor_entropy': -actor_log_probs.mean(),  # For alpha loss computation
        }

    def online_actor_loss(self, batch, grad_params, rng):
        """Compute the FQL online actor loss - ONLINE TRAINING: only Q loss and distill loss with SAC entropy."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape

        # Set BC flow loss to zero for online training
        bc_flow_loss = jnp.zeros(())

        if self.config["actor_type"] == "distill-ddpg":
            # Generate actions using transformer with stochastic sampling
            actor_actions, actor_log_probs = self.compute_transformer_actions_with_log_prob(
                batch['observations'], rng=rng, grad_params=grad_params,
                noise_std=self.config['online_noise_std']
            )

            # Distillation loss - match expert actions
            distill_loss = jnp.mean((actor_actions - batch_actions) ** 2)

            # Get current alpha value
            if self.config['autotune_alpha']:
                alpha_value = self.alpha_state.apply_fn(self.alpha_state.params)
            else:
                alpha_value = self.config['sac_alpha']

            # SAC-style Q loss with entropy regularization
            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -(q - alpha_value * actor_log_probs.reshape(-1)).mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())
            actor_log_probs = jnp.zeros((batch_size, 1))

        # Total loss - only distill loss and Q loss for online training
        actor_loss = self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'actor_entropy': -actor_log_probs.mean(),  # For alpha loss computation
        }

    def alpha_loss(self, actor_entropy):
        """Compute the entropy coefficient loss."""
        if not self.config['autotune_alpha']:
            return 0.0, {'alpha_loss': 0.0}

        alpha_value = self.alpha_state.apply_fn(self.alpha_state.params)
        alpha_loss = (alpha_value * (-actor_entropy - self.config['target_entropy'])).mean()

        return alpha_loss, {
            'alpha_loss': alpha_loss,
            'alpha_value': alpha_value,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        # Alpha loss
        alpha_loss, alpha_info = self.alpha_loss(actor_info['actor_entropy'])
        for k, v in alpha_info.items():
            info[f'alpha/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def online_total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Use same critic loss
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Use online actor loss
        actor_loss, actor_info = self.online_actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        # Alpha loss (same for both online and offline)
        alpha_loss, alpha_info = self.alpha_loss(actor_info['actor_entropy'])
        for k, v in alpha_info.items():
            info[f'alpha/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def update_alpha(self, actor_entropy):
        """Update the alpha coefficient."""
        if not self.config['autotune_alpha']:
            return self.alpha_state

        def alpha_loss_fn(alpha_params):
            alpha_value = self.alpha_state.apply_fn(alpha_params)
            return (alpha_value * (-actor_entropy - self.config['target_entropy'])).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(self.alpha_state.params)
        new_alpha_state = self.alpha_state.apply_gradients(grads=alpha_grads)
        return new_alpha_state

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')

        # Update alpha
        new_alpha_state = agent.update_alpha(info['actor/actor_entropy'])

        return agent.replace(network=new_network, alpha_state=new_alpha_state, rng=new_rng), info

    @staticmethod
    def _online_update(agent, batch):
        """Online update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.online_total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')

        # Update alpha (same as offline)
        new_alpha_state = agent.update_alpha(info['actor/actor_entropy'])

        return agent.replace(network=new_network, alpha_state=new_alpha_state, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def online_update(self, batch):
        return self._online_update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def online_batch_update(self, batch):
        """Online update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._online_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """Sample actions without log probability (for inference)."""
        if self.config["actor_type"] == "distill-ddpg":
            actions = self.compute_transformer_actions_stochastic(
                observations, rng=rng, noise_std=self.config['inference_noise_std']
            )

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_transformer_actions(observations)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions

    @jax.jit
    def sample_actions_with_log_prob(
        self,
        observations,
        rng=None,
    ):
        """Sample actions with log probability (for training)."""
        return self.compute_transformer_actions_with_log_prob(
            observations, rng=rng, noise_std=self.config['inference_noise_std']
        )

    @jax.jit
    def compute_transformer_actions(
        self,
        observations,
        grad_params=None
    ):
        """Compute actions from the Transformer model (deterministic)."""
        if len(observations.shape) == 1:
            observations = observations[None, :]
        return self.network.select('actor_transformer')(
            observations, deterministic=True, params=grad_params
        )

    @jax.jit
    def compute_transformer_actions_stochastic(
        self,
        observations,
        rng=None,
        grad_params=None,
        noise_std=0.1
    ):
        """Compute actions from the Transformer model with stochastic sampling."""
        if len(observations.shape) == 1:
            observations = observations[None, :]
        actions, _ = self.network.select('actor_transformer')(
            observations, rng_key=rng, noise_std=noise_std, deterministic=False, params=grad_params
        )
        return actions

    @jax.jit
    def compute_transformer_actions_with_log_prob(
        self,
        observations,
        rng=None,
        grad_params=None,
        noise_std=0.1
    ):
        """Compute actions from the Transformer model with log probability."""
        if len(observations.shape) == 1:
            observations = observations[None, :]
        return self.network.select('actor_transformer')(
            observations, rng_key=rng, noise_std=noise_std, deterministic=False, params=grad_params
        )

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, alpha_rng = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
            full_action_dim = full_actions.shape[-1]
        else:
            full_actions = ex_actions
            full_action_dim = full_actions.shape[-1]

        # Set target entropy if not provided
        if config.get('target_entropy') is None:
            config['target_entropy'] = -float(full_action_dim) * 0

        # Define encoders - only need one for unified transformer
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_transformer'] = encoder_module()

        # Define networks - only unified transformer and critic
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        # Unified FlowMatchingTransformerActor
        actor_transformer_def = FlowMatchingTransformerActor(
            action_dim=full_action_dim,
            denoising_steps=config['denoising_steps'],
            d_model=config['transformer_d_model'],
            n_head=config['transformer_n_head'],
            n_layers=config['transformer_n_layers'],
            encoder=encoders.get('actor_transformer'),
            layer_norm=config['actor_layer_norm'],
        )
        
        # Prepare example inputs for network initialization
        ex_observations_batched = ex_observations[None, :]
        full_actions_batched = full_actions[None, :]
        ex_times_batched = ex_times[None, :]

        network_info = dict(
            actor_transformer=(actor_transformer_def, (ex_observations_batched,)),  # For stochastic sampling mode
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_transformer') is not None:
            network_info['actor_transformer_encoder'] = (encoders.get('actor_transformer'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params[f'modules_target_critic'] = params[f'modules_critic']

        # Create alpha state if autotune is enabled
        if config['autotune_alpha']:
            entropy_coef = EntropyCoef(config['sac_alpha'])
            alpha_params = entropy_coef.init(alpha_rng)
            alpha_tx = optax.adam(learning_rate=config['alpha_lr'])
            alpha_state = TrainState.create(entropy_coef, alpha_params, tx=alpha_tx)
        else:
            alpha_state = None

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, alpha_state=alpha_state, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql_transformer_ablation_online_sac',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            alpha_lr=3e-4,  # Learning rate for alpha coefficient.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            sac_alpha=0.2,  # Initial SAC entropy regularization coefficient.
            autotune_alpha=True,  # Whether to automatically tune alpha.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (will be set automatically).
            num_qs=2, # critic ensemble size
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            weight_decay=0.,
            # Transformer specific parameters
            denoising_steps=4,  # Number of denoising steps for transformer
            transformer_d_model=128,  # Transformer model dimension
            transformer_n_head=4,  # Number of attention heads
            transformer_n_layers=2,  # Number of transformer layers
            # SAC-specific noise parameters
            offline_noise_std=0.1,   # Noise std for offline training
            online_noise_std=0.05,   # Noise std for online training  
            inference_noise_std=0.02,  # Noise std for inference/evaluation
        )
    )
    return config