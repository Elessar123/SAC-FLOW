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
    Flow Matching Transformer Actor adapted for ACFQL
    Now only outputs velocity given (observation, action, time)
    """
    action_dim: int
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2
    encoder: nn.Module = None
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, actions, time, is_encoded=False):
        """
        Args:
            observations: [batch_size, obs_dim] or [batch_size, seq_len, obs_dim]
            actions: [batch_size, action_dim] or [batch_size, seq_len, action_dim]
            time: [batch_size, 1] or [batch_size, seq_len, 1]
            is_encoded: whether observations are already encoded
        Returns:
            velocity: [batch_size, action_dim]
        """
        batch_size = observations.shape[0]

        # Handle different input shapes
        if len(actions.shape) == 2:
            actions = jnp.expand_dims(actions, axis=1)  # [batch_size, 1, action_dim]
        if len(time.shape) == 2:
            time = jnp.expand_dims(time, axis=1)  # [batch_size, 1, 1]

        # Encode observations if not already encoded
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        # Observation encoding (memory/context)
        obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        if len(observations.shape) == 2:
            obs_emb = obs_encoder(observations)  # [batch_size, d_model]
            obs_emb = jnp.expand_dims(obs_emb, axis=1)  # [batch_size, 1, d_model]
        else:
            obs_emb = obs_encoder(observations)  # [batch_size, seq_len, d_model]

        # Action input projection
        action_proj = nn.Dense(self.d_model, name='action_proj')
        action_emb = action_proj(actions)  # [batch_size, seq_len, d_model]

        # Time embedding
        time_embedding = nn.Sequential([
            nn.Dense(self.d_model // 4),
            nn.silu,
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        time_emb = time_embedding(time)  # [batch_size, seq_len, d_model]

        # Combine action and time embeddings
        input_emb = action_emb + time_emb

        # Transformer decoder layers
        transformer_layers = []
        for i in range(self.n_layers):
            transformer_layers.append(
                TransformerDecoderLayer(self.d_model, self.n_head, name=f'layer_{i}')
            )

        # Apply transformer layers
        seq_len = input_emb.shape[1]
        # Use diagonal mask for flow matching (each position only sees itself)
        diagonal_mask = jnp.full((seq_len, seq_len), -jnp.inf)
        diagonal_mask = diagonal_mask.at[jnp.diag_indices(seq_len)].set(0.0)
        diagonal_mask = jnp.expand_dims(diagonal_mask, axis=(0, 1))  # [1, 1, seq_len, seq_len]

        output = input_emb
        for layer in transformer_layers:
            output = layer(output, obs_emb, tgt_mask=diagonal_mask)

        # Velocity output head
        velocity_head = nn.Dense(self.action_dim, name='velocity_head')
        velocity = velocity_head(output)  # [batch_size, seq_len, action_dim]

        # Return velocity for the last timestep if sequence, otherwise squeeze
        if velocity.shape[1] == 1:
            velocity = velocity.squeeze(1)  # [batch_size, action_dim]
        else:
            velocity = velocity[:, -1, :]  # [batch_size, action_dim]

        return velocity


class ACFQLAgent_TransformerAblationOnline(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with Transformer replacing GRU.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first action

        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)

        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_transformer')(batch['observations'], x_t, t, params=grad_params)

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

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            target_flow_actions = self.compute_flow_actions(batch['observations'], rng=rng)
            actor_actions = self.compute_transformer_actions(batch['observations'], grad_params=grad_params, rng=rng)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss.
            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())
        distill_loss = jnp.zeros(())
        q_loss = jnp.zeros(())
        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
        }

    def online_actor_loss(self, batch, grad_params, rng):
        """Compute the FQL online actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss (set to zero for online training)
        bc_flow_loss = jnp.zeros(())

        if self.config["actor_type"] == "distill-ddpg":
            # Generate actions using transformer
            actor_actions = self.compute_transformer_actions(batch['observations'], grad_params=grad_params, rng=rng)

            # Distillation loss - match expert actions
            distill_loss = jnp.mean((actor_actions - batch_actions) ** 2)

            # Q loss.
            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
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

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @staticmethod
    def _online_update(agent, batch):
        """Online update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.online_total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

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

        if self.config["actor_type"] == "distill-ddpg":
            actions = self.compute_transformer_actions(observations, rng=rng)

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
            actions = self.compute_flow_actions(observations, noises)
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
    def compute_flow_actions(
        self,
        observations,
        noises=None,
        rng=None,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if noises is None:
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )

        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_transformer_actions(
        self,
        observations,
        grad_params=None,
        rng=None
    ):
        """Compute actions from the Transformer model using iterative denoising."""
        if len(observations.shape) == 1:
            observations = observations[None, :]
        
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim'] * \
            (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        
        # Flow Matching time step size
        DELTA_T = 1.0 / self.config['denoising_steps']
        
        # Encode observations if encoder is provided
        if self.config['encoder'] is not None:
            encoded_observations = self.network.select('actor_transformer_encoder')(observations, params=grad_params)
        else:
            encoded_observations = observations

        # Generate initial random action
        if rng is None:
            # Use deterministic seed for reproducibility during training
            key = jax.random.PRNGKey(42)
        else:
            key = rng
        x_current = jax.random.normal(key, (batch_size, action_dim))

        # Flow Matching iterative generation
        for step in range(self.config['denoising_steps']):
            # Current time in [0, 1]
            current_time = jnp.full((batch_size, 1), step / self.config['denoising_steps'])
            
            # Predict velocity using transformer
            velocity = self.network.select('actor_transformer')(
                encoded_observations, x_current, current_time, 
                is_encoded=True, params=grad_params
            )
            
            # Flow Matching update: x_{t+1} = x_t + v_t * Δt
            x_current = x_current + velocity * DELTA_T

        # Apply tanh transformation and return final action
        final_action = jnp.tanh(x_current)
        return final_action

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_transformer'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        actor_transformer_def = FlowMatchingTransformerActor(
            action_dim=full_action_dim,
            d_model=config['transformer_d_model'],
            n_head=config['transformer_n_head'],
            n_layers=config['transformer_n_layers'],
            encoder=encoders.get('actor_transformer'),
            layer_norm=config['actor_layer_norm'],
        )

        # Add batch dimension for network initialization
        ex_observations_batched = ex_observations[None, :]  # [1, obs_dim]
        full_actions_batched = full_actions[None, :]  # [1, action_dim]
        ex_times_batched = ex_times[None, :]  # [1, 1]

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_transformer=(actor_transformer_def, (ex_observations_batched, full_actions_batched, ex_times_batched)),  # 修改：添加batch维度
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
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

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql_transformer_ablation_online',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            # Transformer specific parameters
            denoising_steps=4,  # Number of denoising steps for transformer
            transformer_d_model=64,  # Transformer model dimension
            transformer_n_head=4,  # Number of attention heads
            transformer_n_layers=2,  # Number of transformer layers
        )
    )
    return config