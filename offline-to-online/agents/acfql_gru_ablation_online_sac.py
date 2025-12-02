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
from utils.networks import ActorVectorField, Value, ActorVectorFieldGRU


class EntropyCoef(nn.Module):
    """Learnable entropy coefficient for SAC."""
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ACFQLAgent_GRUAblationOnlineSAC(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking using SAC-style training with learnable alpha.
    
    Unified update logic:
    - Offline training (actor_loss): only BC flow loss
    - Online training (online_actor_loss): only Q loss and distill loss with SAC entropy regularization
    """

    rng: Any
    network: Any
    alpha_state: Any  # Separate TrainState for alpha
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
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
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss - only this is used for offline training
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_onestep_flow')(batch['observations'], x_t, t, params=grad_params)

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
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape

        # Set BC flow loss to zero for online training
        bc_flow_loss = jnp.zeros(())
        
        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            
            actor_actions, actor_log_probs = self.compute_flow_actions_one_step_grad_with_log_prob(
                batch['observations'], noises=noises, grad_params=grad_params,
                noise_std=self.config['online_noise_std']
            )

            # Action regularization loss (similar to distillation but with target actions)
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
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.compute_flow_actions_one_step_stochastic(
                observations, noises, noise_std=self.config['inference_noise_std']
            )

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions_one_step(observations, noises)
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
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        actions, log_probs = self.compute_flow_actions_one_step_with_log_prob(
            observations, noises, noise_std=self.config['inference_noise_std']
        )
        return actions, log_probs

    @jax.jit
    def compute_flow_actions_one_step_stochastic(
        self,
        observations,
        noises,
        noise_std=0.1,
    ):
        """Compute actions from the one-step flow model with stochastic sampling."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_onestep_flow_encoder')(observations)
        actions = noises
        
        # Euler method with stochastic sampling
        dt = 1.0 / self.config['denoising_steps']
        for i in range(self.config['denoising_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['denoising_steps'])
            vels = self.network.select('actor_onestep_flow')(observations, actions, t, is_encoded=True)
            
            # Deterministic step
            mean_next = actions + vels * dt
            
            # Add Gaussian noise
            rng = jax.random.PRNGKey(42)  # Use deterministic key for consistency
            noise = jax.random.normal(rng, mean_next.shape)
            actions = mean_next + noise_std * noise
        
        actions = jnp.tanh(actions)
        return actions

    @jax.jit
    def compute_flow_actions_one_step_with_log_prob(
        self,
        observations,
        noises,
        noise_std=0.1,
    ):
        """Compute actions with log probability computation."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_onestep_flow_encoder')(observations)
        actions = noises
        
        # Initialize log probability with prior N(0, I)
        total_log_prob = jnp.sum(-0.5 * noises**2 - 0.5 * jnp.log(2 * jnp.pi), axis=-1, keepdims=True)
        
        # Euler method with stochastic sampling
        dt = 1.0 / self.config['denoising_steps']
        for i in range(self.config['denoising_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['denoising_steps'])
            vels = self.network.select('actor_onestep_flow')(observations, actions, t, is_encoded=True)
            
            # Deterministic step
            mean_next = actions + vels * dt
            
            # Add Gaussian noise
            rng = jax.random.PRNGKey(42)  # Use deterministic key for consistency
            noise = jax.random.normal(rng, mean_next.shape)
            actions = mean_next + noise_std * noise
            
            # Update log probability
            step_log_prob = jnp.sum(
                -0.5 * ((actions - mean_next) / noise_std)**2 - 
                0.5 * jnp.log(2 * jnp.pi) - jnp.log(noise_std),
                axis=-1, keepdims=True
            )
            total_log_prob += step_log_prob
        
        # Apply tanh and compute Jacobian correction
        y_t = jnp.tanh(actions)
        tanh_correction = jnp.sum(jnp.log(1 - y_t**2 + 1e-6), axis=-1, keepdims=True)
        total_log_prob -= tanh_correction
        
        return y_t, total_log_prob

    @jax.jit
    def compute_flow_actions_one_step(
        self,
        observations,
        noises,
    ):
        """Compute actions from the one-step flow model (deterministic version)."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_onestep_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['denoising_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['denoising_steps'])
            vels = self.network.select('actor_onestep_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['denoising_steps']
        actions = jnp.tanh(actions)
        return actions
    
    @jax.jit
    def compute_flow_actions_one_step_grad_with_log_prob(
        self,
        observations,
        noises,
        grad_params,
        noise_std=0.1
    ):
        """Compute actions with gradients and log probability."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_onestep_flow_encoder')(observations, params=grad_params)
        actions = noises
        
        # Initialize log probability with prior N(0, I)
        total_log_prob = jnp.sum(-0.5 * noises**2 - 0.5 * jnp.log(2 * jnp.pi), axis=-1, keepdims=True)
        
        # Euler method with stochastic sampling
        dt = 1.0 / self.config['denoising_steps']
        for i in range(self.config['denoising_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['denoising_steps'])
            vels = self.network.select('actor_onestep_flow')(observations, actions, t, is_encoded=True, params=grad_params)
            
            # Deterministic step
            mean_next = actions + vels * dt
            
            # Add Gaussian noise
            rng = jax.random.PRNGKey(42)  # Use deterministic key for consistency
            noise = jax.random.normal(rng, mean_next.shape)
            actions = mean_next + noise_std * noise
            
            # Update log probability
            step_log_prob = jnp.sum(
                -0.5 * ((actions - mean_next) / noise_std)**2 - 
                0.5 * jnp.log(2 * jnp.pi) - jnp.log(noise_std),
                axis=-1, keepdims=True
            )
            total_log_prob += step_log_prob
        
        # Apply tanh and compute Jacobian correction
        y_t = jnp.tanh(actions)
        tanh_correction = jnp.sum(jnp.log(1 - y_t**2 + 1e-6), axis=-1, keepdims=True)
        total_log_prob -= tanh_correction
        
        return y_t, total_log_prob

    @jax.jit
    def compute_flow_actions_one_step_grad(
        self,
        observations,
        noises,
        grad_params
    ):
        """Compute actions from the one-step flow model using gradients (deterministic version)."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_onestep_flow_encoder')(observations, params=grad_params)
        actions = noises
        # Euler method.
        for i in range(self.config['denoising_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['denoising_steps'])
            vels = self.network.select('actor_onestep_flow')(observations, actions, t, is_encoded=True, params=grad_params)
            actions = actions + vels / self.config['denoising_steps']
        actions = jnp.tanh(actions)
        return actions

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

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks - only use actor_onestep_flow for unified training
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_onestep_flow_def = ActorVectorFieldGRU(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
            denoising_steps = config['denoising_steps']
        )

        network_info = dict(
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_onestep_flow') is not None:
            # Add actor_onestep_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_onestep_flow_encoder'] = (encoders.get('actor_onestep_flow'), (ex_observations,))
        
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

def save_actor_params(agent, save_dir, step, save_format='flax'):
    actor_save_dir = os.path.join(save_dir, 'actor_checkpoints')
    os.makedirs(actor_save_dir, exist_ok=True)
    
    actor_params = agent.network.params['modules_actor_onestep_flow']
    
    actor_data = {
        'actor_onestep_flow': actor_params,
        'step': step,
        'config': dict(agent.config)  
    }
    
    if 'modules_actor_onestep_flow_encoder' in agent.network.params:
        actor_data['actor_onestep_flow_encoder'] = agent.network.params['modules_actor_onestep_flow_encoder']
    
    if save_format == 'flax':
        filename = f'actor_step_{step}.flax'
        filepath = os.path.join(actor_save_dir, filename)
        
        with open(filepath, 'wb') as f:
            serialized_data = serialization.to_bytes(actor_data)
            f.write(serialized_data)
            
    elif save_format == 'pickle':
        filename = f'actor_step_{step}.pkl'
        filepath = os.path.join(actor_save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(actor_data, f)
    
    latest_link = os.path.join(actor_save_dir, 'latest_actor.txt')
    with open(latest_link, 'w') as f:
        f.write(filename)
    
    print(f"Actor parameters saved to: {filepath}")
    return filepath

def load_actor_params(filepath, save_format='flax'):
    if save_format == 'flax':
        with open(filepath, 'rb') as f:
            serialized_data = f.read()
            actor_data = serialization.from_bytes(target=None, encoded_bytes=serialized_data)
    elif save_format == 'pickle':
        with open(filepath, 'rb') as f:
            actor_data = pickle.load(f)
    
    return actor_data

def save_actor_params_lightweight(agent, save_dir, step):

    actor_save_dir = os.path.join(save_dir, 'actor_checkpoints')
    os.makedirs(actor_save_dir, exist_ok=True)
    

    actor_params = agent.network.params['modules_actor_onestep_flow']
    filename = f'actor_params_step_{step}.npz'
    filepath = os.path.join(actor_save_dir, filename)
    
    flat_params = {}
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_params = flatten_dict(actor_params)
    
    numpy_params = {k: jnp.asarray(v) for k, v in flat_params.items()}
    jnp.savez(filepath, **numpy_params, step=step)
    
    print(f"Lightweight actor parameters saved to: {filepath}")
    return filepath

def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql_gru_ablation_online_sac',  # Agent name.
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
            denoising_steps=4,
            # SAC-specific noise parameters
            offline_noise_std=0.1,  # Noise std for offline training
            online_noise_std=0.05,  # Noise std for online training  
            inference_noise_std=0.02,  # Noise std for inference/evaluation
        )
    )
    return config