from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen.initializers import zeros, constant

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
    
class ActorVectorFieldGRU(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    hidden_dim_gru: int = 256

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)
        self.gate_net = nn.Sequential([
            nn.Dense(self.hidden_dim_gru),
            nn.swish,  # Mish approximation
            nn.Dense(self.action_dim, 
            kernel_init=zeros,  # 权重初始化为0
            bias_init=constant(5.0)),
        ])

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)
        z = nn.sigmoid(self.gate_net(inputs))
        vector_field = z * (v - actions)
        return vector_field

class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation in JAX/Flax"""
    d_model: int
    num_heads: int
    
    def setup(self):
        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.out_proj = nn.Dense(self.d_model)
    
    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key) 
        v = self.v_proj(value)
        
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
        output = self.out_proj(attention_output)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention"""
    d_model: int
    num_heads: int
    
    def setup(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.num_heads)
        self.cross_attn = MultiHeadAttention(self.d_model, self.num_heads)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()
        self.ffn = nn.Sequential([
            nn.Dense(self.d_model * 4),
            nn.gelu,
            nn.Dense(self.d_model),
        ])
    
    def __call__(self, tgt, memory, tgt_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        
        # Cross-attention  
        print(f"The shape of memory {memory.shape}")
        print(f"The shape of tgt {tgt.shape}")
        if memory.shape[1] != tgt.shape[1]:
            memory = jnp.broadcast_to(memory, (memory.shape[0], tgt.shape[1], memory.shape[2]))
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        
        return tgt


class ActorTransformer(nn.Module):
    """
    Transformer-based actor that replaces ActorVectorFieldGRU.
    This generates actions through flow matching with transformer architecture.
    """
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    denoising_steps: int = 4
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2

    def setup(self) -> None:
        # Observation encoding (memory/context)
        self.obs_encoder = nn.Sequential([
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Action input projection
        self.action_proj = nn.Dense(self.d_model)
        
        # Time embedding
        self.time_embedding = nn.Sequential([
            nn.Dense(self.d_model // 4),
            nn.silu,
            nn.Dense(self.d_model // 2),
            nn.silu,
            nn.Dense(self.d_model)
        ])
        
        # Create transformer layers as individual attributes
        for i in range(self.n_layers):
            setattr(self, f'transformer_layer_{i}', 
                   TransformerDecoderLayer(self.d_model, self.n_head))
        
        # Velocity output head
        self.velocity_head = nn.Dense(self.action_dim)

    def __call__(self, observations, actions=None, times=None, is_encoded=False):
        """
        This method is designed to be compatible with the existing framework.
        It can be called in two modes:
        1. Full generation mode: when actions=None, generate actions from scratch
        2. Compatibility mode: when actions is provided, used for gradient computation
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
            
        batch_size = observations.shape[0]
        
        # Observation encoding (memory/context)
        obs_emb = self.obs_encoder(observations)  # [batch_size, d_model]
        obs_emb = jnp.expand_dims(obs_emb, axis=1)  # [batch_size, 1, d_model]
        
        if actions is None:
            # Full generation mode - generate actions from scratch
            return self._generate_actions(obs_emb, batch_size)
        else:
            # Compatibility mode - used for gradient computation
            # This mimics the original ActorVectorFieldGRU interface
            return self._compute_velocity(obs_emb, actions, times)

    def _apply_transformer_layers(self, input_emb, obs_emb, mask=None):
        """Apply all transformer layers to the input."""
        output = input_emb
        for i in range(self.n_layers):
            layer = getattr(self, f'transformer_layer_{i}')
            output = layer(output, obs_emb, tgt_mask=mask)
        return output

    def _generate_actions(self, obs_emb, batch_size):
        """Generate actions from scratch using flow matching"""
        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps
        
        # Generate initial random action (use a fixed seed for deterministic behavior in this context)
        # In practice, this would be passed as input or use the RNG from the caller
        key = jax.random.PRNGKey(42)  # This should ideally be passed from outside
        x_current = jax.random.normal(key, (batch_size, self.action_dim))
        
        # Initialize action sequence with x0
        action_sequence = jnp.expand_dims(x_current, axis=1)  # [batch_size, 1, action_dim]
        
        # Flow Matching iterative generation
        for step in range(self.denoising_steps):
            seq_len = action_sequence.shape[1]
            
            # Project action sequence
            action_emb = self.action_proj(action_sequence)  # [batch_size, seq_len, d_model]
            
            # Add time embedding - Flow Matching style
            time_values = jnp.arange(seq_len, dtype=jnp.float32) / self.denoising_steps
            time_values = jnp.expand_dims(time_values, axis=0)  # [1, seq_len]
            time_values = jnp.expand_dims(time_values, axis=-1)  # [1, seq_len, 1]
            time_values = jnp.broadcast_to(time_values, (batch_size, seq_len, 1))
            
            time_emb = self.time_embedding(time_values)
            
            # Combine action and time embeddings
            input_emb = action_emb + time_emb
            
            # Flow Matching uses diagonal mask (each position only sees itself)
            diagonal_mask = jnp.full((seq_len, seq_len), -jnp.inf)
            diagonal_mask = diagonal_mask.at[jnp.diag_indices(seq_len)].set(0.0)
            diagonal_mask = jnp.expand_dims(diagonal_mask, axis=(0, 1))  # [1, 1, seq_len, seq_len]
            
            # Transformer forward pass
            output = self._apply_transformer_layers(input_emb, obs_emb, diagonal_mask)
            
            # Predict velocity for the last position
            predicted_velocity = self.velocity_head(output[:, -1:, :])  # [batch_size, 1, action_dim]
            velocity = predicted_velocity.squeeze(1)  # [batch_size, action_dim]
            
            # Flow Matching update: x_{t+1} = x_t + v_t * Δt
            x_next = x_current + velocity * DELTA_T
            
            # Append to sequence for next iteration
            action_sequence = jnp.concatenate([
                action_sequence, 
                jnp.expand_dims(x_next, axis=1)
            ], axis=1)
            
            # Update current action
            x_current = x_next
        
        # Final action is the last generated action
        final_action_raw = x_current
        
        # Apply tanh transformation
        final_action = jnp.tanh(final_action_raw)
        
        return final_action

    def _compute_velocity(self, obs_emb, actions, times):
        """
        Compute velocity for given actions and times.
        This is used for compatibility with the existing training framework.
        """
        batch_size = obs_emb.shape[0]
        
        # Project actions - ensure we have the right sequence dimension
        if actions.ndim == 2:
            # actions shape: [batch_size, action_dim] -> [batch_size, 1, action_dim]
            actions_seq = jnp.expand_dims(actions, axis=1)
        else:
            # actions already has sequence dimension
            actions_seq = actions
            
        action_emb = self.action_proj(actions_seq)  # [batch_size, seq_len, d_model]
        
        # Add time embedding if provided
        if times is not None:
            # Handle different possible shapes of times
            if times.ndim == 0:  # scalar
                times = jnp.full((batch_size, 1, 1), times)
            elif times.shape == (1,):  # single element array
                times = jnp.full((batch_size, 1, 1), times[0])
            elif times.shape == (batch_size, 1):  # already correct batch size
                times = jnp.expand_dims(times, axis=-1)  # [batch_size, 1, 1]
            elif times.shape == (batch_size,):  # batch_size length
                times = times.reshape(batch_size, 1, 1)
            else:
                # Fallback: broadcast to correct shape
                times = jnp.broadcast_to(times, (batch_size, action_emb.shape[1], 1))
            
            time_emb = self.time_embedding(times)
            input_emb = action_emb + time_emb
        else:
            input_emb = action_emb
        
        # Apply transformer layers
        output = self._apply_transformer_layers(input_emb, obs_emb, mask=None)
        
        # Handle output based on sequence length
        if output.shape[1] == 1:
            # Single timestep - squeeze the sequence dimension
            velocity = self.velocity_head(output.squeeze(1))  # [batch_size, action_dim]
        else:
            # Multiple timesteps - take the last one or average
            velocity = self.velocity_head(output[:, -1, :])  # [batch_size, action_dim]
        
        return velocity