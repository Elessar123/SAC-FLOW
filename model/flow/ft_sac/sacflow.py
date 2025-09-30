# MIT License

# Copyright (c) 2025 SAC Flow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.mlp_flow import FlowMLP, NoisyFlowMLP

Sample = namedtuple("Sample", "trajectories chains")

class SACFlow(nn.Module):
    def __init__(self, 
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 # randn_clip_value, # REMOVED
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 # denoised_clip_value, # REMOVED
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type,
                 init_temperature=0.1,
                 target_entropy=None,
                 target_ema_rate=0.005
                 ):
        
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps          # number of steps for inference.
        self.ft_denoising_steps = ft_denoising_steps    # could be adjusted
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        
        # --- NEW: Action scaling for tanh transformation ---
        self.act_min_tensor = torch.tensor(act_min, dtype=torch.float32, device=device).reshape(1, 1, -1)
        self.act_max_tensor = torch.tensor(act_max, dtype=torch.float32, device=device).reshape(1, 1, -1)
        self.register_buffer(
            "action_scale",
            (self.act_max_tensor - self.act_min_tensor) / 2.0
        )
        self.register_buffer(
            "action_bias",
            (self.act_max_tensor + self.act_min_tensor) / 2.0
        )
        # --- END NEW ---
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        self.noise_scheduler_type:str = noise_scheduler_type
        
        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std:float = min_sampling_denoising_std

        # Minimum and maximum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std:float = min_logprob_denoising_std
        self.max_logprob_denoising_std:float = max_logprob_denoising_std
        
        # Minimum and maximum logprobability in each batch, cutoff within this range to prevent policy collapse
        self.logprob_min:float= logprob_min
        self.logprob_max:float= logprob_max
        
        self.logprob_debug_sample=logprob_debug_sample
        self.logprob_debug_recalculate=logprob_debug_recalculate
        
        # noise network settings
        self.learn_explore_time_embedding=learn_explore_time_embedding
        self.time_dim_explore=time_dim_explore
        self.use_time_independent_noise=use_time_independent_noise
        self.noise_hidden_dims=noise_hidden_dims
        self.explore_net_activation_type=explore_net_activation_type
        
        # SAC specific parameters
        self.target_ema_rate = target_ema_rate
        
        # Initialize actor (Flow policy)
        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)
        
        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True
        
        self.init_actor_ft(policy_copy)
        logging.info("Cloned policy for fine-tuning")
        
        # Initialize dual critics and target networks
        self.critic = critic.to(self.device) 
        self.critic_target = copy.deepcopy(self.critic)
        
        # Freeze target networks
        for param in self.critic_target.parameters():
            param.requires_grad = False

        
        # Initialize temperature parameter
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        if target_entropy is None:
            self.target_entropy = -self.act_dim_total  # Heuristic value
        else:
            self.target_entropy = target_entropy
        
        self.report_network_params()
    
    def init_actor_ft(self, policy_copy):
        self.actor_ft = NoisyFlowMLP(policy=policy_copy,
                                    denoising_steps=self.inference_steps,
                                    learn_explore_noise_from = self.inference_steps - self.ft_denoising_steps,
                                    inital_noise_scheduler_type=self.noise_scheduler_type,
                                    min_logprob_denoising_std = self.min_logprob_denoising_std,
                                    max_logprob_denoising_std = self.max_logprob_denoising_std,
                                    learn_explore_time_embedding=self.learn_explore_time_embedding,
                                    time_dim_explore=self.time_dim_explore,
                                    use_time_independent_noise=self.use_time_independent_noise,
                                    device=self.device,
                                    noise_hidden_dims=self.noise_hidden_dims,
                                    activation_type=self.explore_net_activation_type
                                    )
    
    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6:.2f} M. "
            f"Actor: {sum(p.numel() for p in self.actor_old.parameters())/1e6:.2f} M. "
            f"Actor (finetune): {sum(p.numel() for p in self.actor_ft.parameters())/1e6:.2f} M. "
            f"Critic: {sum(p.numel() for p in self.critic.parameters())/1e6:.2f} M. "
        )
    
    def load_policy(self, network_path, use_ema=False):
        log.info(f"loading policy from %s" % network_path)
        if network_path:
            print(f"network_path={network_path}, self.device={self.device}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            if use_ema:
                ema_actor_network_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                self.actor_old.load_state_dict(ema_actor_network_data)
                logging.info("Loaded ema actor policy from %s", network_path)
            else:
                self.actor_old.load_state_dict(actor_network_data)
                logging.info("Loaded actor policy from %s", network_path)
            print(f"actor_network_data={actor_network_data.keys()}")
        else:
            logging.warning("No actor policy path provided. Not loading any actor policy. Start from randomly initialized policy.")
    
    @torch.no_grad()
    def sample_first_point(self, B:int)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        B: batchsize
        outputs:
            xt: torch.Tensor of shape `[batchsize, self.horizon_steps, self.action_dim]`
            log_prob: torch.Tensor of shape `[batchsize]`
        '''
        dist = Normal(torch.zeros(B, self.horizon_steps, self.action_dim, device=self.device), 1.0)
        xt = dist.sample()
        log_prob = dist.log_prob(xt).sum(dim=(-2,-1))
        return xt, log_prob

    def _generate_chain(self, cond: dict, use_rsample: bool, save_chains: bool):
        """
        Core private method to generate an action chain and its log probability.
        - Uses rsample for training (gradient flow).
        - Uses sample for inference.
        - Applies tanh transformation and Jacobian correction.
        """
        B = cond["state"].shape[0]
        dt = (1 / self.inference_steps) * torch.ones(B, self.horizon_steps, self.action_dim, device=self.device)
        dt_scalar = torch.tensor(1.0 / self.inference_steps).to(self.device)
        steps = torch.linspace(0, 1 - dt_scalar, self.inference_steps).repeat(B, 1).to(self.device)

        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps + 1, self.horizon_steps, self.action_dim), device=self.device)

        # Sample first point and get initial log_prob
        xt, log_prob = self.sample_first_point(B)
        if save_chains:
            x_chain[:, 0] = xt

        # Iterative denoising loop
        for i in range(self.inference_steps):
            t = steps[:, i]
            vt, nt = self.actor_ft.forward(xt, t, cond, learn_exploration_noise=not use_rsample, step=i)
            
            mean = xt + vt * dt
            std = torch.clamp(nt.unsqueeze(-1).reshape(xt.shape), min=self.min_sampling_denoising_std)
            dist = Normal(mean, std)

            # Use rsample() for training, sample() for inference
            if use_rsample:
                xt = dist.rsample()
            else:
                xt = dist.sample()

            log_prob += dist.log_prob(xt).sum(dim=(-2, -1))
            if save_chains:
                x_chain[:, i + 1] = xt

        # --- NEW: Tanh transformation and Jacobian Correction ---
        # Apply tanh to the unbounded action xt
        xt_tanh = torch.tanh(xt)
        # Scale and bias the action to the environment's range
        action = xt_tanh * self.action_scale + self.action_bias

        # Jacobian correction for the tanh transformation
        # log_prob(action) = log_prob(xt) - log(det(d(action)/d(xt)))
        # The correction term is sum(log(1 - tanh(xt)^2))
        tanh_correction = torch.log(self.action_scale * (1 - xt_tanh.pow(2)) + 1e-6).sum(dim=(-2, -1))
        log_prob -= tanh_correction
        # --- END NEW ---
        
        # Reshape action to be 2D: [batch, horizon * action_dim] for critic
        action_flat = action.reshape(B, -1)
        
        if save_chains:
            return action_flat, log_prob, x_chain
        return action_flat, log_prob, None

    @torch.no_grad()
    def get_actions(self, cond: dict, save_chains=False, **kwargs):
        """
        Get actions for interaction with the environment. NO GRADIENTS.
        """
        action, log_prob, chain = self._generate_chain(cond, use_rsample=True, save_chains=save_chains)
        if save_chains:
            return action, chain, log_prob
        return action, log_prob

    def get_logprobs(self, cond: dict, **kwargs):
        """
        Get actions and log_probs for training. WITH GRADIENTS via rsample.
        This function is now the differentiable action generator.
        """
        action, log_prob, _ = self._generate_chain(cond, use_rsample=True, save_chains=False)
        return action, log_prob

    def update_target_critic(self, tau=None):
        """Soft update of target networks"""
        if tau is None:
            tau = self.target_ema_rate
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma, alpha, **kwargs):
        """SAC critic loss. Uses get_actions (no_grad) for next actions."""
        
        with torch.no_grad():
            # Use the NON-GRADIENT get_actions for sampling next state actions
            next_actions, next_logprobs = self.get_actions(cond=next_obs)
            
            # Clamp logprobs for stability
            next_logprobs = next_logprobs.clamp(min=self.logprob_min, max=self.logprob_max)
            
            # Calculate target Q-value
            q1_next_target, q2_next_target = self.critic_target(next_obs, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Bellman equation for target Q
            target_q = rewards + gamma * (1 - terminated) * (q_next_target - alpha * next_logprobs)
        
        # Current Q-values
        q1_current, q2_current = self.critic(obs, actions)
        
        # Critic loss is the MSE between current and target Q-values
        critic1_loss = F.mse_loss(q1_current, target_q)
        critic2_loss = F.mse_loss(q2_current, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        return critic_loss

    def loss_actor(self, obs, alpha, **kwargs):
        """
        SAC actor loss. Uses the new get_logprobs function to get differentiable actions.
        """
        # Get new actions and their logprobs WITH GRADIENTS
        new_actions, new_logprobs = self.get_logprobs(cond=obs)
        
        # Clamp logprobs for stability
        new_logprobs = new_logprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        
        # Calculate Q-values for the new actions
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # SAC actor loss: E[α*log π - Q]
        actor_loss = (alpha * new_logprobs - q_new).mean()
        
        return actor_loss, new_logprobs

    def loss_temperature(self, obs, alpha, target_entropy, **kwargs):
        """
        Temperature parameter loss.
        """
        # We only need the value of logprobs, so no gradients w.r.t. actor params needed here
        with torch.no_grad():
            _, logprobs = self.get_logprobs(cond=obs)
        
        logprobs = logprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        
        # Loss for alpha is to make the policy entropy match the target entropy
        alpha_loss = -(alpha * (logprobs + target_entropy)).mean()
        return alpha_loss
