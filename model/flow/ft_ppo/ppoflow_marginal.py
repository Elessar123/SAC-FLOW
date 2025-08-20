# MIT License

# Copyright (c) 2025 ReinFlow Authors

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

class PPOFlow(nn.Module):
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
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
                 ):
        
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps          # number of steps for inference.
        self.ft_denoising_steps = ft_denoising_steps    # could be adjusted
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        self.noise_scheduler_type:str = noise_scheduler_type

        # prevent extreme values sampled from gaussian. deviation from mean should stay within `randn_clip_value` times of std.
        self.randn_clip_value:float = randn_clip_value
        
        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std:float = min_sampling_denoising_std

        # Minimum and maximum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std:float = min_logprob_denoising_std
        self.max_logprob_denoising_std:float = max_logprob_denoising_std
        
        # Minimum and maximum logprobability in each batch, cutoff within this range to prevent policy collapse
        self.logprob_min:float= logprob_min
        self.logprob_max:float= logprob_max
        
        self.clip_ploss_coef:float = clip_ploss_coef
        self.clip_ploss_coef_base:float = clip_ploss_coef_base
        self.clip_ploss_coef_rate:float = clip_ploss_coef_rate
        self.clip_vloss_coef:float = clip_vloss_coef
        
        # clip intermediate actions during inference
        self.denoised_clip_value:float = denoised_clip_value
        self.logprob_debug_sample=logprob_debug_sample
        self.logprob_debug_recalculate=logprob_debug_recalculate
        
        # noise network settings
        self.learn_explore_time_embedding=learn_explore_time_embedding
        self.time_dim_explore=time_dim_explore
        self.use_time_independent_noise=use_time_independent_noise
        self.noise_hidden_dims=noise_hidden_dims
        self.explore_net_activation_type=explore_net_activation_type
        
        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)  # it was false for previous experiments for hopper walker halfcheetah
        for param in self.actor_old.parameters():
            param.requires_grad = False             # don't train this copy, just use it to load checkpoint. 
        self.actor_old.to(self.device)
        
        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True
        
        self.init_actor_ft(policy_copy)
        logging.info("Cloned policy for fine-tuning")
        
        self.critic = critic
        self.critic = self.critic.to(self.device)
        
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
    
    def check_gradient_flow(self):
        print(f"{next(self.actor_ft.policy.parameters()).requires_grad}") #True
        print(f"{next(self.actor_ft.mlp_logvar.parameters()).requires_grad}")#True
        print(f"{next(self.actor_ft.time_embedding_explore.parameters()).requires_grad}")#True
        print(f"{self.actor_ft.logvar_min.requires_grad}")#False
        print(f"{self.actor_ft.logvar_max.requires_grad}")#False
        
    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. Actor:{sum(p.numel() for p in self.actor_old.parameters())/1e6} M. Actor (finetune) : {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
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
        dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim), 1.0)
        xt= dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)                    # mean() or sum() 
        xt=xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob
    
    def compute_divergence(self, velocity_field, x, t, cond, step=0):
        """
        Compute the divergence of the velocity field using automatic differentiation
        
        Args:
            velocity_field: the velocity field function (should be self.actor_ft.forward)
            x: input tensor of shape [B, horizon_steps, action_dim]
            t: time tensor of shape [B]
            cond: conditioning dict
            step: current step for the velocity field
            
        Returns:
            div: divergence tensor of shape [B]
        """
        B, horizon_steps, action_dim = x.shape
        x_flat = x.view(B, -1)  # [B, horizon_steps * action_dim]
        
        if not x_flat.requires_grad:
            x_flat = x_flat.requires_grad_(True)
        x_grad = x_flat.view(B, horizon_steps, action_dim)
        
        # Get velocity field with gradient enabled
        v, _ = velocity_field(x_grad, t, cond, learn_exploration_noise=False, step=step)
        v_flat = v.view(B, -1)  # [B, horizon_steps * action_dim]
        
        # Compute divergence using trace of Jacobian
        div = torch.zeros(B, device=x.device)
        
        for i in range(v_flat.shape[1]):  # For each dimension
            grad = torch.autograd.grad(
                outputs=v_flat[:, i].sum(),
                inputs=x_flat, 
                create_graph=True,  
                retain_graph=True,
                allow_unused=True
            )[0]
            if grad is not None:
                div += grad[:, i]  # Add diagonal element of Jacobian
        
        return div
    
    def compute_divergence_hutchinson(self, velocity_field, x, t, cond, step=0, n_samples=10):

        B, horizon_steps, action_dim = x.shape
        x_flat = x.view(B, -1)  # [B, D] where D = horizon_steps * action_dim
        D = x_flat.shape[1]
        
        if not x_flat.requires_grad:
            x_flat = x_flat.requires_grad_(True)
        
        x_reshaped = x_flat.view(B, horizon_steps, action_dim)
        
        v, _ = velocity_field(x_reshaped, t, cond, learn_exploration_noise=False, step=step)
        v_flat = v.view(B, -1)  # [B, D]
        
        # Hutchinson estimator
        div_est = torch.zeros(B, device=x.device)
        
        for _ in range(n_samples):
            epsilon = torch.randint_like(x_flat, 0, 2, dtype=x_flat.dtype) * 2 - 1  # [B, D]
            
            # ε^T * ∇v
            vjp = torch.autograd.grad(
                outputs=v_flat,
                inputs=x_flat,
                grad_outputs=epsilon,
                create_graph=True, 
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if vjp is not None:
                # tr(J) ≈ ε^T * (∇v)
                div_est += (epsilon * vjp).sum(dim=1)
        
        return div_est / n_samples
    
    def get_logprobs(self, 
                     cond:dict, 
                     x_chain:Tensor, 
                     get_entropy =False, 
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        '''
        MODIFIED: Flow Matching probability calculation using divergence-based formula
        
        inputs:
            x_chain: torch.Tensor of shape `[batchsize, self.inference_steps+1, self.horizon_steps, self.action_dim]`
           
        outputs:
            log_prob. tensor of shape `[batchsize]`
            entropy_rate_est: tensor of shape `[batchsize]` (set to zero for deterministic flow)
            chains_stds.mean(): tensor of shape `[batchsize]` (set to zero for deterministic flow)
            
        explanation:
            For Flow Matching:
            log p(x1|s) = log p(x0) - \sum_{t=0}^{K-1} (\nabla \cdot u_t)(x_t) * dt
            
            where u_t is the velocity field and (\nabla \cdot u_t) is its divergence
        '''
        logprob = 0.0
        joint_entropy = 0.0  # Always 0 for deterministic flow
        entropy_rate_est = 0.0  # Always 0 for deterministic flow
        logprob_steps = 0
        
        B = x_chain.shape[0]
        dt = torch.tensor(1.0/self.inference_steps).to(self.device)
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, self.inference_steps]
        
        # Initial probability from standard normal
        init_dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:,0].reshape(B,-1)).sum(-1)   # [batchsize]
        
        if account_for_initial_stochasticity:
            logprob += logprob_init
            logprob_steps += 1
        
        # Flow Matching: accumulate divergence terms
        divergence_sum = torch.zeros(B, device=self.device)
        
        for i in range(self.inference_steps):
            t = steps[:,i]
            xt = x_chain[:,i]  # [batchsize, self.horizon_steps, self.action_dim]
            
            # Compute divergence of velocity field at current position
            with torch.enable_grad():
                div_t = self.compute_divergence_hutchinson(self.actor_ft.forward, xt, t, cond, step=i)
            divergence_sum += div_t * dt
            logprob_steps += 1
        
        # Apply Flow Matching probability formula: log p1 = log p0 - ∫ div(v) dt
        logprob += logprob_init - divergence_sum
        
        if self.logprob_debug_recalculate: 
            log.info(f"logprob_init={logprob_init.mean().item()}, divergence_sum={divergence_sum.mean().item()}")
        
        # For deterministic flow, entropy is always 0
        if get_entropy:
            entropy_rate_est = torch.zeros_like(logprob)
        
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
            
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total
        
        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy (deterministic flow): all zeros")
        
        # Return zeros for std since we're not using noise anymore
        chains_stds_mean = torch.tensor(0.0, device=self.device)
        
        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds_mean
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, chains_stds_mean
            return logprob
    
    def get_actions(self, 
                    cond:dict, 
                    eval_mode:bool, 
                    save_chains=False, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        '''
        MODIFIED: Pure Flow Matching inference without noise
        '''
        B = cond["state"].shape[0]
        dt = (1/self.inference_steps) * torch.ones(B, self.horizon_steps, self.action_dim, device=self.device)
        dt_scalar = torch.tensor(((1)/self.inference_steps)).to(self.device)
        steps = torch.linspace(0, 1-dt_scalar, self.inference_steps).repeat(B, 1).to(self.device)
        
        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.action_dim), device=self.device)
        
        if ret_logprob:
            xt, log_prob_init = self.sample_first_point(B)
            
            if save_chains:
                x_chain[:, 0] = xt
            
            # Flow integration with gradients
            for i in range(self.inference_steps):
                t = steps[:,i]
                vt, _ = self.actor_ft.forward(xt, t, cond, learn_exploration_noise=False, step=i)
                xt = xt + vt * dt
                
                if clip_intermediate_actions:
                    xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
                
                if i == self.inference_steps-1:
                    xt = xt.clamp_(self.act_min, self.act_max)
                
                if save_chains:
                    x_chain[:, i+1] = xt
            
            # Calculate logprob
            if save_chains:
                log_prob = self.get_logprobs(cond, x_chain, 
                                        normalize_denoising_horizon=normalize_denoising_horizon,
                                        normalize_act_space_dimension=normalize_act_space_dimension,
                                        clip_intermediate_actions=clip_intermediate_actions,
                                        account_for_initial_stochasticity=account_for_initial_stochasticity,
                                        get_entropy=False, get_chains_stds=False)
            else:
                # Use current trajectory for logprob calculation
                temp_chain = torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.action_dim), device=self.device)
                temp_xt, _ = self.sample_first_point(B)
                temp_chain[:, 0] = temp_xt
                
                for i in range(self.inference_steps):
                    t = steps[:,i]
                    vt, _ = self.actor_ft.forward(temp_xt, t, cond, learn_exploration_noise=False, step=i)
                    temp_xt = temp_xt + vt * dt
                    if clip_intermediate_actions:
                        temp_xt = temp_xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
                    if i == self.inference_steps-1:
                        temp_xt = temp_xt.clamp_(self.act_min, self.act_max)
                    temp_chain[:, i+1] = temp_xt
                
                log_prob = self.get_logprobs(cond, temp_chain,
                                        normalize_denoising_horizon=normalize_denoising_horizon,
                                        normalize_act_space_dimension=normalize_act_space_dimension,
                                        clip_intermediate_actions=clip_intermediate_actions,
                                        account_for_initial_stochasticity=account_for_initial_stochasticity,
                                        get_entropy=False, get_chains_stds=False)
        else:
            with torch.no_grad():
                xt, log_prob_init = self.sample_first_point(B)
                
                if save_chains:
                    x_chain[:, 0] = xt
                
                for i in range(self.inference_steps):
                    t = steps[:,i]
                    vt, _ = self.actor_ft.forward(xt, t, cond, learn_exploration_noise=False, step=i)
                    xt += vt * dt
                    
                    if clip_intermediate_actions:
                        xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
                    
                    if i == self.inference_steps-1:
                        xt = xt.clamp_(self.act_min, self.act_max)
                    
                    if save_chains:
                        x_chain[:, i+1] = xt
        
        # Return results
        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)  
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt
    
    
    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        bc_loss='W2',
        bc_loss_type=None,
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True
    ):
        """
        PPO loss - MODIFIED for Flow Matching
        obs: dict with key state/rgb; more recent obs at the end
            "state": (B, To, Do)
            "rgb": (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B,)
        advantages: (B,)
        oldlogprobs: (B,)
        use_bc_loss: whether to add BC regularization loss
        normalize_act_space_dimension: whether to normalize logprobs and entropy rates over all horiton steps and action dimensions
        reward_horizon: action horizon that backpropagates gradient, omitted for now.
        Here, B = n_steps x n_envs
        """
        
        # MODIFIED: entropy is always 0 for deterministic flow
        newlogprobs, entropy, noise_std = self.get_logprobs(obs, 
                                                            chains, 
                                                            get_entropy=True, 
                                                            normalize_denoising_horizon=normalize_denoising_horizon,
                                                            normalize_act_space_dimension=normalize_act_space_dimension, 
                                                            verbose_entropy_stats=verbose, 
                                                            clip_intermediate_actions=clip_intermediate_actions,
                                                            account_for_initial_stochasticity=account_for_initial_stochasticity)
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")
        
        
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose:
            if oldlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: old logprobs too high")
        
        # batch normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")
        
        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        ### for debug
        print(ratio)
        
        # Get kl difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # better not use. 
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")
        
        # MODIFIED: Entropy loss is always 0 for deterministic flow
        entropy_loss = -entropy.mean()  # This will be 0
        
        # Monitor policy entropy distribution
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy (deterministic flow): always zero")
        
        # bc loss
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type=='W2':
                # add wasserstein divergence loss via action supervision
                z=torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_ω = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max],z=z)
                a_θ = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max],z=z)
                bc_loss = F.mse_loss(a_ω.detach(), a_θ)
            else:
                raise NotImplementedError
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),  # This will be 0
            newvalues.mean().item(),#Q function
        )