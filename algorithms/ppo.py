import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import torch.multiprocessing as mp
from torch.multiprocessing import Barrier, Queue, Process


# PPO Agent (takes raw dims, works in any process)
class PPOAgent(nn.Module):
    """
    PPO agent containing separate actor and critic networks.
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, obs_dim, act_dim, is_discrete=False,
                 action_high=None, action_low=None):
        """
        Initialize the PPO agent.

        Args:
            obs_dim (int): Dimension of the observation/state space.
            act_dim (int): Dimension of the action space.
            is_discrete (bool): True if the environment has a discrete
                                action space, otherwise False.
            action_high (array-like, optional): Upper bound of the
                                continuous action space.
            action_low (array-like, optional): Lower bound of the
                                continuous action space.

        Expected Output:
            Initializes actor and critic neural networks along with
            parameters required for action scaling in continuous spaces.
        """

        #Your Code goes here
        super(PPOAgent, self).__init__()
        self.is_discrete = is_discrete
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor head
        if is_discrete:
            self.actor = nn.Linear(128, act_dim)
        else:
            self.actor_mean = nn.Linear(128, act_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic head
        self.critic = nn.Linear(128, 1)
        
        # Action scaling for continuous
        if not is_discrete:
            self.action_high = torch.FloatTensor(action_high)
            self.action_low = torch.FloatTensor(action_low)

    
    def get_value(self, x):
        """
        Compute the value estimate for a given state.

        Args:
            x (Tensor): Input state tensor of shape (batch_size, obs_dim).

        Expected Output:
            Tensor containing the predicted state value V(s)
            from the critic network.
        """
        x = self.shared(x)
        return self.critic(x)



    def get_action_and_value(self, x, action=None):
        """
        Compute action, log probability, entropy, and state value.

        Args:
            x (Tensor): Input state tensor of shape (batch_size, obs_dim).
            action (Tensor, optional): Specific action for which the
                                       log probability should be computed.
                                       If None, the function samples an action
                                       from the policy distribution.

        Expected Output:
            action (Tensor): Sampled or provided action.
            log_prob (Tensor): Log probability of the selected action.
            entropy (Tensor): Entropy of the action distribution
                              (used for exploration).
            value (Tensor): Predicted state value from the critic.
        """
        x = self.shared(x)
        value = self.critic(x)
        
        if self.is_discrete:
            logits = self.actor(x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = self.actor_mean(x)
            std = torch.exp(self.actor_log_std)
            dist = Normal(mean, std)
            if action is None:
                raw_action = dist.rsample()
                # Tanh squashing
                action = torch.tanh(raw_action)
                # Scale to env range
                action = self.action_low.to(action.device) + (action + 1) * 0.5 * (
                    self.action_high.to(action.device) - self.action_low.to(action.device)
                )
            else:
                # Inverse scaling: action in env range -> [-1, 1]
                action = action.to(x.device)
                squashed = 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0
                # Clamp to avoid numerical issues with arctanh
                squashed = torch.clamp(squashed, -0.999999, 0.999999)
                # Inverse tanh: arctanh(x) = 0.5 * log((1+x)/(1-x))
                raw_action = 0.5 * torch.log((1 + squashed) / (1 - squashed))
            
            # Calculate log prob (simplified for continuous with tanh)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
    

# PPO Worker + Training Loop (Multiprocessing Version)
def ppo_worker(worker_id, global_agent,
               EnvClass, envConfig, seed, num_steps, num_updates,
               shared_obs, shared_actions, shared_logprobs,
               shared_rewards, shared_dones, shared_values,
               shared_next_obs, shared_next_done,
               collect_barrier, update_barrier, ep_queue):
    """
    PPO worker process responsible for collecting rollouts.

    Args:
        worker_id        : Unique ID of the worker process.
        global_agent     : Shared PPO policy network.
        env_id           : Gym environment name.
        seed             : Random seed for reproducibility.
        num_steps        : Number of steps collected per rollout.
        num_updates      : Number of PPO updates to perform.

        shared_obs       : Shared buffer storing observations.
        shared_actions   : Shared buffer storing actions.
        shared_logprobs  : Shared buffer storing log probabilities.
        shared_rewards   : Shared buffer storing rewards.
        shared_dones     : Shared buffer storing termination flags.
        shared_values    : Shared buffer storing value predictions.

        shared_next_obs  : Shared buffer storing next observations.
        shared_next_done : Shared buffer storing next done flag.

        collect_barrier  : Synchronization barrier for rollout collection.
        update_barrier   : Synchronization barrier after PPO update.
        ep_queue         : Queue used to send episode rewards to main process.

    Expected Output:
        This function does not return values directly.
        Instead, it stores rollout data in shared buffers and
        sends episode rewards to the main process through ep_queue.
    """
    # Create local environment
    env = EnvClass(**envConfig)
    env.reset(seed=seed + worker_id)
    
    # Get dimensions
    obs_dim = 18
    is_discrete = True
    act_dim = 5
    actions = ["L45", "L22", "FW", "R22", "R45"]
    
    # Local state
    state = env.reset()
    done = False
    episode_reward = 0.0
    
    for update in range(num_updates):
        # Sync with global model at start of rollout
        with torch.no_grad():
            local_agent_state_dict = {k: v.clone() for k, v in global_agent.state_dict().items()}
        
        # Storage for this worker's trajectory
        local_obs = []
        local_actions = []
        local_logprobs = []
        local_rewards = []
        local_dones = []
        local_values = []
        
        # Collect trajectory
        for step in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Use local copy of model for inference
            with torch.no_grad():
                # Create temporary agent with local state dict
                temp_agent = PPOAgent(obs_dim, act_dim, is_discrete, None, None)
                temp_agent.load_state_dict(local_agent_state_dict)
                temp_agent.eval()
                
                action, log_prob, _, value = temp_agent.get_action_and_value(state_tensor)
            
            if is_discrete:
                action_item = action.item()
            else:
                action_item = action.squeeze(0).cpu().numpy()
            

            next_state, reward, terminated = env.step(actions[action_item], render=False)
            done_flag = terminated
            
            # Store locally
            local_obs.append(state.copy())
            if is_discrete:
                local_actions.append(action.item())
            else:
                local_actions.append(action_item)  # This is already numpy from above
            local_logprobs.append(log_prob.item())
            local_rewards.append(reward)
            local_dones.append(float(done_flag))
            local_values.append(value.item())
            
            episode_reward += reward
            
            if done_flag:
                # Send episode reward to main process
                ep_queue.put((worker_id, episode_reward))
                episode_reward = 0.0
                state = env.reset()
                done = False
            else:
                state = next_state
                done = done_flag
        
        # Write to shared memory at specific offset for this worker
        offset = worker_id * num_steps
        for i, (o, a, lp, r, d, v) in enumerate(zip(local_obs, local_actions, local_logprobs, 
                                                     local_rewards, local_dones, local_values)):
            idx = offset + i
            # Convert numpy array to torch tensor before assignment
            shared_obs[idx, :] = torch.from_numpy(o) if isinstance(o, np.ndarray) else o
            shared_actions[idx] = a if isinstance(a, (int, float)) else torch.tensor(a)
            shared_logprobs[idx] = lp
            shared_rewards[idx] = r
            shared_dones[idx] = d
            shared_values[idx] = v
        
        # Store next observation and done flag for bootstrap
        shared_next_obs[worker_id, :] = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
        shared_next_done[worker_id] = float(done)
        
        # Wait at collection barrier for all workers to finish
        collect_barrier.wait()
        
        # Wait at update barrier for main process to finish optimization
        update_barrier.wait()


def train_ppo(EnvClass, envConfig, seed,
              total_timesteps,
              num_workers, num_steps,
              lr, gamma,
              gae_lambda, update_epochs,
              num_minibatches,
              clip_coef, clip_vloss,
              ent_coef, vf_coef,
              max_grad_norm, norm_adv,
              anneal_lr, target_kl):
    """
    PPO training loop using multiprocessing workers.

    Args:
        env_id           : Gym environment used for training.
        seed             : Random seed.
        total_timesteps  : Total number of environment interactions.
        num_workers      : Number of parallel workers collecting data.
        num_steps        : Number of rollout steps per worker.

        lr               : Learning rate for optimizer.
        gamma            : Discount factor for future rewards.
        gae_lambda       : Lambda parameter for GAE.

        update_epochs    : Number of PPO update passes per batch.
        num_minibatches  : Number of minibatches for SGD.

        clip_coef        : PPO clipping coefficient.
        clip_vloss       : Whether value loss clipping is used.

        ent_coef         : Entropy coefficient for exploration.
        vf_coef          : Value loss coefficient.
        max_grad_norm    : Maximum gradient norm for clipping.

        norm_adv         : Whether to normalize advantages.
        anneal_lr        : Whether learning rate is annealed.
        target_kl        : Optional KL divergence stopping threshold.

    Expected Output:
        results      : List containing tuples (global_step, episode_reward)
                       used to plot the learning curve.
        global_agent : The trained PPO agent.
    """
    # Get environment specs
    dummy_env = EnvClass(**envConfig)
    obs_dim = 18
    is_discrete = True
    act_dim = 5
    action_high = None
    action_low = None
    
    # Calculate training parameters
    batch_size = num_workers * num_steps
    num_updates = total_timesteps // batch_size
    minibatch_size = batch_size // num_minibatches
    
    print(f"Starting PPO training: {num_updates} updates, batch size {batch_size}")
    
    # Create shared global agent
    global_agent = PPOAgent(obs_dim, act_dim, is_discrete, action_high, action_low)
    global_agent.share_memory()
    
    # Shared memory buffers using torch tensors - ensure they're on CPU
    shared_obs = torch.zeros((num_workers * num_steps, obs_dim), dtype=torch.float32).share_memory_()
    
    if is_discrete:
        shared_actions = torch.zeros((num_workers * num_steps,), dtype=torch.long).share_memory_()
    else:
        shared_actions = torch.zeros((num_workers * num_steps, act_dim), dtype=torch.float32).share_memory_()
    
    shared_logprobs = torch.zeros((num_workers * num_steps,), dtype=torch.float32).share_memory_()
    shared_rewards = torch.zeros((num_workers * num_steps,), dtype=torch.float32).share_memory_()
    shared_dones = torch.zeros((num_workers * num_steps,), dtype=torch.float32).share_memory_()
    shared_values = torch.zeros((num_workers * num_steps,), dtype=torch.float32).share_memory_()
    
    # Next state buffers for bootstrap
    shared_next_obs = torch.zeros((num_workers, obs_dim), dtype=torch.float32).share_memory_()
    shared_next_done = torch.zeros((num_workers,), dtype=torch.float32).share_memory_()
    
    # Synchronization primitives
    collect_barrier = Barrier(num_workers + 1)  # Workers + main process
    update_barrier = Barrier(num_workers + 1)
    
    # Episode reward queue
    ep_queue = Queue()
    
    # Spawn worker processes
    workers = []
    for worker_id in range(num_workers):
        p = Process(target=ppo_worker, args=(
            worker_id, global_agent, EnvClass, envConfig, seed, num_steps, num_updates,
            shared_obs, shared_actions, shared_logprobs,
            shared_rewards, shared_dones, shared_values,
            shared_next_obs, shared_next_done,
            collect_barrier, update_barrier, ep_queue
        ))
        p.start()
        workers.append(p)
    
    # Optimizer
    optimizer = optim.Adam(global_agent.parameters(), lr=lr, eps=1e-5)
    
    # Results tracking
    results = []
    global_step = 0
    
    # Training loop (main process)
    for update in range(num_updates):
        # Anneal learning rate if requested
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Wait for all workers to finish collection
        collect_barrier.wait()
        
        # Collect episode rewards from queue without blocking
        while not ep_queue.empty():
            try:
                worker_id, ep_reward = ep_queue.get_nowait()
                results.append((global_step, ep_reward))
            except:
                break
        
        global_step += batch_size
        
        # Compute GAE advantages and returns
        with torch.no_grad():
            # Get next values for bootstrap
            next_values = global_agent.get_value(shared_next_obs).squeeze()
            
            # Reshape to (num_steps, num_workers)
            rewards = shared_rewards.view(num_workers, num_steps).t()
            dones = shared_dones.view(num_workers, num_steps).t()
            values = shared_values.view(num_workers, num_steps).t()
            next_values = next_values.view(1, num_workers)
            next_dones = shared_next_done.view(1, num_workers)
            
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_dones
                    nextvals = next_values
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvals = values[t+1]
                
                delta = rewards[t] + gamma * nextvals * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
            
            # Flatten
            advantages = advantages.t().reshape(-1)
            returns = returns.t().reshape(-1)
            
            # Normalize advantages
            if norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten batch
        b_obs = shared_obs
        if is_discrete:
            b_actions = shared_actions.long()
        else:
            b_actions = shared_actions
        b_logprobs = shared_logprobs
        b_advantages = advantages
        b_returns = returns
        b_values = shared_values
        
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = global_agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions[mb_inds] if not is_discrete else b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(global_agent.parameters(), max_grad_norm)
                optimizer.step()
            
            # Early stopping on KL divergence
            if target_kl is not None:
                if approx_kl > target_kl:
                    print(f"[worker_id {worker_id}]Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f} > {target_kl}")
                    break
        
        # Signal workers that update is complete
        update_barrier.wait()
        
        if update % 10 == 0:
            '''
            if worker_id == 0:
                torch.save({
                    "model_state_dict": global_agent.state_dict()
                }, f"./model_phase2_sub1/ppo_weights_{update}.pt")
            '''
            print(f"Update {update}/{num_updates}, Step {global_step}, Episodes collected: {len(results)}")
            if len(results) > 0:
                recent_rewards = [r for s, r in results[-10:]]
                print(f"  Mean reward (last 10): {np.mean(recent_rewards):.2f}")
    
    # Cleanup
    for p in workers:
        p.join()
    
    # Collect any remaining episode rewards
    while not ep_queue.empty():
        try:
            worker_id, ep_reward = ep_queue.get_nowait()
            results.append((global_step, ep_reward))
        except:
            break
    
    return results, global_agent