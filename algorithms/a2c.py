import torch
import torch.nn as nn
import time
import numpy as np
import torch.multiprocessing as mp
import sys
import os
sys.path.append("..")
from helper import ValueNetwork, PolicyNetwork, evaluate_policy, SharedAdam, setup_logger

class A2CAgent(nn.Module):
    def __init__(self, env, obs_dim, act_dim, is_discrete=True):
        # Initialize the A2C agent.
        # Tasks:
        # - Determine the dimensions of the observation space.
        # - Determine the number of possible actions.
        # - Create the critic network (takes state, outputs V(s)).
        # - Create the actor network (takes state, outputs action logits).

        # Your code goes here
        super(A2CAgent, self).__init__()
        self.env = env
        self.obs_dim = obs_dim
        self.is_discrete = is_discrete
        self.act_dim = act_dim
        self.actions = ["L45", "L22", "FW", "R22", "R45"]

        self.hDim = [64, 64]
        self.actor_network = PolicyNetwork(self.obs_dim, self.act_dim, hDim=self.hDim)
        self.critic_network = ValueNetwork(self.obs_dim, 1, hDim=self.hDim)

        if not self.is_discrete:
            self.actor_log_std = nn.Parameter(torch.zeros(self.act_dim))

        self.initBookKeeping()

    def initBookKeeping(self):
        self.train_rewards = []
        self.eval_rewards = []
        self.train_time = []
        self.total_steps = []
        self.wallclock_time = []
        self.start_time = time.time()

    def performBookKeeping(self, train=True):
        if train:
            self.train_time.append(time.time() - self.start_time)
        else:
            self.wallclock_time.append(time.time() - self.start_time)

    def get_value(self, x):
        """Return the value of a state from the critic network."""
        # Your code goes here
        return self.critic_network(x)

    def get_action_and_value(self, x, action=None):
        """
        Get an action from the actor and value from the critic.
        Tasks:
        - Use the actor network to produce logits over actions.
        - Convert logits into a probability distribution.
        - Sample an action from the distribution (if action is not provided).
        - Compute the log probability of the chosen action.
        - Compute the entropy of the distribution (helps with exploration).
        - Get the value of the state from the critic.

        Returns: action, log_prob, entropy, value
        """
        # Your code goes here
        value = self.get_value(x)

        if self.is_discrete:
            logits = self.actor_network(x)
            dist = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            else:
                action = action.long().squeeze()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        else:
            mu = self.actor_network(x)
            log_std = torch.clamp(self.actor_log_std, -20, 2)
            std = log_std.exp().expand_as(mu)
            dist = torch.distributions.Normal(mu, std)
            if action is None:
                action = dist.sample()
            else:
                action = action.float()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value
    
def a2c_worker(worker_id, global_agent,
               envClass, envConfig, seed, num_steps, num_updates,
               shared_obs, shared_actions, shared_logprobs,
               shared_rewards, shared_dones, shared_values,
               shared_next_obs, shared_next_done,
               collect_barrier, update_barrier, ep_queue):
    """
    A2C worker process.  Collects rollouts into shared memory buffers,
    then waits at a barrier for the main process to do a centralised update.

    Unlike A3C, this worker NEVER computes gradients or calls optimizer.step().
    The main process owns the gradient update exclusively.
    """

    #Your code goes here
    env = envClass(**envConfig)
    env.reset(seed=seed + worker_id)
    is_discrete = global_agent.is_discrete

    s = env.reset()

    for _ in range(num_updates):
        total_reward = 0.0
        ep_count = 0

        for step in range(num_steps):
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = global_agent.get_action_and_value(s_tensor)

            if is_discrete:
                action_np = action.item()
                action_store = action
            else:
                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.tanh(action_np)
                action_np = action_np * global_agent.act_dim
                action_np = np.clip(action_np, 0, global_agent.act_dim)
                action_store = action.squeeze(0)

            s_next, reward, done = env.step(global_agent.actions[action_np], render=False)
            total_reward += reward

            shared_obs[worker_id, step] = s_tensor.squeeze(0)
            shared_actions[worker_id, step] = action_store
            shared_logprobs[worker_id, step] = logprob
            shared_rewards[worker_id, step] = reward
            shared_dones[worker_id, step] = float(done)
            shared_values[worker_id, step] = value.squeeze(0)

            s = s_next
            if done:
                ep_queue.put(total_reward)
                total_reward = 0.0
                ep_count += 1
                s = env.reset()

        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        shared_next_obs[worker_id] = s_tensor.squeeze(0)
        shared_next_done[worker_id] = float(done)

        collect_barrier.wait()
        update_barrier.wait()


def train_a2c(envClass, envConfig, seed, total_timesteps, num_envs, num_steps, lr,
              gamma, gae_lambda, use_gae, ent_coef, vf_coef,
              max_grad_norm, norm_adv, model_path, log_path):
    """
    A2C training with torch.multiprocessing.

    - num_envs worker processes each run their own environment.
    - Workers collect num_steps of experience into shared memory buffers.
    - A barrier synchronises: all workers must finish before the update.
    - The main process computes GAE advantages and does ONE gradient step.
    - A second barrier signals workers that weights have been updated.
    """

    #Your code goes here
    logger = setup_logger(log_path)
    env = envClass(**envConfig)
    obs_dim = 18
    act_dim = 5

    global_agent = A2CAgent(env, obs_dim, act_dim)
    global_agent.share_memory()
    is_discrete = global_agent.is_discrete

    optimizer = SharedAdam(global_agent.parameters(), lr=lr)

    num_updates = total_timesteps // (num_envs * num_steps)
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // 4
    
    if is_discrete:
        shared_actions = torch.zeros(num_envs, num_steps).long().share_memory_()
    else:
        shared_actions = torch.zeros(num_envs, num_steps, act_dim).share_memory_()
    shared_obs = torch.zeros(num_envs, num_steps, obs_dim).share_memory_()
    shared_logprobs = torch.zeros(num_envs, num_steps).share_memory_()
    shared_rewards = torch.zeros(num_envs, num_steps).share_memory_()
    shared_dones = torch.zeros(num_envs, num_steps).share_memory_()
    shared_values = torch.zeros(num_envs, num_steps).share_memory_()

    shared_next_obs = torch.zeros(num_envs, obs_dim).share_memory_()
    shared_next_done = torch.zeros(num_envs).share_memory_()

    collect_barrier = mp.Barrier(num_envs + 1)
    update_barrier = mp.Barrier(num_envs + 1)
    ep_queue = mp.Queue()

    train_rewards = []
    eval_rewards = []
    train_times = []
    total_steps_list = []
    wallclock_times = []

    total_steps = 0 
    start_time = time.time()

    processes = []
    for i in range(num_envs):
        p = mp.Process(target=a2c_worker, args=(
            i, global_agent,
            envClass, envConfig, seed, num_steps, num_updates,
            shared_obs, shared_actions, shared_logprobs,
            shared_rewards, shared_dones, shared_values,
            shared_next_obs, shared_next_done,
            collect_barrier, update_barrier, ep_queue
        ))
        p.start()
        processes.append(p)

    for update in range(num_updates):
        collect_barrier.wait()

        steps_this_update = num_envs * num_steps
        total_steps += steps_this_update

        with torch.no_grad():
            next_value = global_agent.get_value(shared_next_obs).squeeze(-1)

        advantages = torch.zeros_like(shared_rewards)
        lastgaelam = torch.zeros(num_envs)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - shared_next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - shared_dones[:, t + 1]
                nextvalues = shared_values[:, t + 1]

            delta = shared_rewards[:, t] + gamma * nextvalues * nextnonterminal - shared_values[:, t]

            if use_gae:
                advantages[:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            else:
                advantages[:, t] = delta

        returns = advantages + shared_values

        b_obs = shared_obs.reshape(-1, obs_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        if is_discrete:
            b_actions = shared_actions.reshape(-1)
        else:
            b_actions = shared_actions.reshape(-1, act_dim)

        if norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_inds = np.arange(batch_size)
        num_epochs = 3

        for _ in range(num_epochs):
            np.random.shuffle(batch_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_inds[start:end]

                _, new_logprob, entropy, value = global_agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds].long() if is_discrete else b_actions[mb_inds]
                )
                value = value.view(-1)

                policy_loss = -(b_advantages[mb_inds] * new_logprob).mean()
                value_loss = ((b_returns[mb_inds] - value) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(global_agent.parameters(), max_grad_norm)
                optimizer.step()

        update_barrier.wait()

        train_times.append(time.time() - start_time)
        total_steps_list.append(total_steps)

        ep_rewards_this_update = []
        while not ep_queue.empty():
            try:
                ep_r = ep_queue.get_nowait()
                train_rewards.append(ep_r)
                ep_rewards_this_update.append(ep_r)
            except:
                break

        if (update+1) % 10 == 0:
            eval_r = evaluate_policy(global_agent, envClass, envConfig)
            eval_rewards.append(eval_r)
            wallclock_times.append(time.time() - start_time)

            avg_train_r = np.mean(ep_rewards_this_update) if ep_rewards_this_update else (
                np.mean(train_rewards[-10:]) if train_rewards else 0
            )

            msg = f"     [Update {update+1}/{num_updates}] Steps: {total_steps} | Train Reward: {avg_train_r:.2f} | Eval Reward: {eval_r:.2f} | Loss: {loss.item():.3f}"
            logger.info(msg)

            torch.save(global_agent.state_dict(), f'{model_path}/g_agent_w_e{update+1}.pth')

    for p in processes:
        p.join()

    return train_rewards, eval_rewards, train_times, total_steps_list, wallclock_times