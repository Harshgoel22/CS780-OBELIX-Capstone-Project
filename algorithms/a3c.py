import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configurations.config_p1_sub8_v2 import config
from helper import getStepWiseReturnsAndDiscounts, SharedAdam
from obelix import OBELIX

class A3CAgent(nn.Module):
    """
    A3C agent with separate actor and critic networks.
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, obs_dim, act_dim, is_discrete=False):
        """
        Initialize actor and critic networks.
        obs_dim     : Dimension of observation space
        act_dim     : Dimension of action space
        is_discrete : Whether the action space is discrete
        """

        # Your code goes here
        super(A3CAgent, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.is_discrete = is_discrete

        self.hDim = [64, 64]
        prev_layer_features = obs_dim
        shared_layers = []
        for h in self.hDim:
            shared_layers.append(nn.Linear(prev_layer_features, h))
            shared_layers.append(nn.ReLU())
            prev_layer_features = h
        self.shared_network = nn.Sequential(*shared_layers)

        self.actor_network = nn.Linear(prev_layer_features, self.act_dim)
        if not is_discrete:
                self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        
        self.critic_network = nn.Linear(prev_layer_features, 1)         

    def get_value(self, x):
        """
        Return the value estimate for a given state using the critic.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.critic_network(self.shared_network(x))


    def get_action_and_value(self, x, action=None):
        """
        Compute action, log probability, entropy, and value.

        For discrete actions:
            create a categorical distribution from actor logits.

        For continuous actions:
            create a normal distribution using mean and std.

        If action is None, sample from the distribution.
        Return action, log probability, entropy, and value.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        value = self.get_value(x)

        if self.is_discrete:
            logits = self.actor_network(self.shared_network(x))
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mu = self.actor_network(self.shared_network(x))
            std = self.actor_log_std.exp().expand_as(mu)
            dist = torch.distributions.Normal(mu, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        if not self.is_discrete:
            # since for continuos env actions can be -inf to +inf, so clipping the actions
            action = torch.clamp(action, -2, 2)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, entropy, value
    

# A3C Worker (truly parallel via torch.multiprocessing)

def a3c_worker(worker_id, global_agent, global_optimizer, grad_lock,
               actions, seed, num_episodes,
               gamma, num_steps, ent_coef, vf_coef, max_grad_norm,
               result_queue):
    """
    A3C worker — runs as a separate OS process.

    Each worker independently:
      1. Syncs local weights from the shared global network.
      2. Collects a rollout of num_steps transitions.
      3. Computes n-step returns and advantage.
      4. Computes gradients on its LOCAL network copy.
      5. Pushes those gradients to the GLOBAL network (under lock).
      6. Sends episode reward to main process via result_queue.

    The grad_lock serialises gradient push + optimizer.step() to prevent
    two workers from corrupting each other's gradients.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        wall_obstacles=False,
        difficulty=0,
        box_speed=2
    )
    env.reset(seed=seed+worker_id)
    np.random.seed(seed+worker_id)
    torch.manual_seed(seed+worker_id)

    local_agent = A3CAgent(
        global_agent.obs_dim, global_agent.act_dim,
        global_agent.is_discrete
    )

    local_agent.load_state_dict(global_agent.state_dict())

    for e in range(num_episodes):
        s = env.reset()
        s = torch.tensor(s, dtype=torch.float32)
        n_steps, steps = 0, 0
        done = False
        total_rewards = 0

        log_probs, values, rewards, entropies = [], [], [], []

        while not done:
            steps += 1
            action, log_prob, entropy, value = local_agent.get_action_and_value(s)
            s_next, reward, done = env.step(actions[action.item()], render=False)
            total_rewards += reward

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            if (steps - n_steps) == num_steps or done:            
                # -------- Bootstrap --------
                if done:
                    next_value = torch.zeros(1, 1)
                else:
                    next_value = local_agent.get_value(s_next).detach()

                # -------- Compute Returns --------
                returns, _ = getStepWiseReturnsAndDiscounts(gamma, rewards, next_value=next_value)
                returns = returns.unsqueeze(1)
                values_tensor = torch.cat(values)
                log_probs_tensor = torch.cat(log_probs)

                advantages = returns - values_tensor
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # -------- Loss --------
                entropies_tensor = torch.stack(entropies)

                policy_loss = -(advantages.detach() * log_probs_tensor).mean()
                entropy_loss = -entropies_tensor.mean()
                value_loss = 0.5 * advantages.pow(2).mean()

                total_loss = policy_loss - ent_coef * entropy_loss + vf_coef * value_loss

                # -------- Backprop --------
                local_agent.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), max_grad_norm)

                with grad_lock:
                    global_optimizer.zero_grad()
                    for lp, gp in zip(local_agent.parameters(), global_agent.parameters()):
                        gp._grad = lp.grad.clone()
                    global_optimizer.step()

                local_agent.load_state_dict(global_agent.state_dict())

                # -------- Reset buffers --------
                rewards, log_probs, entropies, values = [], [], [], []
                n_steps = steps

            s = torch.tensor(s_next, dtype=torch.float32)
            if done:
                break

        if worker_id == 0:
            with open("a3c_results/a3c_episodewise_rewards_log_v2.txt", "a") as f:
                print(f"[worker_id: {worker_id}] After episode {e}: Reward = {total_rewards:.2f}", file=f) 
        result_queue.put(total_rewards)


# A3C Launcher — spawns truly parallel worker processes
def train_a3c(seed, num_workers,
              episodes_per_worker,
              gamma, num_steps, lr,
              ent_coef, vf_coef,
              max_grad_norm):
    """
    Launches A3C training with truly parallel worker processes.

    1. Creates global_agent in shared memory.
    2. Creates SharedAdam (optimizer state also in shared memory).
    3. Creates mp.Lock for gradient push serialisation.
    4. Spawns num_workers mp.Process instances (all start simultaneously).
    5. Main process collects rewards from Queue while workers train.
    6. p.join() waits for all workers to finish.
    """

    # Your code goes here
    obs_dim = 18  # OBELIX observation shape
    actions = ["L45", "L22", "FW", "R22", "R45"]
    act_dim = len(actions)
    is_discrete = True

    global_agent = A3CAgent(obs_dim, act_dim, is_discrete=is_discrete)
    global_agent.share_memory()
    global_optimizer = SharedAdam(global_agent.parameters(), lr=lr)
    grad_lock = mp.Lock()

    result_queue = mp.Queue()
    processes = []
    best_reward = -float("inf")
    
    for worker_id in range(num_workers):
        local_workers = mp.Process(
            target=a3c_worker, 
            args=(
                worker_id, global_agent, global_optimizer, 
                grad_lock, actions, seed, episodes_per_worker,
                gamma, num_steps, ent_coef, vf_coef,
                max_grad_norm, result_queue
            )
        )
        # print(f'Spawned worker with id {worker_id}')
        local_workers.start()
        processes.append(local_workers)
    
    rewards = []
    for _ in range(num_workers * episodes_per_worker):
        r = result_queue.get()
        rewards.append(r)
        if r > best_reward:
            best_reward = r
            torch.save({
                "model_state_dict": global_agent.state_dict(),
                "optimizer_state_dict": global_optimizer.state_dict(),
            }, f"a3c_results/weights_v2.pth")
    
    for p in processes:
        p.join()

    return rewards


if __name__ == "__main__":
    os.makedirs("a3c_results", exist_ok=True)
    rewards = train_a3c(**config)
    np.save(f"a3c_results/rewards_list_v2.npy", rewards)
    print("Saved results!")