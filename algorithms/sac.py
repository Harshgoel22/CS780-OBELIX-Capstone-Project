import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import time
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time
from itertools import count


# ============================================
# DISCRETE VALUE NETWORK (Q-Network)
# ============================================

class DiscreteValueNetwork(nn.Module):
    """
    Q-network for discrete actions.
    Outputs Q-value for EACH possible action (no action input needed).
    """
    def __init__(self, stateDim, actionDim, hiddenDims):
        super(DiscreteValueNetwork, self).__init__()
        
        layers = []
        prev_dim = stateDim
        
        for dim in hiddenDims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, actionDim))  # Output Q for each action
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Args:
            state (tensor): shape (batch_size, stateDim)
            
        Returns:
            q_values (tensor): Q(s,a) for all actions, shape (batch_size, actionDim)
        """
        return self.network(state)


# ============================================
# DISCRETE POLICY NETWORK
# ============================================

class DiscretePolicyNetwork(nn.Module):
    """
    Policy network for discrete action spaces.
    Outputs probability distribution over actions.
    """
    def __init__(self, stateDim, actionDim, hiddenDims):
        super(DiscretePolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = stateDim
        
        for dim in hiddenDims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.network = nn.Sequential(*layers)
        self.logits = nn.Linear(prev_dim, actionDim)  # Output action logits

    def forward(self, state, action=None, deterministic=False):
        """
        Args:
            state (tensor): shape (batch_size, stateDim)
            action (tensor, optional): action indices shape (batch_size,)
            deterministic (bool): if True, return argmax action
            
        Returns:
            action (tensor): sampled or selected action indices
            log_prob (tensor): log probability of actions
            entropy (tensor): policy entropy
        """
        features = self.network(state)
        logits = self.logits(features)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy


# ============================================
# REPLAY BUFFER (Unchanged)
# ============================================

class ReplayBuffer:
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.buffer = deque(maxlen=bufferSize)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batchSize):
        batch = random.sample(self.buffer, batchSize)
        return batch

    def splitExperiences(self, experiences):
        states, actions, rewards, nextStates, dones = [], [], [], [], []
        
        for exp in experiences:
            s, a, r, ns, done = exp
            states.append(s)
            actions.append(a)  # Now integer indices
            rewards.append(r)
            nextStates.append(ns)
            dones.append(done)

        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(nextStates), np.array(dones))

    def length(self):
        return len(self.buffer)


# ============================================
# DISCRETE SAC AGENT
# ============================================

class DiscreteSAC:
    def __init__(self, env, gamma, tau, bufferSize, batchSize, entropyLR,
                 updateFrequency, policyOptimizerFn, valueOptimizerFn_1,
                 valueOptimizerFn_2, policyOptimizerLR, valueOptimizerLR,
                 alphaOptimizerFn, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODE, 
                 hDims, minSamples, device=None, model_path=None, target_entropy=None):
        """
        SAC for Discrete Action Spaces.
        
        Args:
            env: gym environment
            gamma: discount factor
            tau: target network update rate
            target_entropy: target entropy (default: -log(1/|A|) * 0.5)
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.minSamples = minSamples
        self.updateFrequency = updateFrequency
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODE = MAX_EVAL_EPISODE
        self.MAX_VALUE_GRAD_NORM = 0.5
        self.MAX_POLICY_GRAD_NORM = 0.5
        self.entropyLR = entropyLR
        self.device = device
        self.model_path = model_path

        # OBELIX Environment Dimensions
        self.stateDim = 18
        self.actionDim = 5
        self.actions = ["L45", "L22", "FW", "R22", "R45"]  # 5 discrete actions

        # Discrete Q-Networks: output Q(s, a) for all actions
        self.targetValueNetwork1 = DiscreteValueNetwork(self.stateDim, self.actionDim, hiddenDims=hDims)
        self.targetValueNetwork2 = DiscreteValueNetwork(self.stateDim, self.actionDim, hiddenDims=hDims)
        self.onlineValueNetwork1 = DiscreteValueNetwork(self.stateDim, self.actionDim, hiddenDims=hDims)
        self.onlineValueNetwork2 = DiscreteValueNetwork(self.stateDim, self.actionDim, hiddenDims=hDims)
        
        # Discrete Policy: outputs action probabilities
        self.policyNetwork = DiscretePolicyNetwork(self.stateDim, self.actionDim, hiddenDims=hDims)

        # Entropy temperature
        self.logAlpha = torch.tensor(0.0, requires_grad=True)
        
        # Target entropy for discrete: usually -log(1/|A|) or scaled version
        if target_entropy is None:
            self.target_entropy = -np.log(1.0 / self.actionDim) * 0.5
        else:
            self.target_entropy = target_entropy

        # Optimizers
        self.alphaOptimizer = alphaOptimizerFn([self.logAlpha], lr=entropyLR)
        self.valueOptimizerFn_1 = valueOptimizerFn_1(self.onlineValueNetwork1.parameters(), lr=valueOptimizerLR)
        self.valueOptimizerFn_2 = valueOptimizerFn_2(self.onlineValueNetwork2.parameters(), lr=valueOptimizerLR)
        self.policyOptimizerFn = policyOptimizerFn(self.policyNetwork.parameters(), lr=policyOptimizerLR)

        self.updateTargetNetworks()
        self.startTime = time.time()
        self.rBuffer = ReplayBuffer(bufferSize)

    def selectAction(self, state, deterministic=False):
        """
        Select action from discrete policy.
        
        Args:
            state (array): current state (18-dim)
            deterministic (bool): if True, use argmax (greedy)
            
        Returns:
            action_idx (int): index 0-4 corresponding to ["L45", "L22", "FW", "R22", "R45"]
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_idx, _, _ = self.policyNetwork(state, deterministic=deterministic)
        return action_idx.item()  # Return integer 0-4

    def trainNetworks(self, batch, episode=None):
        """
        Train critics and policy for discrete actions.
        """
        ss, actions, rs, sNexts, dones = self.rBuffer.splitExperiences(batch)

        # Convert to tensors
        ss = torch.tensor(ss, dtype=torch.float32)           # (batch, 18)
        actions = torch.tensor(actions, dtype=torch.long)      # (batch,) - integer indices!
        rs = torch.tensor(rs, dtype=torch.float32).unsqueeze(1)  # (batch, 1)
        sNexts = torch.tensor(sNexts, dtype=torch.float32)    # (batch, 18)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # (batch, 1)

        # ==================== CRITIC UPDATE ====================
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs, _ = self.policyNetwork(sNexts)  # next_actions: (batch,)
            
            # Get Q-values for all actions, then select the ones we sampled
            q1_next = self.targetValueNetwork1(sNexts)  # (batch, 5)
            q2_next = self.targetValueNetwork2(sNexts)  # (batch, 5)
            
            # Select Q-values for the sampled actions
            q1_next_selected = q1_next.gather(1, next_actions.unsqueeze(1))  # (batch, 1)
            q2_next_selected = q2_next.gather(1, next_actions.unsqueeze(1))  # (batch, 1)
            
            # Soft state value: V(s) = E[Q(s,a) - alpha*log(pi(a|s))]
            q_next = torch.min(q1_next_selected, q2_next_selected)
            alpha = self.logAlpha.exp()
            target_q = rs + self.gamma * (1 - dones) * (q_next - alpha * next_log_probs.unsqueeze(1))

        # Current Q-values for taken actions
        q1 = self.onlineValueNetwork1(ss)  # (batch, 5)
        q2 = self.onlineValueNetwork2(ss)  # (batch, 5)
        
        # Select Q-values for actions that were actually taken
        q1_selected = q1.gather(1, actions.unsqueeze(1))  # (batch, 1)
        q2_selected = q2.gather(1, actions.unsqueeze(1))  # (batch, 1)
        
        # Critic losses
        q1_loss = F.mse_loss(q1_selected, target_q)
        q2_loss = F.mse_loss(q2_selected, target_q)

        # Update critics
        self.valueOptimizerFn_1.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineValueNetwork1.parameters(), self.MAX_VALUE_GRAD_NORM)
        self.valueOptimizerFn_1.step()

        self.valueOptimizerFn_2.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineValueNetwork2.parameters(), self.MAX_VALUE_GRAD_NORM)
        self.valueOptimizerFn_2.step()

        # ==================== POLICY UPDATE ====================
        # Freeze critics
        for p in self.onlineValueNetwork1.parameters():
            p.requires_grad = False
        for p in self.onlineValueNetwork2.parameters():
            p.requires_grad = False

        # Sample actions and get log probs from current policy
        sampled_actions, log_probs, entropy = self.policyNetwork(ss)
        
        # Get Q-values for sampled actions
        q1_current = self.onlineValueNetwork1(ss)
        q2_current = self.onlineValueNetwork2(ss)
        q_current = torch.min(q1_current, q2_current)
        
        # Select Q for sampled actions
        q_selected = q_current.gather(1, sampled_actions.unsqueeze(1)).squeeze(1)
        
        # Policy loss: maximize E[Q(s,a) - alpha*log(pi(a|s))]
        alpha = self.logAlpha.exp()
        policyLoss = (alpha * log_probs - q_selected).mean()

        self.policyOptimizerFn.zero_grad()
        policyLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.MAX_POLICY_GRAD_NORM)
        self.policyOptimizerFn.step()

        # Unfreeze critics
        for p in self.onlineValueNetwork1.parameters():
            p.requires_grad = True
        for p in self.onlineValueNetwork2.parameters():
            p.requires_grad = True

        # ==================== ALPHA UPDATE ====================
        # Adjust temperature to maintain target entropy
        alpha_loss = -(self.logAlpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alphaOptimizer.zero_grad()
        alpha_loss.backward()
        self.alphaOptimizer.step()

    def updateTargetNetworks(self):
        """Polyak averaging for target networks."""
        for target_param, online_param in zip(self.targetValueNetwork1.parameters(), 
                                              self.onlineValueNetwork1.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

        for target_param, online_param in zip(self.targetValueNetwork2.parameters(), 
                                              self.onlineValueNetwork2.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    def trainAgent(self):
        """Run training episodes."""
        self.updateTargetNetworks()
        trainRewards = []
        trainTimes = []

        for e in range(self.MAX_TRAIN_EPISODES):
            rewardPerEpisode = 0.0
            s = self.env.reset()
            done = False
            
            while not done:
                # Select discrete action index (0-4)
                a_idx = self.selectAction(s, deterministic=False)
                
                # Execute action in environment
                s_next, r, done = self.env.step(self.actions[a_idx], render=False)
                
                rewardPerEpisode += float(r)
                
                # Store transition with integer action index
                experience = (s, a_idx, r, s_next, done)  # a_idx is 0,1,2,3,4
                self.rBuffer.store(experience)

                # Train if enough samples
                if self.rBuffer.length() > self.minSamples:
                    experiences = self.rBuffer.sample(self.batchSize)
                    self.trainNetworks(experiences, e)
                
                s = s_next

            # Update target networks periodically
            if e % self.updateFrequency == 0:
                self.updateTargetNetworks()

            trainRewards.append(rewardPerEpisode)
            trainTimes.append(time.time() - self.startTime)
            print(f'[Episode {e+1}] Train reward Sum: {rewardPerEpisode:.2f}')

            if (e+1) % 10 == 0:
                torch.save(self.policyNetwork.state_dict(), f"{self.model_path}/sac_policy_{e+1}.pth")

        return trainRewards, trainTimes

    def evaluateAgent(self):
        """Evaluate with greedy policy."""
        finalEvalRewardsList = []
        
        for e in range(self.MAX_EVAL_EPISODE):
            episodeReward = 0.0
            s = self.env.reset()
            done = False
            
            while not done:
                # Greedy action selection (deterministic)
                a_idx = self.selectAction(s, deterministic=True)
                s_next, r, done = self.env.step(self.actions[a_idx])
                episodeReward += r
                s = s_next
                
            finalEvalRewardsList.append(episodeReward)
            
        return finalEvalRewardsList

    def runSAC(self):
        """Full training pipeline."""
        train_rewards, training_time = self.trainAgent()
        eval_score = self.evaluateAgent()
        return train_rewards, eval_score, training_time