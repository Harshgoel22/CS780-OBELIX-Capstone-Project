import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

def setup_logger(log_file):
    logger = logging.getLogger("A2C")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def plotQuantity(quantityList, totalEpisodeCount, descriptionList):
    """
    Plots a quantity over episodes.

    quantityList: list of values to plot
    totalEpisodeCount: number of episodes
    descriptionList: [x_label, y_label, title, optional legend]
    """
    
    plt.figure(figsize=(10, 5))  # Correct figure size
    episodes = np.arange(totalEpisodeCount)
    estimates = np.array(quantityList)    
    
    label = descriptionList[3] if len(descriptionList) > 3 else None
    plt.plot(episodes, estimates, label=label)

    plt.xlabel(descriptionList[0])
    plt.ylabel(descriptionList[1])
    plt.title(descriptionList[2])

    if label:
        plt.legend()  # Show legend only if label exists

    plt.show()


def decayEpsilon(epsilon, decay_strategy, max_episodes):
    """
    Returns a numpy array of epsilon values per episode according to a decay strategy.

    decay_strategy: list of tuples (decay_type, decay_val)
        decay_val = {'s': start_episode, 'e': end_episode, 'ival': initial_value, 'fval': final_value}
    """
    epsilons = np.zeros(max_episodes, dtype=np.float32)
    next_decay_episode = 0

    for decay_type, decay_val in decay_strategy:
        s = decay_val['s']
        e = decay_val['e']

        # Fill gaps before this decay segment
        if next_decay_episode < s:
            epsilons[next_decay_episode:s] = decay_val.get('ival', epsilon)

        # Length of current decay segment
        length = e - s + 1
        length = min(length, len(epsilons) - s)  # avoid overshooting

        if decay_type == "linear":
            start_val = decay_val['ival']
            end_val = decay_val['fval']
            epsilons[s:s+length] = np.linspace(start_val, end_val, num=length, dtype=np.float32)

        elif decay_type == "exponential":
            start_val = decay_val['ival']
            end_val = decay_val['fval']
            # decay rate
            decay_rate = np.log(end_val / start_val) / (length - 1)
            epsilons[s:s+length] = start_val * np.exp(decay_rate * np.arange(length))

        else:
            # constant value
            epsilons[s:s+length] = decay_val.get('ival', epsilon)

        next_decay_episode = e + 1

    # Fill remaining episodes if any
    if next_decay_episode < max_episodes:
        epsilons[next_decay_episode:] = decay_val.get('fval', epsilon)

    return epsilons


def selectGreedyAction(net, state, useLSTM=False, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)  # [1, state_dim]

    with torch.no_grad():
        if not useLSTM:
            q_values = net(state)
        elif 'time_frac' not in kwargs:
            q_values, hx, cx = net(state, kwargs['hx'], kwargs['cx'])
        else:
            q_values, hx, cx, mask = net(state, kwargs['hx'], kwargs['cx'], kwargs['time_frac'])

    greedyAction = torch.argmax(q_values, dim=1).item()

    return (greedyAction, hx, cx) if useLSTM else greedyAction


def selectEpsilonGreedyAction(net, state, epsilon, useLSTM=False, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    
    sampleAction = False if 'sampleAction' not in kwargs else kwargs['sampleAction']

    with torch.no_grad():
        if not useLSTM:
            q_values = net(state)
        elif 'time_frac' not in kwargs:
            q_values, hx, cx = net(state, kwargs['hx'], kwargs['cx'])
        else:
            q_values, hx, cx, mask = net(state, kwargs['hx'], kwargs['cx'], kwargs['time_frac'])

    num_actions = q_values.shape[1]

    if np.random.rand() < epsilon:
        if sampleAction:
            mask_np = mask.cpu().numpy().flatten()
            mask_np = np.nan_to_num(mask_np, nan=0.0)

            if mask_np.sum() == 0:
                prob = np.ones_like(mask_np) / len(mask_np)
            else:
                # softmax over mask for smooth probabilities
                mask_np = mask_np - mask_np.max()  # numerical stability
                exp_mask = np.exp(mask_np)
                prob = exp_mask / exp_mask.sum()

            action = np.random.choice(num_actions, p=prob)
        else:
            action = np.random.randint(num_actions)
    else:
        action = torch.argmax(q_values, dim=1).item()

    return (action, hx, cx) if useLSTM else action
    

def selectSoftMaxAction(net, state, temp):
    #this function gets q-values via the network and selects an action from q-values using softmax strategy
    #and returns it
    #note this function can be used for decaying temperature softmax strategy,
    #you would need to create a wrapper function that will handle decaying temperature
    #you can create this wrapper in this helper function section
    #for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results

    #Your code goes in here
    state = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = net(state)

    # Apply temperature-scaled softmax
    probabilities = F.softmax(q_values / temp, dim=1).cpu().numpy()[0]

    # Sample action according to probability distribution
    softAction = np.random.choice(len(probabilities), p=probabilities)
    return softAction

#Value Network
def createValueNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value for each possible action

    #Your code goes in here
    class ValueNetwork(nn.Module):
        def __init__(self):
            super(ValueNetwork, self).__init__()
            
            layers = []
            prev_dim = inDim

            # Hidden layers
            for h in hDim:
                layers.append(nn.Linear(prev_dim, h))
                prev_dim = h

            # Output layer
            layers.append(nn.Linear(prev_dim, outDim))

            self.layers = nn.ModuleList(layers)
            self.activation = activation

        def forward(self, x):
            # applying activation to all layers except the last
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))

            # Final layer (no activation for Q-values)
            x = self.layers[-1](x)
            return x

    valueNetwork = ValueNetwork()
    return valueNetwork

#Dueling Network
def createDuelingNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value which is derived
    #internally from action-advantage function and v-function,
    #Note we center the advantage values, basically we subtract the mean from each state-action value

    #Your code goes in here
    class DuelingNetwork(nn.Module):
        def __init__(self):
            super(DuelingNetwork, self).__init__()
            self.activation = activation

            # Common layers
            layers = []
            prev = inDim
            for h in hDim:
                layers.append(nn.Linear(prev, h))
                prev = h
            self.common = nn.ModuleList(layers)

            # Value stream
            self.valueNet = nn.Linear(prev, 1)

            # Advantage stream
            self.advantageNet = nn.Linear(prev, outDim)

        def forward(self, x):
            for layer in self.common:
                x = self.activation(layer(x))

            V = self.valueNet(x)
            A = self.advantageNet(x)

            # Mean advantage
            A_mean = torch.mean(A, dim=1, keepdim=True)

            Q = V + (A - A_mean)
            return Q

    duelNetwork = DuelingNetwork()
    return duelNetwork

def huberLoss(error, delta, weights=None, norm=False):
    #this function calculates the huber loss for the error using the delta parameter

    #Your code goes in here
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic

    loss = 0.5 * quadratic**2 + delta * linear

    if weights is None:
        hLoss = torch.mean(loss)
    elif not norm:
        hLoss = torch.mean(weights * loss)
    else:
        hLoss = weights * loss
        hLoss = hLoss.sum() / weights.sum()
    return hLoss


## ======================== For Policy based RL Agents ======================================

#Policy Network
def createPolicyNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return action logit vector
    #Your code goes in here
    class PolicyNetwork(nn.Module):
        def __init__(self):
            super(PolicyNetwork, self).__init__()
            self.activation = activation
            layers = []
            prev_dim = inDim

            # Hidden layers
            for h in hDim:
                layers.append(nn.Linear(prev_dim, h))
                prev_dim = h

            layers.append(nn.Linear(prev_dim, outDim))
            
            self.layers = nn.ModuleList(layers)
            self.activation = activation

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))

            x = self.layers[-1](x)
            probs = torch.softmax(x, dim=-1)
            return probs

    policyNetwork = PolicyNetwork()
    return policyNetwork


def selectPolicyGreedyAction(policy_network, s, ACTIONS):
    device = next(policy_network.parameters()).device
    state = torch.FloatTensor(s).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = policy_network(state)

    action_idx = torch.argmax(probs, dim=-1)
    action = ACTIONS[action_idx.item()]

    return action


def selectPolicyAction(policy_network, s, ACTIONS):
    device = next(policy_network.parameters()).device
    state = torch.FloatTensor(s).unsqueeze(0).to(device)

    probs = policy_network(state)
    dist = torch.distributions.Categorical(probs)

    action_idx = dist.sample()
    logp_a = dist.log_prob(action_idx)
    entropy = dist.entropy()

    action = ACTIONS[action_idx.item()]

    return action, logp_a.squeeze(), entropy.squeeze()

def getStepWiseReturnsAndDiscounts(gamma, reward, device=torch.device('cpu'), next_value=0.0):
    rewards = torch.tensor(reward, dtype=torch.float32).to(device)
    
    size = rewards.shape[-1]
    returns = torch.zeros(size, dtype=torch.float32).to(device)
    gammas = (gamma ** torch.arange(size, dtype=torch.float32)).to(device)

    G = next_value
    for t in range(size-1,-1,-1):
        G = rewards[t] + gamma * G
        returns[t] = G
        
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns, gammas


def evaluate_policy(agent, envClass, envConfig, episodes=1):
    env = envClass(**envConfig)
    rewards = []

    for _ in range(episodes):
        s = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(s).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(s_tensor)

            if agent.is_discrete:
                action = action.item()
            else:
                action = action.squeeze(0).cpu().numpy()
                action = np.clip(action, 0, agent.act_dim)

            s, r, done = env.step(agent.actions[action])
            total_reward += r
        rewards.append(total_reward)

    return np.mean(rewards)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()


# --------------------------------------------------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, inDim, outDim, hDim=[64,64], activation=F.relu):
        super(PolicyNetwork, self).__init__()
        self.activation = activation
        layers = []
        prev_dim = inDim

        # Hidden layers
        for h in hDim:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, outDim))
        
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        probs = torch.softmax(x, dim=-1)
        return probs
    

class ValueNetwork(nn.Module):
    def __init__(self, inDim, outDim, hDim=[64,64], activation=F.relu):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = inDim

        # Hidden layers
        for h in hDim:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, outDim))
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        # applying activation to all layers except the last
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # Final layer (no activation for Q-values)
        x = self.layers[-1](x)
        return x
    

# ========================================================================================================================

def selectGreedyActionREP(net, state, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)  # [1, state_dim]

    with torch.no_grad():
        q_values, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])

    greedyAction = torch.argmax(q_values, dim=1).item()

    return greedyAction, hx, cx


def selectEpsilonGreedyActionREP(net, state, epsilon, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    
    with torch.no_grad():
        q_values, z_t, hx, cx = net(state, kwargs['hx'], kwargs['cx'])

    num_actions = q_values.shape[1]

    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = torch.argmax(q_values, dim=1).item()

    return action, hx, cx

# ================================================================================

def selectGreedyActionREP_COMP(net, state, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)  # [1, state_dim]

    using_kd = kwargs.get('kd', False)
    with torch.no_grad():
        if not using_kd:
            q_values, _, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])
        else:
            q_values, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])

    greedyAction = torch.argmax(q_values, dim=1).item()

    return greedyAction, hx, cx


def selectEpsilonGreedyActionREP_COMP(net, state, epsilon, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    
    using_kd = kwargs.get('kd', False)
    with torch.no_grad():
        if not using_kd:
            q_values, _, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])
        else:
            q_values, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])

    num_actions = q_values.shape[1]

    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = torch.argmax(q_values, dim=1).item()

    return action, hx, cx


# ================================== bad bump ==========================================


def selectEpsilonGreedyActionREP_NOISY(net, state, epsilon, **kwargs):
    device = next(net.parameters()).device

    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    
    using_kd = kwargs.get('kd', False)
    with torch.no_grad():
        if not using_kd:
            q_values, _, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])
        else:
            q_values, _, hx, cx = net(state, kwargs['hx'], kwargs['cx'])

    q_values = q_values.squeeze(0)
    
    # actions = ["L45", "L22", "FW", "R22", "R45"]
    prob_dist = torch.tensor([0.15, 0.15, 0.40, 0.15, 0.15], device=device)
   
    # Noisy exploration
    if np.random.rand() >= epsilon:
        noise_std = kwargs.get('noise_std', 0.2)
        episode = kwargs.get('episode', 0)
        max_episodes = kwargs.get('max_episodes', 800)

        # decay from noise_std → 0.02 over training
        decay_factor = max(0.1, 1.0 - episode / max_episodes)
        effective_std = noise_std * decay_factor

        noise_q = q_values + effective_std * torch.randn_like(q_values)
        action = torch.argmax(noise_q).item()

    else:
        action = torch.multinomial(prob_dist, 1).item()

    return action, hx, cx


def compute_td_lambda_returns(rewards, max_q, dones, gamma=0.99, lambda_=0.9):
    """
    rewards:  (B, T)
    max_q:    (B, T)   -> Double DQN bootstrap values
    dones:    (B, T)

    returns:  (B, T)
    """
    B, T = rewards.shape
    device = rewards.device

    returns = torch.zeros(B, T, device=device)

    # initialize with last step bootstrap
    returns[:, -1] = rewards[:, -1] + gamma * max_q[:, -1] * (1 - dones[:, -1])

    for t in reversed(range(T - 1)):
        mask = 1 - dones[:, t]

        returns[:, t] = rewards[:, t] + gamma * mask * (
            (1 - lambda_) * max_q[:, t] +
            lambda_ * returns[:, t + 1]
        )

    return returns
