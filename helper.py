import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def selectGreedyAction(net, state):
    #this function gets q-values via the network and selects greedy action from q-values and returns it
    #Your code goes in here

    # converting state to tensor
    device = next(net.parameters()).device
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    # getting Q-values for all the actions from the neural network for some state 's'
    with torch.no_grad():
        q_values = net(state)

    # select action with highest Q-value
    greedyAction = torch.argmax(q_values).item()
    return greedyAction

def decayEpsilon(epsilon, decay_strategy, max_episodes):
    epsilons = np.zeros(max_episodes, dtype=np.float32)
    next_decay_episode = 0

    for decay_type, decay_val in decay_strategy:
        s = decay_val['s']
        e = decay_val['e']

        # Fill gaps before this decay segment
        if next_decay_episode < s:
            epsilons[next_decay_episode:s] = decay_val.get('ival', epsilon)

        if decay_type == "linear":
            epsilons[s:e+1] = np.linspace(decay_val['ival'], decay_val['fval'], e-s+1)

        elif decay_type == "exponential":
            decay_rate = np.log(decay_val['fval']/decay_val['ival']) / (e-s)
            epsilons[s:e+1] = decay_val['ival'] * np.exp(decay_rate * np.arange(e-s+1))

        else:
            epsilons[s:e+1] = decay_val.get('ival', epsilon)

        next_decay_episode = e+1

    # Fill remaining episodes if any
    if next_decay_episode < max_episodes:
        epsilons[next_decay_episode:] = decay_val.get('fval', epsilon)

    return epsilons

def selectEpsilonGreedyAction(net, state, epsilon):
    #this function gets q-values via the network and selects an action from q-values using epsilon greedy strategy
    #and returns it
    #note this function can be used for decaying epsilon greedy strategy,
    #you would need to create a wrapper function that will handle decaying epsilon
    #you can create this wrapper in this helper function section
    #for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results

    #Your code goes in here
    device = next(net.parameters()).device
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = net(state)

    num_actions = q_values.shape[1]

    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = torch.argmax(q_values).item()

    return action

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
            prev = inDim

            for h in hDim:
                layers.append(nn.Linear(prev, h))
                prev = h

            layers.append(nn.Linear(prev, outDim))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))

            logits = self.layers[-1](x)
            return logits

    policyNetwork = PolicyNetwork()
    return policyNetwork

def huberLoss(error, delta):
    #this function calculates the huber loss for the error using the delta parameter

    #Your code goes in here
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic

    hLoss = torch.mean(0.5 * quadratic**2 + delta * linear)
    return hLoss