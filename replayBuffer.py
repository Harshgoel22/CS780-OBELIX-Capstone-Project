from collections import deque
import numpy as np
import torch
import random


class ReplayBuffer():
    def __init__(self, bufferSize, bufferType = 'DQN', **kwargs):
        # this function creates the relevant data-structures, and intializes all relevant variables
        # it can take variable number of parameters like alpha, beta, beta_rate (required for PER)
        # here the bufferType variable can be used to maintain one class for all types of agents
        # using the bufferType parameter in the methods below, you can implement all possible functionalities
        # that could be used for different types of agents
        # permissible values for bufferType = NFQ, DQN, DDQN, D3QN and PER-D3QN

        #Your code goes in here
        self.bufferSize = bufferSize
        self.bufferType = bufferType
        self.actions = ["L45", "L22", "FW", "R22", "R45"]

        # Using deque buffer to store the experiences since deque can manage the experiences in cyclic manner and itself remove the earlier experiences if
        # exceed the limit. We can also use other data structure like torch.array to handle this but it requires another index variable in order to maintain the
        # cyclic nature where we will increment the index as ((index + 1) % bufferSize)
        # So to make the things easier going with deque data structure.
        self.buffer = deque(maxlen=bufferSize)

        # parameters for PER-D3QN
        if bufferType == 'PER-D3QN':
            self.alpha = kwargs.get("alpha")
            self.beta = kwargs.get("beta")
            self.beta_rate = kwargs.get("beta_rate")

            # Selecting deque data-structure for priorities as well for the same reason as above stated
            # One another way is to make deque of pair <experiences, priority>
            # But going with this approach by creating separate data structure for prioirities
            self.priorities = deque(maxlen=bufferSize)

    def store(self, experience):
        #stores the experiences, based on parameters in init it can assign priorities, etc.
        #
        #this function does not return anything
        #
        #Your code goes in here
        if self.bufferType == 'PER-D3QN':
            # Selecting the priority for the observed experience if previously no experience hasn't been collected yet so assigning
            # a priority of 1.0 (we can also assign some other number as well like 2.5, 3.2, etc.) else in order to make this
            # current experience also get selected ahead atleast once assigning it the maximum priority among all the earlier
            # experience's prioirity
            max_priority = 1.0 if len(self.priorities) == 0 else max(self.priorities)

            # finally pushing the experience and max_priority (decided above) into their respective variables
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer.append(experience)

    def update(self, indices, priorities):
        #this is mainly used for PER-DDQN
        #otherwise just have a pass in this method
        #
        #this function does not return anything
        #
        #Your code goes in here
        if self.bufferType != 'PER-D3QN':
            # Algorithm used is not PER-D3QN so simply returning from this method
            return

        # Updating the priority for the collected experieces as per the order of the indices
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def collectExperiences(self, env, state, explorationStrategy, countExperiences, net = None):
        #this method allows the agent to interact with the environment starting from a state and it collects
        #experiences during the interaction, it uses network to get the value function and uses exploration strategy
        #to select action. It collects countExperiences and in case the environment terminates before that it returns
        #the function calling this method needs to handle early termination accordingly.
        #
        #this function does not return anything
        #
        #Your code goes in here
        totalRewards = 0
        totalSteps = 0
        for _ in range(countExperiences):
            # If neural-network is not defined then simply sampling any action from the env
            # else using
            if net is None:
                action = env.action_space.sample()
            else:
                action = explorationStrategy(net, state, self.epsilon)

            # observed the new experience by taking one step in the env
            next_state, reward, done, truncated, _ = env.step(self.actions[action])

            done = done or truncated
            # each experience holds the info of curr_state, action_taken, next_state, reward_got, whether_terminated
            experience = (state, action, reward, next_state, done)

            # finally calling store method defined above to store this observed experience into the replaybuffer after assigning prioirity to it.
            self.store(experience)

            state = next_state

            totalRewards += reward
            totalSteps += 1
            
            # If either the agent reaches the goal or gets truncated because of exceeding the max_steps, then breaks from this loop
            if done:
                break
        return totalRewards, totalSteps

    def sample(self, batchSize, **kwargs):
        # this method returns batchSize number of experiences
        # based on extra arguments, it could do sampling or it could return the latest batchSize experiences or
        # via some other strategy
        #
        # in the case of Prioritized Experience Replay (PER) the sampling needs to take into account the priorities
        #
        # this function returns experiences samples
        #
        #Your code goes in here
        if self.bufferType == 'PER-D3QN':
            # For PER-D3QN, sampling will be done using probability calculation.
            # For all experiences gathered, evaluating probabities using their priority.
            # prob for ith experience will be [ priority[i]^alpha / summation_j (priority[j]^alpha) ]
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            # Based on probabilities, sampling batchSize number of experiences from the replay_buffer by first choosing the indices randomly based on probabilities
            # then simply preparing a list of expeirnces of those slected indices.
            indices = np.random.choice(self.length(), batchSize, replace=True, p=probs)
            experiencesList = [self.buffer[idx] for idx in indices]

            N = self.length()
            updatedBeta = min(1.0, self.beta + self.beta_rate * kwargs['current_step'])
            weights = (N * probs[indices]) ** (updatedBeta)

            self.current_step += 1
            # normalizing the weights for stability
            weights /= weights.max()
            return experiencesList, indices, weights
        
        else:
            experiencesList = random.sample(self.buffer, batchSize)
            return experiencesList
        
    def splitExperiences(self, experiences):
        #it takes in experiences and gives the following:
        #states, actions, rewards, nextStates, dones
        #
        #Your code goes in here
        #
        states = []
        actions = []
        rewards = []
        nextStates = []
        dones = []

        # spliting the curr_states, action_taken, reward_got, next_states, done(whether_terminated)
        # in 5 different lists
        for exp in experiences:
            s, a, r, ns, done = exp
            states.append(s)
            actions.append(a)
            rewards.append(r)
            nextStates.append(ns)
            dones.append(done)

        # converting python list into numpy array
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        nextStates = np.array(nextStates)
        dones = np.array(dones)

        return states, actions, rewards, nextStates, dones
    
    def length(self):
        #tells the number of experiences stored in the internal buffer
        #
        #Your code goes in here
        #
        buffersize = len(self.buffer)
        return buffersize