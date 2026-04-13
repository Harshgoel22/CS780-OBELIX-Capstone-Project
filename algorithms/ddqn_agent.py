import numpy as np
import torch
import random
import time
from helper import createValueNetwork, decayEpsilon, huberLoss
from replayBuffer import ReplayBuffer

class DDQN():
    def __init__(self, env, config):
        #this DDQN method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates tareget and online Q-networks using the createValueNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer

        #Your code goes in here
        self.seed = config['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.env = env
        self.env.reset(seed=self.seed)

        self.gamma = config['gamma']
        self.epochs = config['epochs']
        self.delta = config['delta']
        

        self.MAX_TRAIN_EPISODES = config['MAX_TRAIN_EPISODES']
        self.MAX_EVAL_EPISODES = config['MAX_EVAL_EPISODES']

        self.bufferSize = config['bufferSize']
        self.batchSize = config['batchSize']
        self.updateFrequency = config['updateFrequency']
        self.max_steps = config['max_steps']
        self.device = config['device']

        self.explorationStrategyTrainFn = config['explorationStrategyTrainFn']
        self.explorationStrategyEvalFn = config['explorationStrategyEvalFn']

        self.epsilons = decayEpsilon(
            config['epsilon'],
            config['eps_decay_strategy'],
            self.MAX_TRAIN_EPISODES)
        
        self.model_path = config['model_path']
        self.loss_fn = config['loss_fn']

        # initializing the bookkeeping variables
        self.initBookKeeping()

        # Since the nerural network expects the dimension of states for the first layer
        # so first extract it then extract number of actions because that many neurons will be
        # at the last level of the neural network
        stateDim = 18  # OBELIX observation shape
        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        actionDim = len(self.actions)

        # creating Q network
        self.nnTarget = createValueNetwork(stateDim, actionDim, hDim = config['hDim']).to(self.device)
        self.nnOnline = createValueNetwork(stateDim, actionDim, hDim = config['hDim']).to(self.device)
        self.updateNetwork(self.nnOnline, self.nnTarget)

        # defining the optimizer using optimizerFn with optimizerLR
        self.optimizer = config['optimizerFn'](self.nnOnline.parameters(), lr=config['optimizerLR'])

        # created replay buffer for DQN
        self.rBuffer = ReplayBuffer(self.bufferSize, bufferType='DDQN')

    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method
        #
        # Your code goes in here
        #
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.trainTimeList = []
        self.wallClockTimeList = []
        self.totalStepsList = []
        self.bestEvalScore = None
        self.startTime = time.time()

    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called
        #multiple times during training
        #if you want you can print information using this, so it may help to monitor progress and also help to debug
        #
        # Your code goes in here
        #
        currentTime = time.time()
        if train:
            trainTime = currentTime - self.startTime
            self.trainTimeList.append(trainTime)
        else :
            wallClock = currentTime - self.startTime
            self.wallClockTimeList.append(wallClock)

    def runDDQN(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training,
        #                               note this will include time for BookKeeping and evaluation
        # Note both trainTime and wallClockTime get accumulated as episodes proceed.

        #Your code goes in here
        trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList = self.trainAgent()
        finalEvalReward = self.evaluateAgent()
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList, finalEvalReward
    
    def trainAgent(self):
        #this method collects experiences and trains the agent and does BookKeeping while training.
        #this calls the trainNetwork() method internally, it also evaluates the agent per episode
        #it trains the agent for MAX_TRAIN_EPISODES
        #
        #Your code goes in here
        self.updateNetwork(self.nnOnline, self.nnTarget)
        for episode in range(self.MAX_TRAIN_EPISODES):
            # first of all, getting the current state of the agent
            self.nnOnline.to(self.device)
            self.nnTarget.to(self.device)
            state= self.env.reset()

            done = False
            totalRewards = 0
            totalSteps = 0

            while not done:
                action = self.explorationStrategyTrainFn(self.nnOnline, state, self.epsilons[episode])

                # observed the new experience by taking one step in the env
                next_state, reward, done = self.env.step(self.actions[action], render=False)

                # each experience holds the info of curr_state, action_taken, next_state, reward_got, whether_terminated
                experience = (state, action, reward, next_state, done)

                # finally calling store method defined above to store this observed experience into the replaybuffer after assigning prioirity to it.
                self.rBuffer.store(experience)

                if self.rBuffer.length() >= self.batchSize:
                    experiences = self.rBuffer.sample(self.batchSize)
                    self.trainNetwork(experiences, 1)

                state = next_state
                totalRewards += reward
                totalSteps += 1

            self.performBookKeeping(train=True)

            # appending the totalReward and totalSteps in bookKeeping variables
            self.trainRewardsList.append(totalRewards)
            self.totalStepsList.append(totalSteps)

            evalRewardsList = self.evaluateAgent()
            self.evalRewardsList.append(np.mean(evalRewardsList))
            self.performBookKeeping(train=False)

            if (episode+1) % self.updateFrequency == 0:
                self.updateNetwork(self.nnOnline, self.nnTarget)
            
            if self.bestEvalScore is None or self.bestEvalScore < evalRewardsList[0] or (episode+1) % 10 == 0:
                if self.bestEvalScore is None or self.bestEvalScore < evalRewardsList[0]:
                    self.bestEvalScore = evalRewardsList[0]
                torch.save(self.nnOnline.state_dict(), f"{self.model_path}/ddqn_weights_{episode+1}.pth")

            # printing for interactive console
            print(f"Episode {episode}: TR {self.trainRewardsList[-1]} | ER {self.evalRewardsList[-1]} | TT {self.trainTimeList[-1]} | WC {self.wallClockTimeList[-1]} | TS {self.totalStepsList[-1]}")

        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList
    
    def trainNetwork(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc.
        #
        #Your code goes in here
        ss, acts, rs, sNexts, dones = self.rBuffer.splitExperiences(experiences)

        ss = torch.FloatTensor(ss).to(self.device)
        acts = torch.LongTensor(acts).to(self.device)
        rs = torch.FloatTensor(rs).to(self.device)
        sNexts = torch.FloatTensor(sNexts).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        for _ in range(epochs):
            argmax_a_qs = self.nnOnline(sNexts).detach().max(1)[1]
            qs = self.nnTarget(sNexts).detach()
            max_a_qs = qs.gather(1, argmax_a_qs.unsqueeze(1)).squeeze(1)
            tdTarget = rs + self.gamma * max_a_qs * (1 - dones)

            qs = self.nnOnline(ss).gather(1, acts.unsqueeze(1)).squeeze(1)

            tdErrors = tdTarget - qs

            if self.loss_fn == 'HuberLoss':
                loss = huberLoss(tdErrors, self.delta)
            else:
                loss = torch.mean(0.5 * (tdErrors)**2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def updateNetwork(self, onlineNet, targetNet):
        #this function updates the onlineNetwork with the target network
        #
        # Your code goes in here
        #
        targetNet.load_state_dict(onlineNet.state_dict())

    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1

        #Your code goes in here
        self.nnOnline.eval()
        self.nnTarget.eval()
        finalEvalRewardsList = []

        for e in range(self.MAX_EVAL_EPISODES):
            state= self.env.reset()
            totalReward = 0
            done = False

            while not done:
                action = self.explorationStrategyEvalFn(self.nnOnline, state)
                next_state, reward, done = self.env.step(self.actions[action])

                totalReward += reward
                state = next_state

            finalEvalRewardsList.append(totalReward)

        self.performBookKeeping(train=False)
        return finalEvalRewardsList