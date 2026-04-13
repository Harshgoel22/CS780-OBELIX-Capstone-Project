import torch
import numpy as np
import random
import time
from itertools import count

from helper import createPolicyNetwork, getStepWiseReturnsAndDiscounts

class Reinforce():
    def __init__(self, env, seed, gamma, optimizerFn, optimizerLR,
                MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                explorationStrategyTrainFn, explorationStrategyEvalFn,
                hDim, model_path, device):
        
        self.env = env
        self.seed = seed

        self.env.reset(seed=self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.gamma = gamma
        self.optimizerFn = optimizerFn
        self.optimizerLR = optimizerLR
        self.render = False
        self.device = device
        self.model_path = model_path

        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES

        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn

        stateDim = 18  # OBELIX observation shape
        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        actionDim = len(self.actions)

        self.policy_network = createPolicyNetwork(stateDim, actionDim, hDim=hDim).to(self.device)

        self.optimizer = self.optimizerFn(self.policy_network.parameters(), lr=self.optimizerLR)
        self.initBookKeeping()

    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method

        self.trainRewardsList = []
        self.evalRewardsList = []
        self.trainTimeList = []
        self.wallClockTimeList = []
        self.totalStepsList = []

        self.startTime = time.time()
    
    def performBookKeeping(self, train = True):
       #this method updates relevant variables for the bookKeeping, this can be called
       #multiple times during training
       #if you want you can print information using this, so it may help to monitor progress and also help to debug
       #Your code goes in here
       
       currentTime = time.time()
       if train:
           trainTime = currentTime - self.startTime
           self.trainTimeList.append(trainTime)
       else :
           wallClock = currentTime - self.startTime
           self.wallClockTimeList.append(wallClock)

    def runREINFORCE(self):
        trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList = self.trainAgent()

        self.render = True                        # set this to true before finalEvlaution so that frames will get stored in the class variables
        finalEvalReward = self.evaluateAgent()    # evaluating the agent
        self.render = False                       # after final evaluation, makes render to false
        
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList, finalEvalReward
    
    def trainAgent(self):
        for episode in range(self.MAX_TRAIN_EPISODES):
            s = self.env.reset()
            rewards, logProbs = [], []
            done = False
            train_steps = 0
            while not done:
                a, logp_a, _ = self.explorationStrategyTrainFn(self.policy_network, s, self.actions)
                s, r, done = self.env.step(a, render=False)
                rewards.append(r)
                logProbs.append(logp_a)
                train_steps += 1
            
            self.totalStepsList.append(train_steps)
            self.trainRewardsList.append(np.sum(rewards))
            self.trainPolicyNetwork(rewards, logProbs)
            self.performBookKeeping(train = True)

            evalRmean, _ = self.evaluateAgent()
            self.evalRewardsList.append(evalRmean)
            self.performBookKeeping(train = False)

            if (episode+1) % 10 == 0:
                torch.save(self.policy_network.state_dict(), f"{self.model_path}/reinforce_weights_{episode+1}.pth")

            # printing for interactive console
            print(f"Episode {episode}: TR {self.trainRewardsList[-1]} | ER {self.evalRewardsList[-1]} | TT {self.trainTimeList[-1]} | WC {self.wallClockTimeList[-1]} | TS {self.totalStepsList[-1]}")
        
        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList
    
    def trainPolicyNetwork(self, rewards, logProbs):
        returns, _ = getStepWiseReturnsAndDiscounts(self.gamma, rewards, self.device)
        logProbs = torch.stack(logProbs).squeeze().to(self.device)
        policyLoss = -1.0 * (returns * logProbs).mean()
        
        self.optimizer.zero_grad()
        policyLoss.backward()
        self.optimizer.step()

    def evaluateAgent(self, greedy = True):
        rewards = []
        for _ in range(self.MAX_EVAL_EPISODES):
            rs, done = 0, 0
            s = self.env.reset()
            for _ in count():
                if greedy:
                    a = self.explorationStrategyEvalFn(self.policy_network, s, self.actions)
                else:
                    a, _, _ = self.explorationStrategyTrainFn(self.policy_network, s, self.actions)
                
                s, r, done = self.env.step(a,  render=False)
                rs += r
                if done:
                    rewards.append(rs)
                    break

        if self.render:
            self.performBookKeeping(train = False)
            
        rewards = np.array(rewards)
        return rewards.mean(), rewards.std()