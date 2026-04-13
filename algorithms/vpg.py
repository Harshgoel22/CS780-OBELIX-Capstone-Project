import torch
import numpy as np
import random
import time
from itertools import count

from helper import createPolicyNetwork, createValueNetwork, getStepWiseReturnsAndDiscounts

class VPG():
    def __init__(self, env, seed, gamma, beta, optimizerFn, optimizerLR,
                MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                explorationStrategyTrainFn, explorationStrategyEvalFn,
                MAX_GRAD_NORM_POLICY, MAX_GRAD_NORM_VALUE,
                valuehDim, policyhDim, model_path, device):
        
        self.env = env
        self.seed = seed

        self.env.reset(seed=self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.gamma = gamma
        self.beta = beta
        self.optimizerFn = optimizerFn
        self.optimizerLR = optimizerLR
        self.render = False
        self.device = device
        self.model_path = model_path

        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.MAX_GRAD_NORM_POLICY = MAX_GRAD_NORM_POLICY
        self.MAX_GRAD_NORM_VALUE = MAX_GRAD_NORM_VALUE


        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn

        stateDim = 18  # OBELIX observation shape
        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        actionDim = len(self.actions)
        
        self.policy_network = createPolicyNetwork(stateDim, actionDim, hDim=policyhDim).to(self.device)
        self.value_network = createValueNetwork(stateDim, 1, hDim=valuehDim).to(self.device)
        self.policyOptimizer = self.optimizerFn(self.policy_network.parameters(), lr=self.optimizerLR)
        self.valueOptimizer = self.optimizerFn(self.value_network.parameters(), lr=self.optimizerLR)

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

    def runVPG(self):
        trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList = self.trainAgent()

        self.render = True                        # set this to true before finalEvlaution so that frames will get stored in the class variables
        finalEvalReward = self.evaluateAgent()    # evaluating the agent
        self.render = False                       # after final evaluation, makes render to false
        
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList, finalEvalReward
    
    def trainAgent(self):
        for e in range(self.MAX_TRAIN_EPISODES):
            s = self.env.reset()
            rewards, logProbs = [], []
            entropies, values = [], []
            train_steps = 0
            done = False
            while not done:
                a, logp_a, entropy_pa = self.explorationStrategyTrainFn(self.policy_network, s, self.actions)
                state = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                value = self.value_network(state)
                s, r, done = self.env.step(a, render = False)
                
                rewards.append(r)
                logProbs.append(logp_a)
                entropies.append(entropy_pa)
                values.append(value.squeeze())
                train_steps += 1
            
            self.totalStepsList.append(train_steps)
            self.trainRewardsList.append(np.sum(rewards))
            self.trainPolicyAndValueNetwork(rewards, logProbs, entropies, values)
            self.performBookKeeping(train = True)

            evalRmean, _ = self.evaluateAgent()
            self.evalRewardsList.append(evalRmean)
            self.performBookKeeping(train = False)

            if (e+1) % 10 == 0:
                torch.save(self.policy_network.state_dict(), f"{self.model_path}/vpg_policy_{e+1}.pth")
                torch.save(self.value_network.state_dict(), f"{self.model_path}/vpg_value_{e+1}.pth")

            # printing for interactive console
            print(f"Episode {e}: TR {self.trainRewardsList[-1]} | ER {self.evalRewardsList[-1]} | TT {self.trainTimeList[-1]} | WC {self.wallClockTimeList[-1]} | TS {self.totalStepsList[-1]}")
        
        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList
    
    def trainPolicyAndValueNetwork(self, rewards, logProbs, entropies, values):
        returns, _ = getStepWiseReturnsAndDiscounts(self.gamma, rewards, self.device)
        logProbs = torch.stack(logProbs).squeeze().to(self.device)
        entropies = torch.stack(entropies).squeeze().to(self.device)
        values = torch.stack(values).squeeze().to(self.device)

        deltas = returns - values
        ## ==================== doing advantage normalization ===============
        advantages = (deltas - deltas.mean()) / (deltas.std() + 1e-8)
        baselineLoss = -1.0 * (advantages.detach() * logProbs).mean()
        ## ==================================================================
        #baselineLoss = -1.0 * (deltas.detach() * logProbs).mean()
        entropyLoss = -1.0 * entropies.mean()
        policyLoss = baselineLoss + self.beta * entropyLoss

        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.MAX_GRAD_NORM_POLICY)
        self.policyOptimizer.step()

        valueLoss = 0.5 * (deltas ** 2)
        valueLoss = valueLoss.mean()
        self.valueOptimizer.zero_grad()
        valueLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.MAX_GRAD_NORM_VALUE)
        self.valueOptimizer.step()

    def evaluateAgent(self, greedy = True):
        rewards = []
        for _ in range(self.MAX_EVAL_EPISODES):
            rs, done = 0, 0
            s = self.env.reset()
            for _ in count():
                if greedy:
                    a = self.explorationStrategyEvalFn(self.policy_network, s, self.actions)
                else:
                    a, _ = self.explorationStrategyTrainFn(self.policy_network, s, self.actions)
                
                s, r, done = self.env.step(a, render = False)
                rs += r
                if done:
                    rewards.append(rs)
                    break

        if self.render:
            self.performBookKeeping(train = False)
            
        rewards = np.array(rewards)
        return rewards.mean(), rewards.std()
