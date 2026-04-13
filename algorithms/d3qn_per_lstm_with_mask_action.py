import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from helper import decayEpsilon, huberLoss
from sequenceReplayBuffer import SeqReplayBuffer

class KnowledgeNet(nn.Module):
    def __init__(self, inDim=18, outDim=5, f_hDim=[512, 256], lstm_hDim=128, activation=nn.ReLU):
        super(KnowledgeNet, self).__init__()
        self.activation = activation

        prevDim = inDim
        layers = []
        for nextDim in f_hDim:
            layers.append(nn.Linear(prevDim, nextDim))
            layers.append(self.activation())
            prevDim = nextDim
        self.features = nn.Sequential(*layers)
        self.lstmCell = nn.LSTMCell(prevDim, lstm_hDim)

        self.values = nn.Linear(lstm_hDim, 1)
        self.advantages = nn.Linear(lstm_hDim, outDim)

        self.action_mask = nn.Sequential(
            nn.Linear(lstm_hDim + 1, 64),  # +1 for time input
            nn.ReLU(),
            nn.Linear(64, outDim),
            nn.Sigmoid()                    # soft gate per action
        )

        self._init_weights()
    
    def _init_weights(self):
        # Linear layers in features block
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # LSTMCell has 4 gates, weights are stacked: [4*hidden, input] and [4*hidden, hidden]
        nn.init.xavier_uniform_(self.lstmCell.weight_ih)
        nn.init.xavier_uniform_(self.lstmCell.weight_hh)
        nn.init.zeros_(self.lstmCell.bias_ih)
        nn.init.zeros_(self.lstmCell.bias_hh)

        # Value and advantage heads
        nn.init.xavier_uniform_(self.values.weight)
        nn.init.zeros_(self.values.bias)
        nn.init.xavier_uniform_(self.advantages.weight)
        nn.init.zeros_(self.advantages.bias)

        for layer in self.action_mask:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, hx, cx, time_frac=None):
        x_f = self.features(x)
        hx, cx = self.lstmCell(x_f, (hx, cx))

        v = self.values(hx)
        adv = self.advantages(hx)
        q_t = v + (adv - adv.mean(dim=1, keepdim=True))

        # Compute soft action mask
        if time_frac is None:
            time_frac = torch.zeros(hx.shape[0], 1, device=hx.device)
        
        mask_input = torch.cat([hx.detach(), time_frac], dim=1)
        mask = self.action_mask(mask_input)   # [B, outDim], values in (0,1)

        # Apply mask: suppresses Q-values for bad actions
        q_masked = q_t * mask

        return q_masked, hx, cx, mask
        
    
class D3QN_PER():
    def __init__(self, env, config):
        #Your code goes in here
        self.seed = config['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.env = env
        self.env.reset(seed=self.seed)

        self.gamma = config['gamma']

        self.tau = config['tau']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.beta_rate = config['beta_rate']
        self.delta = config['delta']
        self.seqLen = config['seq_len']
        self.f_hDim = config['f_hDim']
        self.lstm_hDim = config['lstm_hDim']
        self.burn_in = config['burn_in']
        self.mask_lambda = config['mask_lambda']
        self.MASK_THRESHOLD = config['mask_threshold']
        self.minSamples = config['minSamples']
        self.sampleAction = config['sample_action'] if 'sample_action' in config else False

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

        self.initBookKeeping()

        stateDim = 18 
        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        actionDim = len(self.actions)

        self.loss_fn = config['loss_fn']
        self.model_path = config['model_path']

        # creating Q network
        self.nnTarget = KnowledgeNet(stateDim, actionDim, f_hDim = self.f_hDim, lstm_hDim = self.lstm_hDim).to(self.device)
        self.nnOnline = KnowledgeNet(stateDim, actionDim, f_hDim = self.f_hDim, lstm_hDim = self.lstm_hDim).to(self.device)
        self.updateNetwork(self.nnOnline, self.nnTarget)

        # defining the optimizer using optimizerFn with optimizerLR
        self.optimizer = config['optimizerFn'](self.nnOnline.parameters(), lr=config['optimizerLR'])

        # created replay buffer for D3QN-PER
        self.rBuffer = SeqReplayBuffer(
            self.bufferSize, seq_len=self.seqLen, 
            batchSize=self.batchSize, max_steps=self.max_steps,
            alpha=self.alpha, beta=self.beta, beta_rate=self.beta_rate
        )

    def initBookKeeping(self):
        # Your code goes in here
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.trainTimeList = []
        self.wallClockTimeList = []
        self.totalStepsList = []
        self.startTime = time.time()

    def performBookKeeping(self, train = True):
        # Your code goes in here
        currentTime = time.time()
        if train:
            trainTime = currentTime - self.startTime
            self.trainTimeList.append(trainTime)
        else :
            wallClock = currentTime - self.startTime
            self.wallClockTimeList.append(wallClock)

    def runD3QN_PER(self):
        #Your code goes in here
        trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList = self.trainAgent()
        finalEvalReward = self.evaluateAgent()
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, totalStepsList, finalEvalReward
    
    def trainAgent(self):
        #Your code goes in here
        self.updateNetwork(self.nnOnline, self.nnTarget)
        for episode in range(self.MAX_TRAIN_EPISODES):
            # first of all, getting the current state of the agent
            self.nnOnline.to(self.device)
            self.nnTarget.to(self.device)
            state = self.env.reset()
            
            hx = torch.zeros(1, self.lstm_hDim).to(self.device)
            cx = torch.zeros(1, self.lstm_hDim).to(self.device)
            done = False
            totalRewards = 0
            totalSteps = 0

            while not done:
                time_frac = torch.tensor([[totalSteps / self.max_steps]], dtype=torch.float32).to(self.device)
                action, hx, cx = self.explorationStrategyTrainFn(
                    self.nnOnline, state, self.epsilons[episode], 
                    useLSTM=True, hx=hx, cx=cx, time_frac=time_frac,
                    sampleAction = self.sampleAction
                )

                # observed the new experience by taking one step in the env
                next_state, reward, done = self.env.step(self.actions[action], render=False)

                # each experience holds the info of curr_state, action_taken, next_state, reward_got, whether_terminated
                experience = (state, action, reward, next_state, done)

                self.rBuffer.storeStepwiseExperiences(experience)

                if self.rBuffer.length() >= self.minSamples and (totalSteps + 1) % 20 == 0:
                    batch_experiences, indices, batch_weights, start_indices = self.rBuffer.sample(running_step=totalSteps)
                    if batch_experiences is None  or indices is None or batch_weights is None or start_indices is None:
                        continue
                    experiences = (batch_experiences, indices, batch_weights, start_indices)
                    self.trainNetwork(experiences)

                state = next_state
                totalRewards += reward
                totalSteps += 1

                """Now updating the target network based on steps"""
                if (totalSteps+1) % self.updateFrequency == 0:
                    self.updateNetwork(self.nnOnline, self.nnTarget)

            # finally calling store method to store stepwise experiences into the replay buffer
            self.rBuffer.store()
            self.performBookKeeping(train=True)

            # appending the totalReward and totalSteps in bookKeeping variables
            self.trainRewardsList.append(totalRewards)
            self.totalStepsList.append(totalSteps)

            evalRewardsList = self.evaluateAgent()
            self.evalRewardsList.append(np.mean(evalRewardsList))
            self.performBookKeeping(train=False)
            
            if (episode+1) % 20 == 0:
                torch.save(self.nnOnline.state_dict(), f"{self.model_path}/lstm_d3qn_per_w_{episode+1}.pth")

            # printing for interactive console
            print(f"Episode {episode}: TR {self.trainRewardsList[-1]} | ER {self.evalRewardsList[-1]} | TT {self.trainTimeList[-1]} | WC {self.wallClockTimeList[-1]} | TS {self.totalStepsList[-1]}")

        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList
    
    def trainNetwork(self, experiences):
        #Your code goes in here
        self.nnOnline.train()
        self.nnTarget.train()
        batch_experiences, indices, batch_weights, start_indices = experiences
        ss, acts, rs, sNexts, dones = self.rBuffer.splitExperiences(batch_experiences)

        ss = torch.FloatTensor(ss).to(self.device)
        acts = torch.LongTensor(acts).to(self.device)
        rs = torch.FloatTensor(rs).to(self.device)
        sNexts = torch.FloatTensor(sNexts).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(batch_weights).to(self.device)

        B, T, _ = ss.shape

        # Initialize LSTM hidden states (or zeros if MLP)
        hx = torch.zeros(B, self.lstm_hDim).to(self.device)
        cx = torch.zeros(B, self.lstm_hDim).to(self.device)

        loss = 0.0
        mask_loss = 0.0
        td_errors_all = []

        # Burn-in (pass initial steps through LSTM without training)
        with torch.no_grad():
            for t in range(self.burn_in):
                time_frac = torch.full((B, 1), t / self.max_steps, device=self.device)
                _, hx, cx, _ = self.nnOnline(ss[:, t], hx, cx, time_frac)

        for t in range(self.burn_in, T):
            time_frac = torch.full((B, 1), t / self.max_steps, device=self.device)
            qs, hx, cx, mask = self.nnOnline(ss[:,t], hx, cx, time_frac)
            qs_next, _, _, _ = self.nnOnline(sNexts[:,t], hx.detach(), cx.detach(), time_frac)
            a_next = qs_next.argmax(dim=1)

            qt_next, _, _, _ =  self.nnTarget(sNexts[:,t], hx.detach(), cx.detach(), time_frac)
            max_a_qt = qt_next.gather(1, a_next.unsqueeze(1)).squeeze(1)

            tdTarget = rs[:,t] + self.gamma * max_a_qt * (1 - dones[:,t])

            tdError = tdTarget - qs.gather(1, acts[:,t].unsqueeze(1)).squeeze(1)
            td_errors_all.append(tdError)

            # Mask Penalty
            bad_action_mask = (rs[:,t] < self.MASK_THRESHOLD).float()
            chosen_mask_val = mask.gather(1, acts[:, t].unsqueeze(1)).squeeze(1)
            mask_loss += (bad_action_mask * chosen_mask_val).mean()

            if self.loss_fn == 'HuberLoss':
                loss += huberLoss(tdError, self.delta, weights)
            else:
                loss += (weights * tdError.pow(2)).mean()

            done_mask = dones[:,t].unsqueeze(1)
            hx = hx * (1 - done_mask)
            cx = cx * (1 - done_mask)
        
        loss = loss / (T - self.burn_in)
        mask_loss = mask_loss / (T - self.burn_in)

        # conbined loss
        total_loss = loss + self.mask_lambda * mask_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nnOnline.parameters(), 10)
        self.optimizer.step()
            
        self.rBuffer.update(td_errors_all, indices, start_indices)


    def updateNetwork(self, onlineNet, targetNet):
        #this function updates the onlineNetwork with the target network
        # Your code goes in here
        
        for target_param, online_param in zip(targetNet.parameters(), onlineNet.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)


    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #Your code goes in here

        self.nnOnline.eval()
        self.nnTarget.eval()
        finalEvalRewardsList = []

        for e in range(self.MAX_EVAL_EPISODES):
            state= self.env.reset()
            totalReward = 0
            totalSteps = 0
            done = False
            hx = torch.zeros(1, self.lstm_hDim).to(self.device)
            cx = torch.zeros(1, self.lstm_hDim).to(self.device)

            while not done:
                time_frac = torch.tensor([[totalSteps / self.max_steps]], dtype=torch.float32).to(self.device)
                action, hx, cx = self.explorationStrategyEvalFn(
                    self.nnOnline, state, useLSTM=True, 
                    hx=hx, cx=cx, time_frac=time_frac
                )
                next_state, reward, done = self.env.step(self.actions[action], render=False)

                totalReward += reward
                state = next_state
                totalSteps += 1

            finalEvalRewardsList.append(totalReward)

        self.performBookKeeping(train=False)
        return finalEvalRewardsList