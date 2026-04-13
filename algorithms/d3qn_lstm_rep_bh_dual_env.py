import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from helper import decayEpsilon, huberLoss
from sequenceReplayBuffer import SeqReplayBuffer
import os

class KnowledgeNet(nn.Module):
    def __init__(self, inDim=18, outDim=5, f_hDim=[512, 256], lstm_hDim=128, for_hDim=[256, 256], inv_hDim=[256, 256], activation=nn.ReLU):
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

        # Q head 
        self.values = nn.Linear(lstm_hDim, 1)
        self.advantages = nn.Linear(lstm_hDim, outDim)

        # bad state head
        self.bad_state_head = nn.Linear(lstm_hDim, 1)

        # Forward model: (z_t,a_t) -> z_{t+1}
        prevDim = lstm_hDim + outDim
        forward_layers = []
        for nextDim in for_hDim:
            forward_layers.append(nn.Linear(prevDim, nextDim))
            forward_layers.append(self.activation())
            prevDim = nextDim
        forward_layers.append(nn.Linear(prevDim, lstm_hDim))
        self.forward_model = nn.Sequential(*forward_layers)

        # Inverse model: (z_t, z_{t+1}) -> a_t
        prevDim = 2 * lstm_hDim
        inverse_layers = []
        for nextDim in inv_hDim:
            inverse_layers.append(nn.Linear(prevDim, nextDim))
            inverse_layers.append(self.activation())
            prevDim = nextDim
        inverse_layers.append(nn.Linear(prevDim, outDim))
        self.inverse_model = nn.Sequential(*inverse_layers)

        # latent represenation
        self.latent_proj = nn.Sequential(
            nn.Linear(lstm_hDim, lstm_hDim),
            nn.ReLU(),
            nn.LayerNorm(lstm_hDim)
        )

        # Bump Head (for decinding which bump is bad)
        self.bump_head = nn.Sequential(
            nn.Linear(lstm_hDim+2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


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

        # Q Heads
        nn.init.xavier_uniform_(self.values.weight)
        nn.init.zeros_(self.values.bias)

        nn.init.xavier_uniform_(self.advantages.weight)
        nn.init.zeros_(self.advantages.bias)

        # Forward Model
        for layer in self.forward_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Inverse Model
        for layer in self.inverse_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Bump Head
        for layer in self.bump_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


    def forward(self, x, hx, cx):
        x_f = self.features(x)
        hx, cx = self.lstmCell(x_f, (hx, cx))

        z_t = self.latent_proj(hx) # latent representaion

        v = self.values(z_t)
        adv = self.advantages(z_t)
        q = v + (adv - adv.mean(dim=1, keepdim=True))

        bad_state_logit = self.bad_state_head(z_t)

        return q, z_t, bad_state_logit, hx, cx
        
    
class D3QN_PER():
    def __init__(self, env1, env2, config):
        #Your code goes in here
        self.seed = config['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.env1 = env1
        self.env2 = env2
        self.env1.reset(seed = self.seed + 1)
        self.env2.reset(seed = self.seed + 2)

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
        self.minSamples = config['minSamples']
        self.sampleAction = config['sample_action'] if 'sample_action' in config else False
        
        self.for_hDim = config['for_hDim']
        self.inv_hDim = config['inv_hDim']
        self.alpha_f = config['alpha_f']
        self.beta_i = config['beta_i']
        self.eta = config['eta']
        self.noise_std = config['noise_std']

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
            self.MAX_TRAIN_EPISODES
        )

        self.initBookKeeping()

        stateDim = 18 
        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        actionDim = len(self.actions)

        self.loss_fn = config['loss_fn']
        self.model_path = config['model_path']

        # creating Q network
        self.nnTarget = KnowledgeNet(
            stateDim, actionDim, f_hDim = self.f_hDim, 
            lstm_hDim = self.lstm_hDim, for_hDim=self.for_hDim, 
            inv_hDim=self.inv_hDim
        ).to(self.device)

        self.nnOnline = KnowledgeNet(
            stateDim, actionDim, f_hDim = self.f_hDim, 
            lstm_hDim = self.lstm_hDim, for_hDim=self.for_hDim, 
            inv_hDim=self.inv_hDim
        ).to(self.device)

        self.nnOnline._init_weights()
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
        self.evalRewardsList1 = []
        self.evalRewardsList2 = []
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
        trainRewardsList, trainTimeList, evalRewardsList1, evalRewardsList2, wallClockTimeList, totalStepsList = self.trainAgent()
        return trainRewardsList, trainTimeList, evalRewardsList1, evalRewardsList2, wallClockTimeList, totalStepsList
    
    def trainAgent(self):
        #Your code goes in here
        self.updateNetwork(self.nnOnline, self.nnTarget)
        for episode in range(self.MAX_TRAIN_EPISODES):
            # Early Exit 
            if os.path.exists("STOP_TRAINING"):
                print(f"Stop signal detected at episode {episode + 1}. Saving and exiting.")
                torch.save(self.nnOnline.state_dict(), f"{self.model_path}/manual_stop_ep{episode + 1}.pth")
                os.remove("STOP_TRAINING")
                break

            envs = [self.env1, self.env2]
            envName = ["with walls", "without walls"]
            env_idx = random.randint(0, len(envs) - 1)
            env = envs[env_idx]
            # first of all, getting the current state of the agent
            self.nnOnline.to(self.device)
            self.nnTarget.to(self.device)
            state = env.reset()
            
            hx = torch.zeros(1, self.lstm_hDim).to(self.device)
            cx = torch.zeros(1, self.lstm_hDim).to(self.device)
            done = False
            totalRewards = 0
            totalSteps = 0

            while not done:
                action, hx, cx = self.explorationStrategyTrainFn(
                    self.nnOnline, state, self.epsilons[episode], 
                    hx=hx, cx=cx, noise_std=self.noise_std,
                    episode=episode,
                    max_episodes=self.MAX_TRAIN_EPISODES
                )

                # observed the new experience by taking one step in the env
                next_state, reward, done = env.step(self.actions[action], render=False)

                # each experience holds the info of curr_state, action_taken, next_state, reward_got, whether_terminated
                experience = (state, action, reward, next_state, done)

                self.rBuffer.storeStepwiseExperiences(experience)

                if self.rBuffer.length() >= self.minSamples and (totalSteps + 1) % 4 == 0:
                    batch_experiences, indices, batch_weights, start_indices = self.rBuffer.sample(running_step=totalSteps)
                    if batch_experiences is None  or indices is None or batch_weights is None or start_indices is None:
                        continue
                    experiences = (batch_experiences, indices, batch_weights, start_indices)
                    self.trainNetwork(episode, experiences)

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

            if (episode + 1) % 20 == 0:
                evalRewardsFor2Envs = self.evaluateAgent()
                self.evalRewardsList1.append(evalRewardsFor2Envs[0])
                self.evalRewardsList2.append(evalRewardsFor2Envs[1])
                self.performBookKeeping(train=False)
                print(f"Episode {episode+1}: TR({envName[env_idx]}) {self.trainRewardsList[-1]:.3f} | ER(with wall) {self.evalRewardsList1[-1]:.3f} | ER(without wall) {self.evalRewardsList2[-1]:.3f} | TT {self.trainTimeList[-1]:.3f} | WC {self.wallClockTimeList[-1]:.3f} | TS {self.totalStepsList[-1]:.0f}")
                torch.save(self.nnOnline.state_dict(), f"{self.model_path}/lstm_d3qn_rep_bh_dual_w_{episode+1}.pth")

            else:
                print(f"Episode {episode+1}: TR({envName[env_idx]}) {self.trainRewardsList[-1]:.3f} | TT {self.trainTimeList[-1]:.3f} | TS {self.totalStepsList[-1]:.0f}")

        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList1, self.evalRewardsList2, self.wallClockTimeList, self.totalStepsList
    
    def trainNetwork(self, episode, experiences):
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
        td_errors_all = []

        # Burn-in (pass initial steps through LSTM without training)
        with torch.no_grad():
            for t in range(self.burn_in):
                _, _, _, hx, cx = self.nnOnline(ss[:, t], hx, cx)

        for t in range(self.burn_in, T):
            qs, z_t, bad_state_logit, hx, cx = self.nnOnline(ss[:,t], hx, cx)

            hx_next = hx.detach()
            cx_next = cx.detach()
            with torch.no_grad():
                _, z_next, _,  _, _ = self.nnOnline(sNexts[:,t], hx_next, cx_next)
            
            # Forward Model
            a_onehot = F.one_hot(acts[:,t], num_classes=len(self.actions)).float()
            forward_input = torch.cat([z_t, a_onehot], dim=1)

            z_next_pred = self.nnOnline.forward_model(forward_input)
            forward_loss = F.mse_loss(z_next_pred, z_next.detach(), reduction='none').mean(dim=1)

            # inverse model
            inverse_input = torch.cat([z_t, z_next.detach()], dim=1)
            a_pred_logits = self.nnOnline.inverse_model(inverse_input)

            inverse_loss = F.cross_entropy(a_pred_logits, acts[:,t], reduction='none')

            # reward shaping
            rs_env = torch.sign(rs) * torch.log1p(torch.abs(rs))
            rs_env = torch.where(
                rs_env > 0,
                4.0 * rs_env,          # amplify positives
                rs_env                 # keep negatives same
            )

            # Intrinsic Reward
            intrinsic_reward = torch.norm(z_next_pred - z_next.detach(), dim=1)
            intrinsic_reward = intrinsic_reward / (intrinsic_reward.std().detach() + 1e-8)
            eta = self.eta * (1 - episode / (0.7 * self.MAX_TRAIN_EPISODES))

            """
            Doing this if robot is just spinning then it is geting neg reward so giving more neg reward
            in order to make it learn to avoid it
            """
            intrinsic_reward = (intrinsic_reward + 1e-3) * (rs_env[:, t] <= 0)

            # bad state loss
            bump = ss[:, t, 16].unsqueeze(1)
            stuck = ss[:, t, 17].unsqueeze(1)

            bump_input = torch.cat([z_t.detach(), bump, stuck], dim=1)
            bump_bad_logit = self.nnOnline.bump_head(bump_input)
            bump_bad_prob = torch.sigmoid(bump_bad_logit)
            
            bad_state_target = stuck + bump * bump_bad_prob.detach().squeeze(1)
            bad_state_target = torch.clamp(bad_state_target, 0, 1)
            
            bad_penalty = (
                stuck.squeeze(1) + 
                bump.squeeze(1) * bump_bad_prob.detach().squeeze(1)
            )
            
            stuck_loss = F.binary_cross_entropy_with_logits(
                bad_state_logit.squeeze(1),
                stuck.squeeze(1)
            )

            # rewarding if sensor detects box
            progress_reward = 0.2 * torch.mean(ss[:, t,:16], dim=1)

            total_reward = rs_env[:,t] + max(0.01, eta) * intrinsic_reward + progress_reward

            # D3QN Loss     
            qs_next, _, _, _, _ = self.nnOnline(sNexts[:, t], hx_next, cx_next)
            a_next = qs_next.argmax(dim=1)

            qt_next, _, _, _, _ = self.nnTarget(sNexts[:, t], hx_next, cx_next)
            max_q = qt_next.gather(1, a_next.unsqueeze(1)).squeeze(1)

            td_target = total_reward + self.gamma * max_q.detach() * (1 - dones[:, t]) - 0.007 * bad_penalty

            q_val = qs.gather(1, acts[:, t].unsqueeze(1)).squeeze(1)
            td_error = td_target - q_val

            # bump loss
            bump_target = (td_error.detach() < 0).float()
            bump_loss = F.binary_cross_entropy_with_logits(
                bump_bad_logit.squeeze(1),
                bump_target
            )

            td_errors_all.append(td_error)

            if self.loss_fn == 'HuberLoss':
                d3qn_loss = huberLoss(td_error, self.delta, weights, norm=True)
            else:
                d3qn_loss = (weights * td_error.pow(2)).sum().mean() / weights.sum()


            # total loss
            loss += (
                d3qn_loss + 
                self.alpha_f * (weights * forward_loss).mean() +
                self.beta_i * (weights * inverse_loss).mean() +
                0.3 * stuck_loss +
                0.1 * bump_loss
            )

            done_mask = dones[:,t].unsqueeze(1)
            hx = hx * (1 - done_mask)
            cx = cx * (1 - done_mask)
        
        loss = loss / (T - self.burn_in)

        self.optimizer.zero_grad()
        loss.backward()
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
        feval = []

        for env in [self.env1, self.env2]:
            temp_eval = []
            for e in range(self.MAX_EVAL_EPISODES):
                state = env.reset()
                totalReward = 0
                done = False
                hx = torch.zeros(1, self.lstm_hDim).to(self.device)
                cx = torch.zeros(1, self.lstm_hDim).to(self.device)

                while not done:
                    action, hx, cx = self.explorationStrategyEvalFn(self.nnOnline, state, hx=hx, cx=cx)
                    next_state, reward, done = env.step(self.actions[action], render=False)

                    totalReward += reward
                    state = next_state

                temp_eval.append(totalReward)
            feval.append(np.mean(temp_eval).item())

        return feval