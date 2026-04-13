import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import random

from uniformSeqBuffer import SeqReplayBuffer
from helper import decayEpsilon, getStepWiseReturnsAndDiscounts

# here i am adding GAE for better credit assignment and thinking of doing value loss clipping so that it can't force
# the model params to diverge too much

class KnowledgeNet(nn.Module):
    def __init__( self, state_dim=18, action_dim=5, feature_dims=[256, 256],
        lstm_dim=128, forward_dims=[256], inverse_dims=[256]
    ):
        super().__init__()

        # feature extractor
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev = self.state_dim
        for dim in feature_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        self.feature_net = nn.Sequential(*layers)

        # lstm
        self.lstm = nn.LSTMCell(prev, lstm_dim)

        # latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.ReLU(),
            nn.LayerNorm(lstm_dim)
        )

        # value head
        self.value_head = nn.Linear(lstm_dim, 1)
        self.advantages = nn.Linear(lstm_dim, self.action_dim)

        # Forward Model
        f_layers = []
        prev = lstm_dim + self.action_dim
        for dim in forward_dims:
            f_layers.append(nn.Linear(prev, dim))
            f_layers.append(nn.ReLU())
            prev = dim
        f_layers.append(nn.Linear(prev, lstm_dim))
        self.forward_model = nn.Sequential(*f_layers)

        # inverse Model
        i_layers = []
        prev = 2 * lstm_dim
        for dim in inverse_dims:
            i_layers.append(nn.Linear(prev, dim))
            i_layers.append(nn.ReLU())
            prev = dim
        i_layers.append(nn.Linear(prev, self.action_dim))
        self.inverse_model = nn.Sequential(*i_layers)

        # Bump Head (for decinding which bump is bad)
        self.bump_head = nn.Sequential(
            nn.Linear(lstm_dim+2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, hx, cx):
        x = self.feature_net(x)

        hx, cx = self.lstm(x, (hx, cx))
        z = self.latent_proj(hx)
        
        v = self.value_head(z)
        adv = self.advantages(z)
        q = v + (adv - adv.mean(dim=1, keepdim=True))

        return q, z, hx, cx

    # defined this method for predicting the next latent representation basde on current latent
    # repren. and actions available and finding loss using latent represenations of next_states available
    # in order to learn better model parameters
    def predict_next_latent(self, z, action):
        a_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        inp = torch.cat([z, a_onehot], dim=1)
        return self.forward_model(inp)

    # defined this for predicting action based on current latent repren. and predicted represention
    # and usinng available actions, i can define the loss and trained the netwok by backproagting
    # the gradients comnputed in order to make lstm to learn more better latent represenation
    def predict_action(self, z, z_next):
        inp = torch.cat([z, z_next], dim=1)
        return self.inverse_model(inp)
    

class KnowledgeDistillationAgent:
    def __init__(self, env1, env2, teachers, config):
        super().__init__()
        
        self.teachers = teachers
        self.env1 = env1
        self.env2 = env2
        self.gamma = config.get('gamma', 0.99)
        self.device = config['device']

        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        self.action_dim = config.get('action_dim', 5)
        self.state_dim = config.get('state_dim', 18)
        self.lstm_dim = config.get('lstm_hDim', 128)
        self.f_hDim = config.get('f_hDim', [324, 256])
        self.for_hDim = config.get('for_hDim', [256])
        self.inv_hDim = config.get('inv_hDim', [256])
        self.gae_lambda = config.get('gae_lambda', 0.95)

        self.minSamples = config.get('minSamples', 40)
        self.sampleAction = config.get('sample_action', False)
        self.max_steps = config.get('max_steps', 1000)
        self.noise_std = config.get('noise_std', 0.12)
        self.kl_coef = config.get('kl_coef', 0.05)
        self.updateTargetFrequency = config.get('updateTargetFrequency', 20) # after every 20 steps 
        self.tau = config.get('tau', 0.1)

        self.model = KnowledgeNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_dims=self.f_hDim,
            lstm_dim=self.lstm_dim,
            forward_dims=self.for_hDim,
            inverse_dims=self.inv_hDim
        ).to(self.device)

        self.nnTarget = KnowledgeNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_dims=self.f_hDim,
            lstm_dim=self.lstm_dim,
            forward_dims=self.for_hDim,
            inverse_dims=self.inv_hDim
        ).to(self.device)

        self.updateTargetNetwork(1.0)

        self.batch_size = config['batchSize']
        self.seq_len = config['seq_len']
        self.burn_in = config['burn_in']
        self.MAX_TRAIN_EPISODES = config['MAX_TRAIN_EPISODES']
        self.MAX_EVAL_EPISODES = config['MAX_EVAL_EPISODES']
        self.buffer_size = config.get('bufferSize', 1_000_000)
        self.model_path = config['model_path']
        self.beta = 0.5 # decides how much to disagree among parent's common overview

        self.clip_eps = config.get('clip_eps', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.forward_coef = config.get('forward_coef', 0.1)
        self.inverse_coef = config.get('inverse_coef', 0.05)

        self.optimizer = config['optimizerFn'](self.model.parameters(), lr=config['optimLR'])

        self.rBuffer = SeqReplayBuffer(
            self.buffer_size, self.seq_len, self.batch_size, self.max_steps
        );

        # Freezing the teacher model
        for t in self.teachers:
            t.to(self.device)
            t.eval()
            for param in t.parameters():
                param.requires_grad = False
        
        self.explorationStrategyTrainFn = config['explorationStrategyTrainFn']
        self.explorationStrategyEvalFn = config['explorationStrategyEvalFn']

        self.epsilons = decayEpsilon(
            config['epsilon'],
            config['eps_decay_strategy'],
            self.MAX_TRAIN_EPISODES
        )

        self.initBookKeeping()


    def initBookKeeping(self):
        self.trainRewardsList = []
        self.evalRewardsList1 = []
        self.evalRewardsList2 = []
        self.trainTimeList = []
        self.wallClockTimeList = []
        self.totalStepsList = []
        self.KLList = []
        self.entropyList = []
        self.loss = []
        self.startTime = time.time()


    def performBookKeeping(self, train = True):
        currentTime = time.time()
        if train:
            trainTime = currentTime - self.startTime
            self.trainTimeList.append(trainTime)
        else :
            wallClock = currentTime - self.startTime
            self.wallClockTimeList.append(wallClock)

    def updateTargetNetwork(self, tau=0.1):
        for target_param, online_param in zip(self.nnTarget.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

    def runKDAgent(self):
        self.trainAgent()
        return (self.trainRewardsList, self.trainTimeList, self.evalRewardsList1, 
                self.evalRewardsList2, self.wallClockTimeList, self.totalStepsList, 
                self.KLList, self.entropyList, self.loss)
    

    def trainAgent(self):
        for e in range(self.MAX_TRAIN_EPISODES):
            # Early Exit 
            if os.path.exists("STOP_TRAINING_KD"):
                print(f"Stopped at Episode {e + 1}. Saving the model and Exiting.")
                torch.save(self.model.state_dict(), f"{self.model_path}/model_weights/manual_stop_kd_ep{e + 1}.pth")
                os.remove("STOP_TRAINING_KD")
                break

            envs = [self.env1, self.env2]
            envName = ["with walls", "without walls"]
            env_idx = random.randint(0, len(envs) - 1)
            env = envs[env_idx]

            # first of all, getting the current state of the agent
            state = env.reset()
            
            hx = torch.zeros(1, self.lstm_dim).to(self.device)
            cx = torch.zeros(1, self.lstm_dim).to(self.device)

            done = False
            totalRewards = 0
            steps = 0

            while not done:
                action, hx, cx = self.explorationStrategyTrainFn(
                    self.model, state, self.epsilons[e], hx=hx, cx=cx, 
                    noise_std=self.noise_std, episode=e, 
                    max_episodes=self.MAX_TRAIN_EPISODES, kd=True
                )

                # observed the new experience by taking one step in the env
                next_state, reward, done = env.step(self.actions[action], render=False)

                # each experience holds the info of curr_state, action_taken, next_state, reward_got, whether_terminated
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q, _, hx, cx = self.model(state_t, hx, cx)
                    log_probs = F.log_softmax(q, dim=1)
                    log_prob_act = log_probs[0, action]
                experience = (state, action, reward, next_state, done, log_prob_act)

                self.rBuffer.storeStepwiseExperiences(experience)

                if self.rBuffer.length() >= self.minSamples and (steps + 1) % 4 == 0:
                    batch_experiences = self.rBuffer.sample()
                    if batch_experiences is None:
                        continue

                    stats = self.trainNetwork(batch_experiences)
                    self.KLList.append(stats['kl'])
                    self.entropyList.append(stats['entropy'])
                    self.loss.append(stats['loss'])
                    if (steps + 1) % 50 == 0:
                        print(f"Episode {e+1} | step {steps + 1} : KL {stats['kl']} | Loss {stats['loss']} | Entropy {stats['entropy']}")

                state = next_state
                totalRewards += reward
                steps += 1

            # finally calling store method to store stepwise experiences into the replay buffer
            self.rBuffer.store()
            self.performBookKeeping(train=True)

            # appending the totalReward and totalSteps in bookKeeping variables
            self.trainRewardsList.append(totalRewards)
            self.totalStepsList.append(steps)

            if (e + 1) % 20 == 0:
                evalRewardsFor2Envs = self.evaluateAgent()
                self.evalRewardsList1.append(evalRewardsFor2Envs[0])
                self.evalRewardsList2.append(evalRewardsFor2Envs[1])
                self.performBookKeeping(train=False)
                print(f"Episode {e+1}: TR({envName[env_idx]}) {self.trainRewardsList[-1]:.3f} | ER(with wall) {self.evalRewardsList1[-1]:.3f} | ER(without wall) {self.evalRewardsList2[-1]:.3f} | TT {self.trainTimeList[-1]:.3f} | WC {self.wallClockTimeList[-1]:.3f} | TS {self.totalStepsList[-1]:.0f}")
                torch.save(self.model.state_dict(), f"{self.model_path}/model_weights/kd_w_{e+1}.pth")

            else:
                print(f"Episode {e+1}: TR({envName[env_idx]}) {self.trainRewardsList[-1]:.3f} | TT {self.trainTimeList[-1]:.3f} | TS {self.totalStepsList[-1]:.0f}")
    
    
    def trainNetwork(self, batch_experiences):
        ss, acts, rs, sNexts, dones, old_log_probs_act = self.rBuffer.splitExperiences(batch_experiences)

        ss = torch.FloatTensor(ss).to(self.device)
        acts = torch.LongTensor(acts).to(self.device)
        rs = torch.FloatTensor(rs).to(self.device)
        sNexts = torch.FloatTensor(sNexts).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs_act = torch.FloatTensor(old_log_probs_act).to(self.device)

        # reward shaping
        rs_env = torch.sign(rs) * torch.log1p(torch.abs(rs))
        rs_env = torch.where(
            rs_env > 0,
            4.0 * rs_env,          # amplifying positives
            rs_env                 # keeping negatives same
        )

        B, T, _ = ss.shape

        # Initialize LSTM hidden states
        hx = torch.zeros(B, self.lstm_dim).to(self.device)
        cx = torch.zeros(B, self.lstm_dim).to(self.device)

        # ------------------------------------------------------------
        with torch.no_grad():
            hx_old = torch.zeros(B, self.lstm_dim).to(self.device)
            cx_old = torch.zeros(B, self.lstm_dim).to(self.device)

            for t in range(T):
                _, _, hx_old, cx_old = self.model(ss[:, t], hx_old, cx_old)

    
        total_loss = 0
        kl_list, ent_list = [], []

        # Initializing teacher LSTM states (once per batch)
        # this is baseically used to reset the hx, cx for teachers
        teacher_states = [
            (
                torch.zeros(B, self.lstm_dim).to(self.device),
                torch.zeros(B, self.lstm_dim).to(self.device)
            )
            for _ in self.teachers
        ]

        # -----------------------------------------------------------
        # finding advantages using GAE for better credit asignemnt
        advantages = torch.zeros_like(rs, device=self.device)        # [B, T]
        lastgaelam = torch.zeros(B, dtype=torch.float32, device=self.device)  # [B]
        values = torch.zeros_like(advantages, device=self.device)   # [B, T]

        with torch.no_grad():
            hx_tmp = hx.clone()   # carry LSTM hidden state
            cx_tmp = cx.clone()

            for t in reversed(range(T)):
                if t == T - 1:
                    done = 1.0 - dones[:, t]
                    _, z_g, _, _ = self.model(sNexts[:, t], hx_tmp, cx_tmp)
                    vt_1 = self.model.value_head(z_g)
                else:
                    done = 1.0 - dones[:, t + 1].to(self.device)
                    _, z_g, _, _ = self.model(ss[:, t + 1], hx_tmp, cx_tmp)
                    vt_1 = self.model.value_head(z_g)

                # current timestep
                _, z_t, _, _ = self.model(ss[:, t], hx_tmp, cx_tmp)
                vt = self.model.value_head(z_t)
                values[:, t] = vt.squeeze(1)

                delta = rs_env[:, t] + self.gamma * vt_1.squeeze(1) * done - vt.squeeze(1)
                advantages[:, t] = lastgaelam = delta + self.gamma * self.gae_lambda * done * lastgaelam

            returns = advantages + values

            # flatten and normalize
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # -----------------------------------------------------------

        self.updateTargetNetwork(self.tau)

        for t in range(self.burn_in, T):
            q, z, hx, cx = self.model(ss[:, t], hx, cx)

            log_probs = F.log_softmax(q, dim=1)
            probs = log_probs.exp()

            # teacher
            with torch.no_grad():
                teacher_qs = []

                for i, teacher in enumerate(self.teachers):
                    thx, tcx = teacher_states[i]
                    q_t, _, _, thx, tcx = teacher(ss[:, t], thx, tcx)

                    # update stored state
                    teacher_states[i] = (thx.detach(), tcx.detach())
                    teacher_qs.append(q_t)

                # using softmax distribution here since parents has to compete among each other
                # over which actions to prefer. If I will use Normal distribution then it treats each action
                #  equally that will break the comparison.
                teacher_qs = torch.stack(teacher_qs)   # ADD THIS
                mean_q = teacher_qs.mean(dim=0)
                std_q = teacher_qs.std(dim=0)
                teacher_q = mean_q - self.beta * std_q
                teacher_probs = F.softmax(teacher_q, dim=1)

            # findinhg kl-divergence for approximating model prob dist to teacher's prob dist
            # it pulls the model towards teachers
            kl = F.kl_div(log_probs, teacher_probs, reduction='batchmean')

            # using ppo style updates for trusting the advantage signals but in a certain scale 
            # like not updating the prob dist because of some noisy advantage signal which will distort 
            # the learning.
            log_probs_act = log_probs.gather(1, acts[:, t].unsqueeze(1)).squeeze(1)

            ratio = torch.exp(log_probs_act - old_log_probs_act[:, t])
            ppo_loss = -torch.min( ratio * advantages[:,t], torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[:,t])

            # entropy loss: using this so as to prevent the model becoming too confident
            # early by keeping the prob distribution uniform or dispersed
            entropy = -(probs * log_probs).sum(dim=1)

            # forward model for find forward loss so as to improve latent reprensenations
            with torch.no_grad():
                qs_next, z_next, _, _ = self.model(sNexts[:, t], hx.detach(), cx.detach())

            z_pred = self.model.predict_next_latent(z, acts[:, t])
            forward_loss = F.mse_loss(z_pred, z_next.detach(), reduction='none').mean(dim=1)

            # inverse model for finding inverse loss so as to improve latent reprensenations
            inv_logits = self.model.predict_action(z, z_next.detach())
            inverse_loss = F.cross_entropy(inv_logits, acts[:, t], reduction='none')

            # use ddqn type loss
            a_next = qs_next.argmax(dim=1)

            qt_next, _, _, _ = self.nnTarget(sNexts[:, t], hx.detach(), cx.detach())
            max_q = qt_next.gather(1, a_next.unsqueeze(1)).squeeze(1)

            # rewarding if sensor detects box
            progress_reward = 0.2 * torch.mean(ss[:, t,:16], dim=1)

            total_reward = rs_env[:,t] + progress_reward
            td_target = total_reward + self.gamma * max_q.detach() * (1 - dones[:, t])
            td_error = td_target - q.gather(1, acts[:, t].unsqueeze(1)).squeeze(1)

            ddqn_type_loss = 0.5 * (td_error.clamp(-5, 5) ** 2)

            # finding bump loss
            bump = ss[:, t, 16].unsqueeze(1)
            stuck = ss[:, t, 17].unsqueeze(1)
            bump_input = torch.cat([z.detach(), bump, stuck], dim=1)
            bump_bad_logit = self.model.bump_head(bump_input)
            bump_target = (td_error.detach() < 0).float()
            bump_loss = F.binary_cross_entropy_with_logits(
                bump_bad_logit.squeeze(1),
                bump_target
            )

            # single step loss
            step_loss = (
                  self.kl_coef * kl
                + ppo_loss
                + self.forward_coef * forward_loss
                + self.inverse_coef * inverse_loss
                - self.entropy_coef * entropy
                + 0.1 * bump_loss
                + ddqn_type_loss
            )
            mask = (1 - dones[:, t])
            step_loss = step_loss * mask
            total_loss += step_loss.mean()

            kl_list.append(kl.mean().item())
            ent_list.append(entropy.mean().item())

            # reset LSTM if done
            done_mask = dones[:, t].unsqueeze(1)
            hx = hx * (1 - done_mask)
            cx = cx * (1 - done_mask)

            for i in range(len(teacher_states)):
                thx, tcx = teacher_states[i]
                thx = thx * (1 - done_mask)
                tcx = tcx * (1 - done_mask)
                teacher_states[i] = (thx, tcx)

            if (t + 1) % self.updateTargetFrequency == 0:
                self.updateTargetNetwork(self.tau)

        # backpropagating gradients for model parameter's updation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "kl": np.mean(kl_list),
            "entropy": np.mean(ent_list)
        }


    def evaluateAgent(self):
        self.model.eval()
        feval = []

        for env in [self.env1, self.env2]:
            temp_eval = []
            for e in range(self.MAX_EVAL_EPISODES):
                state = env.reset()
                totalReward = 0
                done = False
                hx = torch.zeros(1, self.lstm_dim).to(self.device)
                cx = torch.zeros(1, self.lstm_dim).to(self.device)

                while not done:
                    action, hx, cx = self.explorationStrategyEvalFn(self.model, state, hx=hx, cx=cx, kd=True)
                    next_state, reward, done = env.step(self.actions[action], render=False)

                    totalReward += reward
                    state = next_state

                temp_eval.append(totalReward)
            feval.append(np.mean(temp_eval).item())

        return feval