from collections import deque
import numpy as np
import torch

class SeqReplayBuffer():
    def __init__(self, bufferSize, seq_len = 8, batchSize = 32, **kwargs):

        #Your code goes in here
        self.bufferSize = bufferSize
        self.seqLen = seq_len
        self.batchSize = batchSize
        self.max_steps = kwargs['max_steps']

        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        self.buffer = deque(maxlen=self.bufferSize)
        self.temporaryBuffer = deque(maxlen=self.max_steps)
        self.temporaryPriorities = deque(maxlen=self.max_steps)

        self.alpha = kwargs.get("alpha")
        self.beta = kwargs.get("beta")
        self.beta_rate = kwargs.get("beta_rate")
        self.priorities = deque(maxlen=self.bufferSize)
        self.prioritiesOverEpisodes = deque(maxlen=self.bufferSize)

    def storeStepwiseExperiences(self, experience):
        #Your code goes in here
        max_priority = 1.0 if len(self.temporaryPriorities) == 0 else max(self.temporaryPriorities)
        self.temporaryBuffer.append(experience)
        self.temporaryPriorities.append(max_priority)

    def store(self):
        #Your code goes in here
        timeSeriesExperiences = list(self.temporaryBuffer)
        timeSeriesPriorities = list(self.temporaryPriorities)

        self.temporaryBuffer.clear()
        self.temporaryPriorities.clear()

        self.buffer.append(timeSeriesExperiences)
        self.priorities.append(timeSeriesPriorities)
        max_priority = 1.0 if len(self.prioritiesOverEpisodes) == 0 else max(self.prioritiesOverEpisodes)
        self.prioritiesOverEpisodes.append(max_priority)


    def update(self, td_errors_all, indices, start_indices):
        # td_errors_all shape: (B, seq_len)
        # indices shape: (B, ep_idx)
        if isinstance(td_errors_all, list):
            # list of tensors → stack along dim=1
            td_errors_all = torch.stack(td_errors_all, dim=1).detach().cpu().numpy()
        elif isinstance(td_errors_all, torch.Tensor):
            # single tensor → just detach and convert
            td_errors_all = td_errors_all.detach().cpu().numpy()
        else:
            raise TypeError(f"td_errors_all must be Tensor or list of Tensors, got {type(td_errors_all)}")
        
        for batch_idx, ep_idx in enumerate(indices):
            start = start_indices[batch_idx]
            episode_stepwise = self.priorities[ep_idx]

            # update only the sampled segment
            episode_stepwise[start:start+td_errors_all.shape[1]] = td_errors_all[batch_idx].tolist()

            # update episode-level priority as mean of stepwise
            self.prioritiesOverEpisodes[ep_idx] = np.mean(episode_stepwise) + 1e-6


    def sample(self, **kwargs):
        priorities = np.array(self.prioritiesOverEpisodes, dtype=np.float32)

        # clean priorities
        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1.0, neginf=1e-6)
        priorities = np.abs(priorities) + 1e-6

        probs = priorities ** self.alpha

        # clean probs
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        prob_sum = probs.sum()

        if prob_sum <= 0 or not np.isfinite(prob_sum):
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs = probs / prob_sum

        batch_experiences = []
        batch_weights = []
        start_indices = []

        N = self.length()
        batch = min(self.batchSize, N)

        indices = np.random.choice(N, batch, replace=True, p=probs)

        updatedBeta = min(1.0, self.beta + self.beta_rate * kwargs['running_step'])

        for idx in indices:
            episode = self.buffer[idx]
            ep_prios = self.priorities[idx]

            ep_prios = np.nan_to_num(ep_prios, nan=1e-6, posinf=1.0, neginf=1e-6)
            valid_len = len(episode) - self.seqLen + 1

            if kwargs.get('priority_seq_sampling', False):
                if valid_len > 0:
                    segment_prios = ep_prios[:valid_len]

                    segment_prios = np.abs(segment_prios) + 1e-6
                    segment_probs = segment_prios ** self.alpha
                    segment_probs /= segment_probs.sum()

                    start = np.random.choice(valid_len, p=segment_probs)
                else:
                    start = 0
            else:
                # pick a random contiguous segment
                if valid_len > 0:
                    start = np.random.randint(0, valid_len)
                else:
                    start = 0

            segment = episode[start:start+self.seqLen]
            segment_prios = ep_prios[start:start+self.seqLen]
            segment_prios = list(segment_prios)

            # Pad if episode shorter than seqLen
            while len(segment) < self.seqLen:
                segment.append(segment[-1])
                segment_prios.append(1e-6)

            batch_experiences.append(segment)
            start_indices.append(start)

            # safe weight computation
            seg_prios_np = np.abs(np.array(segment_prios, dtype=np.float32))
            seg_prios_np = np.nan_to_num(seg_prios_np, nan=1e-6)

            mean_prio = max(seg_prios_np.mean(), 1e-6)

            w = (N * mean_prio) ** (-updatedBeta)
            batch_weights.append(w)

        batch_weights = np.array(batch_weights, dtype=np.float32)

        # normalize safely
        max_w = batch_weights.max()
        if max_w <= 0 or not np.isfinite(max_w):
            batch_weights = np.ones_like(batch_weights)
        else:
            batch_weights /= max_w

        return batch_experiences, indices, batch_weights, start_indices

  
    def splitExperiences(self, sequences):
        # sequences: list of list of (s,a,r,ns,d)
        batch_size = len(sequences)
        seq_len = len(sequences[0])

        states = np.zeros((batch_size, seq_len, len(sequences[0][0][0])), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        nextStates = np.zeros((batch_size, seq_len, len(sequences[0][0][0])), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i, seq in enumerate(sequences):
            for t, (s, a, r, ns, d) in enumerate(seq):
                states[i, t] = np.array(s, dtype=np.float32)
                actions[i, t] = a
                rewards[i, t] = r
                nextStates[i, t] = np.array(ns, dtype=np.float32)
                dones[i, t] = d

        return states, actions, rewards, nextStates, dones
    
    
    def length(self):
        #Your code goes in here
        #
        buffersize = len(self.buffer)
        return buffersize