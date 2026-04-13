from collections import deque
import numpy as np
import torch

class SeqReplayBuffer():
    def __init__(self, bufferSize, seq_len = 40, batchSize = 32, max_steps = 1000):

        #Your code goes in here
        self.bufferSize = bufferSize
        self.seqLen = seq_len
        self.batchSize = batchSize
        self.max_steps = max_steps

        self.actions = ["L45", "L22", "FW", "R22", "R45"]
        self.buffer = deque(maxlen=self.bufferSize)
        self.temporaryBuffer = deque(maxlen=self.max_steps)


    def storeStepwiseExperiences(self, experience):
        #Your code goes in here
        self.temporaryBuffer.append(experience)


    def store(self):
        #Your code goes in here
        timeSeriesExperiences = list(self.temporaryBuffer)
        self.temporaryBuffer.clear()
        self.buffer.append(timeSeriesExperiences)


    def sample(self):
        if self.length() == 0:
            return None

        batch_experiences = []
        N = self.length()
        batch = min(self.batchSize, N)

        # sampling batch number of sequence indices without replacement
        indices = np.random.choice(N, batch, replace=False)

        for idx in indices:
            episode = self.buffer[idx]
            ep_len = len(episode)

            # if chosen episode length is smaller than seqLen then simply skipping
            if ep_len < self.seqLen:
                continue

            # randomly selecting a starting index within a sequence
            start = np.random.randint(0, ep_len - self.seqLen + 1)
            subseq = episode[start : start + self.seqLen]
            batch_experiences.append(subseq)

        if len(batch_experiences) == 0:
            return None

        return batch_experiences

  
    def splitExperiences(self, sequences):
        # sequences: list of list of (s,a,r,ns,d, log_p)
        batch_size = len(sequences)
        seq_len = len(sequences[0])

        states = np.zeros((batch_size, seq_len, len(sequences[0][0][0])), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        nextStates = np.zeros((batch_size, seq_len, len(sequences[0][0][0])), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len), dtype=np.float32)
        log_prob = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i, seq in enumerate(sequences):
            for t, (s, a, r, ns, d, log_p) in enumerate(seq):
                states[i, t] = np.array(s, dtype=np.float32)
                actions[i, t] = a
                rewards[i, t] = r
                nextStates[i, t] = np.array(ns, dtype=np.float32)
                dones[i, t] = d
                log_prob[i, t] = log_p

        return states, actions, rewards, nextStates, dones, log_prob
    
    
    def length(self):
        #Your code goes in here
        #
        buffersize = len(self.buffer)
        return buffersize