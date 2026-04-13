import torch
import sys
sys.path.append("..")
from helper import selectEpsilonGreedyActionREP, selectGreedyActionREP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used: ', device)

config = {
    'seed': 333,
    'gamma': 0.99,
    'bufferSize': 1_000_000,
    'batchSize': 64,
    'optimizerFn': torch.optim.Adam,
    'optimizerLR': 7e-4,
    'MAX_TRAIN_EPISODES': 260,
    'MAX_EVAL_EPISODES': 5,
    'updateFrequency': 100,
    'explorationStrategyTrainFn': selectEpsilonGreedyActionREP,
    'explorationStrategyEvalFn': selectGreedyActionREP,
    'max_steps': 1000,
    'epsilon': 0.58,
    'eps_decay_strategy': [
        ("exponential", {'s': 0, 'e': 249, 'ival': 1.0, 'fval': 0.005})
    ],
    'device': device,
    'delta': 0.99,
    'tau': 0.001,
    'alpha': 0.58,
    'beta': 0.42,
    'beta_rate': 0.0002,
    'f_hDim': [512, 256],
    'lstm_hDim': 128,
    'model_path': '../model_weights_phase3_sub3',
    'loss_fn': 'HuberLoss',
    'seq_len': 50,
    'burn_in': 20,
    'minSamples': 1, # represents to collection of one whole sequqnce of experiences
    'sampleAction': False,
    'for_hDim': [256],
    'inv_hDim': [256],
    'alpha_f': 0.1,
    'beta_i': 0.1,
    'eta': 0.01
}