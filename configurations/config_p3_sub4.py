import torch
import sys
sys.path.append("..")
from helper import selectEpsilonGreedyActionREP_COMP, selectGreedyActionREP_COMP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used: ', device)

config = {
    'seed': 333,
    'gamma': 0.99,
    'bufferSize': 1_000_000,
    'batchSize': 64,
    'optimizerFn': torch.optim.Adam,
    'optimizerLR': 5e-4,
    'MAX_TRAIN_EPISODES': 600,
    'MAX_EVAL_EPISODES': 5,
    'updateFrequency': 100,
    'explorationStrategyTrainFn': selectEpsilonGreedyActionREP_COMP,
    'explorationStrategyEvalFn': selectGreedyActionREP_COMP,
    'max_steps': 1000,
    'epsilon': 0.58,
    'eps_decay_strategy': [
        ("exponential", {'s': 0, 'e': 599, 'ival': 1.0, 'fval': 0.002})
    ],
    'device': device,
    'delta': 0.99,
    'tau': 0.001,
    'alpha': 0.58,
    'beta': 0.42,
    'beta_rate': 0.0002,
    'f_hDim': [324, 256],
    'lstm_hDim': 128,
    'model_path': '../model_weights_phase3_sub4',
    'loss_fn': 'HuberLoss',
    'seq_len': 20,
    'burn_in': 6,
    'minSamples': 2, # represents to collection of one whole sequqnce of experiences
    'for_hDim': [224, 128],
    'inv_hDim': [224, 128],
    'alpha_f': 0.12,
    'beta_i': 0.15,
    'eta': 0.07
}