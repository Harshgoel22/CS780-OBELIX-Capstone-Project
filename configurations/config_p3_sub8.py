import torch
import sys
sys.path.append("..")
from helper import selectEpsilonGreedyActionREP_NOISY, selectGreedyActionREP_COMP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used: ', device)

config = {
    'seed': 333,
    'gamma': 0.99,
    'bufferSize': 1_000_000,
    'batchSize': 32,
    'optimizerFn': torch.optim.Adam,
    'optimizerLR': 1e-4,
    'MAX_TRAIN_EPISODES': 1000,
    'MAX_EVAL_EPISODES': 5,
    'updateFrequency': 60,
    'explorationStrategyTrainFn': selectEpsilonGreedyActionREP_NOISY,
    'explorationStrategyEvalFn': selectGreedyActionREP_COMP,
    'max_steps': 1000,
    'epsilon': 0.98,
    'eps_decay_strategy': [
        ("linear", {'s': 0, 'e': 399, 'ival': 1.0, 'fval': 0.4}),
        ("exponential", {'s': 400, 'e': 999, 'ival': 0.4, 'fval': 0.01})
    ],
    'device': device,
    'delta': 0.99,
    'tau': 0.005,
    'alpha': 0.58,
    'beta': 0.42,
    'beta_rate': 0.0002,
    'f_hDim': [324, 256],
    'lstm_hDim': 128,
    'model_path': '../model_weights/model_weights_phase3_sub8',
    'loss_fn': 'HuberLoss',
    'seq_len': 30,
    'burn_in': 10,
    'minSamples': 40, # represents to collection of one whole sequqnce of experiences
    'for_hDim': [224, 128],
    'inv_hDim': [224, 128],
    'alpha_f': 0.12,
    'beta_i': 0.15,
    'eta': 0.07,
    'noise_std': 0.22
}