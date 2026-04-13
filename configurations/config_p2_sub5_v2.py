import torch
import sys
sys.path.append("..")
from helper import selectEpsilonGreedyAction, selectGreedyAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used: ', device)

config = {
    'seed': 333,
    'gamma': 0.999,
    'bufferSize': 100000,
    'batchSize': 32,
    'optimizerFn': torch.optim.Adam,
    'optimizerLR': 3e-4,
    'MAX_TRAIN_EPISODES': 250,
    'MAX_EVAL_EPISODES': 5,
    'updateFrequency': 150,
    'explorationStrategyTrainFn': selectEpsilonGreedyAction,
    'explorationStrategyEvalFn': selectGreedyAction,
    'max_steps': 1000,
    'epochs': 20,
    'epsilon': 0.58,
    'eps_decay_strategy': [
        ("exponential", {'s': 0, 'e': 249, 'ival': 1.0, 'fval': 0.005})
    ],
    'device': device,
    'delta': 0.99,
    'tau': 0.005,
    'alpha': 0.58,
    'beta': 0.42,
    'beta_rate': 0.0002,
    'f_hDim': [512, 512, 256],
    'lstm_hDim': 256,
    'model_path': '../model_weights_phase2_sub5_v2',
    'loss_fn': 'HuberLoss',
    'seq_len': 30,
    'burn_in': 15,
    'minSamples': 2
}