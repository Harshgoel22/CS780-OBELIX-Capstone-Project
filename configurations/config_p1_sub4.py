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
    'batchSize': 64,
    'optimizerFn': torch.optim.Adam,
    'optimizerLR': 0.01,
    'MAX_TRAIN_EPISODES': 150,
    'MAX_EVAL_EPISODES': 1,
    'updateFrequency': 5,
    'explorationStrategyTrainFn': selectEpsilonGreedyAction,
    'explorationStrategyEvalFn': selectGreedyAction,
    'max_steps': 1000,
    'epochs': 20,
    'epsilon': 0.58,
    'eps_decay_strategy': [
        ("exponential", {'s': 0, 'e': 100, 'ival': 1.0, 'fval': 0.005})
    ],
    'device': device,
    'delta': 0.99,
    'tau': 0.1,
    'alpha': 0.58,
    'beta': 0.42,
    'beta_rate': 0.0002,
    'hDim': [256, 128, 32],
    'model_path': '../model_weights_phase1_sub4',
    'loss_fn': 'HuberLoss'
}