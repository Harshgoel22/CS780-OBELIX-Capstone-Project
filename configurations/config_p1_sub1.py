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
    'optimizerLR': 0.0022,
    'MAX_TRAIN_EPISODES': 300,
    'MAX_EVAL_EPISODES': 1,
    'updateFrequency': 5,
    'explorationStrategyTrainFn': selectEpsilonGreedyAction,
    'explorationStrategyEvalFn': selectGreedyAction,
    'max_steps': 500,
    'epochs': 20,
    'epsilon': 0.58,
    'eps_decay_strategy': [
        ("linear", {'s': 15, 'e': 149, 'ival': 1.0, 'fval': 0.645}),
        ("exponential", {'s': 155, 'e': 255, 'ival': 0.645, 'fval': 0.02})
    ],
    'device': device
}