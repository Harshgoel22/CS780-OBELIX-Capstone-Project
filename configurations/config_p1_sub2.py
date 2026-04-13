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
        ("linear", {'s': 5, 'e': 85, 'ival': 1.0, 'fval': 0.499}),
        ("exponential", {'s': 90, 'e': 150, 'ival': 0.499, 'fval': 0.05})
    ],
    'device': device,
    'delta': 0.99,
    'hDim': [64,32],
    'model_path': '../model_weights_phase1_sub2',
    'loss_fn': 'HuberLoss'
}