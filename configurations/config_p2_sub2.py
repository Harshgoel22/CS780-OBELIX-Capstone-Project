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
    'optimizerLR': 0.0002,
    'MAX_TRAIN_EPISODES': 500,
    'MAX_EVAL_EPISODES': 1,
    'updateFrequency': 500, # based on steps
    'explorationStrategyTrainFn': selectEpsilonGreedyAction,
    'explorationStrategyEvalFn': selectGreedyAction,
    'max_steps': 1000,
    'epsilon': 0.58,
    'eps_decay_strategy': [
        ("linear", {'s': 0, 'e': 499, 'ival': 1.0, 'fval': 0.0012})
    ],
    'device': device,
    'delta': 1.0,
    'tau': 0.15,
    'alpha': 0.45,
    'beta': 0.55,
    'beta_rate': 0.0002,
    'hDim': [128, 64],
    'model_path': '../model_weights_phase2_sub2',
    'loss_fn': 'HuberLoss',
}