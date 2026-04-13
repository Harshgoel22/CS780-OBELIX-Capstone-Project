import torch
import sys
sys.path.append("..")
from helper import selectPolicyAction, selectPolicyGreedyAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used: ', device)

config = {
    'seed': 333,
    'gamma': 0.999,
    'optimizerFn': torch.optim.Adam,
    'optimizerLR': 0.001,
    'MAX_TRAIN_EPISODES': 200,
    'MAX_EVAL_EPISODES': 5,
    'explorationStrategyTrainFn': selectPolicyAction,
    'explorationStrategyEvalFn': selectPolicyGreedyAction,
    'device': device,
    'valuehDim': [64, 64],
    'policyhDim': [64, 64],
    'model_path': '../model_weights_phase1_sub7',
    'beta': 0.2,
    'MAX_GRAD_NORM_POLICY': 0.5,
    'MAX_GRAD_NORM_VALUE': 0.5
}