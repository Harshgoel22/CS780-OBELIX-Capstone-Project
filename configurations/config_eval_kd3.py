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
    'optimLR': 1e-4,
    'MAX_TRAIN_EPISODES': 500,
    'MAX_EVAL_EPISODES': 5,
    'explorationStrategyTrainFn': selectEpsilonGreedyActionREP_NOISY,
    'explorationStrategyEvalFn': selectGreedyActionREP_COMP,
    'max_steps': 1000,
    'epsilon': 0.98,
    'eps_decay_strategy': [
        ("linear", {'s': 0, 'e': 200, 'ival': 1.0, 'fval': 0.3}),
        ("exponential", {'s': 201, 'e': 499, 'ival': 0.3, 'fval': 0.03})
    ],
    'device': device,
    'beta': 0.3,
    'f_hDim': [324, 256],
    'lstm_hDim': 128,
    'model_path': '../model_weights/model_weights_eval_kd3',
    'seq_len': 40,
    'burn_in': 10,
    'minSamples': 40, # represents to collection of one whole sequqnce of experiences
    'for_hDim': [224, 128],
    'inv_hDim': [224, 128],
    'noise_std': 0.01,
    'clip_eps': 0.2,
    'entropy_coef': 0.01,
    'forward_coef': 0.1,
    'inverse_coef': 0.04,
    'action_dim': 5,
    'state_dim': 18,
    'tau': 0.2,
    'updateTargetFrequency': 60,
    'kl_coef': 0.12,
    'gae_lambda': 0.95
}