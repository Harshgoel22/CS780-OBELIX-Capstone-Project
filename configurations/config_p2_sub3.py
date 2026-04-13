import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used: ', device)

config = {
    'gamma': 0.99,
    'tau': 0.005,
    'bufferSize': 100_000,
    'entropyLR': 3e-4,
    'updateFrequency': 4,
    'policyOptimizerFn': torch.optim.Adam,
    'valueOptimizerFn_1': torch.optim.Adam,
    'valueOptimizerFn_2': torch.optim.Adam,
    'policyOptimizerLR': 3e-4,
    'valueOptimizerLR': 3e-4,
    'alphaOptimizerFn': torch.optim.Adam,
    'MAX_TRAIN_EPISODES': 500,
    'MAX_EVAL_EPISODE': 1,
    'hDims': [256, 256],
    'batchSize': 64,
    'minSamples': 1000,
    'device': device,
    'model_path': '../model_weights_phase2_sub3',
    'target_entropy': -0.5*np.log(5)
}

