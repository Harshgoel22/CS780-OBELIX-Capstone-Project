"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            layers = []
            prev_dim = 18

            for h in [64, 64]:
                layers.append(nn.Linear(prev_dim, h))
                prev_dim = h

            layers.append(nn.Linear(prev_dim, 5))

            self.layers = nn.ModuleList(layers)
            self.activation = F.relu

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))

            x = self.layers[-1](x)
            return x

    model = Net()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(q))]