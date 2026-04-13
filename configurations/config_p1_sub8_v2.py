import sys
sys.path.append("..")

config = {
    'seed': 333,
    'gamma': 0.995,
    'num_workers': 8,
    'episodes_per_worker': 3000, 
    'num_steps': 10,
    'lr': 0.0002,
    'ent_coef': 0.015,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5
}