import sys
sys.path.append("..")

config = {
    'seed': 333,
    'gamma': 0.999,
    'num_workers': 8,
    'episodes_per_worker': 150, 
    'num_steps': 10,
    'lr': 0.0001,
    'ent_coef': 0.25,
    'vf_coef': 0.15,
    'max_grad_norm': 0.5
}