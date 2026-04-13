import sys
sys.path.append("..")

config = {
    'lr': 3e-4,                    
    'seed': 12,
    'total_timesteps': 10000000,   

    'num_workers': 8,
    'num_steps': 512,              

    'gamma': 0.995,              
    'gae_lambda': 0.97,         

    'update_epochs': 10,
    'num_minibatches': 16,       

    'clip_coef': 0.2,
    'clip_vloss': True,

    'ent_coef': 0.01,            
    'vf_coef': 0.5,

    'max_grad_norm': 0.5,
    'norm_adv': True,
    'anneal_lr': True,

    'target_kl': 0.03,     
}
