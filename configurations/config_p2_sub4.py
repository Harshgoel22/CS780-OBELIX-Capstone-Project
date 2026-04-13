import sys
sys.path.append("..")

config = {                    
    'seed': 33,
    'total_timesteps': 8_000_000,   
    'num_envs': 8,
    'num_steps': 1000,              
    'lr': 1e-4,
    'gamma': 0.99,              
    'gae_lambda': 0.955,         
    'use_gae': True,  
    'ent_coef': 0.002,            
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'norm_adv': True,
    'model_path': './model_weights_phase2_sub4',
    'log_path': './model_weights_phase2_sub4/training_logs.txt' 
}
