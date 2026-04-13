import sys
sys.path.append("..")

from obelix import OBELIX
from algorithms.ppo import train_ppo
from configurations.config_p2_sub1 import config
import torch

''' # used in first sub of phase 2
envConfig = {
        'scaling_factor': 5,
        'arena_size': 500,
        'max_steps': 1000,
        'wall_obstacles': False,
        'difficulty': 2,
        'box_speed': 2
    }
'''

if __name__ == '__main__':

    batch_size =  config['num_workers'] * config['num_steps']
    minibatch_size =  int(batch_size) // int(config['num_minibatches'])

    print(f"PPO: batch={batch_size}, minibatch={minibatch_size}")

    envConfig = {
        'scaling_factor': 10,
        'arena_size': 1000,
        'max_steps': 2500,
        'wall_obstacles': False,
        'difficulty': 2,
        'box_speed': 2
    }
    ppo_results, global_agent = train_ppo(EnvClass=OBELIX, envConfig=envConfig, **config)

    torch.save({
        "model_state_dict": global_agent.state_dict(),
        "results": ppo_results,
    }, "./model_phase2_sub1/ppo_checkpoint2.pt")