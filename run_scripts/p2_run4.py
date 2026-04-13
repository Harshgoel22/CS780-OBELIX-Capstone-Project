import sys
sys.path.append("..")

from obelix import OBELIX
from algorithms.a2c import train_a2c
from configurations.config_p2_sub4 import config
from helper import setup_logger
import torch.multiprocessing as mp
import torch
import numpy as np

envConfig = {
    'scaling_factor': 5,
    'arena_size': 500,
    'max_steps': 1000,
    'wall_obstacles': False,
    'difficulty': 2,
}

if __name__ == '__main__':
    logger = setup_logger(config['log_path'])
    batch_size =  config['num_envs'] * config['num_steps']
    msg = f"     {config['total_timesteps'] // batch_size} gradient updates in {config['total_timesteps']}"
    logger.info(msg)
    
    logger.info("======================== Training Started ============================\n")
    a2c_train_r, a2c_eval_r, a2c_train_t, a2c_steps, a2c_wall_t = train_a2c(envClass=OBELIX, envConfig=envConfig, **config)
    logger.info("======================== Training Finished ===========================\n")

    r_log = f"Mean Train Reward: {np.mean(a2c_train_r)} | Mean Eval Reward: {np.mean(a2c_eval_r)}"
    logger.info(r_log)

    config["A2C"] = {
        "Train Rewards": a2c_train_r,
        "Eval Rewards": a2c_eval_r,
        "Training Time (in sec)": a2c_train_t,
        "Steps": a2c_steps,
        "WallClock Time (in sec)": a2c_wall_t
    }

    torch.save(config, f"{config['model_path']}/a2c_results.pt")
    