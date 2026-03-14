import sys
sys.path.append("..")
from obelix import OBELIX
from algorithms.ddqn_agent1 import DDQN
from configurations.config_p1_sub1 import config

env = OBELIX(
    scaling_factor=5,
    arena_size=500,
    max_steps=1000,
    wall_obstacles=False,
    difficulty=0,
    box_speed=2
)
ddqnAgent = DDQN(env, config)

ddqnTrainRewardsList1, ddqnTrainTimeList1, ddqnEvalRewardsList1, ddqnWallClockTimeList1, ddqnTotalStepsList1, ddqnFinalEvalReward1 = ddqnAgent.runDDQN()