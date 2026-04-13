import numpy as np
import random

# ================================================ Helper =================================================================

def decay_func(initial_value, final_value, episode, max_episode, decay_type, decay_stop=None):
    eps = 1e-8

    if decay_type == 'linear':
      if decay_stop == None or decay_stop >= episode:
        step_size = initial_value + episode * (final_value - initial_value) / (max_episode - 1 + eps)
      else:
        step_size = final_value

    else:
      if decay_stop == None or decay_stop >= episode:
        decay_factor = np.exp((np.log(final_value + eps) - np.log(initial_value + eps)) / (max_episode - 1 + eps))
        step_size = initial_value * decay_factor**episode
      else:
        step_size = final_value

    return step_size

def generate_episode_step_eps(initial_value, final_value, max_episode, decay_type, decay_stop=None):
    step_sizes = []
    for episode in range(max_episode):
      step_sizes.append(decay_func(initial_value, final_value, episode, max_episode, decay_type, decay_stop))

    return step_sizes

def actionSelect(env, s, Q, eps):
    if env.np_random.random() < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])
    
def get_policy_success(environment, policy, config=None):
    # defined some useful variables
    env = environment
    base_env = env.unwrapped
    observation, info = env.reset(seed = config['seed'])
    s = observation['agent']
    sidx = base_env.states_to_index[s]

    gamma = config['discount_factor']
    max_steps = config['max_steps']
    goal_state = config['environment']['goal_state']
    terminated = False
    truncated = False
    discounted_cumm_reward = 0.0
    step = 0

    while not truncated and not terminated:
      action = policy[sidx]
      observation, reward, terminated, truncated, _ = env.step(action)

      s_prime = observation['agent']
      discounted_cumm_reward += ((gamma**step) * reward)

      sidx = base_env.states_to_index[s_prime]
      s = s_prime
      step += 1

    return True if s == goal_state else False, discounted_cumm_reward

# ============================================================================================================================


def trajectory_sampling_control(environment, config=None):
    """
    Trajectory Sampling (Trajectory Learning) Control Algorithm

    Algorithm (from lecture slides):
        TrajectorySampling(env, γ, α0, ε0, noEpisodes, maxTrajectory)

    Goal:
        Learn an optimal action-value function Q(s, a) by combining:
            - Direct interaction with the environment
            - A learned model of the environment
            - Simulated trajectories sampled from the learned model

    The algorithm alternates between:
        1. Real experience updates (Q-learning)
        2. Planning updates via trajectory sampling

    Inputs:
        environment : Random Maze Environment (RME)
        config      : Dictionary containing γ, α0, ε0, number of episodes,
                      maximum trajectory length, and sampling probability
    """
    # reseting the seed for reproducibility
    env = environment
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    env.reset(seed=config['seed'])

    # Unwrapping the environment for mapping either states to index ot index to states
    # (Since I haven't register the wall state in the observation space so handling the mapping explicitly)
    base_env = env.unwrapped

    # Extracting required data from config
    gamma = config['discount_factor']
    max_episode = config['max_episodes']
    total_states = config['environment']['total_states']
    total_actions = config['environment']['total_actions']

    # Generating the episode-wise step sizes for 'max_episodes' number of episodes
    # See Section: generate_episode_step_sizes (above)
    alphadecay = generate_episode_step_eps(**config['step_size'], max_episode=max_episode)

    # Generating the episode-wise epsilons for 'max_episodes' number of episodes
    # See Section: generate_episode_epsilons (above)
    epsilondecay = generate_episode_step_eps(**config['epsilon'], max_episode=max_episode)

    # defining variables to be return in the end as mentioned above in docstring
    Q = np.zeros(shape=(total_states, total_actions), dtype=np.float32)
    V_s = np.zeros(shape=(max_episode, total_states), dtype=np.float32)
    policy = np.empty(total_states)
    Qs = np.zeros(shape=(max_episode, total_states, total_actions))
    policy_success = np.zeros(max_episode, dtype=bool)
    return_reward = np.zeros(max_episode, dtype=np.float32)

    T = np.zeros(shape=(total_states, total_actions, total_states), dtype=np.float32)
    R = np.zeros(shape=(total_states, total_actions, total_states), dtype=np.float32)

    # Trajectory Learning Logic
    for e in range(max_episode):
      alpha = alphadecay[e]
      epsilon = epsilondecay[e]
      E = np.zeros(shape=(total_states, total_actions), dtype=np.float32)
      obs, info = env.reset()
      s = base_env.states_to_index[obs['agent']]
      done = False
      truncated = False
      '''
      Trajectory Learning is also a model based learning and very simimlar to Dyna-Q.
      Only difference is that here state s_p is not sampled from T, instead we use initial state s.

      Q(s,a) = Q(s,a) + alpha * [ r + gamma * max_a'(Q(s',a')) ]
      T[s,a,s'] += 1
      R[s,a,s'] += (r - R[s,a,s']) / T[s,a,s']

      Repeat for max_trajectory:
        s_p = s
        action a_p is chosen either greedily or randomly with probability epsilon
        s'_p ~ S with probability p = T[s_p,a_p] / sum(T[s_p,a_p,:])
        r = R[s_p,a_p,s'_p]
        Q(s_p,a_p) = Q(s_p,a_p) + alpha * [ r + gamma * max_a'(Q(s'_p,a')) ]

      where:
        td_target = r + gamma * max_a'(Q(s',a'))
        td_error = td_target - Q(s,a)
      '''
      
      while not done and not truncated:
        a = actionSelect(env, s, Q, epsilon)
        obs_prime, r, done, truncated, info = env.step(a)
        s_prime = base_env.states_to_index[obs_prime['agent']]
        T[s,a,s_prime] += 1
        rdiff = r - R[s, a, s_prime]
        R[s, a, s_prime] += rdiff / T[s, a, s_prime]

        td_target = r

        if not done:
          td_target += gamma * np.max(Q[s_prime])
        td_error = td_target - Q[s][a]

        Q[s,a] += alpha * td_error
        s_backup = s_prime

        for _ in range(config['maxTrajectory']):
          if np.sum(Q) == 0:
            break
          a = actionSelect(env, s, Q, epsilon)
          if np.sum(T[s,a,:]) == 0:
            break
          prob_s_prime = T[s,a] / np.sum(T[s,a,:])
          s_prime = np.random.choice(total_states, p=prob_s_prime)
          r = R[s,a,s_prime]
          td_target = r + gamma * np.max(Q[s_prime])
          td_error = td_target - Q[s,a]
          Q[s,a] += alpha * td_error
          s = s_prime

        s = s_backup

      Qs[e] = Q.copy()
      V_s[e] = np.max(Q, axis=1)

      # Evaluating Performance Metrics in each episode
      policy = np.argmax(Q, axis=1)
      for _ in range(config['rollouts']):
        # Finding policySucess and returnReward for each rollout using `get_policy_sucess(env,policy,config)`
        # See Section: get_policy_success(env, policy, config)
        policySuccess, returnReward = get_policy_success(env, policy, config)

        # adding policySucess of all rollouts within each episode
        policy_success[e] += policySuccess
        # adding returnReward of all rollouts within each episode
        return_reward[e] += returnReward

      # Taking average of policy_success and return_reward for number of rollouts used
      policy_success[e] /= config['rollouts']
      return_reward[e] /= config['rollouts']

    # Finally returning the mentioned variables above in docstring
    return Q, V_s, policy, Qs, None, policy_success, return_reward
