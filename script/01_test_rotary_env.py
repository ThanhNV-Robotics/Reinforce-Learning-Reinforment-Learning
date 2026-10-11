# 1st step: Getting started with gym.env
# we load and perform a rollout of the env
# load the env name in a config dictionary

# File created:
# 1. infrastructure/config.py: load the env name in a config dictionary
# 2. config/env_cf.yaml

import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import torch
import gymnasium as gym
from RotaryInvertedPendumnDiscrete import RotaryInvertedPendumnDiscrete

def main ():
   # create gym
    env = RotaryInvertedPendumnDiscrete(render_mode="human")
    obs, _ = env.reset(seed=0)
    print("Initial observation:", obs)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n

    print("Observation dimension: ", ob_dim)
    print("Action dimension: ", ac_dim)
    print("sampling time (tau): ", env.unwrapped.tau)

    # perform an episode rollout
    obs,_ = env.reset()
    max_n_step = 200 # maximum env step

    print("Starting random rollout...")
    total_reward = 0
    for step in range(max_n_step):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            print(f"Episode finished after {step+1} steps")
            print(f"Total reward: {total_reward}")
            break
    env.close()
    return

if __name__ == '__main__':
    main()