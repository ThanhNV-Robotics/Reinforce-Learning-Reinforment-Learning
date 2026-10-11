import sys
from pathlib import Path

# Add the project root to sys.path so the env package can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import gymnasium as gym
from env.RotaryInvertedPendumnDiscrete import RotaryInvertedPendumnDiscrete
import numpy as np
from typing import Dict


def sampling_trajectory (env:gym.Env, max_steps: int) :
    """Sample a rollout in the environment using env.action_space.sample() to select the action
    Args:
        env: gym env
        max_steps: max number of step to collect data in one rollout
    Use np.ndarray as gym env runs on CPU. Latter for the training, we need to pass to torch to efficently run on GPU
    """
    ob,__ = env.reset()
    # list of observation, rewards, actions, next_observations, terminals
    obs, acs, rewards, next_obs, terminals = [], [], [], [], [] 
    num_steps = 0
    while True: 
        # sample an action from current observation
        action = env.action_space.sample()
        # perform an env step to get next observation, reward, done, truncated, info
        next_ob, reward, done, truncated, info = env.step(action)
        num_steps += 1
        rollout_done = done or truncated or num_steps >= max_steps

        # append to data lists
        obs.append(ob)
        acs.append(action)
        rewards.append(reward)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        # update observation
        ob = next_ob
        # break if max length is reached
        if rollout_done:
            print("Rollout Done")
            break

    # store the trajectory data
    trajectories = {
        "observations": np.array(obs, dtype=np.float32),
        "actions": np.array(acs, dtype=np.uint8), # for discrete action space
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_obs, dtype=np.float32),
        "terminals": np.array(terminals, dtype=np.float32)
    }
    env.close()
    return trajectories
    
    
def main():
    env = RotaryInvertedPendumnDiscrete(render_mode="human")
    sample_trajector = sampling_trajectory(env, 200)
    print("A rollout trajectory data: ")
    print("Number of steps in trajectory: ", len(sample_trajector["observations"]))
    print("Observations: ", sample_trajector["observations"])
    print("Actions: ", sample_trajector["actions"])
    print("Rewards: ", sample_trajector["rewards"])
    print("Total reward: ", np.asarray(sample_trajector["rewards"]).sum())
    print("Next Observations: ", sample_trajector["next_observations"])
    print("Terminals: ", sample_trajector["terminals"])
    return

if __name__ == '__main__':
    main()

