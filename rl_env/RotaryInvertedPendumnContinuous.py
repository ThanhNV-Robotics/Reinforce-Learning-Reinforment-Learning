# This file demonstrates how to build a custom environment for Reinforcement Learning using Gym.

import gymnasium as gym
import mujoco
from gymnasium import spaces
import numpy as np
# from mujoco import mjx

class RotaryInvertedPendumn (gym.Env):
    """
    Furuta pendulum environment.

    Observation:
        [sin(theta), cos(theta),
         sin(alpha), cos(alpha),
         theta_dot, alpha_dot]

    theta: rotary arm angle
    alpha: pendulum angle, zero when upright

    Action:
        normalized motor torque in [-1, 1]
    """
    # constructor
    def __init__(self, xml_path,
                  render_mode = None,
                  max_episode_step = 500):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.max_episode_step = max_episode_step # maximum env step per rollout

        act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_motor")
        fmin, fmax = self.mj_model.actuator_forcerange[act_id]   # shape (nu, 2)

        self.fmin = fmin
        self.fmax = fmax
        self.action_space = spaces.Box(fmin, fmax, shape=(1,), dtype=np.float32) # must define
        high = np.array([np.pi, np.pi, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32) # must define
        self.step_count = 0   
        

    def _get_obs(self):
        theta, alpha = self.mj_data.qpos[0], self.mj_data.qpos[1]
        theta_dot, alpha_dot = self.mj_data.qvel[0], self.mj_data.qvel[1]
        # wrap angles to [-pi, pi] so they stay inside the observation bounds
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # small random perturbation around upright
        self.mj_data.qpos[0] = 0.0 # set the arm angle at 0
        self.mj_data.qpos[1] = np.pi # set the pendulumn angle at 0
        self.mj_data.qvel[:] = 0.0 # start at rest
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.step_count = 0 # reset step count
        return self._get_obs(), {}

    def step(self, action):
            action = np.clip(action, self.fmin, self.fmax)
            self.mj_data.ctrl[0] = action[0]          # scale by max torque if needed
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.step_count += 1

            obs = self._get_obs()
            alpha = self.mj_data.qpos[1] # pendulumn angle
            reward = float(np.cos(alpha) - 0.01 * action[0] ** 2)
            terminated = bool(abs(alpha) > np.pi / 2)  # fell over
            truncated = self.step_count >= self.max_episode_step
            return obs, reward, terminated, truncated, {}

    def calculate_reward (self, observation, action):
         theta = observation[0]
         alpha = observation[1]

         r_up = np.cos(alpha)
         r_center = np.cos(theta)
         r_force = -0.01*action[0]**2

         r = r_up + r_center + r_force 


def main ():
    print("Hello")
    XML_PATH = "/model/rotary_pendulm.xml"

if __name__ == "__main__":
    main()
    