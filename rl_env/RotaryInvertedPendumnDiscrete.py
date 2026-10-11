# Rotary inverted pendulum (Furuta pendulum) environment, designed to mirror CartPole-v1.

import os
import threading
import time
import gymnasium as gym
import mujoco
from gymnasium import spaces
import numpy as np

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "rotary_pendulumn.xml")


class RotaryInvertedPendumnDiscrete(gym.Env):
    """
    Furuta pendulum balancing task, CartPole-v1 style.

    Observation (4,):
        [theta, alpha, theta_dot, alpha_dot]
        theta: rotary arm angle   (rad), 0 at start
        alpha: pendulum angle     (rad), 0 when upright

    Action: Discrete(2)
        0: apply -torque_mag to the arm (push left)
        1: apply +torque_mag to the arm (push right)

    Reward:
        +1 for every step, including the terminating one (same as CartPole)

    Starting state:
        all state variables uniformly random in [-0.05, 0.05]

    Termination:
        |alpha| > alpha_threshold  (pendulum fell, default 12 deg)
        |theta| > theta_threshold  (arm left its allowed range, default 90 deg)

    Truncation:
        episode length reaches max_episode_step (default 500)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path=XML_PATH,
                 render_mode=None,
                 max_episode_step=500,
                 torque_mag=0.1,
                 frame_skip=20,
                 theta_threshold=np.pi / 2,
                 alpha_threshold=60 * 2 * np.pi / 360):
        super().__init__()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.max_episode_step = max_episode_step  # maximum env step per rollout
        self.torque_mag = torque_mag              # constant torque per action (Nm)
        self.frame_skip = frame_skip              # physics steps per env step (0.001 s * 20 = 0.02 s, like CartPole tau)
        self.tau = self.mj_model.opt.timestep * frame_skip  # seconds between env steps
        self.theta_threshold = theta_threshold    # allowed arm range of motion
        self.alpha_threshold = alpha_threshold    # pendulum angle at which the episode fails

        # joint addresses in qpos / qvel
        arm_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint")
        pend_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        self.arm_qpos, self.arm_qvel = self.mj_model.jnt_qposadr[arm_id], self.mj_model.jnt_dofadr[arm_id]
        self.pend_qpos, self.pend_qvel = self.mj_model.jnt_qposadr[pend_id], self.mj_model.jnt_dofadr[pend_id]

        self.action_space = spaces.Discrete(2)
        # like CartPole: angle bounds are twice the failure thresholds so failing observations are still in bounds
        high = np.array([self.theta_threshold * 2,
                         self.alpha_threshold * 2,
                         np.inf,
                         np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.viewer_threads = set()  # background thread(s) running the passive viewer
        self.last_render_time = None  # wall-clock time of the last human-mode frame
        self.renderer = None
        self.step_count = 0

    def _get_obs(self):
        theta = self.mj_data.qpos[self.arm_qpos]
        alpha = self.mj_data.qpos[self.pend_qpos]
        theta_dot = self.mj_data.qvel[self.arm_qvel]
        alpha_dot = self.mj_data.qvel[self.pend_qvel]
        return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # small random perturbation around upright, like CartPole
        theta, alpha, theta_dot, alpha_dot = self.np_random.uniform(-0.05, 0.05, size=4)
        self.mj_data.qpos[self.arm_qpos] = theta
        self.mj_data.qpos[self.pend_qpos] = alpha
        self.mj_data.qvel[self.arm_qvel] = theta_dot
        self.mj_data.qvel[self.pend_qvel] = alpha_dot
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.step_count = 0  # reset step count

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        self.mj_data.ctrl[0] = self.torque_mag if action == 1 else -self.torque_mag
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=self.frame_skip)
        self.step_count += 1

        obs = self._get_obs()
        theta, alpha = obs[0], obs[1]
        terminated = bool(abs(theta) > self.theta_threshold
                          or abs(alpha) > self.alpha_threshold)
        truncated = self.step_count >= self.max_episode_step
        reward = 1.0

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                threads_before = set(threading.enumerate())
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
                self.viewer_threads = set(threading.enumerate()) - threads_before
            self.viewer.sync()
            # sleep so each env step (tau seconds of sim time) takes tau seconds of wall time
            now = time.perf_counter()
            if self.last_render_time is not None:
                time.sleep(max(0.0, self.tau - (now - self.last_render_time)))
            self.last_render_time = time.perf_counter()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.mj_model)
            self.renderer.update_scene(self.mj_data, camera="fixed")
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            # close() only requests exit; wait for the viewer thread to tear down its
            # OpenGL context, otherwise the interpreter can segfault on shutdown
            for thread in self.viewer_threads:
                thread.join(timeout=5.0)
            self.viewer_threads = set()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def main():
    env = RotaryInvertedPendumnDiscrete()
    obs, _ = env.reset(seed=0)
    print("Initial observation:", obs)

    episode_return, done = 0.0, False
    while not done:
        action = env.action_space.sample()  # random policy
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        done = terminated or truncated
    print("Random policy return:", episode_return)
    env.close()


if __name__ == "__main__":
    main()
