"""Original deterministic two-step Gymnasium task for delayed-reward tests."""
import numpy as np
import gymnasium as gym


class DelayedReward(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=np.array([0,0],np.float32), high=np.array([2,1],np.float32))
        self.action_space = gym.spaces.Discrete(2)
        self.tick = None
        self.choice = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick, self.choice = 0, 0
        return np.array([0,0], np.float32), {}

    def step(self, action):
        if self.tick is None or self.tick >= 2:
            raise RuntimeError('Reset required')
        if not self.action_space.contains(action):
            raise ValueError('Discrete action 0 or 1 required')
        if self.tick == 0:
            self.choice = int(action)
        self.tick += 1
        reward = 1. if self.tick == 1 else float(2*self.choice)
        return np.array([self.tick,self.choice],np.float32), reward, self.tick == 2, False, {}
