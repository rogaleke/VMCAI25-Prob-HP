import gymnasium as gym
import numpy as np
import math
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

class CustomMountainCarEnv(MountainCarEnv):
    def reset(self, init):
        self.state = np.array(init)
        self.steps_beyond_done = None
        return np.array(self.state)

