import gym
import torch
import torch.nn as nn
import numpy as np
import os

from . import atari_wrappers
from ..bench import Monitor
from ..parallel import SubprocVecEnv
from ..parallel import VecPyTorch
from ..parallel import VecPyTorchFrameStack, TransposeImage

import cv2

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


"""
These functions need to be adjusted according to the settings
"""


def make_atari_env(env_id, seed, rank, log_dir=None):
    # define a temp function call
    def _env_func():
        env = atari_wrappers.make_atari(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)))

        #env = atari_wrappers.wrap_deepmind(env)
        env = WarpFrame(env)
        env = TransposeImage(env)
        return env
    return _env_func


def make_parallel_env(env_name, seed, num_workers, num_frame_stack, device, log_dir=None):
    env = [make_atari_env(env_name, seed, i, log_dir)
           for i in range(num_workers)]
    env = SubprocVecEnv(env)
    env = VecPyTorch(env, device)
    #env = VecPyTorchFrameStack(env, num_frame_stack, device)

    return env
