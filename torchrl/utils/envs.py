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

        env = atari_wrappers.wrap_deepmind(env)
        env = TransposeImage(env)
        return env
    return _env_func


def make_parallel_env(env_name, seed, num_workers, num_frame_stack, device, log_dir=None):
    env = [make_atari_env(env_name, seed, i, log_dir)
           for i in range(num_workers)]
    env = SubprocVecEnv(env)
    env = VecPyTorch(env, device)
    env = VecPyTorchFrameStack(env, num_frame_stack, device)

    return env
