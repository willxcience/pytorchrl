import gym
import torch
import torch.nn as nn
import numpy as np
from . import atari_wrappers
from ..parallel import SubprocVecEnv

# def make_env(env_id):
#     def env_func():
#         env = gym.make(env_id)
#         return env
#     return env_func


def make_atari_env(env_id, num_env):

    # define a temp function call
    def make_env():
        def _env_func():
            env = atari_wrappers.make_atari(env_id)
            return atari_wrappers.wrap_deepmind(env)
        return _env_func

    return SubprocVecEnv([make_env() for i in range(num_env)])


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def v_wrap(x, dtype=np.float32):
    return torch.from_numpy(dtype(x))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        #nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0)