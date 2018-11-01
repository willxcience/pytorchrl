import torch
import torch.optim as optim
import numpy as np
from collections import deque
import pdb


class A2C(object):
    def __init__(self, env, model, optimzier, params):
        self.env = env
        self.model = model
        self.optimzier = optimzier
        self.params = params
        self.episode_rewards = deque(maxlen=50)
        self.online_rewards = np.zeros(params["num_workers"])
        self.num_workers = params["num_workers"]
        self.gamma = params["discount_gamma"]
        self.tau = params["gae_tau"]
        self.use_gae = params["use_gae"]
        self.max_grad_norm = params["max_grad_norm"]
        self.use_cuda = params["use_cuda"]
        self.rollout_len = params["rollout_len"]

        self.num_steps = 0


