import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, params):
        self.rollout_len = params["rollout_len"]
        self.num_workers = params["num_workers"]
        self.obs_shape = params["obs_shape"]
        self.action_shape = params["action_shape"]
        self.rnn_size = params["rnn_size"]

        # returns hyper parameters
        self.gamma = params["discount_gamma"]
        self.tau = params["gae_tau"]
        self.use_gae = params["use_gae"]
    

        self.obs = torch.zeros(self.rollout_len + 1,
                               self.num_workers, *self.obs_shape)
        self.rewards = torch.zeros(self.rollout_len, self.num_workers)
        self.values = torch.zeros(
            self.rollout_len + 1, self.num_workers)   # next_value
        self.returns = torch.zeros(
            self.rollout_len + 1, self.num_workers)  # next_return
        self.actions = torch.zeros(self.rollout_len, self.num_workers)
        # masks is shifted downward one step
        self.masks = torch.zeros(self.rollout_len + 1, self.num_workers)

        # used for rnn
        self.rnn_states = torch.zeros(self.rollout_len + 1, self.num_workers, self.rnn_size)

        # local variable
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.rnn_states = self.rnn_states.to(device)

    def insert(self, obs, rewards, masks, actions, values, rnn_states=None):
        self.obs[self.step + 1].copy_(obs)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.actions[self.step].copy_(actions)
        self.values[self.step].copy_(values)
        if rnn_states is not None:
            self.rnn_states[self.step + 1].copy_(rnn_states)
        # increment step index
        self.step = (self.step + 1) % self.rollout_len

    # copy after
    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.rnn_states[0].copy_(self.rnn_states[-1])


    def compute_returns(self, next_value):
        if self.use_gae:
            self.values[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                delta = self.rewards[step] + self.gamma * self.values[step +
                                                                      1] * self.masks[step + 1] - self.values[step]
                gae = delta + self.gamma * self.tau * \
                    self.masks[step + 1] * gae
                self.returns[step] = gae + self.values[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.returns[step + 1] * \
                    self.gamma * self.masks[step + 1] + self.rewards[step]
