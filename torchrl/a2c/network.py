import torch
import torch.nn as nn
import torch.nn.functional as F

from ..network import NatureCNN

import pdb


class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        self.e_coef = params["entropy_coef"]
        self.v_coef = params["value_coef"]
        self.n_actions = params["num_actions"]

        # cnn head
        self.cnn_head = NatureCNN(params)

        # policy function
        self.pf = nn.Linear(512, self.n_actions)
        self.distf = torch.distributions.Categorical

        # value function
        self.vf = nn.Linear(512, 1)

        self.mse = nn.MSELoss()

    def forward(self, x):
        x = self.cnn_head(x)
        p = self.pf(x)
        v = self.vf(x)

        pd = self.distf(logits=p)
        action = pd.sample()
        log_prob = pd.log_prob(action)

        return action, log_prob, pd.entropy(), v

    def value_func(self, x):
        x = self.cnn_head(x)
        v = self.vf(x)
        return v

    def loss_func(self, rollout):
        log_probs, values, returns, advantages, entropys = map(
            lambda x: torch.cat(x, dim=0), zip(*rollout))

        policy_loss = (-advantages * log_probs)
        value_loss = 0.5 * (returns - values).pow(2)
        entropy_loss = entropys.mean()
        loss = (policy_loss - entropy_loss * self.e_coef + value_loss * self.v_coef).mean()

        return policy_loss.mean(), value_loss.mean(), entropy_loss, loss
