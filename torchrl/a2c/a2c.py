import torch
import torch.optim as optim
import numpy as np
from collections import deque

from ..utils import v_wrap
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
        self.curr_states = env.reset()
        self.policy_loss, self.value_loss, self.entropy_loss = 0.0, 0.0, 0.0

    def phi(self, x):
        x = v_wrap(x).permute(0, 3, 1, 2) / 255.0
        if self.use_cuda:
            return x.cuda()
        else:
            return x

    # collect rollout used for training
    def collect_rollout(self, num_rollout=20):

        rollout = []

        # predict and collect
        for i in range(num_rollout):
            # care if states are uint8

            actions, log_probs, entropys, values = self.model(
                self.phi(self.curr_states))
            
            next_states, rewards, dones, _ = self.env.step(actions.cpu().numpy())
            self.online_rewards += rewards
            for i in range(dones.shape[0]):
                if dones[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            
            if self.use_cuda:
                rewards = v_wrap(rewards).cuda()
                not_dones = v_wrap(1 - dones).cuda()

            rollout.append([actions, log_probs, entropys, values,
                            rewards, not_dones])
            self.curr_states = next_states

        # calculate discounted returns and advantages
        last_values = self.model.value_func(
            self.phi(self.curr_states)).squeeze()
        returns = last_values.detach()
        discounted_rollout = []
        advantages = torch.zeros((self.num_workers))
        if self.use_cuda:
            advantages = advantages.cuda()

        for i in reversed(range(len(rollout))):
            actions, log_probs, entropys, values, rewards, not_dones = rollout[i]
            values = values.squeeze()
            returns = rewards + self.gamma * not_dones * returns

            if not self.use_gae:
                advantages = returns - values.detach()
            else:
                next_values = last_values
                gae_returns = rewards + self.gamma * not_dones * next_values.detach()
                td_error = gae_returns - values.detach()
                advantages = advantages * self.tau * self.gamma * not_dones + td_error

            discounted_rollout.append(
                [log_probs, values, returns, advantages, entropys])

        return discounted_rollout

    # train one iter
    def train_step(self):
        # collect rollout
        rollout = self.collect_rollout(self.rollout_len)

        # loss
        p_loss, v_loss, e_loss, loss = self.model.loss_func(
            rollout)
        self.policy_loss = p_loss.item()
        self.value_loss = v_loss.item()
        self.entropy_loss = e_loss.item()

        self.optimzier.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
        self.optimzier.step()

        self.num_steps += self.rollout_len * self.num_workers
