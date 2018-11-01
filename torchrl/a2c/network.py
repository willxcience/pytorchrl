import torch
import torch.nn as nn
import torch.nn.functional as F

from ..network import NatureCNN
import pdb


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0)


class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        self.e_coef = params["entropy_coef"]
        self.v_coef = params["value_coef"]
        self.n_actions = params["action_shape"]

        # cnn head
        self.cnn_head = NatureCNN(params)

        # policy function
        self.pf = nn.Linear(512, self.n_actions)
        self.distf = torch.distributions.Categorical

        # value function
        self.vf = nn.Linear(512, 1)
        self.mse = nn.MSELoss()

        # self.apply(weights_init)

    def forward(self, x):
        raise NotImplementedError
        # x = self.cnn_head(x)
        # p = self.pf(x)
        # v = self.vf(x)

        # pd = self.distf(logits=p)
        # action = pd.sample()
        # log_prob = pd.log_prob(action)

        # return v, action, log_prob, pd.entropy()

    def act(self, obs):
        """
        Get action based on the observations
        """
        x = self.cnn_head(obs / 255.0)
        v = self.vf(x)
        pd = self.distf(logits=self.pf(x))
        return pd.sample(), v

    def eval_action(self, obs, action):
        """
        Get value and log probalities based on the observations and actions
        Return values, log_probs, entropys
        """
        x = self.cnn_head(obs / 255.0)
        pd = self.distf(logits=self.pf(x))
        v = self.vf(x)
        return pd.log_prob(action), v, pd.entropy().mean()

    def get_value(self, obs):
        """Return from value network"""
        x = self.cnn_head(obs / 255.0)
        v = self.vf(x)
        return v

    # def loss_func(self, rollouts):

    #     advantages = rollouts.returns[:-
    #                                   1].view(-1) - rollouts.values[:-1].view(-1)

    #     value_loss = advantages.pow(2).mean()
    #     policy_loss = -(advantages.detach() *
    #                     rollouts.log_probs.view(-1)).mean()
    #     entropy_loss = rollouts.entropys.mean()

    #     loss = (policy_loss - entropy_loss * self.e_coef +
    #             value_loss * self.v_coef).mean()

    #     return loss
