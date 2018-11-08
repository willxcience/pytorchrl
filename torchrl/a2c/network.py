import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..network import NatureCNN
import pdb
import torchvision


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


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# Extremely important
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        def init_(m): return init(m,
                                  nn.init.orthogonal_,
                                  lambda x: nn.init.constant_(x, 0),
                                  gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


# class RNN(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(RNN, self).__init__()
#         self.rnn = nn.GRU(num_inputs, num_outputs)

#     def forward(self, x, masks, hxs, train=False):
#         if not train:
#             x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks.unsqueeze(1)).unsqueeze(0))
#             return x.squeeze(0), hxs.squeeze(0)
#         else:
#             # number of workers
#             N = hxs.size(0)
#             # rollout length
#             T = int(x.size(0) / N)

#             # (seq_len, batch, shape)
#             x = x.view(T, N, x.size(1))
#             masks = masks.view(T, N)

#             # fast computation for sequences
#             has_zeros = ((masks[1:] == 0.0)
#                          .any(dim=-1)
#                          .nonzero()
#                          .squeeze()
#                          .cpu())

#             # scalar
#             if has_zeros.dim() == 0:
#                 has_zeros = [has_zeros.item() + 1]
#             else:
#                 has_zeros = (has_zeros + 1).numpy().tolist()
#              # add t=0 and t=T to the list
#             has_zeros = [0] + has_zeros + [T]
#             hxs = hxs.unsqueeze(0)

#             outputs = []

#             for i in range(len(has_zeros) - 1):
#                 # We can now process steps that don't have any zeros in masks together!
#                 # This is much faster
#                 start_idx = has_zeros[i]
#                 end_idx = has_zeros[i + 1]

#                 rnn_scores, hxs = self.rnn(
#                     x[start_idx:end_idx],
#                     hxs * masks[start_idx].view(1, -1, 1)
#                 )
#                 outputs.append(rnn_scores)

#             x = torch.cat(outputs, dim=0)
#             x = x.view(T * N, -1)
#             hxs = hxs.unsqueeze(0)
#             return x, hxs



class RNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(RNN, self).__init__()
        
        self.rnn = nn.GRUCell(num_inputs, num_outputs)
        nn.init.orthogonal_(self.rnn.weight_ih.data)
        nn.init.orthogonal_(self.rnn.weight_hh.data)
        self.rnn.bias_ih.data.fill_(0)
        self.rnn.bias_hh.data.fill_(0)

    def forward(self, x, masks, hxs, train=False):
        if x.size(0) == hxs.size(0):
            x = hxs = self.rnn(x, (hxs * masks.unsqueeze(1)))
            return x.squeeze(0), hxs.squeeze(0)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # time sequence
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.rnn(x[i], hxs * masks[i])
                outputs.append(hx)

            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs

class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        self.e_coef = params["entropy_coef"]
        self.v_coef = params["value_coef"]
        self.n_actions = params["action_shape"]
        self.rnn_size = params["rnn_size"]

        # cnn head
        # self.cnn_head = torchvision.models.resnet34(pretrained=True)
        #self.cnn_head = NatureCNN(params)
        self.cnn_head = torchvision.models.resnet34(pretrained=True)
        self.cnn_head = nn.Sequential(*list(self.cnn_head.children())[:-1])
        self.rnn = RNN(512, 512)

        # policy function
        #self.pf = nn.Linear(512, self.n_actions)
        self.distf = Categorical(512, self.n_actions)

        # value function
        self.vf = nn.Linear(512, 1)

        # self.apply(weights_init)

    def forward(self, x):
        raise NotImplementedError

    def act(self, obs, masks=None, rnn_hxs=None):
        """
        Get action based on the observations
        """
        obs = F.interpolate(obs, (224, 224))
        x = self.cnn_head(obs / 255.0).squeeze()
        x, rnn_hxs = self.rnn(x, masks, rnn_hxs)
        x = x.squeeze()
        v = self.vf(x)
        pd = self.distf(x)
        return pd.sample(), v, rnn_hxs

    def eval_action(self, obs, action, masks=None, rnn_hxs=None):
        """
        Get value and log probalities based on the observations and actions
        Return values, log_probs, entropys
        """
        obs = F.interpolate(obs, (224, 224))
        x = self.cnn_head(obs / 255.0).squeeze()
        x, rnn_hxs = self.rnn(x, masks, rnn_hxs, True)

        pd = self.distf(x)
        v = self.vf(x)
        return pd.log_prob(action), v, pd.entropy().mean()

    def get_value(self, obs, masks=None, rnn_hxs=None):
        """Return from value network"""
        obs = F.interpolate(obs, (224, 224))
        x = self.cnn_head(obs / 255.0).squeeze()
        x, rnn_hxs = self.rnn(x, masks, rnn_hxs)

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
