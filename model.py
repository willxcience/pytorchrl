import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class NatureCNN(nn.Module):
    def __init__(self, params):
        super(NatureCNN, self).__init__()
        n_c = params['num_in_channels']
        self.cnn_head = nn.Sequential(
            nn.Conv2d(n_c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True)
        )
        self.fc = nn.Linear(3136, 512)
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.cnn_head(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        return x
        
class ActorCritic(nn.Module):
    def __init__(self, cnn_head, params):
        super(ActorCritic, self).__init__()
        self.e_coef = params["entropy_coef"]
        self.v_coef = params["value_coef"]
        self.max_grad_norm = params["max_norm_grad"]

        self.p = 
        self.pdist = torch.distributions.Categorical()

    def forward(self, x):
        pass

    def step(self, obs):


    def loss_func(self):
        entropy = self.pdist.entropy.mean()

        advs = rewards - values
        # policy gradient loss
        pg_loss = advs * neglogpac
        # value function loss
        vf_loss = nn.MSELoss()(value, rewards)
    
        loss = pg_loss - entropy * self.e_coef + vf_loss * self.v_coef

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters, self.max_grad_norm)
        optimz.step()

