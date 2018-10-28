import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(7*7*64, 512)
        
    def forward(self, x):
        x = self.cnn_head(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x