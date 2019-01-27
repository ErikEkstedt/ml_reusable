import torch
import torch.nn as nn
import torch.nn.functional as F

''' https://arxiv.org/pdf/1512.03385.pdf '''


class ResidualBlockMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden, batch_norm=False):
        super(ResidualBlockMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden = hidden
        self.batch_norm = batch_norm

        self.fc1 = nn.Linear(input_size, hidden) 
        self.fc2 = nn.Linear(hidden, output_size) 

        if batch_norm:
            self.bn = nn.BatchNorm1d(output_size)
        
        self.activation = F.relu

    def forward(self, x):
        z = self.fc1(x)
        z = self.activation(z)
        z = self.fc2(x)
        z = self.activation(z)

        if self.batch_norm:
            z = self.bn(z)

        return self.activation(z + x)
