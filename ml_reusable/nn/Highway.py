import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    '''
    Highway Network Paper: 
        https://arxiv.org/pdf/1505.00387.pdf

    "A Highway Network is a network architecture made for deeper networks where
    information has a way of propogating through several layers without being
    attenuated as much as in regular networks. The essential features of a
    highway architecture is that in addition of having a normal layer (affine
    transformation through a nonlinearity) the layer contains a *Transform* and
    a *Carry* gate." 
    '''
    def __init__(self, in_features, out_features):
        '''
        inputs: [N, T, C]
        outputs: [N, T, C]
        '''
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        H = self.linear1(inputs)
        H = F.relu(H)
        T = self.linear2(inputs)
        T = F.sigmoid(T)
        out = H * T + inputs * (1.0 - T)
        return out
