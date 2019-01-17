import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from ml_reusable.utils import conv_output_shape


def mlp_block(in_f, out_f, batchnorm=True, *args, **kwargs):
    if batchnorm:
        return nn.Sequential(
                nn.Linear(in_f, out_f, *args, **kwargs),
                nn.BatchNorm1d(out_f),
                nn.ReLU())
    else:
        return nn.Sequential(
                nn.Linear(in_f, out_f, *args, **kwargs),
                nn.ReLU())


def conv_block(in_channels, out_channels, batchnorm=True, *args, **kwargs):
    if batchnorm:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
    else:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                nn.ReLU())


def deconv_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())


class FrameWiseDimReduction(nn.Module):
    '''
    The goal was to implement a model which processes the frequency bins in the
    same way for each frame in a spectrogram.

    (F, freq_bins) -> ()
    '''
    def __init__(self, freq_bins=128, out_size=20):
        super(FrameWiseDimReduction, self).__init__()
        self.freq_bins = freq_bins
        self.out_size = out_size
        self.fc = nn.Linear(freq_bins, out_size)
        self.out_activation = nn.ReLU()

    def forward(self, x):
        '''
        used on spectrogram: (N, 1, F, freq_bins), where N is batch size and F
        frames
        '''
        x = x.squeeze(1)  # (N, 1, F, freq_bins) -> (N, F, freq_bins)
        N = x.shape[0]  # batch
        F = x.shape[1]  # frames
        x = x.reshape(-1, self.freq_bins)

        z = self.fc(x)
        out = z.reshape(N, 1, F, self.out_size)
        return self.out_activation(out)
