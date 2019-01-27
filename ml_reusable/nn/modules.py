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


class CNNEncoder(nn.Module):
    '''
    Based on the reference encoder in global style tokens but without the RNN

    Layers with:
        CNN -> BatchNorm2d -> ReLU

    used to encode spectrogram "images" to a smaller representation used for
    attention '''

    def __init__(self, 
            input_shape=(1, 60, 128),
            hidden=[64, 64, 128, 128],
            kernel=(3,3),
            stride=[2,2,2,1]):
        super().__init__()
        self.input_shape = input_shape
        self.padding = (kernel[0] // 2, kernel[1] // 2)
        self.kernel = kernel
        self.stride = stride
        self.hidden = hidden
        self.out_channels = hidden[-1]

        filters = [1] + hidden  # input channels=1 for spectrogram
        convs = [conv_block(
            in_channels=filters[i],
            out_channels=filters[i + 1],
            kernel_size=kernel,
            stride=stride[i],
            padding=self.padding) for i in range(len(hidden))]
        self.convs = nn.Sequential(*convs)

        # Shapes of through convolutions
        self.shapes = self._shape()

    def _shape(self):
        ''' Returns a list of shapes based on `h_w` (C, H, W) input.'''
        shapes = []
        h_w = self.input_shape
        shapes.append(h_w)
        for l in self.convs.children():
            for v in l.children():
                if isinstance(v, nn.Conv2d):
                    h_w = conv_output_shape(
                            h_w[1:],
                            out_channels=v.out_channels, 
                            kernel_size=v.kernel_size,
                            stride=v.stride,
                            padding=v.padding)
                    shapes.append(h_w)
        return shapes

    def print_conv_shapes(self):
        for i, s in enumerate(self.shapes):
            if i == 0:
                print(f"Input: {s}")
            else:
                print(f"Conv ({i}) ==> {s}")

    def total_out_features(self):
        return reduce( (lambda x, y: x * y), self.shapes[-1])

    def forward(self, x):
        return self.convs(x)
