import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    '''
    Reference Encoder for the paper:

        * Style Tokens: Unsupervised Style Modeling,
          Control and Transfer in End-to-End Speech Synthesis.
          https://arxiv.org/pdf/1803.09017.pdf

        * First used in: 
          Towards End-to-End Prosody Transfer for Expressive 
          Speech Synthesis with Tacotron
          https://arxiv.org/pdf/1803.09047.pdf

    "For the reference encoder architecture (Figure 2), we use a
    simple 6-layer convolutional network.  Each layer is com-
    posed of 3 × 3 filters with 2 × 2 stride, SAME padding and
    ReLU activation.  Batch normalization (Ioffe & Szegedy,
    2015) is applied to every layer. The number of filters in each
    layer doubles at half the rate of downsampling: 32, 32, 64,
    64, 128, 128."

    The implementation details regarding padding='SAME' differs in keras for
    different backends when stride!=1:
        https://github.com/keras-team/keras/pull/9473

    !!!
    stride=2 in all 6 layers makes the representations too small and stride=1
    is, by default, used in the 3 last layers.
    !!!

    Code modified but based on:
        https://github.com/KinglittleQ/GST-Tacotron

    '''

    def __init__(self, hidden=[32, 32, 64, 64, 128, 128],
            kernel=(3,3), stride=[2,2,2,1,1,1]):
        super().__init__()
        self.padding = (kernel[0] // 2, kernel[1] // 2)
        self.kernel = kernel
        self.stride = stride
        self.hidden = hidden

        filters = [1] + hidden  # input channels=1 for spectrogram
        convs = [conv_block(
            in_channels=filters[i],
            out_channels=filters[i + 1],
            kernel_size=kernel,
            stride=stride[i],
            padding=self.padding) for i in range(len(hidden))]
        self.convs = nn.Sequential(*convs)

        # TODO
        # self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
        #                   hidden_size=hp.E // 2,
        #                   batch_first=True)

    def _shape(self, h_w=(1,70,128)):
        ''' Returns a list of shapes based on `h_w` (C, H, W) input.'''
        shapes = []
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

    def total_out_features(self, h_w=(1, 30, 128)):
        return reduce( (lambda x, y: x * y), self._shape(h_w)[-1])

    def forward(self, x):
        return self.convs(x)


if __name__ == "__main__":

    refenc = ReferenceEncoder()
    [print(s) for s in refenc._shape()]

    N = 5
    x = torch.ones((N, 1, 70, 128))
    out = refenc(x)
    out.shape

