import torch


def total_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num2onehot(label, classes=10):
    '''
    Arguments:
        label: torch.tensor (N,) or (N, 1)
        classes: int, length of onehot vector, C
    Returns:
        torch.tensor (N, C)
    '''
    onehot = torch.zeros(label.shape[0], classes)
    for i, idx in enumerate(label):
        onehot[i, idx] = 1
    return onehot


def conv_output_shape(h_w, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1):
    """
    https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5

    Utility function for computing output of convolutions
    takes a tuple `h_w`: (h,w) and returns a tuple of (h,w)
    """
    if not isinstance(h_w, tuple):
        h_w = (h_w, h_w)
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)

    h = int((h_w[0] + (2 * padding[0]) - \
             ( dilation[0] * (kernel_size[0] - 1)) - 1 )/ stride[0] + 1)
    w = int((h_w[1] + (2 * padding[1]) - \
             ( dilation[1] * (kernel_size[1] - 1)) - 1 )/ stride[0] + 1)
    return out_channels, h, w


def deconv_output_shape(h_w, out_channels=1, kernel_size=3, stride=1,
        padding=0, output_padding=0):
    """ https://pytorch.org/docs/stable/nn.html#convtranspose2d """
    if not isinstance(h_w, tuple):
        h_w = (h_w, h_w)
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(output_padding, tuple):
        output_padding = (output_padding, output_padding)

    h = int((h_w[0]-1) * stride[0] - (2 * padding[0]) + \
                kernel_size[0] + output_padding[0] )
    w = int((h_w[1]-1) * stride[1] - (2 * padding[1]) + \
                kernel_size[1] + output_padding[1] )
    return out_channels, h, w
