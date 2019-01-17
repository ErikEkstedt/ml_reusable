import torch
import json 
import csv 
from datetime import datetime
from os import makedirs
from os.path import join


def read_csv(path):
    data = []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
    return data


def read_json(path):
    with open(path, 'r') as f:
        dialogue = json.loads(f.read())


def read_json(path):
    with open(path, 'r') as jsonfile:
        dialogue = json.loads(jsonfile.read())
    return dialogue


def write_json(dialogue, filename):
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        dialogue = json.dump(dialogue, jsonfile, ensure_ascii=False)
    return dialogue


def read_csv(path):
    data = []
    with open(path, 'r') as f:
        csvReader = csv.reader(f)
        for row in csvReader:
            data.append(row)
    return data 


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


#------------ CNN Output shapes ---------------

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


#------------ RNN Output shapes ---------------
def rnn_output_shape(rnn, seq_len):
    rnn_out = rnn.hidden_size * seq_len 
    if rnn.bidirectional:
        rnn_out *= 2
    return rnn_out


#------------ Checkpoint save/load ------------
def save_checkpoint(model, optimizer, epoch, loss, path=None):
    if not path:
        d = datetime.now()
        dirpath = 'checkpoints'
        makedirs(dirpath, exist_ok=True)
        path = 'checkpoint_'
        path += '-'.join([str(d.date()), str(d.hour), str(d.minute), str(d.second)])
        path += '.pt'
        path = join(dirpath, path)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,}, path)
    return path


def load_checkpoint(model, optimizer, path=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
