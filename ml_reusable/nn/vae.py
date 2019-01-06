from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

from ml_reusable.utils import conv_output_shape, deconv_output_shape, num2onehot
from ml_reusable.datasets import load_mnist
from ml_reusable.nn.modules import conv_block, deconv_block, mlp_block


# MNIST Example. Works
def test_vae_mlp_on_mnist(args):
    train_loader, test_loader = load_mnist(dpath=args.dpath, batch_size=args.batch_size)

    n_epochs = args.epochs
    in_size = 28*28
    z_dim = 128

    if args.conditional:
        vae = CVAE_MLP(in_size, z_dim, _device=args.device)
        print('Conditional VAE')
    else:
        vae = VAE_MLP(in_size, z_dim, _device=args.device)
        print('Regular VAE')
    vae.to(vae._device)
    optimizer = optim.Adam(vae.parameters(), lr=7e-3)

    writer = SummaryWriter()

    for epoch in range(args.epochs):
        train_loss = vae.train_epoch(epoch, n_epochs, optimizer, train_loader)
        test_loss, images = vae.test_epoch(test_loader)

        s = f"CVAE Loss" if args.conditional else f"VAE Loss"
        writer.add_scalars(s, {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_image('Reconstruction', images, epoch) 


def test_vae_conv_on_mnist(args):
    train_loader, test_loader = load_mnist(dpath=args.dpath, batch_size=args.batch_size)

    n_epochs = args.epochs
    z_dim = 64

    print('Regular VAE')
    vae = VAE_CONV(z_dim, _device=args.device)
    vae.to(vae._device)
    optimizer = optim.Adam(vae.parameters(), lr=7e-3)

    writer = SummaryWriter()

    for epoch in range(args.epochs):
        train_loss = vae.train_epoch(epoch, n_epochs, optimizer, train_loader)
        test_loss, images = vae.test_epoch(test_loader)

        s = f"VAE_CONV Loss"
        writer.add_scalars(s, {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_image('Reconstruction', images, epoch) 


# ------------ Models ------------

class BaseModelMNIST(nn.Module):
    def __init__(self):
        super(BaseModelMNIST, self).__init__()
        self.conditional = False  # Overwritten by CVAE
        
    def train_epoch(self, epoch, n_epochs, optimizer, train_loader):
        self.train()
        total_loss, bce_loss, kld_loss = 0, 0, 0

        for batch_idx, (x, c) in enumerate(
                tqdm(train_loader, f'Train: {epoch}/{n_epochs}')):

            x = x.to(self._device)

            if self.conditional:
                c = c.to(self._device).float()
                xc = (x, c)
            else:
                xc = x 

            optimizer.zero_grad()

            loss, recon_batch = self(xc)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

        total_loss /= len(train_loader.dataset)
        print(f"====> Epoch: {epoch}, Average loss: {total_loss:.3f}")
        return total_loss

    def test_epoch(self, test_loader):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (x, c) in enumerate(test_loader):

                x = x.to(self._device)

                if self.conditional:
                    c = c.to(self._device).float()
                    xc = (x, c)
                else:
                    xc = x

                loss, recon_batch = self(xc)

                total_loss += loss.item()
                if i == 0:
                    n = min(x.size(0), 8)
                    bs = x.shape[0]
                    comparison = torch.cat([x[:n], recon_batch.view(bs, 1, 28, 28)[:n]])
                    images = make_grid(comparison, normalize=True, scale_each=True)

        total_loss /= len(test_loader.dataset)
        print(f"====> Test loss: {total_loss:.4f}")
        return total_loss, images

# ----------- VQ-VAE -------------
# utilize 1D latent feature spaces
# TODO
class VQVAE_MLP(BaseModelMNIST):
    def __init__(self, in_size, z_dim=128, k_dim=64, _device='cuda'):
        super(VQVAE_MLP, self).__init__()
        self.in_size = in_size
        self._device = _device
        self.z_dim = z_dim
        self.k_dim = k_dim

        self.encoder = EncoderMLP(in_size, latent_size=2*z_dim)
        dec_hidden = self.encoder.layer_sizes[::-1]
        self.decoder = DecoderMLP(hidden=dec_hidden, latent_size=z_dim)


        # Encoder MLP
        self.encoder = EncoderMLP() 

        # Embedding Book
        self.embd = nn.Embedding(self.k_dim, self.z_dim)

        # Decoder MLP
        self.decode = DecoderMLP()

    def find_nearest(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index]

    def forward(self, X):
        Z_enc = self.encode(X.view(-1,784))
        Z_dec = self.find_nearest(Z_enc,self.embd.weight)
        Z_dec.register_hook(self.hook)

        X_recon = self.decode(Z_dec).view(-1,1,28,28)
        Z_enc_for_embd = self.find_nearest(self.embd.weight,Z_enc)
        return X_recon, Z_enc, Z_dec, Z_enc_for_embd

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad


# ------------ MLP ---------------
class EncoderMLP(nn.Module):
    def __init__(self, in_size, hidden=[512, 256], latent_size=128):
        super(EncoderMLP, self).__init__()
        self.layer_sizes = [in_size, *hidden]
        self.latent_size = latent_size
        
        layers = [mlp_block(in_, out_) 
                for in_, out_ in zip(self.layer_sizes, self.layer_sizes[1:])]

        # No activation on latent space
        layers.append(nn.Linear(self.layer_sizes[-1], latent_size))
    
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        bs = x.shape[0]
        return self.layers(x.reshape(bs, -1))


class DecoderMLP(nn.Module):
    def __init__(self, hidden=[256, 512, 784], latent_size=128, out_activation='sigmoid'):
        super(DecoderMLP, self).__init__()
        self.layer_sizes = [latent_size, *hidden]
        
        layers = [mlp_block(in_, out_) 
                for in_, out_ in zip(self.layer_sizes, self.layer_sizes[1:])]

        self.layers = nn.Sequential(*layers)
        if out_activation.lower()  == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        elif out_activation.lower()  == 'relu':
            self.out_activation = nn.ReLU()

    def forward(self, z):
        bs = z.shape[0]
        out = self.layers(z.reshape(bs, -1))
        return self.out_activation(out)


class VAE_MLP(BaseModelMNIST):
    def __init__(self, in_size, z_dim=48, _device='cuda'):
        super(VAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.in_size = in_size
        self._device = _device

        self.encoder = EncoderMLP(in_size, latent_size=2*z_dim)
        dec_hidden = self.encoder.layer_sizes[::-1]
        self.decoder = DecoderMLP(hidden=dec_hidden, latent_size=z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        bs = x.shape[0]
        z_tot = self.encoder(x)
        mu, logvar = z_tot[:, :self.z_dim], z_tot[:, self.z_dim:]
        z_sampled = self.reparameterize(mu, logvar) 
        recon_x = self.decoder(z_sampled)
        loss = self.loss_function(recon_x, x.reshape(bs, -1), mu, logvar)
        return loss, recon_x

    # Reconstruction + KL divergence losses summed over all elements and batch
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


class CVAE_MLP(BaseModelMNIST):
    def __init__(self, in_size, z_dim=48, _device='cuda'):
        super(CVAE_MLP, self).__init__()
        self.conditional = True
        self.z_dim = z_dim
        self.in_size = in_size + 1  # conditional on input
        self._device = _device

        self.encoder = EncoderMLP(self.in_size, latent_size=2*z_dim)
        dec_hidden = self.encoder.layer_sizes[::-1]
        dec_hidden[-1] = dec_hidden[-1] - 1
        self.decoder = DecoderMLP(hidden=dec_hidden, latent_size=z_dim+1) # conditional on input

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, data):
        x, c = data
        bs = x.shape[0]
        x, c = x.flatten(1), c.flatten(1)
        xc = torch.cat([x, c], dim=1)  # concat image and label

        z_tot = self.encoder(xc)  # q(z|x,c)
        mu, logvar = z_tot[:, :self.z_dim], z_tot[:, self.z_dim:]
        z = self.reparameterize(mu, logvar) 

        zc = torch.cat([z, c], dim=1) # concat latent and label
        recon_x = self.decoder(zc) # p(x|z,c)
        loss = self.loss_function(recon_x, x.flatten(1), mu, logvar)
        return loss, recon_x

    # Reconstruction + KL divergence losses summed over all elements and batch
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


# ------------ Convs ---------------
class EncoderConv(nn.Module):
    ''' Convolution -> Linear -> out '''
    def __init__(self, in_channels=1, latent_size=128, hidden=[32, 32, 64, 64, 128, 128],
            kernel=(3,3), stride=[2,1,1,1,1,1]):
        super(EncoderConv, self).__init__()
        self.layer_sizes = [in_channels, *hidden]
        self.latent_size = latent_size
        
        convs = [conv_block(in_, out_, kernel_size=kernel, stride=stride[i]) 
                for i, (in_, out_) in enumerate(zip(self.layer_sizes, self.layer_sizes[1:]))]
        self.convs = nn.Sequential(*convs)

        self.conv_out_shape = self._conv_shape()[-1]
        conv_out = self.conv_out_features()

        # No activation on latent space
        self.head = nn.Linear(conv_out, latent_size)
        self.out_shape = latent_size

    def _conv_shape(self, h_w=(1,32,128)):
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

    def conv_out_features(self):
        return reduce( (lambda x, y: x * y), self._conv_shape()[-1])

    def forward(self, x):
        x = self.convs(x)
        return self.head(x.flatten(1))


class DecoderConv(nn.Module):
    ''' DeConvolution -> out '''
    def __init__(self, out_channels=1,
            latent_size=128,
            shape_after_linear=(8, 16, 64),
            hidden=[32 , 64, 128, 128],
            kernel=(3, 3),
            padding=[(1,2),(1,2),(2,2),(1,0),(0,0),(0,0)],
            dilation=[(2,8),(2,8),(4,8),(3,8),(2,6),(1,1)],
            output_padding=[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
            stride=[1,1,1,1,1,1],
            ):
        super(DecoderConv, self).__init__()
        self.dilation = dilation

        self.shape_after_linear = shape_after_linear
        self.fc_out = reduce( (lambda x, y: x * y), shape_after_linear )
        self.fc1 = nn.Linear(latent_size, self.fc_out)

        # Deconv block
        self.layer_sizes = [shape_after_linear[0], *hidden, out_channels]
        self.out_channels = out_channels
        
        deconvs = [deconv_block(in_, out_, kernel_size=kernel,
            stride=stride[i],
            padding=padding[i],
            dilation=dilation[i],
            output_padding=output_padding[i]) 
            for i, (in_, out_) in enumerate(zip(self.layer_sizes, self.layer_sizes[1:]))]
        self.layers = nn.Sequential(*deconvs)

        self.out_shape = self.deconv_out_features()
        self.out_activation = nn.Sigmoid()

    def _deconv_shape(self, h_w=None):
        ''' Returns a list of shapes based on `h_w` (C, H, W) input.'''
        if h_w is None:
            h_w = self.shape_after_linear

        shapes = []
        shapes.append(h_w)
        for l in self.layers.children():
            for v in l.children():
                if isinstance(v, nn.ConvTranspose2d):
                    h_w = deconv_output_shape(
                            h_w[1:],
                            out_channels=v.out_channels, 
                            kernel_size=v.kernel_size,
                            stride=v.stride,
                            padding=v.padding,
                            output_padding=v.output_padding)
                    shapes.append(h_w)
        return shapes

    def deconv_out_features(self, h_w=(1, 30, 128)):
        return self._deconv_shape(h_w)[-1]

    def forward(self, x):
        bs = x.shape[0]
        x = self.fc1(x)
        x = x.reshape(bs, *self.shape_after_linear)
        x = self.layers(x)
        return self.out_activation(x) 


class VAE_CONV(nn.Module):
    def __init__(self, z_dim=64, _device='cpu'):
        super(VAE_CONV, self).__init__()
        self._device = _device
        self.z_dim = z_dim

        self.encoder = EncoderConv(in_channels=1, latent_size=2*z_dim)
        self.decoder = DecoderConv(latent_size=z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        print('in')
        z_mu_logvar = self.encoder(x)  # q(z|x)
        mu, logvar = z_mu_logvar[:, :self.z_dim], z_mu_logvar[:, self.z_dim:]
        z = self.reparameterize(mu, logvar) 
        print('enc out')
        recon_x = self.decoder(z) # p(x|z)
        loss = self.loss_function(recon_x, x, mu, logvar)
        print('out')
        return loss, recon_x

    # Reconstruction + KL divergence losses summed over all elements and batch
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train_epoch(self, epoch, n_epochs, optimizer, train_loader):
        self.train()
        total_loss, bce_loss, kld_loss = 0, 0, 0

        for batch_idx, data in enumerate(
                tqdm(train_loader, f'Train: {epoch}/{n_epochs}')):

            x = data['mel']
            x = x.to(self._device)

            optimizer.zero_grad()

            loss, recon_batch = self(x)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

        total_loss /= len(train_loader.dataset)
        print(f"====> Epoch: {epoch}, Average loss: {total_loss:.3f}")
        return total_loss

    def test_epoch(self, test_loader):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (x, c) in enumerate(test_loader):

                x = x.to(self._device)
                loss, recon_batch = self(x)

                total_loss += loss.item()
                if i == 0:
                    n = min(x.size(0), 8)
                    bs = x.shape[0]
                    comparison = torch.cat([x[:n], recon_batch.view(bs, 1, 28, 28)[:n]])
                    images = make_grid(comparison, normalize=True, scale_each=True)

        total_loss /= len(test_loader.dataset)
        print(f"====> Test loss: {total_loss:.4f}")
        return total_loss, images


# TODO: VAE superclass containing loss_fn, reparameterize

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description='VAE')
    # parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    # parser.add_argument('--conditional', action='store_true', help='CVAE')
    # parser.add_argument('--dpath', type=str)
    # parser.add_argument('-b', '--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=100, help='Disable CUDA')
    # args = parser.parse_args()
    # args.device = None
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')
    # else:
    #     args.device = torch.device('cpu')
    # test_vae_mlp_on_mnist(args)

    # z_dim = 32
    # enc = EncoderConv(latent_size=z_dim*2)
    # print('Encoder conv: ', enc.conv_out_shape)
    # print('Encoder out: ', enc.out_shape)

    # dec = DecoderConv(latent_size=z_dim)
    # print('Decoder: ', dec.out_shape)

    # d = torch.ones(1,1,32,128)
    # z = enc(d)

    # print('Z: ', z.shape)

    # mu, logvar = z[:, :z_dim], z[:, z_dim:]
    # out = dec()

    model = VAE_CONV(z_dim=64)
    d = torch.ones(1,1,32,128)
    loss, reconstruction = model(d)
    print('loss: ', loss)
    print('recon: ', reconstruction.shape)
