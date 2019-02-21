from os import makedirs

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(dpath='DATA/mnist_data', batch_size=128, **kwargs):
    '''
    Arguments:
        batch_size:     Int
        data_dir:       Pathwhere to save data

    Returns:
        train_loader:      Dataloader training
        test_loader:       Dataloader testing
    '''
    makedirs(dpath, exist_ok=True)

    train_loader = DataLoader(
            datasets.MNIST(dpath, train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
            datasets.MNIST(dpath, train=False, transform=transforms.Compose([
                transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader
