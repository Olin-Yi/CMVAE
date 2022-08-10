import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image
from utils import resize_img
from numpy import sqrt
from match_mnist_svhn_idx import pair_mnist_svhn


def get_MNIST(batch_size, shuffle=True, device="cuda"):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
    tx = transforms.ToTensor()
    train = DataLoader(datasets.MNIST('/home/xal/CMVAE/data', train=True, download=True, transform=tx),
                       batch_size=batch_size, shuffle=shuffle, **kwargs)
    test = DataLoader(datasets.MNIST('/home/xal/CMVAE/data', train=False, download=True, transform=tx),
                      batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train, test


def get_SVHN(batch_size, shuffle=True, device='cuda'):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    tx = transforms.ToTensor()
    train = DataLoader(datasets.SVHN('/home/xal/CMVAE/data', split='train', download=True, transform=tx),
                       batch_size=batch_size, shuffle=shuffle, **kwargs)
    test = DataLoader(datasets.SVHN('/home/xal/CMVAE/data', split='test', download=True, transform=tx),
                      batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train, test


def get_MNIST_SVHN(batch_size, shuffle=True, device='cuda', times=10):
    if not (os.path.exists('/home/xal/CMVAE/data/match_datasets/train-ms-mnist-idx-{}.pt'.format(str(times)))
            and os.path.exists('/home/xal/CMVAE/data/match_datasets/train-ms-svhn-idx-{}.pt'.format(str(times)))
            and os.path.exists('/home/xal/CMVAE/data/match_datasets/test-ms-mnist-idx-{}.pt'.format(str(times)))
            and os.path.exists('/home/xal/CMVAE/data/match_datasets/test-ms-svhn-idx-{}.pt'.format(str(times)))):
        print('Generate transformed indices with the script in match_mnist_svhn_idx')
        pair_mnist_svhn(times)

    # get transformed indices
    t_mnist = torch.load('/home/xal/CMVAE/data/match_datasets/train-ms-mnist-idx-{}.pt'.format(str(times)))
    t_svhn = torch.load('/home/xal/CMVAE/data/match_datasets/train-ms-svhn-idx-{}.pt'.format(str(times)))
    s_mnist = torch.load('/home/xal/CMVAE/data/match_datasets/test-ms-mnist-idx-{}.pt'.format(str(times)))
    s_svhn = torch.load('/home/xal/CMVAE/data/match_datasets/test-ms-svhn-idx-{}.pt'.format(str(times)))

    t1, s1 = get_MNIST(batch_size, shuffle, device)
    t2, s2 = get_SVHN(batch_size, shuffle, device)

    train_mnist_svhn = TensorDataset([
        ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
        ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
    ])
    test_mnist_svhn = TensorDataset([
        ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
        ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
    ])

    kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
    train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train, test

