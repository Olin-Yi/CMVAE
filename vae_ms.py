import torch.nn as nn
import torch
from numpy import prod
import torch.nn.functional as F


# Constants
mnist_dataSize = torch.Size([1, 28, 28])
data_dim = int(prod(mnist_dataSize))

eta = 1e-6
svhn_dataSize = torch.Size([3, 32, 32])
imgChans = svhn_dataSize[0]
fBase = 32  # base size of filter channels


def extra_hidden_layer(hidden_dim):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


class mnist_Enc(nn.Module):
    """ Generate latent parameters for MNIST image data. """

    def __init__(self, params):
        super(mnist_Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, params.hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(params.hidden_dim) for _ in range(params.num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(params.hidden_dim, params.latent_dim)
        self.fc22 = nn.Linear(params.hidden_dim, params.latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data

        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + eta


class mnist_Dec(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """

    def __init__(self, params):
        super(mnist_Dec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(params.latent_dim, params.hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(params.hidden_dim) for _ in range(params.num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(params.hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))

        # return p
        d = torch.sigmoid(p.view(*z.size()[:-1], *mnist_dataSize))  # reshape data
        d = d.clamp(eta, 1 - eta)
        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class svhn_Enc(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, params):
        super(svhn_Enc, self).__init__()
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, params.latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 4, params.latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)

        lv = self.c2(e).squeeze()
        return self.c1(e).squeeze(), F.softmax(lv, dim=-1) * lv.size(-1) + eta


class svhn_Dec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, params):
        super(svhn_Dec, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(params.latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))

        # return out
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(z.device)  # mean, length scale