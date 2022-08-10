import torch.nn as nn


class MNIST_classifier(nn.Module):
    def __init__(self):
        super(MNIST_classifier, self).__init__()
        # input 1x28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2), # output 10x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output 10x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=2), # output 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2), # output (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class SVHN_classifier(nn.Module):
    def __init__(self):
        super(SVHN_classifier, self).__init__()
        # input 3x32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=2), # output 10x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output 10x16x16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=2), # output 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(2), # output (32, 8, 8)
        )
        self.out = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Latent_Classifier(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, in_n, out_n):
        super(Latent_Classifier, self).__init__()
        self.mlp = nn.Linear(in_n, out_n)

    def forward(self, x):
        return self.mlp(x)