import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import set_random_seed, Logger
import os
from classifier import SVHN_classifier


tx = transforms.ToTensor()
device = torch.device('cuda:1')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
batch_size = 64
seed = 10

set_random_seed(seed)

train = DataLoader(datasets.SVHN('./data', split='train', download=False, transform=tx),
                       batch_size=batch_size, shuffle=True, **kwargs)
test = DataLoader(datasets.SVHN('./data', split='test', download=False, transform=tx),
                      batch_size=batch_size, shuffle=True, **kwargs)


classifier = SVHN_classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)


def accuracy(predict, label):
    _, pred_y = torch.max(predict, 1)
    acc = (pred_y == label).type(torch.float32).mean().item()
    return acc


def trian_epoch(epoch):
    classifier.train()
    print("\nTrain Epoch: {}".format(epoch))
    for batch_idx, (data, target) in enumerate(train):
        classifier.train()
        data, target = data.to(device), target.to(device)
        output = classifier(data)
        loss = criterion(output, target)
        train_right = accuracy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % batch_size == 0:
            print("\t iteration[{}/{} ({:.0f}%) \t Loss: {:.6f} \t Train Accuracy: {:.2f}%".format(
                batch_idx * 64,
                len(train.dataset),
                100. * batch_idx / len(train),
                loss.item(),
                100. * train_right))


def test_epoch(epoch):
    classifier.eval()

    print("\nTest Epoch: {}".format(epoch))
    for (data, target) in test:
        data, target = data.to(device), target.to(device)
        output = classifier(data)
        loss = criterion(output, target)
        test_right = accuracy(output, target)

    print("\t Loss: {:.6f} \t Test Accuracy: {:.2f}%".format(loss.item(), 100. * test_right))


if __name__ == '__main__':
    modelsave_path = '/3.7TB/xal/pretrained_classifer_model/' + 'test'
    if not os.path.isdir(modelsave_path):
        os.mkdir(modelsave_path)

    sys.stdout = Logger('{}/run.log'.format(modelsave_path))

    for i in range(50):
        trian_epoch(i+1)
        test_epoch(i+1)

        torch.save(classifier.state_dict(), '{}/{}.pth.tar'.format(modelsave_path, i + 1))



