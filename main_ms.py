import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
from numpy import prod
import torch
from torch import optim
import torch.nn as nn
import os
from torchvision.utils import save_image, make_grid
import argparse
from dataloader_ms import get_MNIST_SVHN
from CMVAE_ms import CMVAE
from utils import unpack_data, set_random_seed, Timer, Logger
from objectives import m_iwae_looser, m_dreg_looser


parser = argparse.ArgumentParser()
parser.add_argument('--mark', type=str, default='analysis-latent_20-hidden_400-m_iwae_looser20-bs_256',
                    help='name or parameters of current strategy')
parser.add_argument('--modality-num', type=int, default=2,
                    help='number of modalities [default: 2]')
parser.add_argument('--latent-dim', type=int, default=20,
                    help='size of the latent embedding [default: 64]')
parser.add_argument('--hidden-dim', type=int, default=400,
                    help='size of the hidden embedding [default: 64]')
parser.add_argument('--feature-dim', type=int, default=128,
                    help='size of the feature embedding [default: 64]')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training [default: 100]')
parser.add_argument('--epoch', type=int, default=30, metavar='N',
                    help='number of epochs to train [default: 500]')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate [default: 1e-3]')
parser.add_argument('--K', type=int, default=20, metavar='K',
                    help='number of particles to use for iwae (default: 10)')
parser.add_argument('--llik-scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--times', type=int, default=20,
                    help='times of extended datasets')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default='3', type=str,
                    help='GPU id to use.')
parser.add_argument('--pre-trained-conv', action='store_true', default=False)
args = parser.parse_args()

# set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda:0')

# set random seed
set_random_seed(args.seed)

runPath = '/3.7TB/xal/CMVAE/{}'.format(args.mark)
if not os.path.isdir(runPath):
    os.mkdir(runPath)

sys.stdout = Logger('{}/run.log'.format(runPath))

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

if args.pre_trained_conv is not True:
    model = CMVAE(args).cuda()
else:
    # load pre-trained converter
    converter_path = '/3.7TB/xal/pretrained_classifer_model/converter/test1/best_conv.pkl'
    pred_dict = torch.load(converter_path, map_location='cpu')

    # insert pre-trained converter parameters into CMVAE model
    model = CMVAE(args).cuda()
    model_dict = model.state_dict()
    Pred_dict = {k: v for k, v in pred_dict.items() if k in model_dict}
    model_dict.update(Pred_dict)
    model.load_state_dict(model_dict)

trainloader, testloader = get_MNIST_SVHN(args.batch_size, times=args.times)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
loss_list = defaultdict(list)


def train(epoch, loss_list):
    model.train()
    b_loss = 0

    for i, dataT in enumerate(trainloader):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss = -m_iwae_looser(model, data, K=args.K)
        loss.backward(retain_graph=True)
        optimizer.step()
        b_loss += loss.item()

        if i % 100 == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))

    loss_list['train_loss'].append(b_loss / len(trainloader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, loss_list['train_loss'][-1]))


def test(epoch, loss_list):
    model.eval()
    b_loss = 0

    with torch.no_grad():
        for i, dataT in enumerate(testloader):
            data = unpack_data(dataT, device=device)
            loss = -m_iwae_looser(model, data, K=args.K)
            b_loss += loss.item()

            # reconstruct modalities
            if i == 0:
                model.reconstruct(data, runPath, epoch)

    loss_list['test_loss'].append(b_loss / len(testloader.dataset))
    print('====> Epoch: {:03d} Test loss: {:.4f}'.format(epoch, loss_list['test_loss'][-1]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with Timer('CMVAE') as t:
        for epoch in range(args.epoch):
            train(epoch + 1, loss_list)
            test(epoch + 1, loss_list)

            modelsave_path = runPath + '/save_models'
            if not os.path.isdir(modelsave_path):
                os.mkdir(modelsave_path)

            torch.save(model.state_dict(), '{}/{}.pth.tar'.format(modelsave_path, epoch + 1))

        with open('{}/Loss.json'.format(runPath), 'w') as f_loss:
            json.dump(loss_list, f_loss)
