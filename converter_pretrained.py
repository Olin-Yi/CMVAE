import torch
import torch.nn as nn
from converter import Converter
import sys
from vae_ms import *
import os
import argparse
import torch.distributions as dist
from classifier import MNIST_classifier, SVHN_classifier
import torch.optim as optim
from dataloader_ms import get_MNIST_SVHN
from utils import Timer, unpack_data, Logger, set_random_seed, log_mean_exp, is_multidata, accuracy
import json


parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', type=str, default='/3.7TB/xal/pretrained_classifer_model/converter/test5-20-400-bs_128-iwae',
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
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training [default: 100]')
parser.add_argument('--epoch', type=int, default=10, metavar='N',
                    help='number of epochs to train [default: 500]')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate [default: 1e-3]')
parser.add_argument('--times', type=int, default=30,
                    help='times of extended datasets')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU id to use.')
parser.add_argument('--llik-scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
args = parser.parse_args()

# set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda:0')

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

# set random seed
set_random_seed(args.seed)

sys.stdout = Logger('{}/run.log'.format(args.save_dir))

# save args to run
with open('{}/args.json'.format(args.save_dir), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(args.save_dir))

# load the pre-trained MNIST and SVHN classifier
pretrained_classifer_path = '/3.7TB/xal/pretrained_classifer_model/{}/50.pth.tar'
mnist_net = MNIST_classifier().cuda()
svhn_net = SVHN_classifier().cuda()
mnist_net.load_state_dict(torch.load(pretrained_classifer_path.format('MNIST_classifer'), map_location='cpu'))
svhn_net.load_state_dict(torch.load(pretrained_classifer_path.format('SVHN_classifer'), map_location='cpu'))


class Train_converter(nn.Module):
    def __init__(self, params,
                 prior_dist=dist.Normal,
                 likelihood_dist=dist.Normal,
                 post_dist=dist.Normal):
        super(Train_converter, self).__init__()
        self.params = params
        self.pz = prior_dist
        self.px_z = [likelihood_dist for _ in range(params.modality_num)]
        self.qz_x = [post_dist for _ in range(params.modality_num)]
        self.encoder = nn.ModuleList([mnist_Enc(params), svhn_Enc(params)])
        self.decoder = nn.ModuleList([mnist_Dec(params), svhn_Dec(params)])
        self._pz_param = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=True)  # logvar
        ])
        self.converter = Converter(params)
        llik_mnist = prod(svhn_dataSize) / prod(mnist_dataSize) if params.llik_scaling == 0 else 1
        self.llik_scaling = [llik_mnist, 1]

    @property
    def pz_params(self):
        return self._pz_param[0], F.softmax(self._pz_param[1], dim=1) * self._pz_param[1].size(-1)

    def forward(self, x, K=1):
        mu, var = [], []
        px_z_cross, qz_xs = [], []
        zss = [[None for _ in range(self.params.modality_num)] for _ in range(self.params.modality_num)]

        for m in range(len(self.encoder)):
            mu_, var_ = self.encoder[m](x[m])
            qz_x = self.qz_x[m](mu_, var_)
            zs = qz_x.rsample(torch.Size([K]))

            qz_xs.append(qz_x)
            zss[m][m] = zs
            mu.append(mu_)
            var.append(var_)

        # create cross-modal latent representation
        mu_trans, var_trans = self.converter(mu, var)

        # i=0, j=1, px_z_cross[0]: s->m ; i=1, j=0, px_z_cross[1]: m->s
        for i, dec in enumerate(self.decoder):
            for j, (mu_ms, var_ms) in enumerate(zip(mu_trans, var_trans)):
                if i != j:
                    qz_x_cross = self.qz_x[i](mu_ms, var_ms)
                    zs = qz_x_cross.rsample(torch.Size([K]))
                    px_z = self.px_z[i](*dec(zs))

                    zss[j][i] = zs
                    px_z_cross.append(px_z)

        return qz_xs, px_z_cross, zss


def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r][r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for zs in zss[r]]))
        lpx_z = px_zs[r].log_prob(x[r]).view(*px_zs[r].batch_shape[:2], -1).mul(model.llik_scaling[r]).sum(-1)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae_looser(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae_looser(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def train(model, dataloader, optimizer, epoch):
    model.train()

    train_loss = 0

    for i, dataT in enumerate(dataloader):
        data = unpack_data(dataT, device=device)
        # label = dataT[0][1].to(device)
        optimizer.zero_grad()

        loss = -m_iwae_looser(model, data, K=20)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if i % 100 == 0:
            print("iteration {:04d}: loss: {:.6f}".format(i, loss.item() / args.batch_size))

    print('====> Epoch: {:03d} Train loss: {:.6f}'.format(epoch, train_loss / len(trainloader.dataset)))


def test(model, dataloader, epoch):
    model.eval()
    mnist_net.eval()
    svhn_net.eval()

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    test_loss, acc_1, acc_2 = 0, 0, 0

    with torch.no_grad():
        for i, dataT in enumerate(dataloader):
            data = unpack_data(dataT, device=device)
            label = dataT[0][1].to(device)
            qz_xs, px_z_cross, zss = model(data)

            svhn_mnist = mnist_net(px_z_cross[0].mean.squeeze(0))
            mnist_svhn = svhn_net(px_z_cross[1].mean.squeeze(0))

            loss1 = criterion1(svhn_mnist, label) + criterion1(mnist_svhn, label) \
                   + criterion2(px_z_cross[0].mean.squeeze(0), data[0]) + criterion2(px_z_cross[1].mean.squeeze(0), data[1])
            loss2 = -m_iwae_looser(model, data, K=20)
            test_loss += loss2.item()

            acc1 = accuracy(svhn_mnist, label)
            acc2 = accuracy(mnist_svhn, label)

            acc_1 += acc1
            acc_2 += acc2

        print('====> Epoch: {:03d} Test loss: {:.6f}'.format(epoch, test_loss / len(testloader.dataset)))
        print('====> \t Classify loss: {:.6f}'.format(loss1))
        print('Test Accuracy: \n\tSVHN->MNIST: {:.2f}% \n\tMNIST->SVHN: {:.2f}%'.format(acc_1 / (i+1) * 100,
                                                                                        acc_2 / (i+1) * 100))


if __name__ == '__main__':
    model = Train_converter(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainloader, testloader = get_MNIST_SVHN(args.batch_size, times=args.times)
    with Timer('CMVAE') as t:
        for epoch in range(args.epoch):
            train(model, trainloader, optimizer, epoch+1)
            test(model, testloader, epoch+1)

            modelsave_path = args.save_dir + '/save_models'
            if not os.path.isdir(modelsave_path):
                os.mkdir(modelsave_path)

            torch.save(model.state_dict(), '{}/{}.pkl'.format(modelsave_path, epoch+1))
