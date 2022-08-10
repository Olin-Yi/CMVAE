import torch.nn as nn
from classifier import Latent_Classifier
import os
import torch
import torch.optim as optim
from utils import set_random_seed, unpack_data, accuracy, Logger
from classifier import MNIST_classifier, SVHN_classifier
from CMVAE_all import CMVAE
from dataloader_ms import get_MNIST_SVHN
import sys

model_path = '/3.7TB/xal/CMVAE/without_all'
print(model_path)
args = torch.load(model_path + '/args.rar', map_location='cpu')
args.gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda:0')

# set random seed
set_random_seed(args.seed)

# sys.stdout = Logger('{}/run.log'.format('/3.7TB/xal/classify_result'))

# load CMVAE model
model = CMVAE(args).cuda()
model.load_state_dict(torch.load(model_path + '/save_models/2.pth.tar', map_location='cpu'), strict=False)

# load the pre-trained MNIST and SVHN classifier
pretrained_classifer_path = '/3.7TB/xal/pretrained_classifer_model/{}/50.pth.tar'
mnist_net = MNIST_classifier().cuda()
svhn_net = SVHN_classifier().cuda()
mnist_net.load_state_dict(torch.load(pretrained_classifer_path.format('MNIST_classifer'), map_location='cpu'))
svhn_net.load_state_dict(torch.load(pretrained_classifer_path.format('SVHN_classifer'), map_location='cpu'))

trainloader, testloader = get_MNIST_SVHN(args.batch_size, times=args.times)


def result_classify():
    model.eval()
    mnist_net.eval()
    svhn_net.eval()

    total = 0
    corr = 0

    corr_1, corr_2, corr_3 = 0, 0, 0
    acc_1, acc_2, acc_3, acc_4 = 0, 0, 0, 0

    with torch.no_grad():
        pzs = model.pz(*model.pz_params).sample([10000])
        mnist = model.dec_list[0](pzs)
        svhn = model.dec_list[1](pzs)

        mnist_mnist = mnist_net(mnist[0].squeeze(1))
        svhn_svhn = svhn_net(svhn[0].squeeze(1))

        _, pred_m = torch.max(mnist_mnist.data, 1)
        _, pred_s = torch.max(svhn_svhn.data, 1)
        total += pred_m.size(0)
        corr += (pred_m == pred_s).sum().item()

        for j, dataT in enumerate(testloader):
            data = unpack_data(dataT, device=device)
            label = dataT[0][1].to(device)
            _, px_z_cross, _ = model(data, K=1)

            mnist_mnist = mnist_net(px_z_cross[0][0].mean.squeeze(0))
            svhn_mnist = mnist_net(px_z_cross[1][0].mean.squeeze(0))
            mnist_svhn = svhn_net(px_z_cross[0][1].mean.squeeze(0))
            svhn_svhn = svhn_net(px_z_cross[1][1].mean.squeeze(0))

            # reconstruct accuracy
            acc1 = accuracy(mnist_mnist, label)
            acc2 = accuracy(svhn_mnist, label)
            acc3 = accuracy(mnist_svhn, label)
            acc4 = accuracy(svhn_svhn, label)

            # consistency
            _, pred_mm = torch.max(mnist_mnist.data, 1)
            _, pred_sm = torch.max(svhn_mnist.data, 1)
            _, pred_ss = torch.max(svhn_svhn.data, 1)
            _, pred_ms = torch.max(mnist_svhn.data, 1)

            corr1 = (pred_mm == pred_ss).type(torch.float32).mean().item()
            corr2 = (pred_sm == pred_mm).type(torch.float32).mean().item()
            corr3 = (pred_ss == pred_ms).type(torch.float32).mean().item()

            acc_1 += acc1
            acc_2 += acc2
            acc_3 += acc3
            acc_4 += acc4

            corr_1 += corr1
            corr_2 += corr2
            corr_3 += corr3

    print('Reconstruct accuracy: \n\tMNIST -> MNIST {:.2f}% \n\tSVHN -> MNIST {:.2f}% \n\t'
          'MNIST -> SVHN {:.2f}% \n\tSVHN -> SVHN {:.2f}%'.format(acc_1 / (j + 1) * 100,
                                                                  acc_2 / (j + 1) * 100,
                                                                  acc_3 / (j + 1) * 100,
                                                                  acc_4 / (j + 1) * 100))
    print('Consistency: \n\tjoint org {:.2f}% \n\tjoint {:.2f}% \n\tcross(S->M) {:.2f}% \n\tcross(M->S) {:.2f}%'.format(
        corr / total * 100,
        corr_1 / (j + 1) * 100,
        corr_2 / (j + 1) * 100,
        corr_3 / (j + 1) * 100))


def latent_classify(epochs):
    model.eval()

    # load the latent variables classifier
    classifier = Latent_Classifier(args.latent_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    for epoch in range(epochs):
        running_loss = 0.0
        total_iters = len(trainloader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, dataT in enumerate(trainloader):
            data = unpack_data(dataT, device=device)
            label = dataT[0][1].to(device)
            qz_xs, px_z_cross, zss = model(data, K=1)
            zs_fuse = model.get_zs()

            optimizer.zero_grad()
            outputs = classifier(zs_fuse.squeeze())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 1000 == 0:
                print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training, calculating test loss...')

    classifier.eval()
    correct = 0
    with torch.no_grad():
        for j, dataT in enumerate(testloader):
            data = unpack_data(dataT, device=device)
            label = dataT[0][1].to(device)
            qz_xs, px_z_cross, zss = model(data, K=1)
            zs_fuse = model.get_zs()

            outputs = classifier(zs_fuse.squeeze())
            acc = accuracy(outputs, label)
            correct += acc

        print('The classify accuracy: {:.2f}%'.format(correct / (j + 1) * 100))


if __name__ == '__main__':
    result_classify()

    latent_classify(10)
