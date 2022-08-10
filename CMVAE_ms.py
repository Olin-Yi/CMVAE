import torch.distributions as dist
from numpy import sqrt
from torchvision.utils import save_image
from utils import *
from converter import Converter
from vae_ms import *
from itertools import combinations

lambda_mnist = 1
lambda_svhn = 1


class CMVAE(nn.Module):
    def __init__(self, params,
                 prior_dist=dist.Normal,
                 likelihood_dist=dist.Normal,
                 post_dist=dist.Normal):
        super(CMVAE, self).__init__()
        self.params = params
        self.pz = prior_dist
        self.pz_x_fuse = post_dist
        self.px_z = [likelihood_dist for _ in range(params.modality_num)]
        self.qz_x = [post_dist for _ in range(params.modality_num)]
        self.enc_list = nn.ModuleList([mnist_Enc(params), svhn_Enc(params)])
        self.dec_list = nn.ModuleList([mnist_Dec(params), svhn_Dec(params)])
        self.converter = Converter(params)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=True)  # logvar
        ])
        self.poe = PoE()
        llik_mnist = prod(svhn_dataSize) / prod(mnist_dataSize) if params.llik_scaling == 0 else lambda_mnist
        self.llik_scaling = [llik_mnist, lambda_svhn]
        self.zs_fusion = None

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def forward(self, x, K=1):
        mu, var = [], []
        qz_xs = []
        px_z_cross = [[None for _ in range(self.params.modality_num)] for _ in range(self.params.modality_num)]
        zss = [[None for _ in range(self.params.modality_num)] for _ in range(self.params.modality_num)]
        zss_org = []

        for m in range(len(self.enc_list)):
            mu_, var_ = self.enc_list[m](x[m])
            qz_x = self.qz_x[m](mu_, var_)
            zs = qz_x.rsample(torch.Size([K]))

            qz_xs.append(qz_x)
            zss[m][m] = zs
            zss_org.append(zs)
            mu.append(mu_)
            var.append(var_)

        # create cross-modal latent representation
        mu_trans, var_trans = self.converter(mu, var)

        # fuse the latent representation from different modalites
        mu_pz, logvar_pz = prior_expert((len(mu[0]), self.params.latent_dim), use_cuda=True)
        mu_trans.append(mu_pz)
        var_trans.append(logvar_pz)
        mu_all = torch.stack(mu_trans)
        var_all = torch.stack(var_trans)
        mu_fuse, var_fuse = self.poe(mu_all, var_all)

        # sampled from the fused latent representation -> latent variables(zs_fuse)
        pz_x_fuse = self.pz_x_fuse(mu_fuse, var_fuse)
        zs_fuse = pz_x_fuse.rsample(torch.Size([K]))

        self.zs_fusion = zs_fuse

        # i=0, j=1, px_z_cross[1][0]: s->m ; i=1, j=0, px_z_cross[0][1]: m->s
        for i, dec in enumerate(self.dec_list):
            px_z_cross[i][i] = self.px_z[i](*self.dec_list[i](zs_fuse))
            for j, (mu_ms, var_ms) in enumerate(zip(mu_trans[:2], var_trans[:2])):
                if i != j:
                    qz_x_cross = self.qz_x[i](mu_ms, var_ms)
                    zs = qz_x_cross.rsample(torch.Size([K]))
                    px_z_cross[j][i] = self.px_z[i](*dec(zs))
                    zss[j][i] = zs

        return qz_xs, px_z_cross, zss

    def reconstruct(self, data, runPath, epoch):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward([d[:8] for d in data])
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]

        for r, recons_list in enumerate(recons):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8]
                recon = recon.squeeze(0)
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                _data = _data if r == 1 else resize_img(_data, svhn_dataSize)
                recon = recon if o == 1 else resize_img(recon, svhn_dataSize)
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def get_zs(self):
        return self.zs_fusion



