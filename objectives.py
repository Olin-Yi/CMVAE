import torch
from utils import log_mean_exp, is_multidata, kl_divergence, get_mean
from numpy import prod
import torch.nn as nn


# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def my_iwae_looser(model, x, K=1):
    qz_xs, px_zs, zss = model(x, K)
    lws = []

    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r][r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_xs[d].log_prob(zs).sum(-1) for d, zs in enumerate(zss[r])]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.llik_scaling[d]).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r][r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r][r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.llik_scaling[d]).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
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

