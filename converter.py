import torch.nn as nn


def converter_layer(params):
    converter = nn.Sequential(
        nn.Linear(params.latent_dim, params.feature_dim),
        nn.ReLU(True),
        nn.Dropout(p=0.1),
        nn.Linear(params.feature_dim, params.feature_dim),
        nn.ReLU(True),
        nn.Dropout(p=0.1),
        nn.Linear(params.feature_dim, params.latent_dim)
    )
    return converter


class Converter(nn.Module):
    def __init__(self, params):
        super(Converter, self).__init__()
        mu_conv, var_conv = None, None
        if params.modality_num == 2:
            mu_conv = nn.ModuleList([converter_layer(params) for _ in range(params.modality_num)])
            var_conv = nn.ModuleList([converter_layer(params) for _ in range(params.modality_num)])
        elif params.modality_num > 2:
            mu_conv = nn.ModuleList([
                nn.ModuleList([converter_layer(params) for _ in range(params.modality_num - 1)])
                for _ in range(params.modality_num)
            ])
            var_conv = nn.ModuleList([
                nn.ModuleList([converter_layer(params) for _ in range(params.modality_num - 1)])
                for _ in range(params.modality_num)
            ])
        else:
            print('This is not a multimodal task!')

        self.mu_conv = mu_conv
        self.var_conv = var_conv
        self.params = params

    def forward(self, mu, var):
        """
        :param mu:  a list of modal mean
        :param var: a list of modal log variance
        :return: cross-modal mu and var

        if modality_num == 2: (MNIST <-> SVHN)
            result = [M->S, S->M]
        else if modality_num > 2: (text <-> visual <-> acoustic)
            result = [[T->V, T->A],
                      [V->T, V->A],
                      [A->T, A->V]]
        """
        mu_trans, var_trans = None, None
        if self.params.modality_num == 2:
            mu_trans, var_trans = [], []

            for i in range(self.params.modality_num):
                mu_ = self.mu_conv[i](mu[i])
                var_ = self.var_conv[i](var[i])

                mu_trans.append(mu_)
                var_trans.append(var_)
        elif self.params.modality_num > 2:
            mu_trans = [[None for _ in range(self.params.modality_num - 1)] for _ in range(self.params.modality_num)]
            var_trans = [[None for _ in range(self.params.modality_num - 1)] for _ in range(self.params.modality_num)]

            for i in range(self.params.modality_num):
                for j in range(self.params.modality_num - 1):
                    mu_ = self.mu_conv[i][j](mu[i])
                    var_ = self.var_conv[i][j](var[i])

                    mu_trans[i][j] = mu_
                    var_trans[i][j] = var_

        return mu_trans, var_trans
