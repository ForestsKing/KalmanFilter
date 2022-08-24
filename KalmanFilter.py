import torch
from torch import nn
from torch.nn.parameter import Parameter


class KalmanFilter(nn.Module):

    def __init__(self, n_feature, device):
        super(KalmanFilter, self).__init__()

        self.n_feature = n_feature
        self.I = torch.eye(self.n_feature).to(device)
        self.F = torch.eye(self.n_feature).to(device)
        self.H = torch.eye(self.n_feature).to(device)

        self.Q = Parameter(torch.randn(n_feature, 1), requires_grad=True)
        self.R = Parameter(torch.randn(n_feature, 1), requires_grad=True)

    def _step(self, x_old, P_old, z_new):
        Q = self.Q.unsqueeze(0).repeat(z_new.shape[0], 1, 1)
        R = self.R.unsqueeze(0).repeat(z_new.shape[0], 1, 1)
        F = self.F.unsqueeze(0).repeat(z_new.shape[0], 1, 1)
        H = self.H.unsqueeze(0).repeat(z_new.shape[0], 1, 1)
        I = self.I.unsqueeze(0).repeat(z_new.shape[0], 1, 1)

        Q = Q * I
        R = R * I

        # predict
        x_new_prior = torch.matmul(F, x_old)
        P_new_prior = torch.matmul(torch.matmul(F, P_old), F.permute(0, 2, 1)) + Q

        # update
        K_new = torch.matmul(torch.matmul(P_new_prior, H.permute(0, 2, 1)),
                             torch.linalg.inv(torch.matmul(torch.matmul(H, P_new_prior), H.permute(0, 2, 1)) + R))
        x_new = x_new_prior + torch.matmul(K_new, (z_new - torch.matmul(H, x_new_prior)))
        P_new = torch.matmul(I - torch.matmul(K_new, H), P_new_prior)
        return x_new, P_new

    def forward(self, arr):
        arr = arr.permute(0, 2, 1)
        res = []

        x = torch.zeros(arr.shape[0], arr.shape[1], 1).to(arr.device)
        P = torch.ones(arr.shape[0], arr.shape[1], arr.shape[1]).to(arr.device)

        for i in range(arr.shape[-1]):
            z = arr[:, :, i].unsqueeze(-1)
            x, P = self._step(x, P, z)
            res.append(x)
        res = torch.cat(res, dim=-1).permute(0, 2, 1)
        return res