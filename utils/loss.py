import torch
import torch.nn.functional as F
import torch.nn as nn

class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(reduction='batchmean')):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, pred, label):
        # The original implementation from ActionCLIP uses 'loss = self.error_metric(y_pred, y_true) * batch_size',
        # however, it seems more reasonable to use batchmean instead.
        y_pred = F.log_softmax(input=pred, dim=1)
        y_true = F.softmax(input=label*10, dim=1) # if introduce temperature, distribution after softmax will become more 'sharp'
        loss = self.error_metric(y_pred, y_true)
        return loss

class suppressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape # 512
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten() # Remove the 0th column, which contains all the diagonal elements of the original matrix

    def forward(self, f_a, f_b):
        # empirical cross-correlation matrix
        # apply column-wise z-score normalization to input feature matrix:
        f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)    # (bs, 512)
        f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)    # (bs, 512)
        c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)   # (512, 512)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = self.off_diagonal(c).pow_(2).mean()
        loss = on_diag + 0.005 * off_diag

        return loss
    