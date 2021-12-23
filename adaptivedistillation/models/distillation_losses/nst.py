import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES
from .base_distill_loss import BaseDistillLoss


@DISTILL_LOSSES.register_module()
class NST(BaseDistillLoss):
    '''
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    '''

    def __init__(self, mode='feature', loss_weight=1.0, loss_name='loss_kd_nst'):
        super(NST, self).__init__(mode, loss_weight, loss_name)

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)

        gram_s = self.gram_matrix(fm_s)
        gram_t = self.gram_matrix(fm_t)

        loss = F.mse_loss(gram_s, gram_t)

        return loss

    def gram_matrix(self, fm):
        return torch.bmm(fm, fm.transpose(1, 2))
