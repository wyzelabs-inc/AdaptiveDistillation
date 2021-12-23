import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES
from .base_distill_loss import BaseDistillLoss


@DISTILL_LOSSES.register_module()
class AttentionTransfer(BaseDistillLoss):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, mode='feature', p=2, loss_weight=1.0, loss_name='loss_kd_attention_transfer'):
        super(AttentionTransfer, self).__init__(mode, loss_weight, loss_name)
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm+eps)

        return am
