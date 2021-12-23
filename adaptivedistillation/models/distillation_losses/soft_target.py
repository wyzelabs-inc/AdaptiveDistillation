import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES
from .base_distill_loss import BaseDistillLoss


@DISTILL_LOSSES.register_module()
class SoftTarget(BaseDistillLoss):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, mode='logits', T=4, loss_weight=0.1, loss_name='loss_kd_soft_target'):
        super(SoftTarget, self).__init__(mode, loss_weight, loss_name)
        self.T = T

    def forward(self, out_s, out_t, gt_label=None):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss
