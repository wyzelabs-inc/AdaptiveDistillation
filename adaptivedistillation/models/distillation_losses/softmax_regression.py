import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES
from .logits import Logits


@DISTILL_LOSSES.register_module()
class SoftmaxRegression(Logits):
    '''
    Knowledge Distillation via softmax regression representation learning
    https://openreview.net/pdf?id=ZzwDy_wiWv
    '''

    def __init__(self,
                 mode='softmax_regression',
                 loss_weight=1.0,
                 loss_name='loss_kd_softmax_regression'):
        super(SoftmaxRegression, self).__init__(mode, loss_weight, loss_name)

    def forward(self, out_s, out_t, gt_label=None):
        loss = F.mse_loss(out_s, out_t)

        return loss
