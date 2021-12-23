import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES
from .base_distill_loss import BaseDistillLoss


@DISTILL_LOSSES.register_module()
class FeatureMap(BaseDistillLoss):
    '''
    Knowledge Distillation via softmax regression representation learning
    https://openreview.net/pdf?id=ZzwDy_wiWv
    '''

    def __init__(self, mode='feature', loss_weight=1.0, loss_name='loss_kd_feature_map'):
        super(FeatureMap, self).__init__(mode, loss_weight, loss_name)

    def forward(self, fm_s, fm_t):
        x = fm_s.view(fm_s.size(0),fm_s.size(1),-1)
        y = fm_t.view(fm_t.size(0),fm_t.size(1),-1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()
