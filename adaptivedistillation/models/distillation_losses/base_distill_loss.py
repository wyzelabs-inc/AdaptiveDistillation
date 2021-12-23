import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class BaseDistillLoss(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, mode, loss_weight=1.0, loss_name='loss_kd_base'):
        super(BaseDistillLoss, self).__init__()
        self.mode = mode
        self.loss_weight=loss_weight
        self.loss_name=loss_name

    def forward(self, student_x, teacher_x):
        pass
