import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES
from .base_distill_loss import BaseDistillLoss


@DISTILL_LOSSES.register_module()
class Logits(BaseDistillLoss):
	'''
	Do Deep Nets Really Need to be Deep?
	http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
	'''
	def __init__(self, mode='logits', loss_weight=1.0, loss_name='loss_kd_logits'):
		super(Logits, self).__init__(mode, loss_weight, loss_name)

	def forward(self, out_s, out_t, gt_label=None):
		loss = F.mse_loss(out_s, out_t)

		return loss