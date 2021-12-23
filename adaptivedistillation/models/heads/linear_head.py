import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.heads import LinearClsHead

from ..builder import HEADS

# Modified LinearClsHead for knowledge distillation
@HEADS.register_module()
class LinearClsHeadKD(LinearClsHead):
    """Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHeadKD, self).__init__(num_classes, in_channels, init_cfg,  *args, **kwargs)

    def forward_train(self, x, gt_label, return_loss=True):
        cls_score = self.fc(x)
        if return_loss:
            losses = self.loss(cls_score, gt_label)
            return losses
        else:
            return cls_score
