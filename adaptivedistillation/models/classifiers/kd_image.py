import copy
import mmcv
import re
import torch
import torch.nn as nn
import warnings
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.cnn.bricks.registry import NORM_LAYERS
from mmcv.runner import load_checkpoint
from numpy.random import rand
from operator import pos
from torch.nn import functional as F

from mmcls.models.builder import build_backbone, build_neck
from mmcls.models.classifiers.base import BaseClassifier

from ..builder import (CLASSIFIERS, build_distill_losses,
                         build_classifier, build_head)

ADAPTATION = "adaptation_layer_{}"
NORM_LAYER = "adaptation_norm_layer_{}"


@CLASSIFIERS.register_module()
class KnowledgeDistillationImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 teacher_config,
                 distill_losses,
                 adaptation=None,
                 neck=None,
                 head=None,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 adaptive=False,
                 pretrained=None,
                 init_cfg=None,
                 add_layer_loss=True,
                 alpha_distill=1.0):
        super(KnowledgeDistillationImageClassifier, self).__init__(init_cfg)
        """
            backbone (dict): student backbone configuration.
            teacher_config (str): path to teacher configuration.add()
            distillation_losses (list): list of distillation losses
            adaptation (dict): configuration for adaptation layers
            neck (dict): student neck configuration. Default:None
            head (dict): student prediciton head configuration. Default:None
            teacher_ckpt (str): path to teacher checkpoint file
            eval_teacher (bool): flag to change teacher mode. Default:True
            adaptive (bool): flag to specify whether to use adaptive distillation. Default:False
            pretrained (str): path to pretrained checkpoint for student model.add(). Default:None
            init_cfg (dict): student model initialization configuration. Default:None
            add_layer_loss (bool): flag to switch between adaptive or adaptive-layerwise methods.add()
            alpha_distill (float): relative importance between distillation loss and empirical loss.
        """

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)  # build student backbone
        self.eval_teacher = eval_teacher
        self.adaptive = adaptive
        self.distill_losses = {}
        self.distill_paths = []
        self.add_layer_loss = add_layer_loss
        self.adaptation = adaptation
        self.alpha_distill = alpha_distill

        # Build adaptation layers
        if adaptation:
            for i, (in_c, out_c) in enumerate(zip(adaptation['in_channels'], adaptation['out_channels'])):
                conv = build_conv_layer(backbone.get('conv_cfg', None), in_c, out_c,
                                        kernel_size=1,
                                        stride=1, bias=False)
                _, norm = build_norm_layer(backbone.get('norm_cfg', 'BN'), out_c, postfix=i)
                kaiming_init(conv)
                constant_init(norm, 1)
                self.add_module(ADAPTATION.format(i), conv)
                self.add_module(NORM_LAYER.format(i), norm)

        # Build teacher model and load from checkpoint
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_classifier(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt)

        # Build distillation losses
        for distill_loss_cfg in distill_losses:
            distill_loss = build_distill_losses(distill_loss_cfg)
            self.distill_losses[distill_loss.loss_name] = distill_loss
            if not add_layer_loss and distill_loss.mode == 'feature':
                for ii in self.backbone.out_indices:
                    self.distill_paths.append(distill_loss.loss_name + ":" + str(ii))
            else:
                self.distill_paths.append(distill_loss.loss_name)

        # Build loss scale for adaptive disitllation
        self.loss_scaler = AdaptiveLossScaler(
            list(self.distill_paths)) if self.adaptive else None

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to devices when calling cuda function."""
        self.teacher_model.cuda(device=device)
        for dl_name in self.distill_losses:
            self.distill_losses[dl_name].cuda(device=device)
        if self.loss_scaler:
            self.loss_scaler.cuda(device=device)
        return super().cuda(device=device)

    def extract_feat(self, img, with_neck=True):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.with_neck and with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        # ----Extract student outputs------
        # x is tuple of student backbone output layers
        # gap_x is global average pooled output from last layer of student backbone
        # out is the output logit from student
        x = self.extract_feat(img, with_neck=False)
        gap_x = self.neck(x[-1])
        out = self.head.forward_train(gap_x, gt_label=None, return_loss=False)
        loss = self.head.loss(out, gt_label)
        losses['loss_cls'] = loss['loss']

        # Transform student features to teacher using adaptation layers
        if self.adaptation:
            feats = []
            for i, feat in enumerate(x):
                feat = getattr(self, ADAPTATION.format(i))(x[i])
                feat = getattr(self, NORM_LAYER.format(i))(feat)
                feats.append(feat)

        # ----Extract teacher outputs------
        # teacher_x is tuple of teacher backbone output layers
        # teacher_gap_x is global average pooled output from last layer of teacher backbone
        # out_teacher is the output logit from teacher
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img, with_neck=False)
            teacher_gap_x = self.teacher_model.neck(teacher_x[-1])
            out_teacher = self.teacher_model.head.forward_train(
                teacher_gap_x, gt_label=None, return_loss=False)

        # Compute different distillation losses using features or logits
        for dl_name in self.distill_losses:
            distill_loss = self.distill_losses[dl_name]
            if distill_loss.mode == 'feature':
                s_feats = x if 'attention_transfer' in distill_loss.loss_name else feats
                if isinstance(s_feats, tuple) or isinstance(s_feats, list):
                    if self.add_layer_loss:
                        loss = sum([distill_loss(s_x, t_x)
                                    for s_x, t_x in zip(s_feats, teacher_x)])
                        losses[dl_name] = loss*distill_loss.loss_weight
                    else:
                        for ii, (s_x, t_x) in enumerate(zip(s_feats, teacher_x)):
                            dl_name = distill_loss.loss_name + ":" + str(ii)
                            losses[dl_name] = distill_loss(s_x, t_x)*distill_loss.loss_weight
                else:
                    loss = distill_loss(x, teacher_x)
            elif distill_loss.mode == 'logits':
                loss = distill_loss(out, out_teacher, gt_label)
                losses[dl_name] = loss*distill_loss.loss_weight
            elif distill_loss.mode == 'softmax_regression':
                adapted_student_gap_x = self.neck(feats[-1])
                out_cross_student = self.teacher_model.head.forward_train(
                    adapted_student_gap_x, gt_label=None, return_loss=False)
                loss = distill_loss(out_teacher, out_cross_student)
                losses[dl_name] = loss*distill_loss.loss_weight

        # scale distillation losses using adaptive distillation loss scaler
        if self.loss_scaler:
            losses.update(self.loss_scaler(losses))

        # change the relative importance of distillation loss using alpha distill
        for key, loss in losses.items():
            if 'loss_kd' in key or 'loss_alphas' in key:
                losses[key] = loss*self.alpha_distill

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img, with_neck=False)
        x = self.neck(x[-1])
        out = self.head.simple_test(x)
        teacher_x = self.teacher_model.extract_feat(img, with_neck=False)
        teacher_x = self.teacher_model.neck(teacher_x[-1])
        teacher_out = self.teacher_model.head.simple_test(teacher_x)
        return torch.tensor([out, teacher_out])

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


class AdaptiveLossScaler(nn.Module):
    def __init__(self, positions):
        super().__init__()
        self.positions = positions
        self.alphas = nn.Parameter(torch.zeros(len(positions)))

    def forward(self, losses):
        loss_cls = losses.pop('loss_cls')
        scaled_losses = self.get_scaled_losses(losses, torch.exp(-self.alphas))
        scaled_losses.update(dict(loss_alphas=self.alphas.sum()))
        scaled_losses.update(dict(loss_cls=loss_cls))
        return scaled_losses

    def get_scaled_losses(self, losses, alphas):
        if len(list(losses.keys())) != len(self.positions):
            raise ValueError('Check distillation positions. Losses: {}, Positions: {}'.format(
                list(losses.keys()), self.positions))
        scaled_losses = {}
        for index, position in enumerate(self.positions):
            scaled_losses.update(self.scale_losses(losses.pop(position), alphas[index], position))
            scaled_losses.update({'alpha_{}'.format(position.strip("loss_")): alphas[index]})
        return scaled_losses

    def scale_losses(self, losses, alpha, position=None):
        # Scale losses with alpha.
        if isinstance(losses, dict):
            for task, loss in losses.items():
                if isinstance(loss, list):
                    losses[task] = [l*alpha for l in loss]
                else:
                    losses[task] = loss*alpha
        elif isinstance(losses, list):
            losses = {'{}'.format(position): [l*alpha for l in losses]}
        elif isinstance(losses, torch.Tensor):
            losses = {'{}'.format(position): losses*alpha}

        return losses
