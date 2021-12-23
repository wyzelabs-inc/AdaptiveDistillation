from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry, build_from_cfg
from mmcls.models import CLASSIFIERS, HEADS
from torch import nn


DISTILL_LOSSES = Registry('distill_losses')

def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)


def build_distill_losses(cfg):
    """Build distill losses."""
    return build(cfg, DISTILL_LOSSES)

