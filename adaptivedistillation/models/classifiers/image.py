from mmcls.models.classifiers import ImageClassifier

from ..builder import CLASSIFIERS, build_head



# The modified ImageClassifier used in teacher model of KnowledgeDistillationImageClassifier
@CLASSIFIERS.register_module()
class ImageClassifierAD(ImageClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifierAD, self).__init__(backbone, neck, head,
                                                pretrained, train_cfg, init_cfg)

    def extract_feat(self, img, with_neck=True):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.with_neck and with_neck:
            x = self.neck(x)
        return x