_base_ = [
    '../../_base_/datasets/cifar100_bs128.py', '../../_base_/schedules/cifar10_bs128.py',
    '../../_base_/default_runtime.py'
]
dataset_type = 'CIFAR100KD'
data = dict(
    train = dict(type=dataset_type),
    val = dict(type=dataset_type),
    test = dict(type=dataset_type)
)
# norm_cfg = dict(type="SyncBN", requires_grad=True)
# model settings
teacher_ckpt = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth"
model = dict(
    type='KnowledgeDistillationImageClassifier',
    teacher_config="configs/kd/cifar100/resnet50_b128_cifar100.py",
    teacher_ckpt=teacher_ckpt,
    alpha_distill=1.0,
    distill_losses=[
        dict(type='AttentionTransfer', mode='feature', p=2, loss_weight=1.0),
        dict(type='SoftTarget', mode='logits', T=4, loss_weight=1.0)
    ],
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # norm_cfg=norm_cfg,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHeadKD',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
evaluation = dict(interval=1, metric=['accuracy','agreement'])
checkpoint_config = dict(interval=40)
