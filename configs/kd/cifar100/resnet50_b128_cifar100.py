_base_ = [
    '../../_base_/models/resnet50_cifar.py', '../../_base_/datasets/cifar100_bs128.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
        type='ImageClassifierAD',
        backbone=dict(
            out_indices=(0, 1, 2, 3)
            ), 
        head=dict(
            type='LinearClsHeadKD', 
            num_classes=100
            )
        )
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
