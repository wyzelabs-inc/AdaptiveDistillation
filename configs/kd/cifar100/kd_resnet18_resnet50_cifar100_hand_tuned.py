_base_ = './kd_resnet18_resnet50_cifar100_equal.py'

# model settings
model = dict(
    distill_losses=[
        dict(type='AttentionTransfer', mode='feature', p=2, loss_weight=1000.0),
        dict(type='SoftTarget', mode='logits', T=4, loss_weight=0.1)
    ])
