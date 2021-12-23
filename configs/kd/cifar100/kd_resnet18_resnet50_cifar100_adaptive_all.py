_base_ = './kd_resnet18_resnet50_cifar100_equal.py'
# model settings
model = dict(
    adaptive=True,
    adaptation = dict(out_channels=[256, 512, 1024, 2048], in_channels=[64, 128, 256, 512]),
    distill_losses=[
        dict(type='NST', mode='feature', loss_weight=1.0),
        dict(type='Logits', mode='logits', loss_weight=1.0),
        dict(type='AttentionTransfer', mode='feature', p=2, loss_weight=1.0),
        dict(type='SoftTarget', mode='logits', T=4, loss_weight=1.0)
    ])
