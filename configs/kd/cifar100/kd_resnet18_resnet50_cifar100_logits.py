_base_ = './kd_resnet18_resnet50_cifar100_equal.py'

# model settings
model = dict(
    distill_losses=[
        dict(type='Logits', mode='logits', loss_weight=0.1)
    ])
