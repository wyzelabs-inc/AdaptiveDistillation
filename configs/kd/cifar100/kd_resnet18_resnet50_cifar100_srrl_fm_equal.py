_base_ = './kd_resnet18_resnet50_cifar100_equal.py'
# model settings
model = dict(
    adaptation = dict(out_channels=[256, 512, 1024, 2048], in_channels=[64, 128, 256, 512]),
    distill_losses=[
        dict(type='FeatureMap', mode='feature', loss_weight=1.0),
        dict(type='SoftmaxRegression', mode='softmax_regression', loss_weight=1.0),
    ])
