_base_ = './kd_resnet18_resnet50_cifar100_adaptive_all.py'
model = dict(add_layer_loss=False)
