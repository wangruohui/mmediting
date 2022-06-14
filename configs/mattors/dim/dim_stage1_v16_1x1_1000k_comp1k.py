_base_ = ['dim_base.py']

# model settings
model = dict(
    pretrained='open-mmlab://mmedit/vgg16',
    train_cfg=dict(train_backbone=True, train_refiner=False),
    test_cfg=dict(refine=False))
