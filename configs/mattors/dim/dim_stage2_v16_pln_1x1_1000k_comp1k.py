_base_ = ['dim_base.py']

# model settings
model = dict(
    refiner=dict(type='PlainRefiner'),
    pretrained=None,  # make sure to set load_from to dim stage 1
    loss_refine=dict(type='CharbonnierLoss'),
    train_cfg=dict(train_backbone=False, train_refiner=True),
    test_cfg=dict(refine=True),
)

load_from = './checkpoints/dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth'
