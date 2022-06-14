_base_ = ['dim_base.py']

# model settings
model = dict(
    refiner=dict(type='PlainRefiner'),
    pretrained=None,  # make sure to set load_from to dim stage 2
    loss_refine=dict(type='CharbonnierLoss'),
    train_cfg=dict(train_backbone=True, train_refiner=True),
    test_cfg=dict(refine=True, ))

load_from = './checkpoints/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth'
