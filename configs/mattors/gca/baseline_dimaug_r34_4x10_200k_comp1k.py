_base_ = ['baseline_r34_4x10_200k_comp1k.py']

# model settings
model = dict(backbone=dict(encoder=dict(in_channels=4)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged']),
    dict(
        type='Resize',
        keys=['alpha', 'merged'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    # dict(type='RescaleToZeroOne', keys=['merged', 'alpha']),
    # dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    # dict(type='Collect', keys=['merged', 'alpha', 'trimap'], meta_keys=[]),
    # dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap']),
    dict(type='FormatTrimap', to_onehot=False),
    dict(type='PackEditInputs'),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    # dict(type='Pad', keys=['trimap', 'merged'], mode='reflect'),
    # dict(type='RescaleToZeroOne', keys=['merged']),
    # dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    # dict(
    #     type='Collect',
    #     keys=['merged', 'trimap'],
    #     meta_keys=[
    #         'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
    #     ]),
    # dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='FormatTrimap', to_onehot=False),
    dict(type='PackEditInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader
