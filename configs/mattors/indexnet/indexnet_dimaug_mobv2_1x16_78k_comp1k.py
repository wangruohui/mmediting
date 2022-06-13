_base_ = ['../comp1k.py', '../default_runtime.py']

# model settings
model = dict(
    type='IndexNet',
    data_preprocessor=dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        proc_inputs='normalize',
        proc_trimap='rescale_to_zero_one',
        proc_gt='rescale_to_zero_one',
    ),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='IndexNetEncoder', in_channels=4, freeze_bn=True),
        decoder=dict(type='IndexNetDecoder')),
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5, sample_wise=True),
    loss_comp=dict(
        type='CharbonnierCompLoss', loss_weight=1.5, sample_wise=True),
    pretrained='open-mmlab://mmedit/mobilenet_v2',
    test_cfg=dict(resize_method='pad', resize_mode='reflect', size_divisor=32))

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = 'data/adobe_composition-1k'

train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    #     dict(
    #         type='RescaleToZeroOne',
    #         keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg']),
    #     dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    #     dict(
    #         type='Collect',
    #         keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],
    #         meta_keys=[]),
    #     dict(
    #         type='ImageToTensor',
    #         keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),
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
    # dict(type='RescaleToZeroOne', keys=['merged', 'trimap']),
    # dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    # dict(
    #     type='Collect',
    #     keys=['merged', 'trimap'],
    #     meta_keys=[
    #         'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
    #     ]),
    # dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='PackEditInputs'),
]
# data = dict(
#     workers_per_gpu=8,
#     train_dataloader=dict(samples_per_gpu=16, drop_last=True),
#     val_dataloader=dict(samples_per_gpu=1),
#     test_dataloader=dict(samples_per_gpu=1),
#     train=dict(
#         type=dataset_type,
#         ann_file=f'{data_root}/training_list.json',
#         data_prefix=data_root,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=f'{data_root}/test_list.json',
#         data_prefix=data_root,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=f'{data_root}/test_list.json',
#         data_prefix=data_root,
#         pipeline=test_pipeline))

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=78_000,
    val_interval=2600,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-2),
    paramwise_cfg=dict(custom_keys={'encoder.layers': dict(lr_mult=0.01)}),
)
# optimizers = dict(
#     constructor='DefaultOptimizerConstructor',
#     type='Adam',
#     lr=1e-2,
#     paramwise_cfg=dict(custom_keys={'encoder.layers': dict(lr_mult=0.01)}))
# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[52000, 67600],
    gamma=0.1,
    by_epoch=False,
)
# lr_config = dict(policy='Step', step=[52000, 67600], gamma=0.1, by_epoch=False)

# # checkpoint saving
# checkpoint_config = dict(interval=2600, by_epoch=False)
# evaluation = dict(interval=2600, save_image=False)
# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         # dict(type='TensorboardLoggerHook'),
#         # dict(type='PaviLoggerHook', init_kwargs=dict(project='IndexNet'))
#     ])

default_hooks = dict(checkpoint=dict(interval=2600))

# # runtime settings
# total_iters = 78000
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = './work_dirs/indexnet'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
