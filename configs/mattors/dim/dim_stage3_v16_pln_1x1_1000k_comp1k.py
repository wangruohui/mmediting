_base_ = ['../comp1k.py', '../default_runtime.py']

# model settings
model = dict(
    type='DIM',
    data_preprocessor=dict(
        type='ImageAndTrimapPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        trimap_proc='rescale_to_zero_one',
        size_divisor=32,
        resize_method='pad',
        resize_mode='reflect',
    ),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='VGG16', in_channels=4),
        decoder=dict(type='PlainDecoder')),
    refiner=dict(type='PlainRefiner'),
    pretrained=None,  # make sure to set load_from to dim stage 1
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
    loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5),
    loss_refine=dict(type='CharbonnierLoss'),
    train_cfg=dict(train_backbone=True, train_refiner=True),
    test_cfg=dict(refine=True, pad_multiple=32, pad_mode='reflect'))

# dataset settings
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
    # dict(
    #     type='RescaleToZeroOne',
    #     keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg', 'trimap']),
    # dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    # dict(
    #     type='Collect',
    #     keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],
    #     meta_keys=[]),
    # dict(
    #     type='ImageToTensor',
    #     keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),
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
    #         'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha',
    # 'ori_trimap'
    #     ]),
    # dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='PackEditInputs'),
]

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
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

train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1_000_000,
    val_interval=40000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.00001),
    ))
# optimizers = dict(type='Adam', lr=0.00001)
# # learning policy
# lr_config = dict(policy='Fixed')

# # checkpoint saving
# checkpoint_config = dict(interval=40000, by_epoch=False)
# evaluation = dict(interval=40000, save_image=False)
# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         # dict(type='TensorboardLoggerHook'),
#         # dict(type='PaviLoggerHook', init_kwargs=dict(project='dim'))
#     ])

# # runtime settings
# total_iters = 1000000
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = './work_dirs/dim_finetune'
# load_from = './work_dirs/dim_stage2/latest.pth'
# resume_from = None
# workflow = [('train', 1)]

load_from = './checkpoints/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth'
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=40000))
