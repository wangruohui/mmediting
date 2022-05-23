_base_ = ['../default_runtime.py']

# model settings
model = dict(
    type='DIM',
    preprocess_cfg=dict(
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='VGG16', in_channels=4),
        decoder=dict(type='PlainDecoder')),
    pretrained='open-mmlab://mmedit/vgg16',
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
    loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5),
    train_cfg=dict(train_backbone=True, train_refiner=False),
    test_cfg=dict(refine=False, pad_multiple=32, pad_mode='reflect'))

# dataset settings
dataset_type = 'AdobeComp1kDataset'
# data_root = 's3://openmmlab/datasets/editing/adobe_composition-1k'  # cluster=openmmlab in ~/petreloss.conf
data_root = 'data/adobe_composition-1k'
# img_norm_cfg = dict(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged', save_original_img=True),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    # dict(
    #     type='RescaleToZeroOne',
    #     keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg', 'trimap']),
    # dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],
        meta_keys=[]),
    # dict(
    #     type='ImageToTensor',
    #     keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),
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

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_list.json',
        test_mode=True,
        pipeline=test_pipeline))
# test_dataloader = val_dataloader

test_evaluator = [
    dict(type='SAD'),
    dict(type='MSE'),
    dict(type='GradientError'),
    dict(type='ConnectivityError'),
]
# test_evaluator = val_evaluator

# val_cfg = dict(interval=1)
test_cfg = dict()

# # optimizer
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
# work_dir = './work_dirs/dim_stage1'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
