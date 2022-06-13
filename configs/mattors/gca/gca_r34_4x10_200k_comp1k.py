_base_ = ['../comp1k.py', '../default_runtime.py']

# model settings
model = dict(
    type='GCA',
    data_preprocessor=dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        trimap_proc='onehot',
        inputs_only=True,
        size_divisor=32,
        resize_method='pad',
        resize_mode='reflect',
    ),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='ResGCAEncoder',
            block='BasicBlock',
            layers=[3, 4, 4, 2],
            in_channels=6,
            with_spectral_norm=True),
        decoder=dict(
            type='ResGCADecoder',
            block='BasicBlockDec',
            layers=[2, 3, 3, 2],
            with_spectral_norm=True)),
    loss_alpha=dict(type='L1Loss'),
    pretrained='open-mmlab://mmedit/res34_en_nomixup',
    train_cfg=dict(train_backbone=True),
    test_cfg=dict(pad_multiple=32, pad_mode='reflect'))

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = 'data/adobe_composition-1k'
bg_dir = './data/coco/train2017'
# img_norm_cfg = dict(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    dict(
        type='CompositeFg',
        fg_dirs=[
            f'{data_root}/Training_set/Adobe-licensed images/fg',
            f'{data_root}/Training_set/Other/fg'
        ],
        alpha_dirs=[
            f'{data_root}/Training_set/Adobe-licensed images/alpha',
            f'{data_root}/Training_set/Other/alpha'
        ]),
    dict(
        type='RandomAffine',
        keys=['alpha', 'fg'],
        degrees=30,
        scale=(0.8, 1.25),
        shear=10,
        flip_ratio=0.5),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='CropAroundCenter', crop_size=512),
    dict(type='RandomJitter'),
    dict(type='MergeFgAndBg'),
    # dict(type='RescaleToZeroOne', keys=['merged', 'alpha']),
    # dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    # dict(type='Collect', keys=['merged', 'alpha', 'trimap'], meta_keys=[]),
    # dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap']),
    # dict(type='FormatTrimap', to_onehot=True),
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
    # dict(type='FormatTrimap', to_onehot=True),
    dict(type='PackEditInputs'),
]
# data = dict(
#     workers_per_gpu=8,
#     train_dataloader=dict(samples_per_gpu=10, drop_last=True),
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
    # batch_size=10,
    # num_workers=8,
    batch_size=4,
    num_workers=1,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    # num_workers=8,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=200_000,
    val_interval=10_000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=4e-4, betas=[0.5, 0.999]))
# optimizers = dict(type='Adam', lr=4e-4, betas=[0.5, 0.999])
# learning policy

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=5000,
        by_epoch=False,  # 按迭代更新学习率
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=200_000,  ## TODO, need more check
        eta_min=0,
        begin=5000,
        end=200_000,
        by_epoch=False,  # 按迭代更新学习率
    )
]
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     by_epoch=False,
#     warmup='linear',
#     warmup_iters=5000,
#     warmup_ratio=0.001)

# # checkpoint saving
# checkpoint_config = dict(interval=2000, by_epoch=False)
# evaluation = dict(interval=2000, save_image=False, gpu_collect=False)
# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         # dict(type='TensorboardLoggerHook'),
#         # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca'))
#     ])

# # runtime settings
# total_iters = 200000
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = './work_dirs/gca'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
