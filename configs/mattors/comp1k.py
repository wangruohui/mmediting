# Base config for Composition-1K dataset

# dataset settings
dataset_type = 'AdobeComp1kDataset'
# data_root = 's3://openmmlab/datasets/editing/adobe_composition-1k'
# cluster=openmmlab in ~/petreloss.conf
data_root = 'data/adobe_composition-1k'
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

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='training_list.json',
        test_mode=False,
    ))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_list.json',
        test_mode=True,
    ))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SAD'),
    dict(type='MSE'),
    dict(type='GradientError'),
    dict(type='ConnectivityError'),
]

test_evaluator = val_evaluator
