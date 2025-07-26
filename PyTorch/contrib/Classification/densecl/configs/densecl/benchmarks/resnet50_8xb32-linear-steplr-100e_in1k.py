_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_sgd_steplr_100e.py',
    '../../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220825-3078723b.pth', prefix='backbone.')))

train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.))

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
