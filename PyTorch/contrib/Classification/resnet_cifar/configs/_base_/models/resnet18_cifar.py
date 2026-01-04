# model settings
model = dict(
    backbone=dict(
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet_CIFAR'),
    head=dict(
        in_channels=512,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=10,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
