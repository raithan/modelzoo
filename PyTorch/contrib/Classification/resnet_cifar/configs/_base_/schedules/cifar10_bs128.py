optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        100,
        150,
    ], type='MultiStepLR')
# randomness = dict(deterministic=False, seed=None)
# resume = False

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)
