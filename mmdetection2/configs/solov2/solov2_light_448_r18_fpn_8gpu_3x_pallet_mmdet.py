_base_ = './solov2_light_r50_fpn_3x_coco.py'

dataset_type = 'CocoDataset'
classes = ('front',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/pallet_train/pallet_train_annotations.json',
        img_prefix='data/pallet_train'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/pallet_val/pallet_val_annotations.json',
        img_prefix='data/pallet_val'))

# model settings
model = dict(
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
