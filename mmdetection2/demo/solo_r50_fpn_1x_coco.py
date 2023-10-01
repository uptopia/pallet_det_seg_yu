auto_scale_lr = dict(base_batch_size=16, enable=False)
checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='NumClassCheckHook'),
]
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    train=dict(
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(img_scale=(
                1333,
                800,
            ), keep_ratio=True, type='Resize'),
            dict(flip_ratio=0.5, type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_labels',
                    'gt_masks',
                ],
                type='Collect'),
        ],
        type='CocoDataset'),
    val=dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    workers_per_gpu=2)
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
dist_params = dict(backend='nccl')
evaluation = dict(metric=[
    'bbox',
    'segm',
])
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        8,
        11,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    mask_head=dict(
        cls_down_index=0,
        feat_channels=256,
        in_channels=256,
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_mask=dict(loss_weight=3.0, type='DiceLoss', use_sigmoid=True),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=80,
        num_grids=[
            40,
            36,
            24,
            16,
            12,
        ],
        pos_scale=0.2,
        scale_ranges=(
            (
                1,
                96,
            ),
            (
                48,
                192,
            ),
            (
                96,
                384,
            ),
            (
                192,
                768,
            ),
            (
                384,
                2048,
            ),
        ),
        stacked_convs=7,
        strides=[
            8,
            8,
            16,
            32,
            32,
        ],
        type='SOLOHead'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        filter_thr=0.05,
        kernel='gaussian',
        mask_thr=0.5,
        max_per_img=100,
        nms_pre=500,
        score_thr=0.1,
        sigma=2.0),
    type='SOLO')
mp_start_method = 'fork'
opencv_num_threads = 0
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
resume_from = None
runner = dict(max_epochs=12, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(img_scale=(
        1333,
        800,
    ), keep_ratio=True, type='Resize'),
    dict(flip_ratio=0.5, type='RandomFlip'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
        'gt_masks',
    ], type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
