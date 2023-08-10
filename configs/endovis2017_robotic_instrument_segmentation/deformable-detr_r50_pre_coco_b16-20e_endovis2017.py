_base_ = [
    # "../_base_/datasets/coco_detection.py",
    "../_base_/default_runtime.py",
]

load_from = "https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth"
num_classes = 7
data_root = "/data2/shared/miccai/EndoVis2017/train"
annotation_filename = "coco.json"
classes = (
    "prograsp forceps",
    "bipolar forceps",
    "needle driver",
    "grasping reactor",
    "vessel sealer",
    "monopolar curved scissors",
    "other",
)
dataset_type = "CocoDataset"
backend_args = None

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(384, 600), allow_negative_crop=True),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_1",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_2",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_3",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_4",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_5",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                test_mode=True,
                pipeline=test_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_7",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
            dict(
                type=dataset_type,
                metainfo=dict(classes=classes),
                data_root=f"{data_root}/instrument_dataset_8",
                ann_file=annotation_filename,
                data_prefix=dict(img="left_frames/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            ),
        ],
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=f"{data_root}/instrument_dataset_6",
        ann_file=annotation_filename,
        data_prefix=dict(img="left_frames/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/instrument_dataset_6/{annotation_filename}",
    metric="bbox",
    classwise=True,
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        }
    ),
)

# learning policy
max_epochs = 20
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[40], gamma=0.1)]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=1, max_keep_ckpts=-1),
    visualization=dict(type="DetVisualizationHook", draw=True, interval=50),
)

model = dict(
    type="DeformableDETR",
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(embed_dims=256, batch_first=True),  # MultiScaleDeformableAttention
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1),
        ),
    ),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),  # MultiheadAttention
            cross_attn_cfg=dict(embed_dims=256, batch_first=True),  # MultiScaleDeformableAttention
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type="DeformableDETRHead",
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=100),
)
