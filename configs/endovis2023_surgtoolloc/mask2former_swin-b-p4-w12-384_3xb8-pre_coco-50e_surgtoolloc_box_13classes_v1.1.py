_base_ = [
    "../_base_/default_runtime.py",
]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth"  # noqa
load_from = "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic_20220329_230021-05ec7315.pth"  # noqa

# hyperparameters
num_gpus = 4
batch_size = 4
lr = 0.0001
image_size = (750, 1333)  # height, width
max_epochs = 30

# custom
data_root = "/data2/shared/miccai/EndoVis2023/SurgToolLoc/v1.1"
annotation_filename = "coco_box_13classes.json"
classes = (
    "bipolar_forceps",
    "cadiere_forceps",
    "clip_applier",
    "force_bipolar",
    "grasping_retractor",
    "monopolar_curved_scissors",
    "needle_driver",
    "permanent_cautery_hook_spatula",
    "prograsp_forceps",
    "stapler",
    "suction_irrigator",
    "tip_up_fenestrated_grasper",
    "vessel_sealer",
)
train_list = [
    # "1243_clip_000000-000299",
    "1244_clip_000300-000599",
    "1245_clip_000600-000899",
    "1246_clip_000900-001199",
    "1247_clip_001200-001499",
    "1248_clip_001500-001799",
    "1249_clip_001800-002099",
    "1250_clip_002100-002399",
    "1251_clip_002400-002699",
    "1252_clip_002700-002999",
    "1253_clip_003000-003300",
    "1254_clip_003301-003600",
    "1255_clip_003601-003901",
    "1256_clip_003902-004201",
    "1257_clip_004202-004501",
    "1258_clip_004502-004802",
    "1259_clip_004803-005102",
    "1260_clip_005103-005402",
    "1261_clip_005403-005703",
    "1262_clip_005704-006003",
    "1263_clip_006004-006303",
    "1264_clip_006304-006604",
    "1265_clip_006605-006904",
    "1266_clip_006905-007204",
    "1267_clip_007205-007504",
    "1268_clip_007505-007804",
    "1269_clip_007805-008105",
    "1270_clip_008106-008405",
    "1271_clip_008406-008705",
    "1272_clip_008706-009006",
    "1273_clip_009007-009307",
    "1274_clip_009308-009608",
    "1275_clip_009609-009908",
    "1276_clip_009909-010209",
    "1277_clip_010210-010510",
    "1278_clip_010511-010810",
    "1279_clip_010811-011110",
    "1280_clip_011111-011410",
    "1281_clip_011411-011710",
    "1282_clip_011711-012011",
    "1283_clip_012012-012312",
    "1284_clip_012313-012612",
    "1285_clip_012613-012912",
    "1286_clip_012913-013212",
    "1287_clip_013213-013512",
    "1288_clip_013513-013812",
    "1289_clip_013813-014112",
    "1290_clip_014113-014412",
    "1291_clip_014413-014713",
    "1292_clip_014714-015013",
    "1293_clip_015014-015314",
    "1294_clip_015315-015614",
    "1295_clip_015615-015914",
    "1296_clip_015915-016215",
    "1297_clip_016216-016516",
    "1298_clip_016517-016816",
    "1299_clip_016817-017116",
    "1300_clip_017117-017416",
    "1301_clip_017417-017716",
    "1302_clip_017717-018017",
    "1303_clip_018018-018317",
    "1304_clip_018318-018619",
    "1305_clip_018620-018919",
    "1306_clip_018920-019219",
    "1307_clip_019220-019519",
    "1308_clip_019520-019819",
    "1309_clip_019820-020119",
    "1310_clip_020120-020419",
    "1311_clip_020420-020719",
    "1312_clip_020720-021020",
    "1313_clip_021021-021320",
    "1314_clip_021321-021621",
    "1315_clip_021622-021922",
    "1316_clip_021923-022222",
    "1317_clip_022223-022523",
    "1318_clip_022524-022823",
    "1319_clip_022824-023123",
    "1320_clip_023124-023423",
    "1321_clip_023424-023724",
    "1322_clip_023725-024024",
    "1323_clip_024025-024325",
    "1324_clip_024326-024625",
    "1325_clip_024626-024720",
]


# dataset setting
backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True, backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    # large scale jittering
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type="Resize",
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=image_size,
        crop_type="absolute",
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True, backend_args=backend_args),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

dataset_type = "CocoDataset"
train_dataloader = dict(
    batch_size=batch_size,
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
                data_root=f"{data_root}/{case}",
                ann_file=annotation_filename,
                data_prefix=dict(img="frame/"),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args,
            )
            for case in train_list
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
        data_root=f"{data_root}/1243_clip_000000-000299",
        # data_root=f"{data_root}/{train_list[0]}",
        ann_file=annotation_filename,
        data_prefix=dict(img="frame/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/1243_clip_000000-000299/{annotation_filename}",
    # ann_file=f"{data_root}/{train_list[0]}/{annotation_filename}",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator


# learning policy
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning policy
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1,
        begin=0,
        end=1500,
        by_epoch=False,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min_ratio=1e-2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=batch_size * num_gpus)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best="coco/bbox_mAP_50",
    ),
    visualization=dict(type="DetVisualizationHook", draw=True, interval=50),
)


# model setting
depths = [2, 2, 18, 2]
num_things_classes = len(classes)
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type="Mask2Former",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0.32858519781689005 * 255, 0.15265839395622285 * 255, 0.14655234887549404 * 255],
        std=[0.07691241763785549 * 255, 0.053818967599625046 * 255, 0.056615884572508365 * 255],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
        # batch_augments=[
        #     dict(
        #         type="BatchFixedSizePad",
        #         size=image_size,
        #         img_pad_value=0,
        #         pad_mask=True,
        #         mask_pad_value=0,
        #         pad_seg=False,
        #     )
        # ],
    ),
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    panoptic_head=dict(
        type="Mask2FormerHead",
        in_channels=[128, 256, 512, 1024],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type="MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
            ),
            positional_encoding=dict(num_feats=128, normalize=True),
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
                ),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0
        ),
        loss_dice=dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
    ),
    panoptic_fusion_head=dict(
        type="MaskFormerFusionHead",
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None,
    ),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="ClassificationCost", weight=2.0),
                dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
            ],
        ),
        sampler=dict(type="MaskPseudoSampler"),
    ),
    test_cfg=dict(
        panoptic_on=False,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True,
    ),
    init_cfg=None,
)

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    "backbone": dict(lr_mult=0.1, decay_mult=1.0),
    "backbone.patch_embed.norm": backbone_norm_multi,
    "backbone.norm": backbone_norm_multi,
    "absolute_pos_embed": backbone_embed_multi,
    "relative_position_bias_table": backbone_embed_multi,
    "query_embed": embed_multi,
    "query_feat": embed_multi,
    "level_embed": embed_multi,
}
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.blocks.{block_id}.norm": backbone_norm_multi
        for stage_id, num_blocks in enumerate(depths)
        for block_id in range(num_blocks)
    }
)
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.downsample.norm": backbone_norm_multi
        for stage_id in range(len(depths) - 1)
    }
)
# optimizer
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2),
)
