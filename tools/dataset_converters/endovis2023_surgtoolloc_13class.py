import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm


def main():
    # convert parts instance segmentation to each class of instance instruments dataset
    # SurgToolLoc(EndoVis2023) -> COCO format Instance segmentation
    parent_dir = Path("/data1/shared/miccai/EndoVis2023/SurgToolLoc/v1.1")
    annotation_csv = Path("/data2/src/atsushi/SurgToolLoc/result/cls_labels/cls_labels_modify.csv")
    child_dir_ptn = "*_clip_*"
    seg_dir_name = "mask"
    save_file = "coco_13classes.json"

    # semantic mask to instance id
    cmap = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 3,
        10: 4,
        11: 4,
        12: 4,
    }

    categories = [
        {"id": 0, "name": "bipolar_forceps"},
        {"id": 1, "name": "cadiere_forceps"},
        {"id": 2, "name": "clip_applier"},
        {"id": 3, "name": "force_bipolar"},
        {"id": 4, "name": "grasping_retractor"},
        {"id": 5, "name": "monopolar_curved_scissors"},
        {"id": 6, "name": "needle_driver"},
        {"id": 7, "name": "permanent_cautery_hook_spatula"},
        {"id": 8, "name": "prograsp_forceps"},
        {"id": 9, "name": "stapler"},
        {"id": 10, "name": "suction_irrigator"},
        {"id": 11, "name": "tip_up_fenestrated_grasper"},
        {"id": 12, "name": "vessel_sealer"},
        # {"id": 13, "name": "bipolar_dissector"},  # NOTE: bipolar dissector was removed from evaluation
    ]

    # taskごとにcoco.jsonを作成する
    df = pl.read_csv(annotation_csv)
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        task_name = "_".join(video_dir.name.split("_")[1:])
        df_cls_annotations = df.filter(pl.col("task_dirname") == task_name).select(
            ["filename", "instance_id", "bbox", "label"]
        )
        coco_dataset = create_coco_format(
            video_dir / seg_dir_name, cmap, categories, df_cls_annotations
        )
        save_annotation(coco_dataset, video_dir / save_file)


def create_coco_format(mask_parent_dir, cmap, categories, df_cls_annotations):
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # instance通し番号
    instance_id = 0
    with tqdm(enumerate(mask_parent_dir.glob("*.png"))) as pbar:
        for image_id, label_file in pbar:
            # cv2.imreadで開くとpaletteのRGB値で開かれるのでPILで開く
            mask = np.array(Image.open(label_file).convert("P"))

            # image info
            pbar.set_postfix(dict(image=label_file.name))
            height, width = mask.shape
            image_info = {
                "id": image_id,
                "file_name": label_file.name,  # maskとframeは同じファイル名
                "width": width,
                "height": height,
            }
            coco_dataset["images"].append(image_info)
            # annotation info
            pbar.set_description(mask_parent_dir.parent.name)
            mask = index_remap(mask, cmap)
            instance_uniques = np.unique(mask)[1:]  # ignore background
            # label of instances
            anno = df_cls_annotations.filter(pl.col("filename") == label_file.name).to_dicts()
            # maskからinstanceを取り出す
            for _id in instance_uniques:
                instance_mask = get_target_mask(mask, _id)
                bbox = mask2bbox(instance_mask)
                area = calc_mask_area(instance_mask)

                # determine which label is targeted
                label = rename_label(
                    anno[
                        np.argmax(
                            [
                                calc_bbox_iou(parse_bbox(anno_instance["bbox"], xyxy=False), bbox)
                                for anno_instance in anno
                            ]
                        )
                    ]["label"]
                )
                label_id = get_label_id(categories, label)

                # 術具が写っていない場合真っ黒
                if area > 0:
                    annotation_info = {
                        "id": instance_id,
                        "image_id": image_id,
                        "category_id": label_id,
                        "bbox": mask2bbox(instance_mask),
                        "segmentation": [
                            contour.flatten().tolist() for contour in mask2contours(instance_mask)
                        ],
                        "area": area,
                        "iscrowd": 0,
                    }
                    coco_dataset["annotations"].append(annotation_info)
                    instance_id += 1

    return coco_dataset


def index_remap(mask, cmap):
    seg_map = np.zeros_like(mask)
    for old_id, new_id in cmap.items():
        seg_map += get_target_mask(mask, old_id) * new_id
    return seg_map


def get_target_mask(mask, index):
    return np.where(mask == index, 1, 0).astype(mask.dtype)


def mask2bbox(mask):
    # 2d array -> [xmin, ymin, width, height]
    contours = mask2contours(mask)
    all_contours = np.concatenate(contours)
    xmin, ymin = all_contours.min(axis=0)
    xmax, ymax = all_contours.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    return list(map(int, [xmin, ymin, w, h]))


def mask2contours(mask):
    # 2d array -> [contour([x1, y1], [x2, y2], [x3, y3], [x4, y4],...)]
    _, bw = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [contour[:, 0] for contour in contours]


def calc_mask_area(mask):
    return int(np.sum(mask > 0))


def rename_label(name):
    space = name[-1]
    name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
    return name[:-1] if space == " " else name


def parse_bbox(bbox_str, xyxy=True):
    x1, y1, x2, y2 = tuple(map(int, bbox_str.split(",")))
    return [x1, y1, x2, y2] if xyxy else [x1, y1, x2 - x1, y2 - y1]


def calc_bbox_iou(pred, gt):
    # bboxの座標を取得
    pred_x, pred_y, pred_w, pred_h = pred
    gt_x, gt_y, gt_w, gt_h = gt

    # bboxの形式を (x1, y1, x2, y2) に変換
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_x, pred_y, pred_x + pred_w, pred_y + pred_h
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_x, gt_y, gt_x + gt_w, gt_y + gt_h

    # 交差の矩形の座標を計算
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    # 交差の矩形の面積を計算
    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 各bboxの面積を計算
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    union = float(pred_area + gt_area - intersection)
    return intersection / union


def get_label_id(categories, label_name):
    for category in categories:
        if category["name"] == label_name:
            return category["id"]
    return


def save_annotation(anno, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fw:
        json.dump(anno, fw, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
