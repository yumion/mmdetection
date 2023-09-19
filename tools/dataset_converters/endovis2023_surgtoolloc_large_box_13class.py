import json
from pathlib import Path

import cv2
import numpy as np
from lxml import etree
from PIL import Image
from tqdm import tqdm


def main():
    # convert CVAT 1.1 box annotation to each class of instance instruments dataset
    # SurgToolLoc(EndoVis2023) -> COCO format Instance segmentation
    parent_dir = Path("/data1/shared/miccai/EndoVis2023/SurgToolLoc/v1.3")
    annotation_parent_dir = Path(
        "/data2/src/atsushi/JmeesBrain/projects/Annotation/data/cvat/annotations/xml/miccai2023-box"
    )
    child_dir_ptn = "*_clip_*"
    seg_dir_name = "mask"
    save_file = "coco_large_box_13classes_anno.json"

    # semantic mask to instance id
    cmap = {
        1: 1,  # clasper
        2: 1,  # wrist
        3: 1,  # shaft
        4: 2,  # clasper
        5: 2,  # wrist
        6: 2,  # shaft
        7: 3,  # clasper
        8: 3,  # wrist
        9: 3,  # shaft
        10: 4,  # clasper
        11: 4,  # wrist
        12: 4,  # shaft
    }

    clasper_or_wrist = {
        "bipolar_forceps": [2, 5, 8, 11],  # wrist
        "cadiere_forceps": [2, 5, 8, 11],  # wrist
        "clip_applier": [2, 5, 8, 11],  # wrist
        "force_bipolar": [2, 5, 8, 11],  # wrist
        "grasping_retractor": [1, 4, 7, 10],  # clasper
        # "monopolar_curved_scissors": [1, 4, 7, 10, 3, 6, 9, 12],  # clasper and shaft
        "monopolar_curved_scissors": [1, 4, 7, 10],  # clasper
        "needle_driver": [2, 5, 8, 11],  # wrist
        "permanent_cautery_hook_spatula": [2, 5, 8, 11],  # wrist
        "prograsp_forceps": [2, 5, 8, 11],  # wrist
        "stapler": [1, 4, 7, 10, 2, 5, 8, 11],  # clasper and wrist
        "suction_irrigator": [1, 4, 7, 10],  # clasper
        "tip_up_fenestrated_grasper": [1, 4, 7, 10, 2, 5, 8, 11],  # clasper and wrist
        "vessel_sealer": [2, 5, 8, 11],  # wrist
        "other": [-1],  # ignore
        "bipolar_dissector": [-1],  # ignore
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
    for task_dir in parent_dir.glob(child_dir_ptn):
        if not task_dir.is_dir():
            print(f"{task_dir} is not a directory")
            continue

        task_name = "_".join(task_dir.name.split("_")[1:])
        if task_name not in [
            "clip_000000-000299",
            "clip_000300-000599",
            "clip_000600-000899",
            "clip_000900-001199",
            "clip_001200-001499",
            "clip_001500-001799",
            "clip_001800-002099",
            "clip_002100-002399",
            "clip_002400-002699",
            "clip_002700-002999",
            "clip_003000-003300",
            "clip_003301-003600",
            "clip_003601-003901",
            "clip_003902-004201",
            "clip_004202-004501",
            "clip_004502-004802",
            "clip_004803-005102",
            "clip_005103-005402",
            "clip_005403-005703",
            "clip_005704-006003",
            "clip_006004-006303",
            "clip_006304-006604",
            "clip_006605-006904",
            "clip_006905-007204",
            "clip_007205-007504",
            "clip_007505-007804",
            "clip_007805-008105",
            "clip_008106-008405",
            "clip_008406-008705",
            "clip_008706-009006",
            "clip_009007-009307",
            "clip_009308-009608",
            # "clip_009609-009908",
            # "clip_009909-010209",
            # "clip_010210-010510",
            # "clip_010511-010810",
            # "clip_010811-011110",
            # "clip_011111-011410",
            # "clip_011411-011710",
            # "clip_024025-024325",
            "clip_024326-024625",
            "clip_024626-024720",
        ]:
            continue

        annotation_xml = next(annotation_parent_dir.glob(f"*{task_name}*.xml"))
        if annotation_xml is None:
            print(f"Annotation file of {task_name} is not found in {annotation_parent_dir}")
            continue
        annotation_dict = read_cvat_xml(annotation_xml)

        coco_dataset = create_coco_format(
            task_dir / seg_dir_name, cmap, categories, annotation_dict, clasper_or_wrist
        )
        save_annotation(coco_dataset, task_dir / save_file)


def read_cvat_xml(annotation_xml):
    annotations = {}

    tree = etree.parse(annotation_xml)
    root = tree.getroot()
    for image_tag in root.findall("image"):
        image_name = image_tag.attrib["name"]
        anno = []
        for box_tag in image_tag.findall("box"):
            label = box_tag.attrib["label"]
            bbox = _get_bbox_from_tag(box_tag)
            anno.append({"label": label, "bbox": bbox})
        annotations[image_name] = anno
    return annotations


def _get_bbox_from_tag(box_tag):
    x1 = float(box_tag.attrib["xtl"])
    y1 = float(box_tag.attrib["ytl"])
    x2 = float(box_tag.attrib["xbr"])
    y2 = float(box_tag.attrib["ybr"])
    # return x, y, w, h
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=int)


def create_coco_format(mask_parent_dir, cmap, categories, annotation_dict, clasper_or_wrist):
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
            panoptic_mask = np.array(Image.open(label_file).convert("P"))

            # image info
            pbar.set_postfix(dict(image=label_file.name))
            height, width = panoptic_mask.shape
            image_info = {
                "id": image_id,
                "file_name": label_file.name,  # maskとframeは同じファイル名
                "width": width,
                "height": height,
            }
            coco_dataset["images"].append(image_info)

            # annotation info
            pbar.set_description(mask_parent_dir.parent.name)
            instance_masks = index_remap(panoptic_mask, cmap)
            instance_uniques = np.unique(instance_masks)[1:]  # ignore background
            # label of instances
            anno = annotation_dict[label_file.name]
            if len(anno) == 0:
                print("No labels")
                continue
            # instance_masksからinstanceを取り出す
            for _id in instance_uniques:
                instance_mask = get_target_mask(instance_masks, _id)
                instrument_whole_bbox = mask2bbox(instance_mask)

                # determine which label is targeted
                target_instance_idx = np.argmax(
                    [
                        calc_bbox_iou(anno_instance["bbox"], instrument_whole_bbox)
                        for anno_instance in anno
                    ]
                )
                label = anno[target_instance_idx]["label"]
                label_id = get_label_id(categories, label)
                bbox = anno[target_instance_idx]["bbox"]

                # 先端か手首にbboxを絞る
                target_indexes = clasper_or_wrist[label]
                target_mask = (
                    instance_mask.astype(bool)
                    & np.isin(panoptic_mask, target_indexes)
                    & generate_box_mask(bbox, (width, height)).astype(bool)
                ).astype(np.uint8)
                area = calc_mask_area(target_mask)
                # 術具が写っていない場合真っ黒
                if area > 0:
                    annotation_info = {
                        "id": instance_id,
                        "image_id": image_id,
                        "category_id": label_id,
                        "bbox": instrument_whole_bbox,
                        "segmentation": [
                            contour.flatten().tolist() for contour in mask2contours(target_mask)
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


def generate_box_mask(bbox, img_size):
    img_w, img_h = img_size
    x, y, w, h = bbox
    box_mask = np.zeros((img_h, img_w))
    box_mask[y : y + h, x : x + w] = 1
    return box_mask


def calc_mask_area(mask):
    return int(np.sum(mask > 0))


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
    assert f"there is no category to match in {label_name}"


def save_annotation(anno, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fw:
        json.dump(anno, fw, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
