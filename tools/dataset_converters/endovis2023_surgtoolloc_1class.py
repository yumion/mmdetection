import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    # convert parts instance segmentation to one instance instrument dataset
    # SurgToolLoc(EndoVis2023) -> COCO format Instance segmentation
    parent_dir = Path("/data1/shared/miccai/EndoVis2023/SurgToolLoc/v0.2")
    child_dir_ptn = "*_clip_*"
    seg_dir_name = "mask"
    save_file = "coco.json"

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
        {"id": 0, "name": "instrument"},
    ]

    # taskごとにcoco.jsonを作成する
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        coco_dataset = create_coco_format(video_dir / seg_dir_name, cmap, categories)
        save_annotation(coco_dataset, video_dir / save_file)


def create_coco_format(mask_parent_dir, cmap, categories):
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
            # maskからinstanceを取り出す
            for _id in instance_uniques:
                instance_mask = get_target_mask(mask, _id)
                area = sum_area(instance_mask)
                # 術具が写っていない場合真っ黒
                if area > 0:
                    annotation_info = {
                        "id": instance_id,
                        "image_id": image_id,
                        "category_id": 0,
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


def sum_area(mask):
    return int(np.sum(mask > 0))


def save_annotation(anno, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fw:
        json.dump(anno, fw, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
