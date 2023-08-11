import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    # convert 1 instance instrument dataset
    # RoboticInstrumentSegmentation(EndoVis2017) -> COCO format Instance segmentation
    parent_dir = Path("/data1/shared/miccai/EndoVis2017/train")
    child_dir_ptn = "instrument_dataset_*"
    seg_dir_name = "ground_truth"
    save_file = "coco.json"

    categories = [
        {"id": 0, "name": "instrument"},
    ]

    create_coco_format(parent_dir, child_dir_ptn, seg_dir_name, save_file, categories)


def create_coco_format(parent_dir, child_dir_ptn, seg_dir_name, save_file, categories):
    # 評価用にすべてをまとめて1つのcoco.jsonを作成する
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    # 通し番号
    image_id = 0
    instance_id = 0
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        sub_seg_dirs = list((video_dir / seg_dir_name).iterdir())
        with tqdm(sub_seg_dirs[0].glob("*.png")) as pbar:
            for label_file in pbar:
                # image info
                image_file = video_dir / "left_frames" / label_file.name
                pbar.set_postfix(dict(image=image_file.name))
                # 同じフォルダに画像を保存するのでファイル名を変える
                new_image_path = parent_dir / "images" / f"{video_dir.name}_{image_file.name}"
                # 共通フォルダにコピーする
                if new_image_path.is_symlink():
                    new_image_path.unlink()
                new_image_path.symlink_to(image_file)

                height, width = cv2.imread(str(image_file), 0).shape
                image_info = {
                    "id": image_id,
                    "file_name": new_image_path.name,
                    "width": width,
                    "height": height,
                }
                coco_dataset["images"].append(image_info)

                # annotation info
                for subdir in sub_seg_dirs:
                    this_category_name = subdir.name.lower().replace("_", " ")
                    # otherのinstanceは術具ではないので無視する
                    if "other" in this_category_name:
                        continue
                    pbar.set_description(f"{video_dir.name}/{this_category_name}")

                    mask = cv2.imread(str(subdir / label_file.name), 0)
                    area = sum_area(mask)
                    if area > 0:
                        annotation_info = {
                            "id": instance_id,
                            "image_id": image_id,
                            "category_id": 0,
                            "bbox": mask2bbox(mask),
                            "segmentation": [
                                contour.flatten().tolist() for contour in mask2contours(mask)
                            ],
                            "area": area,
                            "iscrowd": 0,
                        }
                        coco_dataset["annotations"].append(annotation_info)
                        instance_id += 1
                image_id += 1
    save_annotation(coco_dataset, parent_dir / save_file)


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
    return [contour[:, 0] for contour in contours if len(contour) > 1]  # 1点の輪郭を除外


def sum_area(mask):
    return int(np.sum(mask > 0))


def save_annotation(anno, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fw:
        json.dump(anno, fw, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
