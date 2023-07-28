import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    # convert 6 + 1 instance instrument dataset
    # RoboticInstrumentSegmentation(EndoVis2017) -> COCO format Instance segmentation
    dataset_type = "train"
    parent_dir = Path(f"/data1/shared/miccai/EndoVis2017/{dataset_type}")
    child_dir_ptn = "instrument_dataset_*"
    seg_dir_name = "ground_truth"
    save_file = "coco.json"

    categories = [
        {"id": 0, "name": "prograsp forceps"},
        {"id": 1, "name": "bipolar forceps"},
        {"id": 2, "name": "needle driver"},
        {"id": 3, "name": "grasping reactor"},
        {"id": 4, "name": "vessel sealer"},
        {"id": 5, "name": "monopolar curved scissors"},
        {"id": 6, "name": "other"},
    ]

    create_coco_format(parent_dir, child_dir_ptn, seg_dir_name, save_file, categories)


def create_coco_format(parent_dir, child_dir_ptn, seg_dir_name, save_file, categories):
    # 動画ごとにcoco.jsonを作成する
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        coco_dataset = {
            "images": [],
            "annotations": [],
            "categories": categories,
        }

        # instanceごと
        sub_seg_dirs = list((video_dir / seg_dir_name).iterdir())
        instance_id = 0
        with tqdm(enumerate(sub_seg_dirs[0].glob("*.png"))) as pbar:
            for image_id, label_file in pbar:
                pbar.set_postfix(dict(image=label_file.name))

                height, width = cv2.imread(str(label_file), 0).shape
                image_info = {
                    "id": image_id,
                    "file_name": label_file.name,
                    "width": width,
                    "height": height,
                }
                coco_dataset["images"].append(image_info)
                for subdir in sub_seg_dirs:
                    this_category_name = subdir.name.lower().replace("_", " ")
                    pbar.set_description(f"{video_dir.name}/{this_category_name}")
                    for cat in categories:
                        if cat["name"] in this_category_name:
                            mask = cv2.imread(str(subdir / label_file.name), 0)
                            area = sum_area(mask)
                            # 術具が写っていない場合真っ黒
                            if area > 0:
                                annotation_info = {
                                    "id": instance_id,
                                    "image_id": image_id,
                                    "category_id": cat["id"],
                                    "bbox": mask2bbox(mask),
                                    "segmentation": [contour.flatten().tolist() for contour in mask2contours(mask)],
                                    "area": area,
                                    "iscrowd": 0,
                                }
                                coco_dataset["annotations"].append(annotation_info)
                                instance_id += 1
        save_annotation(coco_dataset, video_dir / save_file)


def mask2bbox(mask):
    contours = mask2contours(mask)
    all_contours = np.concatenate(contours)
    xmin, ymin = all_contours.min(axis=0)
    xmax, ymax = all_contours.max(axis=0)
    return list(map(int, [xmin, ymin, xmax, ymax]))


def mask2contours(mask):
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
