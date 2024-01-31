import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple COCO format datasets")
    parser.add_argument("datasets", nargs="+", type=str, help="datasets to be merged")
    parser.add_argument("--data-root", "--data_root", type=Path, help="root path of datasets")
    parser.add_argument("--out", type=Path, required=True, help="output path")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    new_dataset = update_coco_dataset(args.datasets, args.data_root)
    write_coco_dataset(new_dataset, args.out)


def read_coco_dataset(
    dataset_path: Union[str, Path]
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], Dict[str, List[dict]]]:
    with open(dataset_path) as fr:
        dataset = json.load(fr)
    images = dataset["images"]
    annotations = dataset["annotations"]
    categories = dataset["categories"]
    return images, annotations, categories


def update_coco_dataset(
    datasets: List[Union[str, Path]], data_root: Optional[Path] = None
) -> Dict[str, List[dict]]:
    new_dataset = {"images": [], "annotations": [], "categories": []}
    image_id = 0  # consective number
    annotation_id = 0  # consective number

    for dataset_path in datasets:
        if data_root is not None:
            dataset_path = data_root / dataset_path
        images, annotations, categories = read_coco_dataset(dataset_path)
        # update image_id and annotation_id
        for image in images:
            image["id"] = image_id
            image_id += 1
        for annotation in annotations:
            annotation["image_id"] = image_id
            annotation["id"] = annotation_id
            annotation_id += 1
        # copy dataset to merge
        new_dataset["images"].extend(images)
        new_dataset["annotations"].extend(annotations)
        if not new_dataset["categories"]:
            new_dataset["categories"] = categories
    return new_dataset


def write_coco_dataset(new_dataset: Dict[str, List[dict]], out_path: str) -> None:
    assert str(out_path).endswith(".json"), "output path must be json file"

    assert (
        new_dataset.get("categories") is not None
        and new_dataset.get("images") is not None
        and new_dataset.get("annotations") is not None
    ), "COCO dataset must have categories, images and annotations"

    assert (
        len(new_dataset["images"]) > 0
        and len(new_dataset["annotations"]) > 0
        and len(new_dataset["categories"]) > 0
    ), "COCO dataset must have at least one image and annotation"

    with open(out_path, "w") as fw:
        json.dump(new_dataset, fw, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
