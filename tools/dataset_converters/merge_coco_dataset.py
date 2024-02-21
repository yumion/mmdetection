import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple COCO format datasets")
    parser.add_argument("datasets", nargs="+", type=str, help="datasets to be merged")
    parser.add_argument("--out", type=Path, required=True, help="output path")
    parser.add_argument("--data-root", "--data_root", type=Path, help="root path of datasets")
    parser.add_argument(
        "--include-images", "--include_images", action="store_true", help="include images"
    )
    parser.add_argument(
        "--image-prefix", "--image_prefix", type=str, default="images", help="image prefix"
    )
    parser.add_argument(
        "--symlink", action="store_true", help="create symlink instead of copying images"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    new_dataset = update_coco_dataset(args.datasets, args.data_root)
    write_coco_dataset(new_dataset, args.out)
    if args.include_images:
        copy_images(args.datasets, args.out, args.data_root, args.image_prefix, args.symlink)


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
            original_image_id = image["id"]
            image["id"] = image_id
            for annotation in annotations:
                # find annotation_id corresponding original image_id
                if annotation["image_id"] != original_image_id:
                    continue
                annotation["image_id"] = image_id
                annotation["id"] = annotation_id
                annotation_id += 1
            image_id += 1
        # copy dataset to merge
        new_dataset["images"].extend(images)
        new_dataset["annotations"].extend(annotations)
        if not new_dataset["categories"]:
            new_dataset["categories"] = categories
    return new_dataset


def write_coco_dataset(new_dataset: Dict[str, List[dict]], out_path: Path) -> None:
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fw:
        json.dump(new_dataset, fw, indent=4, ensure_ascii=False)


def copy_images(
    datasets: List[Union[str, Path]],
    out_path: Path,
    data_root: Optional[Path] = None,
    image_prefix: str = "images",
    is_symlink: bool = False,
) -> None:
    out_images_dir = out_path.parent / image_prefix
    out_images_dir.mkdir(exist_ok=True, parents=True)
    for dataset_path in datasets:
        if data_root is not None:
            dataset_path = data_root / dataset_path
        images, _, _ = read_coco_dataset(dataset_path)
        for image in images:
            image_path = dataset_path.parent / image_prefix / image["file_name"]
            out_image_path = out_images_dir / image["file_name"]
            if is_symlink:
                out_image_path.symlink_to(image_path)
            else:
                out_image_path.write_bytes(image_path.read_bytes())


if __name__ == "__main__":
    main()
