import argparse
import json
from collections import Counter
from pathlib import Path
from pprint import pprint

from matplotlib import pyplot as plt


def parser_args():
    parser = argparse.ArgumentParser(description="Count categories in COCO annotation file")
    parser.add_argument("annotation_files", nargs="+", type=Path, help="COCO annotation json")
    parser.add_argument("--output", type=Path, help="Output png file path")
    return parser.parse_args()


def main():
    args = parser_args()
    if args.output is None:
        args.output = args.annotation_files[0].with_suffix(".png")

    total_category2count = Counter()
    for annotation_file in args.annotation_files:
        coco_dataset = read_coco_json(annotation_file)
        annotations = coco_dataset["annotations"]
        id2category = coco_dataset["id2category"]

        category2count = count_categories(annotations, id2category)
        print(f"Category count in {annotation_file}")
        pprint(category2count)

        total_category2count += Counter(category2count)
    total_category2count = dict(total_category2count)
    print("Total category count")
    pprint(total_category2count)
    plot_count(
        total_category2count.keys(),
        total_category2count.values(),
        save_path=args.output,
    )


def read_coco_json(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)
    images = data["images"]
    annotations = data["annotations"]
    id2category = {category["id"]: category["name"] for category in data["categories"]}
    return {"images": images, "annotations": annotations, "id2category": id2category}


def count_categories(annotations, id2category):
    category2count = {category: 0 for category in id2category.values()}
    for annotation in annotations:
        category = id2category[annotation["category_id"]]
        category2count[category] += 1
    return category2count


def plot_count(category, counts, save_path):
    plt.bar(category, counts)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.title(f"Category count (Total: {sum(counts)})")
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    main()
