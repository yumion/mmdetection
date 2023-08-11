import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO


def parser_args():
    parser = argparse.ArgumentParser(description="Visualize COCO annotations.")
    parser.add_argument(
        "data_root",
        type=Path,
        help="Path to the annotation dataset root directory.",
    )
    parser.add_argument(
        "out",
        type=Path,
        help="Path to the directory for saving output images.",
    )
    parser.add_argument("--img-ids", nargs="+", type=int, help="Image IDs to visualize.")
    parser.add_argument(
        "--annotation",
        type=str,
        help="COCO annotation file name (JSON).",
        default="coco.json",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Directory name containing images.",
        default="images",
    )
    return parser.parse_args()


def main():
    args = parser_args()

    annotation_file = args.data_root / args.annotation
    image_directory = args.data_root / args.image
    args.out.mkdir(parents=True, exist_ok=True)

    visualizer = CocoVisualizer(annotation_file, image_directory, args.out)

    if args.img_ids is None:
        visualizer.visualize_all_images()
    else:
        for img_id in args.img_ids:
            visualizer.visualize_image(img_id)


class CocoVisualizer:
    def __init__(self, annotation_file, image_directory, output_directory):
        self.coco = COCO(annotation_file)
        self.image_directory = image_directory
        self.output_directory = output_directory
        self.colors = list(mcolors.TABLEAU_COLORS.values())

    def visualize_image(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        image_path = self.image_directory / img_info["file_name"]
        image = Image.open(image_path)
        width, height = image.size

        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)

        for idx, ann in enumerate(anns):
            color = self.colors[idx % len(self.colors)]
            polygons = ann["segmentation"]
            label_name = self.coco.loadCats(ann["category_id"])[0]["name"]

            for polygon in polygons:
                poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                plt.plot(poly[:, 0], poly[:, 1], color=color, linewidth=4)

            bbox = ann["bbox"]
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            plt.gca().add_patch(rect)

            plt.text(
                bbox[0],
                bbox[1],
                label_name,
                color=color,
                fontsize=10,
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0),
            )

        output_path = self.output_directory / f"img{img_id}_{img_info['file_name']}.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    def visualize_all_images(self):
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            self.visualize_image(img_id)


if __name__ == "__main__":
    main()
