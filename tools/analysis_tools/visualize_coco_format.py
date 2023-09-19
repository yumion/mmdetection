import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hex2color
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.mask import decode
from tqdm import tqdm


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
    parser.add_argument(
        "--img-ids",
        nargs="+",
        type=int,
        help="Image IDs to visualize.",
    )
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
    parser.add_argument(
        "--draw-mask",
        action="store_true",
        help="Overlay mask to image.",
    )
    return parser.parse_args()


def main():
    args = parser_args()

    annotation_file = args.data_root / args.annotation
    image_directory = args.data_root / args.image
    args.out.mkdir(parents=True, exist_ok=True)

    visualizer = CocoVisualizer(annotation_file, image_directory, args.out, args.draw_mask)

    if args.img_ids is None:
        visualizer.visualize_all_images()
    else:
        for img_id in args.img_ids:
            visualizer.visualize_image(img_id)


class CocoVisualizer:
    def __init__(self, annotation_file, image_directory, output_directory, do_draw_mask=False):
        self.coco = COCO(annotation_file)
        self.image_directory = image_directory
        self.output_directory = output_directory
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.do_draw_mask = do_draw_mask

    def visualize_image(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        print(img_info)

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

            # polygon
            for polygon in polygons:
                poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                plt.plot(poly[:, 0], poly[:, 1], color=color, linewidth=4)

            # mask
            # annToRLEのデバッグ用
            if self.do_draw_mask:
                # RLEからバイナリマスクへの変換
                rle = self.coco.annToRLE(ann)
                binary_mask = decode(rle)
                # マスク描画（指定されたRGB色で描画）
                colored_mask = np.zeros_like(image)
                colored_mask[binary_mask == 1] = np.array(hex2color(color)) * 255
                ax.imshow(colored_mask, alpha=0.5)

            # bbox
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

            # label
            plt.text(
                bbox[0],
                bbox[1],
                label_name,
                color=color,
                fontsize=10,
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0),
            )

        output_path = self.output_directory / f"img{img_id:06d}_{img_info['file_name']}"
        plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    def visualize_all_images(self):
        img_ids = self.coco.getImgIds()
        for img_id in tqdm(img_ids):
            self.visualize_image(img_id)


if __name__ == "__main__":
    main()
