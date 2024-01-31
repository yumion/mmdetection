import argparse
import os
from glob import glob
from pathlib import Path

import cv2
from mmdet.apis import DetInferencer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet inferencer model")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file")
    parser.add_argument(
        "device", type=int, default=0, help="device used for inference. `-1` means using cpu."
    )
    parser.add_argument(
        "--target_dir",
        "--target-dir",
        type=str,
        help="input directory of images to be predicted. It can be used wildcard.",
    )
    parser.add_argument(
        "--show_dir",
        "--show-dir",
        type=Path,
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument(
        "--pred_score_thr",
        "--pred-score-thr",
        type=float,
        default=0.3,
        help="Minimum score of bboxes to draw. Defaults to 0.3.",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    args = parse_args()
    assert (
        args.show_dir is not None or args.out is not None
    ), "Either --show-dir or --out should be specified."

    model = DetInferencer(
        weights=str(args.checkpoint),
        device=f"cuda:{args.device}" if args.device >= 0 else "cpu",
        show_progress=False,
    )

    with tqdm(sorted(glob(f"{args.target_dir}/*.png", recursive=True))) as pbar:
        for img_path in pbar:
            verbose = {}
            img_path = Path(img_path)

            # inputの画像のフォルダ構成を保ったまま保存する
            parent_dir = img_path.parent.name
            if "*" in args.target_dir:
                parent_dir = str(img_path.parent).replace(args.target_dir.split("*")[0], "")
            pbar.set_description(f"{parent_dir}/{img_path.name}")

            # --outの場合、APIで結果をjsonで保存するためパスを指定する
            if args.out is not None:
                out_dir = args.out / parent_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                verbose["pred"] = str(out_dir)
            # --showの場合、APIで画像を保存せず描画した画像を受け取って自分で保存するためout_dirは""にする
            else:
                out_dir = ""

            result = model(
                inputs=str(img_path),
                return_vis=args.show_dir is not None,
                no_save_vis=args.show_dir is None,
                pred_score_thr=args.pred_score_thr,
                no_save_pred=args.out is None,
                out_dir=out_dir,
            )

            if args.show_dir is not None:
                show_dir = args.show_dir / parent_dir / "vis"
                show_dir.mkdir(parents=True, exist_ok=True)
                blend = result["visualization"][0]
                cv2.imwrite(str(show_dir / img_path.name), blend[..., ::-1])
                verbose["show"] = str(show_dir / img_path.name)

            pbar.set_postfix(verbose)


if __name__ == "__main__":
    main()
