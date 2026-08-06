"""Score a folder of restored images against their ground-truth folder.

    python evaluate.py --restored outputs/restored --gt data/test/gt

Prints PSNR / SSIM / LPIPS per image plus the means, and writes
outputs/metrics.csv — the means go straight into the README and slides.

Images pair up by filename: restored/chip_07.png <-> gt/chip_07.png.

WHY a separate script (instead of scoring inside train.py):
    Judges and teammates can verify results without touching training —
    point it at any two folders and get numbers. It also works on OTHER
    methods' outputs (e.g. plain bicubic upscaling), which is how we
    build the "ours vs baseline" comparison for the slides.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from utils.image_io import list_images, load_image
from utils.metrics import LPIPSMetric, psnr, ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Score restored images against ground truth")
    parser.add_argument("--restored", required=True, help="folder of restored images")
    parser.add_argument("--gt", required=True, help="folder of ground-truth images")
    parser.add_argument("--csv", default="outputs/metrics.csv",
                        help="where to write the per-image table")
    parser.add_argument("--no-lpips", action="store_true",
                        help="skip LPIPS (avoids the one-time model download)")
    return parser.parse_args()


def main():
    args = parse_args()

    # LPIPS runs a neural net, so use the GPU if there is one.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_metric = None if args.no_lpips else LPIPSMetric(device=device)

    # Look up restored images by filename so pairing is order-independent.
    restored_by_name = {p.stem: p for p in list_images(args.restored)}

    rows = []
    for gt_path in list_images(args.gt):
        if gt_path.stem not in restored_by_name:
            print(f"skipping {gt_path.name}: no restored image with that name")
            continue

        truth = load_image(gt_path)
        result = load_image(restored_by_name[gt_path.stem])

        # Guard against 1-2 pixel size mismatches (e.g. odd-sized inputs):
        # compare only the overlapping region instead of crashing.
        height = min(truth.shape[-2], result.shape[-2])
        width = min(truth.shape[-1], result.shape[-1])
        truth = truth[:, :height, :width]
        result = result[:, :height, :width]

        row = {"name": gt_path.stem,
               "psnr": psnr(result, truth),
               "ssim": ssim(result, truth)}
        if lpips_metric is not None:
            row["lpips"] = lpips_metric(result, truth)
        rows.append(row)

        print("  ".join(f"{key}={value:.4f}" if key != "name" else value
                        for key, value in row.items()))

    if not rows:
        raise SystemExit("No matching image pairs found — check the two folders.")

    # Means: the headline numbers for the README/slides.
    metric_names = [key for key in rows[0] if key != "name"]
    print("-" * 48)
    for name in metric_names:
        direction = "lower=better" if name == "lpips" else "higher=better"
        print(f"mean {name.upper():5s} = "
              f"{np.mean([row[name] for row in rows]):.4f}   ({direction})")

    # CSV so results can be inspected in Excel / attached to the submission.
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-image table written to {csv_path}")


if __name__ == "__main__":
    main()
