"""Build a frozen degraded test set (plus a bicubic baseline) from clean images.

Two situations:

A) You only have CLEAN images — degrade them yourself:
    python make_testset.py --clean data/test/clean --out data/test --scale 4

B) You already have degraded/clean PAIRS (organiser data) and only need
   the bicubic baseline folder for the comparison table:
    python make_testset.py --lq data/test/lq --gt data/test/gt --out data/test

Creates (situation A creates all three, B only the last one):
    data/test/gt/       ground truth (clean, trimmed to a multiple of scale)
    data/test/lq/       degraded inputs (low-res + speckle + Gaussian noise)
    data/test/bicubic/  lq upscaled with plain bicubic — the BASELINE

WHY a frozen test set (fixed random seed)?
    Training data uses fresh random noise every step — good for learning,
    terrible for benchmarking: scores would change every run. Here every
    image gets noise from a FIXED seed, so all teammates, all experiments
    and all report numbers use the exact same degraded pixels. Fair,
    reproducible comparison.

WHY the bicubic folder?
    "Our model: 32 dB" means nothing alone. "Plain upscaling: 24 dB, our
    model: 32 dB" is the story. evaluate.py can score the bicubic folder
    exactly like ours, giving the baseline row of the results table.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset.degradation import DegradationSettings, degrade_image
from utils.image_io import list_images, load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Create a reproducible degraded benchmark")
    parser.add_argument("--clean", help="folder of clean images (situation A)")
    parser.add_argument("--lq", help="existing degraded folder (situation B)")
    parser.add_argument("--gt", help="existing ground-truth folder (situation B)")
    parser.add_argument("--out", required=True, help="output folder (gt/, lq/, bicubic/)")
    parser.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=0,
                        help="change to generate a different (but again frozen) test set")
    return parser.parse_args()


def make_baseline_from_pairs(args):
    """Situation B: pairs already exist — only build the bicubic folder.

    The scale is read from the images themselves (gt width / lq width),
    so there is nothing to configure or get wrong.
    """
    out = Path(args.out) / "bicubic"
    out.mkdir(parents=True, exist_ok=True)

    gt_by_name = {p.stem: p for p in list_images(args.gt)}
    for path in list_images(args.lq):
        degraded = load_image(path)
        if path.stem not in gt_by_name:
            print(f"skipping {path.name}: no ground truth with that name")
            continue
        truth = load_image(gt_by_name[path.stem])
        s = round(truth.shape[-1] / degraded.shape[-1])   # scale from the data

        if s > 1:
            baseline = F.interpolate(degraded.unsqueeze(0), scale_factor=s,
                                     mode="bicubic", align_corners=False)
            baseline = baseline.squeeze(0).clamp(0, 1)
        else:
            baseline = degraded
        save_image(baseline, out / path.name)   # keep the input's format
        print(f"{path.name}: bicubic x{s} baseline written")

    print(f"baseline ready under {out}")


def main():
    args = parse_args()

    if args.lq and args.gt:                    # situation B: pairs exist
        make_baseline_from_pairs(args)
        return
    if not args.clean:
        raise SystemExit("Give either --clean (situation A) "
                         "or --lq and --gt together (situation B).")
    settings = DegradationSettings(scale=args.scale)

    out = Path(args.out)
    for name in ("gt", "lq", "bicubic"):
        (out / name).mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(list_images(args.clean)):
        clean = load_image(path)

        # Trim so height/width divide evenly by the scale — otherwise the
        # restored image would be a pixel or two off from the ground truth.
        s = args.scale
        clean = clean[:, : clean.shape[-2] // s * s, : clean.shape[-1] // s * s]

        # One fixed seed per image: image #3 gets the same noise on every
        # computer, every day. (Different images still get different noise.)
        degraded = degrade_image(clean, settings, seed=args.seed * 100003 + index)

        # The baseline: what you get WITHOUT any AI — just resize back up.
        if s > 1:
            baseline = F.interpolate(degraded.unsqueeze(0), scale_factor=s,
                                     mode="bicubic", align_corners=False)
            baseline = baseline.squeeze(0).clamp(0, 1)
        else:
            baseline = degraded          # scale 1: the noisy image itself

        save_image(clean, out / "gt" / f"{path.stem}.png")
        save_image(degraded, out / "lq" / f"{path.stem}.png")
        save_image(baseline, out / "bicubic" / f"{path.stem}.png")
        print(f"{path.name}: gt {tuple(clean.shape[-2:])} "
              f"-> lq {tuple(degraded.shape[-2:])}")

    print(f"test set ready under {out} (gt/, lq/, bicubic/)")


if __name__ == "__main__":
    main()
