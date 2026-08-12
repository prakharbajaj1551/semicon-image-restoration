"""Turn any folder of natural images into training pairs matching KLA's data.

    python make_external_pairs.py --images path/to/DIV2K --out data/external

WHY this exists (out-of-distribution robustness):
    KLA's hidden test set contains BOTH in-distribution content (similar to
    the training images) and out-of-distribution content - urban scenes,
    architecture, object classes never seen in training - and reconstruction
    quality is scored on both. A model that has only ever seen 3,000 images
    from one distribution is at its weakest exactly where those OOD samples
    live. The rules explicitly permit external datasets for this reason.

WHY these pairs are trustworthy:
    We did not guess the degradation. We recovered it from the organisers'
    own pairs (see dataset/degradation.py): plain bicubic downsampling with
    no antialiasing, speckle sigma ~0.155 and Gaussian sigma ~0.029, and
    inputs deliberately left unclamped. Our synthetic inputs reproduce their
    statistics closely - mean 0.469 vs 0.469, std 0.212 vs 0.207.

Output matches the organisers' format exactly, so training needs no changes:
    out/gt/name_0.npy   256x256 float32 in [0,1]
    out/lq/name_0.npy   128x128 float32, values allowed outside [0,1]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset.degradation import DegradationSettings, degrade_image
from src.utils.image_io import save_image

# Pillow can open these; .npy is excluded on purpose (external sets are photos)
PHOTO_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True,
                        help="folder of external images (searched recursively)")
    parser.add_argument("--out", default="data/external",
                        help="where to write gt/ and lq/")
    parser.add_argument("--tile", type=int, default=256,
                        help="ground-truth tile size (KLA's GT is 256x256)")
    parser.add_argument("--per-image", type=int, default=4,
                        help="how many tiles to cut from each source image")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many source images (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def to_grayscale_tensor(path):
    """Load a colour photo as a single-channel float tensor in [0,1].

    WHY grayscale: KLA's data is single-channel, so colour would be a
    distribution mismatch, not extra information.
    """
    img = Image.open(path).convert("L")
    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, :, :]


def main():
    args = parse_args()
    settings = DegradationSettings(scale=args.scale)
    rng = np.random.default_rng(args.seed)

    sources = sorted(p for p in Path(args.images).rglob("*")
                     if p.suffix.lower() in PHOTO_TYPES)
    if not sources:
        raise SystemExit(f"No images found under {args.images}")
    if args.limit:
        sources = sources[:args.limit]

    out = Path(args.out)
    # np.save (used below for the unclamped input) does not create folders,
    # unlike our save_image helper, so make both up front.
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "lq").mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for index, path in enumerate(sources):
        clean = to_grayscale_tensor(path)
        height, width = clean.shape[-2:]
        if height < args.tile or width < args.tile:
            skipped += 1                   # too small to cut a full tile from
            continue

        for k in range(args.per_image):
            top = int(rng.integers(0, height - args.tile + 1))
            left = int(rng.integers(0, width - args.tile + 1))
            tile = clean[:, top:top + args.tile, left:left + args.tile]

            # Seeded per tile so the whole set is reproducible.
            degraded = degrade_image(tile, settings,
                                     seed=args.seed * 1000003 + index * 16 + k)

            name = f"{path.stem}_{k}.npy"
            save_image(tile, out / "gt" / name)          # clipped to [0,1]: correct
            np.save(out / "lq" / name,                   # NOT clipped: matches KLA
                    degraded[0].numpy().astype(np.float32))
            written += 1

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(sources)} images -> {written} pairs")

    if skipped:
        print(f"\nskipped {skipped} image(s) smaller than {args.tile}x{args.tile} px")
    if not written:
        raise SystemExit(
            f"No pairs written. Every source image was smaller than "
            f"{args.tile}x{args.tile}. Use larger images (DIV2K is 2K) "
            f"or pass a smaller --tile.")

    print(f"\n{written} pairs written to {out}/ (gt/ and lq/)")
    print("train on them together with the organisers' data, e.g.:")
    print(f"  python train.py --mode paired \\\n"
          f"      --train-lq {out}/lq --train-gt {out}/gt \\\n"
          f"      --val-lq data/val/lq --val-gt data/val/gt \\\n"
          f"      --scale {args.scale} --epochs 60 --resume weights/model.pth")


if __name__ == "__main__":
    main()
