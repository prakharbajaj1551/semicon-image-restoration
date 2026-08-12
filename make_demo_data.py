"""Create a small demo dataset so anyone can run this project immediately.

    python make_demo_data.py

WHY this script exists:
    The hackathon dataset is provided by the organisers and is not part of
    this repository. Without it, someone who clones the repo has the model
    but nothing to run it on. This script generates a handful of synthetic
    clean images, degrades them with the same corruption pipeline used in
    training, and writes them to demo_data/ — so `inference.py`,
    `evaluate.py` and `app.py` all work out of the box.

    These are SYNTHETIC images for a smoke test. The results reported in
    the README come from the organisers' real held-out test pairs.
"""

import argparse

import numpy as np
import torch

from src.dataset.degradation import DegradationSettings, degrade_image
from src.utils.image_io import save_image


def make_clean_image(index, size=256):
    """Build one synthetic 'inspection-like' image: periodic line/via patterns.

    WHY periodic patterns: real semiconductor images are dominated by
    repeated dies, traces and vias. Straight edges and regular spacing are
    exactly what a restoration model must preserve, so they make an honest
    smoke test.
    """
    rng = np.random.default_rng(index)
    y, x = np.mgrid[0:size, 0:size]

    # horizontal + vertical line grid (traces)
    pitch = int(rng.integers(12, 28))
    image = 0.5 + 0.25 * np.sign(np.sin(2 * np.pi * x / pitch))
    image += 0.15 * np.sign(np.sin(2 * np.pi * y / (pitch * 2)))

    # a few bright square "vias"
    for _ in range(rng.integers(3, 8)):
        cy, cx = rng.integers(20, size - 20, size=2)
        half = int(rng.integers(4, 10))
        image[cy - half:cy + half, cx - half:cx + half] = 0.95

    # gentle illumination gradient, as in real microscopy
    image = image * (0.85 + 0.3 * y / size)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6, help="how many images")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--out", default="demo_data")
    args = parser.parse_args()

    settings = DegradationSettings(scale=args.scale)

    for index in range(args.count):
        clean = torch.from_numpy(make_clean_image(index))[None, :, :]
        # fixed seed per image -> everyone gets identical demo files
        degraded = degrade_image(clean, settings, seed=1000 + index)

        name = f"demo_{index:02d}.npy"
        save_image(clean, f"{args.out}/gt/{name}")
        save_image(degraded, f"{args.out}/lq/{name}")
        print(f"{name}: lq {tuple(degraded.shape[-2:])} -> gt {tuple(clean.shape[-2:])}")

    print(f"\ndemo data written to {args.out}/ (lq/ and gt/)\n"
          f"try it:\n"
          f"  python inference.py --ckpt weights/model.pth "
          f"--input {args.out}/lq --output outputs/demo\n"
          f"  python evaluate.py --restored outputs/demo --gt {args.out}/gt\n"
          f"\nNOTE: these are SYNTHETIC smoke-test images, deliberately\n"
          f"unlike the training data (hard geometric edges). Scores here\n"
          f"run ~6 dB BELOW our reported benchmark and are NOT comparable\n"
          f"to it. The README's 28.00 dB comes from the organisers' real\n"
          f"held-out test pairs. This script only proves the code runs.")


if __name__ == "__main__":
    main()
