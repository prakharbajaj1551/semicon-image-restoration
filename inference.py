"""Restore every image in a folder using a trained checkpoint.

    python inference.py --ckpt checkpoints/best.pth --input data/test/lq --output outputs/restored

No code editing needed: the checkpoint file stores the model's own
configuration (width, blocks, scale), so the right network is rebuilt
automatically no matter which experiment produced it.

WHY images are processed in TILES (default 256x256):
    A 2000x2000 input at 4x scale would need several GB of GPU memory in
    one go. Instead we cut the input into overlapping tiles, restore each
    tile, and stitch the results. Small images (<= one tile) skip this
    entirely. The tiles OVERLAP because pixels at a tile's edge see less
    context and restore slightly worse — overlapping lets every stitched
    pixel come from a tile where it was comfortably inside.
"""

import argparse
import time
from pathlib import Path

import torch

from models.nafnet import NAFNetSR
from utils.image_io import list_images, load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Restore a folder of degraded images")
    parser.add_argument("--ckpt", default="checkpoints/best.pth",
                        help="trained checkpoint (best.pth)")
    parser.add_argument("--input", required=True, help="folder of degraded images")
    parser.add_argument("--output", default="outputs/restored",
                        help="folder for restored images")
    parser.add_argument("--tile", type=int, default=256,
                        help="tile size for large images; 0 = whole image at once")
    parser.add_argument("--overlap", type=int, default=32,
                        help="how much neighbouring tiles overlap")
    return parser.parse_args()


def load_model(ckpt_path, device):
    """Rebuild the network from the config saved inside the checkpoint,
    then load the trained weights into it."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = NAFNetSR(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["config"]


@torch.no_grad()   # inference only — no gradients means less memory, more speed
def restore_image(model, degraded, scale, device, tile, overlap):
    """Restore one (C,H,W) image, tiling it if it is larger than `tile`."""
    channels, height, width = degraded.shape

    # Small image: one pass, no tiling needed.
    if tile <= 0 or (height <= tile and width <= tile):
        restored = model(degraded.unsqueeze(0).to(device))  # add batch dim
        return restored.squeeze(0).clamp(0, 1).cpu()

    # Large image: restore overlapping tiles and paste them into a canvas.
    canvas = torch.zeros(channels, height * scale, width * scale)
    step = tile - overlap

    # All tile top-left corners; the extra last position guarantees the
    # bottom/right borders are covered even when sizes don't divide evenly.
    row_starts = list(range(0, max(height - tile, 0) + 1, step))
    col_starts = list(range(0, max(width - tile, 0) + 1, step))
    if row_starts[-1] + tile < height:
        row_starts.append(height - tile)
    if col_starts[-1] + tile < width:
        col_starts.append(width - tile)

    for top in row_starts:
        for left in col_starts:
            patch = degraded[:, top:top + tile, left:left + tile]
            restored = model(patch.unsqueeze(0).to(device))
            restored = restored.squeeze(0).clamp(0, 1).cpu()
            # Paste at the scaled-up position. Later tiles overwrite the
            # overlap zone — those pixels sit deeper inside the later tile,
            # so the overwrite is the better-restored version.
            canvas[:, top * scale:(top + tile) * scale,
                   left * scale:(left + tile) * scale] = restored
    return canvas


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model(args.ckpt, device)
    print(f"model loaded from {args.ckpt} "
          f"(scale x{config['scale']}, width {config['width']}, "
          f"{config['n_blocks']} blocks) on {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    image_paths = list_images(args.input)
    for path in image_paths:
        degraded = load_image(path, config["channels"])

        start = time.time()
        restored = restore_image(model, degraded, config["scale"],
                                 device, args.tile, args.overlap)
        seconds = time.time() - start
        total_time += seconds

        # Keep the input's format: .npy stays .npy (float precision, the
        # organisers' submission format), images become PNG (lossless —
        # JPEG would add its own artifacts to our restored pixels).
        extension = ".npy" if path.suffix.lower() == ".npy" else ".png"
        save_image(restored, out_dir / f"{path.stem}{extension}")
        print(f"{path.name}  {tuple(degraded.shape[-2:])} -> "
              f"{tuple(restored.shape[-2:])}  in {seconds:.2f}s")

    average = total_time / max(1, len(image_paths))
    print(f"done: {len(image_paths)} images in {total_time:.1f}s "
          f"(average {average:.2f}s per image) -> {out_dir}")


if __name__ == "__main__":
    main()
