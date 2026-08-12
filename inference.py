"""Restore every image in a folder using a trained checkpoint.

    python inference.py --input data/test_submission --output outputs/submission

No code editing needed: the checkpoint stores the model's own configuration
(width, blocks, scale), so the right network is rebuilt automatically no
matter which experiment produced it.

WHY BATCHING (this is a scored criterion):
    Evaluators measure END-TO-END throughput: disk reading, preprocessing,
    CPU-to-GPU transfer, model execution, GPU-to-CPU transfer, post-
    processing and saving. Sending one image at a time leaves the GPU
    mostly idle waiting for transfers. Stacking N images into one tensor
    keeps it busy and cuts total time substantially.

    Images can only be batched together if they have the SAME size, so we
    group by shape first. Real test sets are usually one size, so this
    normally produces a single group.

WHY TILING is still here:
    A very large image at 2x scale can exhaust GPU memory in one pass. Any
    image bigger than --tile is cut into overlapping tiles, restored, and
    stitched. Tiles overlap because pixels at a tile edge see less context;
    overlapping means every stitched pixel comes from a tile where it sat
    comfortably inside.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.models.nafnet import NAFNetSR
from src.utils.image_io import list_images, load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Restore a folder of degraded images")
    parser.add_argument("--input", required=True, help="folder of degraded images")
    parser.add_argument("--output", required=True, help="folder for restored images")
    parser.add_argument("--ckpt", default="weights/model.pth",
                        help="trained checkpoint (best.pth)")
    # Default adapts to the device: batching hides transfer latency on a GPU,
    # but a CPU is already compute-bound and large batches only add memory
    # pressure (measured: batch 1 beats batch 8 by ~17% on CPU).
    parser.add_argument("--batch-size", type=int, default=None,
                        help="images processed together "
                             "(default: 8 on GPU, 1 on CPU); lower it if GPU memory runs out")
    parser.add_argument("--tile", type=int, default=512,
                        help="images larger than this are processed in tiles; 0 = never tile")
    parser.add_argument("--overlap", type=int, default=32,
                        help="how much neighbouring tiles overlap")
    parser.add_argument("--amp", action="store_true",
                        help="half-precision on GPU: faster, tiny numerical difference")
    return parser.parse_args()


def load_model(ckpt_path, device):
    """Rebuild the network from the config saved inside the checkpoint,
    then load the trained weights into it."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = NAFNetSR(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["config"]


@torch.no_grad()   # inference only — no gradients means less memory, more speed
def restore_batch(model, batch, device, use_amp=False):
    """Restore a stack of same-sized images: (N,C,H,W) in -> (N,C,sH,sW) out."""
    batch = batch.to(device, non_blocking=True)
    if use_amp and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            restored = model(batch)
        restored = restored.float()
    else:
        restored = model(batch)
    # Clip inside our own pipeline: KLA scores exactly what we save, and
    # ground truth lives in [0,1], so values outside it can only be wrong.
    return restored.clamp(0, 1).cpu()


@torch.no_grad()
def restore_image(model, degraded, scale, device, tile=512, overlap=32, use_amp=False):
    """Restore ONE (C,H,W) image, tiling it if it is larger than `tile`.

    Kept as a separate function because the Streamlit app restores single
    uploaded images, and because very large images must avoid batching.
    """
    channels, height, width = degraded.shape

    # Small enough: one pass, no tiling needed.
    if tile <= 0 or (height <= tile and width <= tile):
        return restore_batch(model, degraded.unsqueeze(0), device, use_amp).squeeze(0)

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
            restored = restore_batch(model, patch.unsqueeze(0), device,
                                     use_amp).squeeze(0)
            # Later tiles overwrite the overlap zone — those pixels sit
            # deeper inside the later tile, so they are the better version.
            canvas[:, top * scale:(top + tile) * scale,
                   left * scale:(left + tile) * scale] = restored
    return canvas


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.batch_size is None:
        args.batch_size = 8 if device.type == "cuda" else 1

    model, config = load_model(args.ckpt, device)
    scale = config["scale"]
    print(f"model: NAFNet-SR x{scale}, width {config['width']}, "
          f"{config['n_blocks']} blocks | device: {device.type} | "
          f"batch size: {args.batch_size} | amp: {args.amp}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(args.input)

    # ---- END-TO-END TIMING starts here: disk read is part of the runtime ----
    start_total = time.time()

    # 1. READ every image and group by shape (only same-sized images batch).
    start_read = time.time()
    groups = defaultdict(list)
    for path in image_paths:
        image = load_image(path, config["channels"])
        groups[tuple(image.shape)].append((path, image))
    read_seconds = time.time() - start_read

    # 2. RESTORE, then 3. SAVE — timed separately so the report can show
    #    where the time actually goes.
    compute_seconds = 0.0
    save_seconds = 0.0

    for shape, items in groups.items():
        too_big = args.tile > 0 and (shape[-2] > args.tile or shape[-1] > args.tile)

        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]

            t0 = time.time()
            if too_big:
                # Oversized images: one at a time, through the tiling path.
                results = [restore_image(model, img, scale, device, args.tile,
                                         args.overlap, args.amp)
                           for _, img in chunk]
            else:
                batch = torch.stack([img for _, img in chunk])
                results = list(restore_batch(model, batch, device, args.amp))
            if device.type == "cuda":
                torch.cuda.synchronize()   # GPU work is async; wait before timing
            compute_seconds += time.time() - t0

            t0 = time.time()
            for (path, _), restored in zip(chunk, results):
                # Keep the input's format: .npy stays .npy (the organisers'
                # format), images become PNG (lossless).
                suffix = ".npy" if path.suffix.lower() == ".npy" else ".png"
                save_image(restored, out_dir / f"{path.stem}{suffix}")
            save_seconds += time.time() - t0

    total_seconds = time.time() - start_total
    count = len(image_paths)

    print(f"\nrestored {count} images -> {out_dir}")
    print(f"  read + preprocess : {read_seconds:6.2f} s")
    print(f"  model + transfers : {compute_seconds:6.2f} s")
    print(f"  post + save       : {save_seconds:6.2f} s")
    print(f"  END-TO-END TOTAL  : {total_seconds:6.2f} s "
          f"({total_seconds / max(1, count) * 1000:.1f} ms per image, "
          f"{count / max(1e-9, total_seconds):.1f} images/second)")


if __name__ == "__main__":
    main()
