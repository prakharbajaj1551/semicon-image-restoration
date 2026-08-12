"""Helpers for loading and saving images (PNG/JPG/... AND .npy arrays).

WHY this file exists:
    Training, evaluation, inference AND the Streamlit UI all need to
    load images from disk and save results. If every script had its own
    copy of this code, a bug fixed in one script would stay broken in
    the others. So all image reading/writing lives here, in ONE place.

WHY .npy support:
    The hackathon organisers provide data as raw float32 NumPy arrays,
    NOT image files. This matters more than it sounds: the degraded
    inputs contain values below 0 and above 1 (unclipped noise). Saving
    them as PNG would CLIP those values and quantise to 255 levels —
    silently destroying information the model could have used. So .npy
    files are loaded/saved as-is, no rescaling, no clipping of inputs.

The one convention used everywhere in this project:
    An image in memory is a PyTorch tensor of shape (channels, height, width),
    float32, where clean images live in [0, 1] (degraded inputs may
    slightly overshoot that range — that is real, keep it).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# File types we treat as images. Anything else in a folder is ignored,
# so a stray notes.txt in the data folder will not crash training.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
                    ".npy"}


def list_images(folder):
    """Return a sorted list of every image file inside `folder` (including subfolders).

    WHY sorted: file systems return files in random order. Sorting makes
    the order identical on every computer, so image #7 is the same image
    for every teammate — important when comparing results.
    """
    folder = Path(folder)
    image_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in '{folder}'. "
            f"Did you put your dataset in the right place? (see README)"
        )
    return image_paths


def pil_to_tensor(img, channels=1):
    """Convert an already-opened PIL image -> float tensor (channels, H, W) in [0, 1].

    Steps, and WHY each one is done:

    1. convert("L")  -> force grayscale (1 channel).
       WHY grayscale: SEM inspection images carry no color information.
       Using 1 channel instead of 3 means the model has 3x less data to
       process — faster training, less GPU memory, zero quality loss.

    2. divide by 255 -> "normalization" from pixel values 0..255 to 0.0..1.0.
       WHY normalize: neural networks learn by nudging weights with small
       gradient steps. If inputs are big numbers (like 255), the gradients
       explode or oscillate and training becomes unstable. Small values in
       a fixed 0-1 range keep every image on the same scale and make one
       learning rate work for all of them.

    3. rearrange to (channels, height, width).
       WHY: PIL/numpy store images as (H, W, C) but PyTorch layers expect
       (C, H, W). We convert once, here, so no other file has to think
       about it.
    """
    img = img.convert("L" if channels == 1 else "RGB")

    array = np.asarray(img, dtype=np.float32) / 255.0  # step 2: normalize

    if channels == 1:
        array = array[None, :, :]           # (H, W)    -> (1, H, W)
    else:
        array = array.transpose(2, 0, 1)    # (H, W, 3) -> (3, H, W)

    return torch.from_numpy(np.ascontiguousarray(array))


def npy_to_tensor(array):
    """Convert a loaded .npy array -> float tensor (1, H, W).

    NO division by 255 (the arrays are already on a ~0-1 float scale)
    and NO clipping (out-of-range values in degraded inputs are real
    signal about the noise — the model should see them).
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:                     # (H, W) -> (1, H, W)
        array = array[None, :, :]
    return torch.from_numpy(np.ascontiguousarray(array))


def load_image(path, channels=1):
    """Load an image OR .npy file -> float tensor (channels, H, W).

    One function for both formats so every script in the project can
    stay completely format-agnostic.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return npy_to_tensor(np.load(path))
    return pil_to_tensor(Image.open(path), channels)


def save_image(tensor, path):
    """Save a (C, H, W) float tensor as .npy or as an image file.

    .npy   -> float32 array clipped to [0, 1]. Clipping the OUTPUT is
              correct (unlike clipping inputs): ground truth lives in
              [0, 1], so predictions outside that range can only be
              wrong — clipping them never hurts, only helps, the score.
    image  -> classic 0..255 grayscale PNG/JPG (for viewing/reports).
    """
    path = Path(path)
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)   # create folder if missing

    if path.suffix.lower() == ".npy":
        array = tensor.numpy().astype(np.float32)
        if array.shape[0] == 1:
            array = array[0]                # save (H, W), same as organisers
        np.save(path, array)
        return

    array = (tensor.numpy() * 255.0).round().astype(np.uint8)
    if array.shape[0] == 1:
        # A 2-D uint8 array is understood as grayscale automatically
        # (passing mode="L" explicitly is deprecated in newer Pillow).
        img = Image.fromarray(array[0])
    else:
        img = Image.fromarray(array.transpose(1, 2, 0))    # back to (H, W, C)
    img.save(path)
