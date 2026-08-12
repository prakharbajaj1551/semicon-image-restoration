"""PyTorch Dataset classes: they feed (degraded, clean) pairs to training.

WHY a "Dataset" class at all:
    PyTorch's DataLoader needs an object that can answer two questions:
      __len__()        -> "how many images do you have?"
      __getitem__(i)   -> "give me training pair number i"
    Given that, the DataLoader handles batching, shuffling and loading
    with several CPU workers in parallel for us — code we don't have
    to write or debug.

Two classes for the two situations we might face at the hackathon:

  SyntheticPairs  -> we only have CLEAN images. The degraded partner is
                     manufactured on the fly by dataset/degradation.py.
                     Fresh random noise each epoch = unlimited unique
                     training pairs from a small image collection.

  PairedFolders   -> the organisers give us degraded + clean folders,
                     where pairs share the same filename. We just load both.

WHY train on small PATCHES (crops), not whole images:
    A 1024x1024 image doesn't fit many times into GPU memory, but a
    192x192 crop does — so we can use bigger batches. Noise removal is a
    LOCAL problem: the model doesn't need to see the whole die to clean
    an edge. Random crops also mean every epoch shows different regions,
    which is free extra variety.
"""

import random

import torch
from torch.utils.data import Dataset

from src.dataset.degradation import degrade_image
from src.utils.image_io import list_images, load_image


def random_crop(image, crop_size):
    """Cut a random square of side `crop_size` out of the image.

    If the image is smaller than the crop, mirror-pad its borders first
    (reflection looks natural; black padding would teach the model that
    images have fake black frames).
    """
    pad_h = max(0, crop_size - image.shape[-2])
    pad_w = max(0, crop_size - image.shape[-1])
    if pad_h or pad_w:
        image = torch.nn.functional.pad(
            image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
        ).squeeze(0)

    top = random.randint(0, image.shape[-2] - crop_size)
    left = random.randint(0, image.shape[-1] - crop_size)
    return image[:, top:top + crop_size, left:left + crop_size]


def augment_pair(*images):
    """Randomly flip/rotate — the SAME way for every image passed in.

    WHY augmentation: a flipped chip image is still a perfectly valid chip
    image. Flips (x2) x flips (x2) x rotation (x2) turn every training
    image into up to 8 different-looking ones for free, which helps a lot
    when the dataset is small and reduces overfitting (memorising).

    WHY "the same way for every image": input and ground truth must stay
    perfectly aligned — if we flipped only one of them, the model would
    be asked to learn a mirror operation instead of denoising!
    """
    flip_horizontal = random.random() < 0.5
    flip_vertical = random.random() < 0.5
    rotate_90 = random.random() < 0.5

    results = []
    for img in images:
        if flip_horizontal:
            img = torch.flip(img, dims=[-1])
        if flip_vertical:
            img = torch.flip(img, dims=[-2])
        if rotate_90:
            img = torch.rot90(img, 1, dims=[-2, -1])
        results.append(img)
    return results if len(results) > 1 else results[0]


class SyntheticPairs(Dataset):
    """Mode 1: only clean images exist; we corrupt them ourselves.

    training=True  -> random crop + augmentation + FRESH random noise.
    training=False -> fixed center crop + FIXED noise (seeded per image),
                      so validation scores are comparable across epochs.
    """

    def __init__(self, clean_folder, settings, patch_size=192,
                 channels=1, training=True):
        self.image_paths = list_images(clean_folder)
        self.settings = settings
        self.channels = channels
        self.training = training
        # Patch side must divide evenly by the scale factor, otherwise the
        # shrunken input and the clean target sizes wouldn't line up.
        self.patch_size = patch_size - patch_size % settings.scale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        clean = load_image(self.image_paths[index], self.channels)

        if self.training:
            clean = random_crop(clean, self.patch_size)
            clean = augment_pair(clean)
            degraded = degrade_image(clean, self.settings)      # fresh noise
        else:
            clean = self._center_crop(clean, max_size=512)
            degraded = degrade_image(clean, self.settings,
                                     seed=1234 + index)         # frozen noise

        return degraded, clean

    def _center_crop(self, image, max_size):
        """Validation crop: centered and deterministic (no randomness),
        trimmed so both sides divide evenly by the scale factor."""
        s = self.settings.scale
        height = min(image.shape[-2], max_size) // s * s
        width = min(image.shape[-1], max_size) // s * s
        top = (image.shape[-2] - height) // 2
        left = (image.shape[-1] - width) // 2
        return image[:, top:top + height, left:left + width]


class PairedFolders(Dataset):
    """Mode 2: organisers provide matching degraded (lq) and clean (gt)
    folders. "lq" = low quality, "gt" = ground truth. Files pair up by
    name: lq/chip_07.png <-> gt/chip_07.png.

    WHY we crop input and target TOGETHER: the crop taken from the clean
    image must show exactly the same region as the crop from the degraded
    one (just `scale` times bigger), or the model would train on
    mismatched answers.
    """

    def __init__(self, degraded_folder, clean_folder, scale=4,
                 patch_size=192, channels=1, training=True):
        # Build a name -> path lookup for degraded files, then keep only
        # clean files that have a degraded partner with the same name.
        self.degraded_by_name = {p.stem: p for p in list_images(degraded_folder)}
        self.clean_paths = [p for p in list_images(clean_folder)
                            if p.stem in self.degraded_by_name]
        if not self.clean_paths:
            raise FileNotFoundError(
                "No filename-matched pairs found. Files must share names, "
                "e.g. lq/chip_07.png and gt/chip_07.png")

        self.scale = scale
        self.channels = channels
        self.training = training
        self.patch_size = patch_size - patch_size % scale

        # Safety check: does --scale match the actual data? If gt images
        # aren't `scale` times bigger than lq images, the crop coordinates
        # below would silently produce MISALIGNED training pairs and the
        # model would learn garbage. Fail loudly instead.
        first_lq = load_image(self.degraded_by_name[self.clean_paths[0].stem])
        first_gt = load_image(self.clean_paths[0])
        actual = round(first_gt.shape[-1] / first_lq.shape[-1])
        if actual != scale:
            raise ValueError(
                f"--scale {scale} doesn't match the data: gt is {actual}x "
                f"bigger than lq ({first_gt.shape[-1]} vs {first_lq.shape[-1]} "
                f"px wide). Re-run with --scale {actual}.")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = load_image(clean_path, self.channels)
        degraded = load_image(self.degraded_by_name[clean_path.stem], self.channels)

        s = self.scale
        if self.training:
            # Pick the crop position on the SMALL image, then multiply the
            # coordinates by `scale` to get the matching big-image crop.
            small_patch = self.patch_size // s
            top = random.randint(0, degraded.shape[-2] - small_patch)
            left = random.randint(0, degraded.shape[-1] - small_patch)

            degraded = degraded[:, top:top + small_patch, left:left + small_patch]
            clean = clean[:, top * s:(top + small_patch) * s,
                          left * s:(left + small_patch) * s]
            degraded, clean = augment_pair(degraded, clean)
        else:
            # Validation: use full images, just trim the clean one so it is
            # exactly `scale` times the degraded size (guards against
            # off-by-a-few-pixels datasets).
            clean = clean[:, :degraded.shape[-2] * s, :degraded.shape[-1] * s]

        return degraded, clean
