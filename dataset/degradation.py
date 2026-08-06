"""The degradation simulator: turns CLEAN images into DEGRADED ones.

WHY this file is the heart of the project:
    A restoration model learns from pairs (degraded image, clean image).
    If the organisers only give us clean images, we manufacture the pairs
    ourselves by corrupting clean images in code. Every big restoration
    project (SwinIR, Real-ESRGAN, ...) trains exactly this way.

The corruption pipeline (applied in this order):

    CLEAN IMAGE
        |
        v
    [1] slight blur (sometimes)     <- imitates imperfect microscope focus
        |
        v
    [2] shrink by 2x or 4x          <- creates the "low resolution" problem
        |
        v
    [3] speckle noise               <- MULTIPLIES pixels:  pixel * (1 + noise)
        |                              bright areas get noisier (SEM physics)
        v
    [4] Gaussian noise              <- ADDS to pixels:     pixel + noise
        |                              same strength everywhere (sensor noise)
        v
    DEGRADED IMAGE  (this is what the model sees as input)

WHY speckle and Gaussian are different formulas:
    Speckle scales with brightness (a bright pixel gets a big kick, a dark
    pixel a small one) so it must MULTIPLY. Sensor noise is the same
    everywhere so it must ADD. Modelling both correctly is what makes our
    synthetic training data look like real degraded SEM images.

WHY noise strength is RANDOM for every training patch:
    We don't know how noisy the judges' test images will be. By training
    on a whole RANGE of noise levels, the model learns to handle any
    strength it meets ("blind" restoration) instead of overfitting to one.
"""

import random

import torch
import torch.nn.functional as F


class DegradationSettings:
    """All the knobs of the corruption pipeline in one small object.

    Each range is (lowest, highest); a value is drawn from it per image.
    Defaults are sensible for SEM-like images — tune only if the
    organisers reveal how strong the real test degradation is.
    """

    def __init__(self, scale=4):
        self.scale = scale                    # 2 -> 128->256 task, 4 -> 128->512
        self.blur_probability = 0.3           # only 30% of patches get blurred
        self.blur_sigma_range = (0.2, 1.0)    # how strong the blur can be
        self.speckle_range = (0.05, 0.25)     # speckle noise strength range
        self.gaussian_range = (0.01, 0.08)    # gaussian noise strength range


def apply_blur(image, sigma):
    """Step [1]: soften the image with a Gaussian blur of strength `sigma`.

    We build a small bell-curve shaped filter and slide it over the image
    (a convolution). Each output pixel becomes a weighted average of its
    neighbours -> fine detail is smeared, like a slightly out-of-focus lens.
    """
    radius = max(1, int(round(3 * sigma)))          # bell curve ~ 0 beyond 3*sigma
    positions = torch.arange(2 * radius + 1, dtype=image.dtype) - radius
    bell = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
    bell = bell / bell.sum()                        # weights must sum to 1 so
                                                    # brightness stays unchanged
    # A 2D blur = 1D bell applied horizontally times vertically.
    kernel = bell[:, None] * bell[None, :]
    kernel = kernel.expand(image.shape[0], 1, *kernel.shape)  # one per channel

    # conv2d wants a batch dimension, so temporarily add one with unsqueeze.
    blurred = F.conv2d(image.unsqueeze(0), kernel,
                       padding=radius, groups=image.shape[0])
    return blurred.squeeze(0)


def shrink(image, scale):
    """Step [2]: downsample, e.g. 512x512 -> 128x128 when scale=4.

    WHY bicubic with antialias: that is the standard, realistic way
    resolution is lost; it's also what the judges most likely used to
    create their test set.
    """
    if scale == 1:
        return image                                # scale 1 = denoising only
    height, width = image.shape[-2:]
    small = F.interpolate(image.unsqueeze(0),
                          size=(height // scale, width // scale),
                          mode="bicubic", antialias=True, align_corners=False)
    return small.squeeze(0)


def add_speckle_noise(image, strength, generator=None):
    """Step [3]: multiplicative noise.  new_pixel = pixel * (1 + strength * random)

    Where the image is bright, `pixel` is large, so the noise kick is
    large too — exactly how speckle behaves in real SEM/laser imaging.
    """
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
    return image * (1.0 + strength * noise)


def add_gaussian_noise(image, strength, generator=None):
    """Step [4]: additive noise.  new_pixel = pixel + strength * random

    Same strength for bright and dark pixels — like electronic sensor hiss.
    """
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
    return image + strength * noise


def degrade_image(clean, settings, seed=None):
    """Run the full pipeline: clean (C,H,W in [0,1]) -> degraded image.

    seed=None  -> fresh random corruption every call (used for TRAINING,
                  so the model never sees the same corruption twice).
    seed=1234  -> the exact same corruption every call (used for VALIDATION,
                  so scores are comparable between epochs — if the noise
                  changed every time, we couldn't tell whether the model
                  improved or the noise just got easier).
    """
    if seed is None:
        rng, generator = random, None
    else:
        rng = random.Random(seed)                       # seeded choice-maker
        generator = torch.Generator().manual_seed(seed)  # seeded noise-maker

    degraded = clean

    if rng.random() < settings.blur_probability:                       # [1]
        degraded = apply_blur(degraded, rng.uniform(*settings.blur_sigma_range))

    degraded = shrink(degraded, settings.scale)                        # [2]

    degraded = add_speckle_noise(
        degraded, rng.uniform(*settings.speckle_range), generator)     # [3]

    degraded = add_gaussian_noise(
        degraded, rng.uniform(*settings.gaussian_range), generator)    # [4]

    # Noise can push pixels below 0 or above 1; a real camera can't record
    # that either, so clamp back to the valid range.
    return degraded.clamp(0.0, 1.0)
