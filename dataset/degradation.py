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

    WHERE THESE NUMBERS COME FROM (we measured them, we did not guess):
        The organisers give paired data but never state the noise levels.
        We recovered them from the pairs themselves. For a clean pixel
        value c, the residual r = lq - downsample(gt) satisfies
            Var(r | c) = c^2 * speckle^2 + gaussian^2
        so binning pixels by intensity and fitting a straight line to
        Var(r) against c^2 gives speckle from the slope and gaussian from
        the intercept. Over 120 training pairs this yielded
            speckle  ~ 0.158  (10th-90th percentile 0.129 - 0.181)
            gaussian ~ 0.040  (10th-90th percentile 0.014 - 0.074)
        and comparing candidate downsamplers showed plain bicubic left the
        least structure in the residual, so antialiasing is NOT used.

        The ranges below are set slightly wider than the measured spread:
        training on a little more variety than we expect costs nothing and
        makes the model robust if the hidden test set drifts.
    """

    def __init__(self, scale=4):
        self.scale = scale                    # 2 -> 128->256 task, 4 -> 128->512
        self.blur_probability = 0.3           # only 30% of patches get blurred
        self.blur_sigma_range = (0.2, 1.0)    # how strong the blur can be
        self.speckle_range = (0.10, 0.22)     # measured 0.129-0.181
        self.gaussian_range = (0.01, 0.09)    # measured 0.014-0.074


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

    WHY bicubic WITHOUT antialiasing: we tested five downsamplers against
    the organisers' real pairs and compared how much residual each left
    after subtracting from their degraded image. Plain bicubic won:

        bicubic (no antialias)  0.00860   <- best match
        area / average pool     0.00868
        bicubic + antialias     0.00897
        bilinear + antialias    0.00936
        nearest neighbour       0.01076

    Using antialiasing here would blur our synthetic inputs slightly more
    than theirs, so a model trained on them would expect the wrong input.
    """
    if scale == 1:
        return image                                # scale 1 = denoising only
    height, width = image.shape[-2:]
    small = F.interpolate(image.unsqueeze(0),
                          size=(height // scale, width // scale),
                          mode="bicubic", align_corners=False)
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

    # DO NOT CLAMP. Noise legitimately pushes pixels outside [0,1], and the
    # organisers' real degraded images do exactly that: measured range
    # -0.099 to 1.714, with 2.4% of pixels outside [0,1]. Clamping here
    # would make our synthetic inputs a different distribution from the
    # real ones, and would throw away signal the model can use to tell
    # noise apart from true bright detail.
    return degraded
