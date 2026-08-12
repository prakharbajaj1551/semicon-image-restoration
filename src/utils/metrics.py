"""Image quality metrics (numbers for the scoreboard, not for training).

METRIC vs LOSS — what's the difference?
    The LOSS guides learning (must be differentiable, computed on GPU
    thousands of times). A METRIC is just a report card computed now and
    then to answer "is the model actually getting better?".

The three standard restoration metrics, in plain words:

    PSNR   higher = better   "how small is the average pixel error?"
    SSIM   higher = better   "do edges/patterns look structurally right?"
    LPIPS  LOWER  = better   "would a human say these look similar?"

WHY three metrics and not one?
    Each can be fooled alone. A slightly blurry image can score good
    PSNR (small average error) while looking bad; SSIM catches that.
    LPIPS catches perceptual weirdness both miss. An image that wins on
    all three is genuinely well restored.
"""

import math

import torch


def psnr(pred, target, max_value=1.0):
    """Peak Signal-to-Noise Ratio, in decibels (dB). Higher = better.

    WHAT IT MEANS: log-scale measure of average pixel error.
        mse   = mean((pred - target)^2)
        psnr  = 10 * log10(max_value^2 / mse)
    Rough guide for restoration: 25 dB = visibly noisy, 30 dB = decent,
    35+ dB = very clean. +3 dB means the pixel error was cut in HALF —
    the scale is logarithmic, so small dB gains are real improvements.
    """
    mse = float(torch.mean((pred - target) ** 2))
    if mse <= 1e-12:        # identical images; cap instead of infinity so
        return 99.0         # averages over a folder stay meaningful
    return 10.0 * math.log10(max_value ** 2 / mse)


def ssim(pred, target):
    """Structural SIMilarity, 0..1. Higher = better (1 = identical).

    WHAT IT MEANS: instead of comparing pixels one by one, SSIM slides a
    small window over both images and compares three local properties:
    brightness (mean), contrast (variance) and structure (correlation).
    That is much closer to how an engineer judges whether edges and
    patterns survived — which is why it's a standard judging metric.

    We use scikit-image's reference implementation: for REPORTING scores,
    everyone should use the same battle-tested code so numbers are
    comparable. (Training uses pytorch-msssim instead because the loss
    must be differentiable — different job, different tool.)
    """
    from skimage.metrics import structural_similarity

    a = pred.detach().cpu().numpy()
    b = target.detach().cpu().numpy()
    if a.shape[0] == 1:                       # grayscale (1,H,W) -> (H,W)
        return float(structural_similarity(a[0], b[0], data_range=1.0))
    return float(structural_similarity(a.transpose(1, 2, 0),
                                       b.transpose(1, 2, 0),
                                       data_range=1.0, channel_axis=-1))


class LPIPSMetric:
    """LPIPS = Learned Perceptual Image Patch Similarity. LOWER = better.

    WHAT IT MEANS: both images are passed through a small pretrained
    vision network (AlexNet); LPIPS is the distance between their
    internal feature maps. Networks "see" images somewhat like humans
    do, so this correlates with human judgement far better than pixel
    math — it's the standard "does it LOOK right?" metric.

    WHY a class and not a function: the AlexNet weights (~9 MB,
    downloaded automatically on first use) should be loaded ONCE, not
    once per image. Create one LPIPSMetric, then call it like a function.
    """

    def __init__(self, device="cpu"):
        import lpips  # imported here so evaluate.py --no-lpips works
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)   # we only measure, never train it

    @torch.no_grad()
    def __call__(self, pred, target):
        def prepare(x):
            x = x.unsqueeze(0)                    # add batch dimension
            if x.shape[1] == 1:                   # AlexNet expects 3 channels,
                x = x.repeat(1, 3, 1, 1)          # so copy gray 3 times
            return x.to(self.device)
        # normalize=True tells lpips our images are in [0,1] (it wants [-1,1])
        return float(self.model(prepare(pred), prepare(target), normalize=True))
