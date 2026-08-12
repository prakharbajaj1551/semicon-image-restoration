"""NAFNet-SR: our restoration model (denoise + super-resolve in one network).

WHY NAFNet ("Nonlinear Activation Free Network", ECCV 2022):
    It reaches state-of-the-art denoising quality with the SIMPLEST block
    design of any modern restoration network — no transformer math, not
    even a ReLU. The paper's whole point: the complicated parts of other
    models weren't what made them good. Perfect for a team that must
    explain and debug everything itself.

The full picture:

    Degraded input (e.g. 1 x 128 x 128)
        |
        |----------------------------------------------.
        v                                              |
    HEAD: 3x3 conv                                     |
    (1 gray channel -> 48 feature channels)            |
        v                                              v
    BODY: 24 x NAFBlock                        bicubic upscale
    (each block cleans the features            (cheap blurry 4x resize
     a little more)                             of the raw input)
        v                                              |
    FUSE: 3x3 conv + skip from head                    |
        v                                              |
    UPSAMPLE: PixelShuffle                             |
    (features grow 128 -> 512)                         |
        v                                              |
    TAIL: 3x3 conv                                     |
    (48 channels -> 1 gray channel)                    |
        |                                              |
        '-------------------> ( + ) <------------------'
                                |
                                v
                Restored output (1 x 512 x 512)

The final (+) is the GLOBAL RESIDUAL SKIP: the network only predicts the
CORRECTION on top of a cheap bicubic upscale, not the whole image.
Learning a small correction is far easier than learning a full image, so
training is stable — and the output can never drift far from the real
input, which matters for inspection: recover structure, never invent it.

WHAT IS A CONVOLUTION (the building block of everything below)?
    A tiny filter (e.g. 3x3 numbers) slides across the image; at each
    position it multiplies-and-sums the pixels under it. Different
    filters detect different local patterns (edges, corners, blobs).
    A "conv layer" learns many such filters at once; its output is a
    stack of "feature channels" — one map per filter showing where its
    pattern occurs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Normalises each pixel's channel vector to mean 0, variance 1.

    WHY: as data flows through many layers, its scale drifts (values grow
    or shrink). Re-standardising at the start of every block keeps every
    layer's input in a predictable range, which makes deep networks train
    stably. The learnable weight/bias let the network undo the
    normalisation where it isn't helpful.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps  # tiny number so we never divide by zero

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """NAFNet's replacement for ReLU: split channels in half, multiply.

    WHY: a network needs SOME non-linear step, otherwise stacking layers
    is pointless (many linear steps collapse into one). Instead of a
    fixed activation function, one half of the channels acts as a learned
    "gate" that scales the other half — where the gate is near 0 the
    signal is suppressed, where it's large the signal passes. Simpler
    than attention, works as well (that's the paper's headline result).
    """

    def forward(self, x):
        first_half, second_half = x.chunk(2, dim=1)
        return first_half * second_half


class NAFBlock(nn.Module):
    """One "cleaning step". The BODY stacks 24 of these.

    Each block = two small sub-units, each wrapped in a residual skip:

      Part A (spatial): look at each pixel's 3x3 NEIGHBOURHOOD
          LayerNorm -> 1x1 conv (mix channels, widen 2x)
                    -> 3x3 depthwise conv (look at neighbours)
                    -> SimpleGate -> channel attention -> 1x1 conv (back)

      Part B (channel): rethink each pixel's own FEATURE MIX
          LayerNorm -> 1x1 conv (widen) -> SimpleGate -> 1x1 conv (back)

    WHAT IS A RESIDUAL SKIP (the "x + ..." lines)?
        The block outputs  x + correction  instead of a brand-new tensor.
        If a block has nothing useful to add, it can output correction=0
        and simply pass x through unchanged. This is why we can stack 24
        blocks without the signal (or the gradient during learning)
        getting lost — there is always a clean highway straight through.

    WHY "depthwise" 3x3 conv?
        A normal 3x3 conv mixes all 96 channels at every position —
        expensive. Depthwise means each channel is filtered separately
        (cheap), and the 1x1 convs around it handle channel mixing.
        Same power, far fewer computations.
    """

    def __init__(self, channels, expand=2):
        super().__init__()
        wide = channels * expand  # width inside Part A (e.g. 48 -> 96)

        # ---- Part A: spatial cleaning ----
        self.norm1 = LayerNorm2d(channels)
        self.expand_conv = nn.Conv2d(channels, wide, kernel_size=1)
        self.spatial_conv = nn.Conv2d(wide, wide, kernel_size=3,
                                      padding=1, groups=wide)  # depthwise
        self.gate = SimpleGate()  # halves the channels: wide -> wide//2

        # Simplified channel attention: average each channel over the whole
        # image (one number per channel = "how active is this feature?"),
        # pass through a 1x1 conv, multiply back. Lets the block boost
        # useful feature types and mute useless ones GLOBALLY — the only
        # place the block sees beyond a 3x3 neighbourhood.
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(wide // 2, wide // 2, kernel_size=1),
        )
        self.shrink_conv = nn.Conv2d(wide // 2, channels, kernel_size=1)

        # ---- Part B: per-pixel feature re-mixing ----
        self.norm2 = LayerNorm2d(channels)
        self.mix_expand = nn.Conv2d(channels, wide, kernel_size=1)
        self.mix_shrink = nn.Conv2d(wide // 2, channels, kernel_size=1)

        # Learnable "volume knobs" for each residual, starting at 0:
        # the block begins as a pure pass-through and gradually turns
        # itself up during training. Makes deep stacks train smoothly.
        self.scale_a = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale_b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Part A with residual skip
        y = self.norm1(x)
        y = self.spatial_conv(self.expand_conv(y))
        y = self.gate(y)
        y = y * self.channel_attention(y)
        y = self.shrink_conv(y)
        x = x + y * self.scale_a

        # Part B with residual skip
        y = self.norm2(x)
        y = self.mix_shrink(self.gate(self.mix_expand(y)))
        return x + y * self.scale_b


class NAFNetSR(nn.Module):
    """The full model: HEAD -> BODY -> FUSE -> UPSAMPLE -> TAIL (+ bicubic skip).

    width    = number of feature channels (48 default; 64 = stronger, slower)
    n_blocks = how many NAFBlocks in the body (24 default; 32 = stronger)
    scale    = 4 for 128->512, 2 for 128->256, 1 for pure denoising

    WHY the body runs at LOW resolution and we upsample only at the end:
        Processing 128x128 features is 16x cheaper than 512x512. All the
        heavy thinking happens small; the PixelShuffle enlarges once,
        at the very end.
    """

    def __init__(self, channels=1, width=48, n_blocks=24, scale=4):
        super().__init__()
        self.scale = scale

        # HEAD (feature extraction / "encoder"): lift the gray image into
        # `width` feature channels so the body has a rich representation
        # to work with.
        self.head = nn.Conv2d(channels, width, kernel_size=3, padding=1)

        # BODY: the stack of cleaning blocks.
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(n_blocks)])

        # FUSE: one conv after the body; combined with the head skip below,
        # it merges "raw features" with "cleaned features".
        self.fuse = nn.Conv2d(width, width, kernel_size=3, padding=1)

        # UPSAMPLE via PixelShuffle, one 2x stage at a time (4x = two stages).
        # WHAT IS PIXELSHUFFLE? A conv first makes 4x the channels; pixel-
        # shuffle then rearranges every group of 4 channel values into a
        # 2x2 block of pixels: (4*C, H, W) -> (C, 2H, 2W). The network
        # LEARNS how to fill in new pixels — much sharper than any fixed
        # interpolation, with no checkerboard artifacts.
        upsample_layers = []
        remaining = scale
        while remaining > 1:
            factor = 2 if remaining % 2 == 0 else remaining
            upsample_layers += [
                nn.Conv2d(width, width * factor * factor, kernel_size=3, padding=1),
                nn.PixelShuffle(factor),
            ]
            remaining //= factor
        self.upsample = nn.Sequential(*upsample_layers) if upsample_layers \
            else nn.Identity()

        # TAIL (output layer): collapse feature channels back to a gray image.
        self.tail = nn.Conv2d(width, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # The cheap "base guess" for the global residual skip.
        if self.scale == 1:
            base = x  # pure denoising: base is the input itself
        else:
            base = F.interpolate(x, scale_factor=self.scale,
                                 mode="bicubic", align_corners=False)

        features = self.head(x)
        features = features + self.fuse(self.body(features))  # body skip
        features = self.upsample(features)

        # base + learned correction = restored image
        return base + self.tail(features)


def count_parameters(model):
    """How many learnable numbers the model has — nice for the slides."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
