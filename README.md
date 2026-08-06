# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Team BYTE BRIGADE** — SEMICON India Hackathon 2026
Prakhar Bajaj · Raghav Soni · Parth Kumar · Mayank Lodhi

A single lightweight neural network (0.53 M parameters, 6.8 MB) that takes a
noisy 128×128 inspection image and outputs a clean, sharp 256×256 image —
joint **denoising + 2× super-resolution** in one forward pass.

---

## Results

Measured on 100 held-out image pairs never seen during training:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upscaling (no AI) | 22.96 dB | 0.537 | 0.433 |
| **Ours (NAFNet-SR)** | **28.00 dB** | **0.759** | **0.175** |

**+5.0 dB PSNR** (≈ 69 % of pixel error removed), **+0.22 SSIM**, and
**LPIPS reduced by 60 %** — better on every judged metric.

Inference speed: **~1.2 s/image on a laptop CPU**, ~0.03 s/image on a T4 GPU.
No GPU is required to run the demo.

![Comparison: degraded vs bicubic vs ours vs ground truth](assets/comparison.png)

## How it works

```
Degraded input (1×128×128)
    │
    ├────────────────────────────┐
    ▼                            │
HEAD   3×3 conv (1→48 ch.)       │
    ▼                            ▼
BODY   24 × NAFBlock        bicubic ×2 upscale
    ▼                       (cheap "base guess")
FUSE   3×3 conv + skip           │
    ▼                            │
UP     PixelShuffle ×2           │
    ▼                            │
TAIL   3×3 conv (48→1 ch.)       │
    └─────────▶ ( + ) ◀──────────┘
                 ▼
   Restored output (1×256×256)
```

Design decisions, each chosen for **simplicity + proven results**:

- **NAFNet blocks** (ECCV 2022): state-of-the-art restoration quality from a
  block simple enough to draw on a whiteboard — LayerNorm → conv →
  SimpleGate → channel attention. No transformers, not even ReLU.
- **Global bicubic skip**: the network only learns the *correction* on top of
  a plain upscale. It recovers structure — it cannot invent it. For defect
  inspection, never hallucinating is a hard requirement, not a preference.
- **Loss = Charbonnier + 0.15·SSIM + 0.05·LPIPS**: pixel fidelity in charge,
  structural and perceptual terms directly optimising the judged metrics.
  Deliberately **no GAN loss** (GANs fabricate texture).
- **Native float32 `.npy` I/O**: degraded inputs contain values outside
  [0, 1]; converting to PNG would clip real signal. The pipeline never does.
- **Blind to noise level**: trained across the full range of degradations in
  the data, so one model handles any test image without tuning.

## Folder structure

```
├── dataset/
│   ├── degradation.py    synthetic corruption pipeline (for augmentation/demo)
│   └── loader.py         paired + synthetic PyTorch Datasets
├── models/
│   └── nafnet.py         NAFNet-SR architecture
├── utils/
│   ├── image_io.py       .npy / image loading & saving (one shared place)
│   ├── losses.py         Charbonnier + SSIM + light LPIPS
│   └── metrics.py        PSNR / SSIM / LPIPS
├── data/                 datasets (not in git)
│   ├── train/lq|gt       3000 training pairs
│   ├── val/lq|gt         100 validation pairs
│   ├── test/lq|gt        100 held-out benchmark pairs (+ bicubic/ baseline)
│   └── test_submission/  400 official test inputs
├── checkpoints/          best.pth — the trained model (config stored inside)
├── outputs/              restored images, metrics CSVs, submission files
├── app.py                Streamlit demo UI
├── train.py              training loop
├── evaluate.py           folder-vs-folder PSNR/SSIM/LPIPS report
├── inference.py          batch restoration (the submission generator)
├── make_testset.py       benchmark / bicubic-baseline builder
└── requirements.txt
```

Every file is written for readability: small modules, beginner-friendly
naming, and comments that explain **why**, not just what.

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Dataset setup

The organisers provide float32 `.npy` files: `NoisyLR` (128×128 degraded)
and `GT` (256×256 clean), paired by filename. Arrange as:

```
data/train/lq + data/train/gt     data/val/lq + data/val/gt
data/test/lq  + data/test/gt      data/test_submission/
```

(Our split: 3000 / 100 / 100, seeded shuffle for reproducibility.)

## Training

```bash
python train.py --mode paired \
    --train-lq data/train/lq --train-gt data/train/gt \
    --val-lq data/val/lq --val-gt data/val/gt \
    --scale 2 --patch 128 --epochs 100 --batch-size 16 --out checkpoints
```

~2 h on a free Colab T4 (74 s/epoch). Resume an interrupted run with
`--resume checkpoints/last.pth`. The wrong `--scale` fails immediately with
a message telling you the correct value.

## Evaluation

```bash
python make_testset.py --lq data/test/lq --gt data/test/gt --out data/test   # baseline
python evaluate.py --restored outputs/restored --gt data/test/gt             # score any folder
```

Prints per-image and mean PSNR / SSIM / LPIPS and writes a CSV.

## Inference (submission generation)

```bash
python inference.py --ckpt checkpoints/best.pth \
    --input data/test_submission --output outputs/submission
```

Restores every image in a folder — `.npy` in, `.npy` out, no code editing.
The checkpoint stores its own architecture config, so the right network is
rebuilt automatically. Large images are processed in overlapping tiles.

## Demo app

```bash
streamlit run app.py
```

Upload a degraded image (`.npy` or PNG) → side-by-side original/restored
with PSNR and inference time. A second mode degrades a clean image live and
restores it, showing the bicubic baseline vs. ours.

## Future improvements

- **Bigger model** (`--width 64 --blocks 32`): more capacity, ~2× slower.
- **Test-time ensemble** (average over 8 flips/rotations): ~0.1–0.2 dB extra
  at 8× the inference cost — a knob between "fast" and "max quality".
- **EMA weights & mixed precision**: standard training extras worth ~0.1–0.3 dB.
- **Degradation-matched fine-tuning** if organisers reveal exact noise stats.

## References

- Chen et al., *Simple Baselines for Image Restoration* (NAFNet), ECCV 2022 — [arXiv:2204.04676](https://arxiv.org/abs/2204.04676)
- Shi et al., *Real-Time Single Image SR Using an Efficient Sub-Pixel CNN* (PixelShuffle), CVPR 2016
- Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS), CVPR 2018
- Libraries: PyTorch, pytorch-msssim, lpips, scikit-image, Streamlit
