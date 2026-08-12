"""Train the restoration model.

Synthetic mode (you only have clean images — pairs are manufactured live):
    python train.py --mode synthetic --train-dir data/train --val-dir data/val --scale 4

Paired mode (organisers gave degraded+clean folders, matched filenames):
    python train.py --mode paired --train-lq data/train/lq --train-gt data/train/gt
                    --val-lq data/val/lq --val-gt data/val/gt --scale 4

If training is interrupted, continue with:  --resume weights/last.pth

HOW TRAINING WORKS (one "step", repeated thousands of times):
    1. FORWARD PASS   feed a batch of degraded patches through the model
                      -> it produces its best-guess restored patches
    2. LOSS           compare guesses to the clean ground truth
                      -> one number: "how wrong was that?"  (utils/losses.py)
    3. BACKPROPAGATION  torch traces the loss backwards through every layer
                      and computes, for each of the ~600k weights, "which
                      direction would reduce the loss?" (the gradient)
    4. OPTIMIZER STEP  nudge every weight a tiny bit in that direction

One pass through the whole training set = one EPOCH. Every few epochs we
pause and measure PSNR on validation images the model never trains on —
that tells us if it is genuinely learning or just memorising.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.degradation import DegradationSettings
from src.dataset.loader import PairedFolders, SyntheticPairs
from src.models.nafnet import NAFNetSR, count_parameters
from src.utils.losses import RestorationLoss
from src.utils.metrics import psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Train the restoration model")
    parser.add_argument("--mode", choices=["synthetic", "paired"], default="synthetic")
    # synthetic mode folders (clean images only)
    parser.add_argument("--train-dir", help="clean training images")
    parser.add_argument("--val-dir", help="clean validation images")
    # paired mode folders (lq = degraded input, gt = clean ground truth)
    parser.add_argument("--train-lq"); parser.add_argument("--train-gt")
    parser.add_argument("--val-lq");   parser.add_argument("--val-gt")

    parser.add_argument("--scale", type=int, default=4, choices=[1, 2, 4],
                        help="4: 128->512, 2: 128->256, 1: denoise only")
    parser.add_argument("--patch", type=int, default=192,
                        help="training crop size (on the clean image)")
    parser.add_argument("--width", type=int, default=48,
                        help="model feature channels (bigger = stronger, slower)")
    parser.add_argument("--blocks", type=int, default=24,
                        help="number of NAFBlocks in the body")

    # BATCH SIZE: how many patches are processed together in one step.
    # Bigger = steadier gradients and better GPU use, but more memory.
    # If you hit a CUDA out-of-memory error, halve this first.
    parser.add_argument("--batch-size", type=int, default=16)

    # EPOCHS: how many times we sweep the whole training set. Restoration
    # models improve for a long time; 200-300 is typical for a final run.
    parser.add_argument("--epochs", type=int, default=200)

    # LEARNING RATE: the size of each weight nudge. Too big -> training
    # explodes (loss jumps around). Too small -> takes forever.
    # 2e-4 with AdamW is the standard, proven setting for NAFNet.
    parser.add_argument("--lr", type=float, default=2e-4)

    # Loss weights. LPIPS is a judged metric, so a LIGHT lpips term is on
    # by default (see utils/losses.py for why it must stay small).
    # --w-lpips 0 disables it for slightly faster training.
    parser.add_argument("--w-pixel", type=float, default=1.0)
    parser.add_argument("--w-ssim", type=float, default=0.15)
    parser.add_argument("--w-lpips", type=float, default=0.05)

    parser.add_argument("--val-every", type=int, default=5,
                        help="validate every N epochs")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel CPU processes loading data (0 on weak PCs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="fixes all randomness so runs are repeatable")
    parser.add_argument("--resume", default=None,
                        help="checkpoint path to continue an interrupted run")
    parser.add_argument("--out", default="weights",
                        help="where best.pth / last.pth are saved")
    return parser.parse_args()


def build_datasets(args):
    """Create train/val datasets for whichever data situation we're in."""
    if args.mode == "synthetic":
        assert args.train_dir and args.val_dir, \
            "synthetic mode needs --train-dir and --val-dir"
        settings = DegradationSettings(scale=args.scale)
        train_set = SyntheticPairs(args.train_dir, settings, args.patch, training=True)
        val_set = SyntheticPairs(args.val_dir, settings, training=False)
    else:
        assert args.train_lq and args.train_gt and args.val_lq and args.val_gt, \
            "paired mode needs --train-lq --train-gt --val-lq --val-gt"
        train_set = PairedFolders(args.train_lq, args.train_gt, args.scale,
                                  args.patch, training=True)
        val_set = PairedFolders(args.val_lq, args.val_gt, args.scale,
                                training=False)
    return train_set, val_set


@torch.no_grad()   # no gradients needed when only measuring -> faster, less memory
def validate(model, val_loader, device):
    """Average PSNR over the validation set (images never trained on).

    WHY validate on unseen images: training loss always goes down, even
    when the model is just memorising. Only performance on NEW images
    proves it learned a general skill. This score also picks which
    checkpoint we keep as 'best'.
    """
    model.eval()      # switch off training-only behaviours
    scores = []
    for degraded, clean in val_loader:
        restored = model(degraded.to(device)).clamp(0, 1).cpu()
        scores.append(psnr(restored[0], clean[0]))
    model.train()     # back to training mode
    return float(np.mean(scores))


def main():
    args = parse_args()

    # Seed every random source -> two runs with the same seed behave the
    # same. Essential for debugging ("did my change help, or was it luck?").
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use the GPU if there is one; everything else in this file is
    # identical either way.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_datasets(args)
    # The DataLoader batches and shuffles for us. WHY shuffle: if the
    # model saw images in the same order every epoch it could pick up
    # ordering patterns; shuffling keeps every batch a fresh random mix.
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0)

    model_config = {"channels": 1, "width": args.width,
                    "n_blocks": args.blocks, "scale": args.scale}
    model = NAFNetSR(**model_config).to(device)
    print(f"device: {device} | parameters: {count_parameters(model)/1e6:.2f}M "
          f"| train images: {len(train_set)} | val images: {len(val_set)}")

    loss_fn = RestorationLoss(channels=1, weight_pixel=args.w_pixel,
                              weight_ssim=args.w_ssim,
                              weight_lpips=args.w_lpips).to(device)

    # OPTIMIZER: AdamW. It gives each weight its own adaptive step size
    # (weights with noisy gradients get gentler nudges), which is why it
    # "just works" without hand-tuning. The betas value is taken from the
    # NAFNet paper; weight_decay gently shrinks weights to fight overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.9), weight_decay=1e-4)

    # SCHEDULER: cosine decay — the learning rate glides smoothly from
    # 2e-4 down to ~0 over the whole run. Big early steps learn the rough
    # task fast; tiny late steps let the model settle into a sharp,
    # precise optimum instead of bouncing around it.
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7)

    # CHECKPOINTS: everything needed to continue or reuse the run.
    # last.pth = newest state (crash insurance), best.pth = highest
    # validation PSNR so far (the one we actually ship).
    start_epoch, best_psnr = 1, 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint["best_psnr"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running = {}

        for degraded, clean in train_loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            restored = model(degraded)                 # 1. forward pass
            loss, parts = loss_fn(restored, clean)     # 2. loss

            optimizer.zero_grad()                      # clear old gradients
            loss.backward()                            # 3. backpropagation

            # Safety belt: if one weird batch produces a huge gradient,
            # cap it so a single step can't wreck the model.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()                           # 4. nudge the weights
            scheduler.step()                           # decay the LR a little

            for name, value in parts.items():          # accumulate for the log
                running[name] = running.get(name, 0.0) + value

        n_batches = max(1, len(train_loader))
        log = "  ".join(f"{k}={v/n_batches:.4f}" for k, v in running.items())
        print(f"epoch {epoch:3d}/{args.epochs} | {log} "
              f"| lr={scheduler.get_last_lr()[0]:.2e} "
              f"| {time.time()-epoch_start:.1f}s")

        # ---- periodic validation + checkpointing ----
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_psnr = validate(model, val_loader, device)
            print(f"  [val] PSNR = {val_psnr:.3f} dB (best so far {best_psnr:.3f})")

            checkpoint = {"model": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(),
                          "epoch": epoch,
                          "best_psnr": max(best_psnr, val_psnr),
                          "config": model_config}
            torch.save(checkpoint, out_dir / "last.pth")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(checkpoint, out_dir / "best.pth")
                print(f"  ** new best model saved -> {out_dir/'best.pth'}")

    print(f"done. best validation PSNR: {best_psnr:.3f} dB "
          f"| final model: {out_dir/'best.pth'}")


if __name__ == "__main__":
    main()
