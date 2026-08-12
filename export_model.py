"""Strip a training checkpoint down to what inference actually needs.

    python export_model.py

WHY:
    train.py saves everything required to RESUME training - the weights,
    the AdamW optimizer state, the scheduler, the epoch counter. That file
    is 6.5 MB, but 4.6 MB of it is optimizer moment buffers that inference
    never touches. Shipping the slim file makes the download smaller and
    lets us state the model size honestly: 0.53 M parameters, 2.3 MB.

    The training checkpoint is still kept so training can be resumed.
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="weights/best.pth",
                        help="full training checkpoint (weights + optimizer)")
    parser.add_argument("--out", default="weights/model.pth",
                        help="slim inference checkpoint to write")
    args = parser.parse_args()

    full = torch.load(args.ckpt, map_location="cpu")

    # Keep only what load_model() reads: the weights and the architecture
    # config that rebuilds the network.
    slim = {"model": full["model"], "config": full["config"]}
    torch.save(slim, args.out)

    before = Path(args.ckpt).stat().st_size / 1e6
    after = Path(args.out).stat().st_size / 1e6
    params = sum(v.numel() for v in full["model"].values())
    print(f"{args.ckpt}: {before:.2f} MB (weights + optimizer + scheduler)")
    print(f"{args.out}: {after:.2f} MB (weights + config only)")
    print(f"{params/1e6:.3f} M parameters = {params*4/1e6:.2f} MB of float32 weights")
    print(f"saved {before - after:.2f} MB")


if __name__ == "__main__":
    main()
