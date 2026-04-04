#!/usr/bin/env python3
"""
train_all.py
============
Convenience script that runs the full pipeline in sequence:
  1. data_prep.py   – build vocab + sequence datasets
  2. pretrain.py    – pre-train GRU on next-token prediction
  3. finetune.py    – fine-tune on conversational pairs

Then launches the Flask web app.

Usage:
    python train_all.py
    python train_all.py --skip_train   # skip training, just launch app
    python train_all.py --no_app       # train only, don't launch app
"""

import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print(f"\n{'=' * 60}")
    print(f"$ {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true", help="Skip data prep + training; go straight to web app")
    parser.add_argument("--no_app", action="store_true", help="Run training but do not launch the web app")
    # Forward common hyper-params
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_train:
        # Step 1 – data prep
        run([py, "data_prep.py"])

        # Step 2 – pre-train
        run([py, "pretrain.py", "--epochs", str(args.pretrain_epochs), "--batch_size", str(args.batch_size)])

        # Step 3 – fine-tune
        run([py, "finetune.py", "--epochs", str(args.finetune_epochs)])

    if not args.no_app:
        print("\n🚀  Launching web app…  (Ctrl-C to stop)")
        run([py, "app.py"])


if __name__ == "__main__":
    main()
