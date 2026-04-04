"""
pretrain.py
===========
Step 2: Pre-train the GRU language model on next-token prediction.

CPU-optimised: subsamples sequences per epoch so each epoch finishes
in ~60–90 seconds on a typical laptop CPU.

Usage:
    python pretrain.py
    python pretrain.py --epochs 15 --steps_per_epoch 300
"""

import argparse
import pickle
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from model import GRULanguageModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def run_epoch(model, loader, optimizer, criterion, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_steps = len(loader)

    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            # Inline progress every 50 steps
            if train and step % 50 == 0:
                print(f"   step {step:4d}/{total_steps}  loss={total_loss / step:.4f}", end="\r")

    if train:
        print(" " * 60, end="\r")  # clear progress line

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension (default: 64)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden size (default: 128)")
    parser.add_argument("--n_layers", type=int, default=1, help="GRU layers (default: 1)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=400,
        help="Max batches per epoch — keeps each epoch ~60-90s on CPU (default: 400)",
    )
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(exist_ok=True)

    print(f"Device : {device}")

    # ── Load sequences ─────────────────────────────────────────────────────
    with open(data / "train.pkl", "rb") as f:
        train_seq = pickle.load(f)
    with open(data / "val.pkl", "rb") as f:
        val_seq = pickle.load(f)

    train_ds = SequenceDataset(train_seq)
    val_ds = SequenceDataset(val_seq)

    # ── Vocab size ─────────────────────────────────────────────────────────
    vocab_data = json.load(open(data / "vocab.json"))
    vocab_size = len(vocab_data["token2id"])
    print(f"Vocab size      : {vocab_size:,}")
    print(f"Train sequences : {len(train_ds):,}")
    print(f"Val   sequences : {len(val_ds):,}")

    # Max batches per epoch = steps_per_epoch
    max_train_samples = args.steps_per_epoch * args.batch_size
    print(f"Samples/epoch   : {min(max_train_samples, len(train_ds)):,}  (steps_per_epoch={args.steps_per_epoch})")

    # ── Model ──────────────────────────────────────────────────────────────
    model = GRULanguageModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters      : {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Fixed val loader (use all val data)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val = float("inf")
    patience_ctr = 0
    history = []

    print(f"{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'LR':>8}  {'Time':>6}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Subsample training sequences each epoch (fresh random subset)
        indices = random.sample(range(len(train_ds)), min(max_train_samples, len(train_ds)))
        subset = Subset(train_ds, indices)
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

        tr_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "lr": lr_now})
        print(f"{epoch:5d}  {tr_loss:11.4f}  {val_loss:9.4f}  {lr_now:8.6f}  {elapsed:5.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                outdir / "pretrain_best.pt",
            )
            print(f"         ✓ saved best checkpoint (val={val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stop — no improvement for {args.patience} epochs.")
                break

    torch.save(model.state_dict(), outdir / "pretrain_final.pt")
    with open(outdir / "pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss : {best_val:.4f}")
    print(f"Saved to      : '{outdir}/'")
    print("\nNext → python finetune.py")


if __name__ == "__main__":
    main()
