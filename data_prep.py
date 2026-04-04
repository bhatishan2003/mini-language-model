"""
data_prep.py
============
Step 1: Text cleaning, tokenisation, vocabulary construction, and
sequence dataset creation.

Dataset: TinyStories (HuggingFace) — simple short stories, great for chat.

Usage:
    python data_prep.py                        # default: 50,000 stories
    python data_prep.py --max_stories 20000    # faster, fewer stories
    python data_prep.py --input my_file.txt    # your own .txt file
"""

import re
import json
import pickle
import argparse
import random
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# TinyStories Loader
# ---------------------------------------------------------------------------


def load_tinystories(max_stories: int = 50000) -> str:
    """
    Downloads TinyStories from HuggingFace and returns text.
    Install: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please run: pip install datasets")

    print("[data_prep] Downloading TinyStories from HuggingFace...")
    print(f"[data_prep] Loading {max_stories:,} stories...")

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    stories = []
    for i, row in enumerate(ds):
        if i >= max_stories:
            break
        text = row["text"].strip()
        if text:
            stories.append(text)

    full_text = "\n\n".join(stories)
    print(f"[data_prep] Loaded {len(stories):,} stories → {len(full_text):,} characters")
    return full_text


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Lowercase and normalise whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer, keeps punctuation as separate tokens."""
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    return [t for t in text.split() if t]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    def build(self, tokens: list[str], max_vocab: int = 10000):
        counts = Counter(tokens)
        self.token2id = dict(SPECIAL)
        for tok, freq in counts.most_common(max_vocab):
            if freq >= self.min_freq and tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)
        self.id2token = {v: k for k, v in self.token2id.items()}

        oov = sum(1 for t in tokens if t not in self.token2id)
        oov_pct = oov / max(len(tokens), 1) * 100
        print(f"Vocabulary size : {len(self.token2id):,}")
        print(f"OOV rate        : {oov_pct:.2f}%  (good if < 5%)")

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token2id["<UNK>"]
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, "<UNK>") for i in ids]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)

    @classmethod
    def load(cls, path) -> "Vocabulary":
        v = cls()
        with open(path) as f:
            data = json.load(f)
        v.token2id = data["token2id"]
        v.id2token = {int(v2): k for k, v2 in v.token2id.items()}
        return v

    def __len__(self):
        return len(self.token2id)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------


def build_sequences(token_ids: list[int], seq_len: int) -> list[tuple[list[int], list[int]]]:
    """Sliding-window (input, target) pairs with 50% overlap."""
    sequences = []
    step = seq_len // 2
    for i in range(0, len(token_ids) - seq_len, step):
        x = token_ids[i : i + seq_len]
        y = token_ids[i + 1 : i + seq_len + 1]
        sequences.append((x, y))
    return sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to your own .txt file (skips TinyStories download)")
    parser.add_argument("--max_stories", type=int, default=50000, help="Number of TinyStories to load (default: 50000)")
    parser.add_argument("--seq_len", type=int, default=64, help="Sliding-window sequence length (default: 64)")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum token frequency for vocab (default: 2)")
    parser.add_argument("--max_vocab", type=int, default=10000, help="Maximum vocabulary size (default: 10000)")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation set fraction (default: 0.05)")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    # ── Load text ──────────────────────────────────────────────────────────
    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
        print(f"[data_prep] Loaded '{args.input}': {len(raw):,} chars")
    else:
        raw = load_tinystories(max_stories=args.max_stories)

    # ── Clean & tokenize ───────────────────────────────────────────────────
    cleaned = clean_text(raw)
    tokens = tokenize(cleaned)
    print(f"Total tokens after cleaning : {len(tokens):,}")

    # ── Build vocabulary ───────────────────────────────────────────────────
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build(tokens, max_vocab=args.max_vocab)
    vocab.save(out / "vocab.json")

    # ── Encode + build sequences ───────────────────────────────────────────
    token_ids = vocab.encode(tokens)
    sequences = build_sequences(token_ids, args.seq_len)
    print(f"Total sequences             : {len(sequences):,}")

    # ── Train / val split ──────────────────────────────────────────────────
    random.shuffle(sequences)
    split = int(len(sequences) * (1 - args.val_split))
    train_seq = sequences[:split]
    val_seq = sequences[split:]

    with open(out / "train.pkl", "wb") as f:
        pickle.dump(train_seq, f)
    with open(out / "val.pkl", "wb") as f:
        pickle.dump(val_seq, f)

    print(f"Train sequences : {len(train_seq):,}")
    print(f"Val   sequences : {len(val_seq):,}")
    print(f"Saved to        : '{out}/'")
    print("\nNext step: python pretrain.py --epochs 20 --hidden_dim 128 --n_layers 1 --batch_size 128")


if __name__ == "__main__":
    main()
