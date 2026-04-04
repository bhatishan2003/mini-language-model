"""
finetune.py
===========
Improved Fine-tuning for GRU Chat Model

Fixes:
- Data augmentation
- Stronger supervision
- Label smoothing
- Stable masking
- Better defaults
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from model import GRULanguageModel
from data_prep import Vocabulary, clean_text, tokenize


# ---------------------------------------------------------------------------
# Built-in conversational pairs
# ---------------------------------------------------------------------------

BUILTIN_CHAT_DATA = [
    # ───────────── BASIC CONVERSATION (IMPORTANT) ─────────────
    ("hello", "Hello! Would you like me to tell you a story?"),
    ("hi", "Hi there! I can tell you a story if you like."),
    ("hey", "Hey! Ready for a story?"),
    ("thanks", "You're welcome! Would you like to hear another story?"),
    ("thank you", "You're welcome! I can tell you another story if you want."),
    ("bye", "Goodbye! Hope you enjoyed the story."),
    ("goodbye", "See you later! Come back for more stories."),
    # ───────────── STORY STARTERS ─────────────
    (
        "tell me a story",
        "Once upon a time, a small fox lived in a quiet forest. One day, he decided to explore beyond his home and discovered something surprising.",
    ),
    (
        "tell me a good story",
        "Once there was a brave little bird who was afraid to fly. One day, it gathered courage and spread its wings for the first time.",
    ),
    (
        "can you tell a story",
        "Long ago, in a peaceful village, there lived a young boy who dreamed of finding hidden treasure.",
    ),
    ("i want a story", "Once upon a time, a curious cat wandered into a magical garden filled with glowing flowers."),
    ("start a story", "In a distant land, a lonely dragon watched humans from afar, wishing to be their friend."),
    # ───────────── CONTINUATIONS ─────────────
    ("what happened next", "As the fox walked deeper into the forest, he heard a strange noise coming from behind the trees."),
    ("then what happened", "The boy followed the map and reached a dark cave where something unexpected awaited him."),
    ("and then", "The bird jumped from the branch and, for a moment, felt like it would fall, but then it began to glide."),
    ("what happened after that", "The cat touched one of the glowing flowers and suddenly found itself in another world."),
    ("continue the story", "The dragon finally gathered courage and approached the village, hoping to make a friend."),
    # ───────────── MULTI-STEP STORY FLOW ─────────────
    ("tell me more", "The fox carefully stepped closer and saw a small rabbit trapped under a fallen branch."),
    ("go on", "The boy lit a torch and slowly walked into the cave, hearing echoes all around him."),
    ("keep going", "The bird flapped its wings harder and soon realized it was flying high in the sky."),
    ("continue", "The magical world was filled with colors and creatures the cat had never seen before."),
    ("what happened next in the story", "The dragon spoke gently, and the villagers slowly began to trust him."),
    # ───────────── STORY VARIATIONS ─────────────
    ("tell me a short story", "A small ant once got lost but found its way home by following the scent trail."),
    (
        "tell me a long story",
        "Once upon a time, a young girl set out on a journey across mountains and rivers to find her lost family.",
    ),
    ("tell me a bedtime story", "Once there was a sleepy bunny who loved watching the stars before going to bed."),
    ("tell me a fun story", "A monkey once tried to fly using leaves and ended up learning a funny lesson."),
    ("tell me a magical story", "In a hidden kingdom, a wizard discovered a spell that could bring dreams to life."),
    # ───────────── STORY ENDINGS ─────────────
    ("how did the story end", "In the end, the fox helped the rabbit, and they became best friends."),
    ("what is the ending", "The boy found the treasure and returned home happily."),
    ("finish the story", "The bird became the bravest flyer in the forest."),
    ("how does it end", "The cat returned home, carrying memories of a magical adventure."),
    ("end the story", "The dragon finally found friendship and was no longer lonely."),
    # ───────────── EXTRA DEPTH (IMPORTANT) ─────────────
    ("tell me another story", "Once there was a turtle who dreamed of racing with the fastest animals."),
    ("another story", "A young girl once found a hidden door in her house that led to a secret world."),
    ("one more story", "A tiny fish once explored the deepest part of the ocean."),
    ("different story", "A boy once discovered he could talk to animals."),
    ("new story", "A girl once followed a glowing light into a mysterious forest."),
    # ───────────── EMOTIONAL STORIES ─────────────
    ("tell me a sad story", "Once there was a lonely dog waiting for its owner who never returned."),
    ("tell me a happy story", "A small child found a lost puppy and gave it a loving home."),
    ("tell me an adventure story", "A group of friends went on a journey to climb the tallest mountain."),
    ("tell me a mystery story", "A strange shadow appeared every night near the old house."),
    ("tell me a brave story", "A young warrior stood up to protect his village from danger."),
    # ───────────── STRONG CONTINUATION TRAINING ─────────────
    ("what happened after", "The friends faced many challenges but worked together to overcome them."),
    ("what happened later", "As time passed, the mystery slowly began to unfold."),
    ("what did they do next", "They decided to follow the path deeper into the unknown."),
    ("what came next", "A sudden storm changed everything for them."),
    ("and after that", "They discovered something that changed their lives forever."),
    # ───────────── CLOSING FLOW ─────────────
    (
        "tell me the full story",
        "Once upon a time, a fox went on an adventure, helped a friend, and returned home wiser and happier.",
    ),
    ("summarize the story", "It is a story about courage, friendship, and discovery."),
    ("what is the lesson", "The lesson is that courage and kindness can lead to great things."),
    ("what do we learn", "We learn that helping others brings happiness."),
    ("moral of the story", "Always be brave and kind."),
]
# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------


def augment_pairs(pairs):
    augmented = []
    for p, r in pairs:
        augmented.extend(
            [
                (p, r),
                (p.lower(), r),
                (p + " please", r),
                ("please " + p, r),
                ("can you " + p, r),
                ("could you " + p, r),
            ]
        )
    return augmented


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ChatDataset(Dataset):
    def __init__(self, pairs, vocab: Vocabulary, seq_len: int = 64, repeat: int = 10):
        self.samples = []

        bos = vocab.token2id.get("<BOS>", 2)
        eos = vocab.token2id.get("<EOS>", 3)
        pad = vocab.token2id.get("<PAD>", 0)

        max_len = seq_len + 1

        for _ in range(repeat):
            for prompt, response in pairs:
                user_ids = vocab.encode(["user", ":"] + tokenize(clean_text(prompt)))
                assistant_ids = vocab.encode(["assistant", ":"] + tokenize(clean_text(response)))

                full_ids = [bos] + user_ids + assistant_ids + [eos]

                mask = [0] + [0] * len(user_ids) + [1] * len(assistant_ids) + [1]

                full_ids = full_ids[:max_len]
                mask = mask[:max_len]

                while len(full_ids) < max_len:
                    full_ids.append(pad)
                    mask.append(0)

                x = full_ids[:-1]
                y = full_ids[1:]
                y_mask = mask[1:]

                self.samples.append(
                    (
                        torch.tensor(x, dtype=torch.long),
                        torch.tensor(y, dtype=torch.long),
                        torch.tensor(y_mask, dtype=torch.float),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--chat_data", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Path(args.data_dir)
    ckpt = Path(args.ckpt_dir)
    ckpt.mkdir(exist_ok=True)

    # ── Load vocab ─────────────────────────────────────────────
    vocab = Vocabulary.load(data / "vocab.json")
    print(f"Vocab size : {len(vocab):,}")

    # ── Load data ──────────────────────────────────────────────
    if args.chat_data:
        pairs = json.load(open(args.chat_data))
    else:
        pairs = BUILTIN_CHAT_DATA

    pairs = augment_pairs(pairs)
    print(f"Training pairs: {len(pairs)}")

    # ── Load pretrained model ──────────────────────────────────
    ckpt_path = ckpt / "pretrain_best.pt"
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = saved["args"]

    model = GRULanguageModel(
        vocab_size=len(vocab),
        embed_dim=a["embed_dim"],
        hidden_dim=a["hidden_dim"],
        n_layers=a["n_layers"],
        dropout=0.2,
    ).to(device)

    model.load_state_dict(saved["model_state"])
    print("Loaded pre-trained model")

    # ── Dataset ────────────────────────────────────────────────
    dataset = ChatDataset(pairs, vocab, seq_len=args.seq_len, repeat=args.repeat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Samples: {len(dataset)} | Batches: {len(loader)}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

    best_loss = float("inf")

    # ── Training loop ──────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        t0 = time.time()

        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            logits, _ = model(x)

            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            loss = loss * mask.view(-1)

            n = mask.sum()
            if n == 0:
                continue

            loss = loss.sum() / n

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item() * n.item()
            total_tokens += n.item()

        avg_loss = total_loss / max(total_tokens, 1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d} | Loss {avg_loss:.4f} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": model.state_dict(), "base_args": a}, ckpt / "finetune_best.pt")
            print("✓ Saved best model")

    print(f"\nBest Loss: {best_loss:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
