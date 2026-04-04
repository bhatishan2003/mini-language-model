"""
inference.py
============
Step 4: Load a fine-tuned checkpoint and generate chat responses.

Can be used standalone (CLI) or imported by the web app.

CLI usage:
    python inference.py
    python inference.py --message "What is deep learning?"
    python inference.py --temperature 0.6 --top_k 20
"""

import argparse
import re

import torch

from model import GRULanguageModel
from data_prep import Vocabulary, clean_text, tokenize


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------


class ChatEngine:
    """
    Wraps vocab + model and exposes a single `respond(message)` method.
    """

    def __init__(
        self,
        vocab_path: str = "data/vocab.json",
        ckpt_path: str = "checkpoints/finetune_best.pt",
        device: str = "auto",
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 30,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.vocab = Vocabulary.load(vocab_path)
        self.eos_id = self.vocab.token2id.get("<EOS>", 3)
        self.bos_id = self.vocab.token2id.get("<BOS>", 2)
        self.max_new = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        a = ckpt["base_args"]
        self.model = GRULanguageModel(
            vocab_size=len(self.vocab),
            embed_dim=a["embed_dim"],
            hidden_dim=a["hidden_dim"],
            n_layers=a["n_layers"],
            dropout=0.0,  # no dropout at inference
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[ChatEngine] Loaded '{ckpt_path}' on {self.device}")

    # ------------------------------------------------------------------
    def _build_prompt_ids(self, message: str) -> list[int]:
        """Convert a user message into the prompt-ID sequence the model expects."""
        cleaned = clean_text(message)
        tokens = ["user", ":"] + tokenize(cleaned) + ["assistant", ":"]
        ids = [self.bos_id] + self.vocab.encode(tokens)
        return ids

    # ------------------------------------------------------------------
    def _ids_to_text(self, ids: list[int]) -> str:
        """Decode generated IDs back to a readable string."""
        tokens = self.vocab.decode(ids)
        # stop at EOS token string if present
        if "<EOS>" in tokens:
            tokens = tokens[: tokens.index("<EOS>")]
        # Remove assistant/user markers that leaked into output
        text = " ".join(tokens)
        # Clean up spacing around punctuation
        text = re.sub(r"\s([.,!?;:])", r"\1", text)
        text = text.strip()
        return text if text else "I'm not sure how to respond to that."

    # ------------------------------------------------------------------
    def respond(self, message: str) -> str:
        prompt_ids = self._build_prompt_ids(message)
        generated = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=self.max_new,
            temperature=self.temperature,
            top_k=self.top_k,
            eos_id=self.eos_id,
            device=self.device,
        )
        return self._ids_to_text(generated)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="data/vocab.json")
    parser.add_argument("--ckpt", default="checkpoints/finetune_best.pt")
    parser.add_argument("--message", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=80)
    args = parser.parse_args()

    engine = ChatEngine(
        vocab_path=args.vocab,
        ckpt_path=args.ckpt,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
    )

    if args.message:
        print(f"User: {args.message}")
        reply = engine.respond(args.message)
        print(f"Assistant: {reply}")
        return

    # Interactive loop
    print("\nMiniGPT Chat  (type 'quit' to exit)\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("quit", "exit", "q"):
            break
        if not msg:
            continue
        reply = engine.respond(msg)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
