"""
model.py
========
GRU-based language model — optimised for CPU training.

Architecture:
  Embedding → Dropout → GRU (n_layers) → LayerNorm → Linear (vocab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,  # smaller default for CPU
        hidden_dim: int = 128,  # smaller default for CPU
        n_layers: int = 1,  # single layer for CPU speed
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input_ids, hidden=None):
        x = self.emb_drop(self.embedding(input_ids))
        out, hidden = self.gru(x, hidden)
        out = self.layer_norm(out)
        out = self.out_drop(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 60,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: int | None = None,
        device: torch.device | None = None,
    ) -> list[int]:
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        ids = list(prompt_ids)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        _, hidden = self.forward(inp)

        for _ in range(max_new_tokens):
            last = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
            logits, hidden = self.forward(last, hidden)
            logits = logits[0, -1, :]

            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = F.softmax(topk_vals, dim=-1)
                    sampled = torch.multinomial(probs, 1)
                    next_id = int(topk_idx[sampled])
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, 1))

            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return ids[len(prompt_ids) :]


if __name__ == "__main__":
    m = GRULanguageModel(500)
    x = torch.randint(0, 500, (4, 16))
    logits, h = m(x)
    print(f"logits: {logits.shape}  hidden: {h.shape}")
    print(f"params: {sum(p.numel() for p in m.parameters()):,}")
