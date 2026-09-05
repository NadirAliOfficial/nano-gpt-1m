"""NanoGPT: an autoregressive byte-level Transformer decoder."""

import math
from typing import Generator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with scaled dot-product attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")
        # kept for the manual-attention fallback on pre-2.0 torch; harmless otherwise
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self._has_sdpa:
            # No dropout_p here: the MPS SDPA kernel does not support it, and
            # resid/mlp/embedding dropout already regularize a model this small.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        rows = [f"{'Component':<35} | {'Shape':<20} | {'Parameters':>12}", "-" * 74]
        total = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            total += p.numel()
            rows.append(f"{name:<35} | {str(list(p.shape)):<20} | {p.numel():>12,}")
        rows.append("-" * 74)
        rows.append(f"{'TOTAL TRAINABLE PARAMETERS':<35} | {'':<20} | {total:>12,}")
        return "\n".join(rows)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        logits = self.lm_head(x[:, [-1], :])  # last position only
        return logits, None

    def _next_logits(self, idx, temperature, top_k):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        return logits

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int,
        temperature: float = 0.8, top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self._next_logits(idx, temperature, top_k)
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def stream_generate(
        self, idx: torch.Tensor, max_new_tokens: int,
        temperature: float = 0.8, top_k: Optional[int] = 40,
    ) -> Generator[int, None, None]:
        for _ in range(max_new_tokens):
            logits = self._next_logits(idx, temperature, top_k)
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield int(idx_next.item())
