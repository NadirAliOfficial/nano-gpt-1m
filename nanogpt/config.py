"""Model configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 256      # byte-level tokenizer (UTF-8 bytes)
    block_size: int = 256      # maximum context window (tokens)
    n_layer: int = 4           # number of Transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 144          # embedding dimension
    dropout: float = 0.05
    bias: bool = False         # no bias in Linear / LayerNorm (modern LLM default)


# Named presets used by the CLIs.
SHAKESPEARE = GPTConfig(block_size=256, n_layer=4, n_head=6, n_embd=144, dropout=0.05)
QA = GPTConfig(block_size=128, n_layer=6, n_head=8, n_embd=256, dropout=0.0)
