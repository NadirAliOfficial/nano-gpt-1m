"""Tokenizers. Byte-level is the default: 256 tokens, zero out-of-vocab."""

from typing import List, Union

import torch

Tokens = Union[List[int], "torch.Tensor"]


class ByteTokenizer:
    """Maps raw UTF-8 bytes to token IDs (0-255)."""

    vocab_size = 256
    eos_token_id = 0

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: Tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
        return bytes(max(0, min(255, int(t))) for t in tokens).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size


class CharTokenizer:
    """Character-level tokenizer, optionally fit to a corpus."""

    def __init__(self, text: str = None):
        if text is not None:
            chars = sorted(set(text))
        else:
            chars = sorted([chr(i) for i in range(32, 127)] + ["\n", "\t", "\r"])
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, tokens: Tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
        return "".join(self.itos.get(int(t), "?") for t in tokens)

    def __len__(self) -> int:
        return self.vocab_size


def get_tokenizer(name: str = "byte", text_corpus: str = None):
    name = name.lower()
    if name == "byte":
        return ByteTokenizer()
    if name == "char":
        return CharTokenizer(text_corpus)
    raise ValueError(f"unknown tokenizer '{name}' (choose 'byte' or 'char')")
