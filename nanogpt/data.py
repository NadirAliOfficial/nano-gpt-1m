"""Text dataset loading, caching, and batch sampling."""

import os
import urllib.request
from typing import Tuple

import torch

from nanogpt.tokenizer import ByteTokenizer

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_CACHE_DIR = "data"


def get_dataset(name: str = "shakespeare", path: str = None, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Return raw text for a named dataset, or read `path` directly if given."""
    if path:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"Loaded '{path}': {len(text):,} chars.")
        return text

    os.makedirs(cache_dir, exist_ok=True)
    data_file = os.path.join(cache_dir, f"{name}.txt")

    if not os.path.exists(data_file):
        if name != "shakespeare":
            raise ValueError(f"unknown dataset '{name}'; pass path= for a custom corpus")
        print(f"Downloading tinyshakespeare -> {data_file}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, data_file)

    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"Loaded '{name}': {len(text):,} chars.")
    return text


class TextDataset:
    """Holds train/val token tensors and samples random contiguous batches."""

    def __init__(self, text: str, tokenizer: ByteTokenizer = None, train_val_split: float = 0.9):
        self.tokenizer = tokenizer or ByteTokenizer()
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(train_val_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        print(f"Tokens: {len(self.train_data):,} train | {len(self.val_data):,} val")

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        hi = len(data) - block_size - 1
        if hi <= 0:
            raise ValueError(f"dataset ({len(data)} tokens) too small for block_size {block_size}")
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
