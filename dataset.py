"""
Dataset Loader for NanoGPT-1M.

Handles downloading, caching, and batching text datasets for training.
Supports:
- Shakespeare (classic Karpathy benchmark)
- TinyStories snippet
- Custom user-provided text files
"""

import os
import urllib.request
from typing import Tuple
import torch
from tokenizer import ByteTokenizer


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_or_get_dataset(dataset_name: str = "shakespeare", custom_path: str = None) -> str:
    """Download standard dataset or load custom local file."""
    if custom_path and os.path.isfile(custom_path):
        with open(custom_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    data_file = os.path.join(DATA_DIR, f"{dataset_name}.txt")

    if not os.path.exists(data_file):
        print(f"Downloading {dataset_name} dataset...")
        try:
            if dataset_name == "shakespeare":
                urllib.request.urlretrieve(SHAKESPEARE_URL, data_file)
            else:
                # Default fallback text
                urllib.request.urlretrieve(SHAKESPEARE_URL, data_file)
            print(f"Downloaded to {data_file}")
        except Exception as e:
            print(f"Download failed ({e}), creating a fallback built-in training corpus...")
            # Create a rich fallback corpus of stories and dialogue
            corpus = [
                "Once upon a time, there was a little robot named Pip who lived inside a powerful Apple Silicon chip.",
                "Pip loved to learn. Every single day, Pip would read millions of words and calculate attention matrices.",
                "'Look at my parameters!' Pip exclaimed happily to the other threads. 'I have one million parameters!'",
                "The CPU smiled and said, 'You are fast and nimble, Pip. With Metal Performance Shaders, you can think in milliseconds.'",
                "One sunny afternoon, a human programmer sat down at their keyboard. 'Hello, little AI,' the human typed.",
                "Pip processed the byte tokens one by one: 72, 101, 108, 108, 111. The self-attention heads lit up with excitement.",
                "'Hello, creator!' Pip replied with high confidence and low cross-entropy loss.",
                "They worked together to solve puzzles, write poems, and explore algorithms.",
                "The human asked Pip, 'What is the secret to good intelligence?'",
                "Pip answered, 'Curiosity, practice, and a good loss function that converges smoothly with AdamW optimizer.'",
                "And from that day on, Pip and the programmer built wonderful models together, living happily ever after."
            ] * 500  # Repeat to make an ample training set
            with open(data_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(corpus))

    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    print(f"Loaded dataset '{dataset_name}': {len(text):,} characters ({len(text.encode('utf-8')):,} bytes).")
    return text


class TextDataset:
    """Manages train and validation token tensors and batch sampling."""

    def __init__(self, text: str, tokenizer: ByteTokenizer, train_val_split: float = 0.9):
        self.tokenizer = tokenizer
        print("Encoding dataset to byte tokens...")
        encoded = tokenizer.encode(text)
        self.data_tensor = torch.tensor(encoded, dtype=torch.long)

        n = int(train_val_split * len(self.data_tensor))
        self.train_data = self.data_tensor[:n]
        self.val_data = self.data_tensor[n:]

        print(f"Dataset split: {len(self.train_data):,} train tokens | {len(self.val_data):,} val tokens.")

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a small batch of inputs x and targets y."""
        data = self.train_data if split == "train" else self.val_data
        # Sample random offsets
        max_idx = len(data) - block_size - 1
        if max_idx <= 0:
            raise ValueError(f"Dataset length ({len(data)}) is too small for block_size ({block_size})")

        ix = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        # Pin memory or transfer to MPS/CPU device
        x, y = x.to(device), y.to(device)
        return x, y


if __name__ == "__main__":
    tok = ByteTokenizer()
    text = download_or_get_dataset("shakespeare")
    ds = TextDataset(text, tok)
    dev = torch.device("cpu")
    xb, yb = ds.get_batch("train", batch_size=4, block_size=16, device=dev)
    print("Batch x shape:", xb.shape)
    print("Batch y shape:", yb.shape)
    print("Sample x[0]:", tok.decode(xb[0]))
    print("Sample y[0]:", tok.decode(yb[0]))
