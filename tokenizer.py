"""
Byte-level and Character-level Tokenizer for NanoGPT-1M.

Byte-level encoding maps every UTF-8 byte (0-255) to a token ID.
This ensures:
1. Zero Out-Of-Vocabulary (OOV) tokens - any text or code works.
2. Compact vocabulary (256 tokens), reserving parameter capacity
   for the attention layers rather than embedding matrices.
"""

from typing import List, Union
import torch


class ByteTokenizer:
    """Byte-level tokenizer that maps raw UTF-8 bytes to token IDs (0-255)."""

    def __init__(self):
        self.vocab_size = 256
        self.eos_token_id = 0

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of byte integers [0, 255]."""
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """Convert a list or tensor of byte integers back to a UTF-8 string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
        # Filter to valid 0-255 range
        byte_list = [max(0, min(255, int(t))) for t in tokens]
        return bytes(byte_list).decode("utf-8", errors="replace")

    def __len__(self):
        return self.vocab_size


class CharTokenizer:
    """Character-level tokenizer built from a specific training corpus."""

    def __init__(self, text: str = None):
        if text is not None:
            chars = sorted(list(set(text)))
            self.chars = chars
            self.vocab_size = len(chars)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
        else:
            # Default printable ASCII + newlines
            chars = [chr(i) for i in range(32, 127)] + ["\n", "\t", "\r"]
            self.chars = sorted(chars)
            self.vocab_size = len(self.chars)
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
        return "".join([self.itos.get(int(t), "?") for t in tokens])

    def __len__(self):
        return self.vocab_size


def get_tokenizer(name: str = "byte", text_corpus: str = None):
    """Factory function to get the appropriate tokenizer."""
    if name.lower() == "byte":
        return ByteTokenizer()
    elif name.lower() == "char":
        return CharTokenizer(text_corpus)
    else:
        raise ValueError(f"Unknown tokenizer type: {name}. Choose 'byte' or 'char'.")


if __name__ == "__main__":
    tok = ByteTokenizer()
    sample = "Hello, Apple M4 Pro! 🚀 1M Parameter AI Model."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Original: {sample}")
    print(f"Encoded ({len(encoded)} bytes): {encoded[:10]}...")
    print(f"Decoded:  {decoded}")
    assert sample == decoded, "ByteTokenizer roundtrip test failed!"
    print("Tokenizer self-test passed successfully!")
