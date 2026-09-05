"""Sanity checks: tokenizer round-trip, param count, forward/backward, generation.

    python -m pytest -q          # or:  python tests/test_model.py
"""

import math

import torch

from nanogpt import GPTConfig, NanoGPT
from nanogpt.config import QA, SHAKESPEARE
from nanogpt.qa import build
from nanogpt.sampling import END
from nanogpt.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Hello, Mac! 1M model test \U0001F680 éàü"
    assert tok.decode(tok.encode(text)) == text
    assert len(tok) == 256


def test_param_counts_in_range():
    for preset, lo, hi in [(SHAKESPEARE, 900_000, 1_200_000), (QA, 4_000_000, 6_000_000)]:
        n = NanoGPT(preset).count_parameters()
        assert lo <= n <= hi, f"{n} out of range [{lo}, {hi}]"


def test_forward_backward_and_generate():
    cfg = GPTConfig(block_size=64, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    model = NanoGPT(cfg)

    B, T = 2, 16
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    logits, loss = model(x, y)
    assert logits.shape == (B, T, 256)
    assert loss.item() > 0
    # untrained loss should sit near ln(vocab)
    assert abs(loss.item() - math.log(256)) < 1.0

    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    out = model.generate(x[:1, :4], max_new_tokens=10, temperature=0.8, top_k=20)
    assert out.shape == (1, 14)


def test_qa_corpus_shape():
    text = build(target_examples=2000, seed=0)
    assert text.count(END) > 500
    assert "### Question: " in text and "### Answer: " in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall passed")
