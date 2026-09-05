"""
Unit tests and sanity checks for NanoGPT-1M.
Run: python test_model.py
"""

import torch
from tokenizer import ByteTokenizer, CharTokenizer
from model import GPTConfig, NanoGPT


def test_tokenizer():
    print("Testing ByteTokenizer...")
    tok = ByteTokenizer()
    text = "Hello Mac M4 Pro! 1M Model Test 🚀"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert text == decoded, f"Mismatch: {text} != {decoded}"
    assert len(tok) == 256
    print("✓ ByteTokenizer passed.")


def test_model_architecture():
    print("\nTesting Model Architecture...")
    config = GPTConfig(
        vocab_size=256,
        block_size=128,
        n_layer=4,
        n_head=6,
        n_embd=144,
        dropout=0.0,
        bias=False,
    )
    model = NanoGPT(config)
    total_params = model.count_parameters()
    print(f"Total trainable parameters: {total_params:,}")

    # Verify ~1M target
    assert 900_000 <= total_params <= 1_200_000, f"Parameter count {total_params} out of target ~1M range!"
    print(f"✓ Parameter count is {total_params:,} (within 1M target).")

    # Forward pass test
    B, T = 2, 16
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    logits, loss = model(x, y)
    assert logits.shape == (B, T, 256), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss computation failed!"
    print(f"✓ Forward pass and loss calculation successful (Loss: {loss.item():.4f}).")

    # Autoregressive generation test
    gen = model.generate(x[:1, :4], max_new_tokens=10, temperature=0.8, top_k=20)
    assert gen.shape == (1, 14), f"Generated shape mismatch: {gen.shape}"
    print(f"✓ Autoregressive generation successful (Tokens: {gen.shape[1]}).")

    # Backward pass test
    loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Some parameters missing gradients!"
    print("✓ Backward pass & gradient propagation successful.")


if __name__ == "__main__":
    test_tokenizer()
    test_model_architecture()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
