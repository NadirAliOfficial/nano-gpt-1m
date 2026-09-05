"""
Training Pipeline for NanoGPT-1M.

Optimized for Apple Silicon (M4 Pro) using PyTorch Metal Performance Shaders (MPS).
Includes:
- Auto hardware detection (MPS / CPU)
- Cosine learning rate schedule with warmup
- Gradient clipping & AdamW optimizer
- Periodic validation loss and live text generation demos
- Checkpoint saving
"""

import argparse
import math
import os
import time
import torch

from model import GPTConfig, NanoGPT
from tokenizer import ByteTokenizer
from dataset import download_or_get_dataset, TextDataset


def get_device() -> torch.device:
    """Detect and return fastest available device."""
    if torch.backends.mps.is_available():
        print("🚀 Apple Silicon GPU detected! Using 'mps' backend (Metal Performance Shaders).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🚀 CUDA GPU detected! Using 'cuda'.")
        return torch.device("cuda")
    else:
        print("⚠️ GPU not detected. Using CPU.")
        return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model: NanoGPT, dataset: TextDataset, batch_size: int, block_size: int, device: torch.device, eval_iters: int = 40):
    """Estimate average loss across multiple random batches."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    dataset_name: str = "shakespeare",
    max_steps: int = 1500,
    batch_size: int = 32,
    block_size: int = 256,
    max_lr: float = 8e-4,
    min_lr: float = 8e-5,
    warmup_steps: int = 100,
    eval_interval: int = 250,
    sample_interval: int = 250,
    checkpoint_path: str = "model_1m.pt",
    custom_dataset_path: str = None,
):
    device = get_device()
    tokenizer = ByteTokenizer()

    # 1. Load Dataset
    raw_text = download_or_get_dataset(dataset_name, custom_dataset_path)
    dataset = TextDataset(raw_text, tokenizer)

    # 2. Instantiate Model
    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=block_size,
        n_layer=4,
        n_head=6,
        n_embd=144,
        dropout=0.05,
        bias=False,
    )
    model = NanoGPT(config).to(device)

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE & PARAMETERS:")
    print("=" * 60)
    print(model.parameter_summary())
    total_params = model.count_parameters()
    print(f"\nTarget: ~1,000,000 | Actual: {total_params:,} parameters ({total_params / 1e6:.2f}M)")
    print("=" * 60 + "\n")

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    best_val_loss = float("inf")
    start_time = time.time()
    print(f"Starting training for {max_steps} steps (batch_size={batch_size}, context={block_size})...\n")

    for step in range(max_steps):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample batch & forward
        x, y = dataset.get_batch("train", batch_size, block_size, device)
        logits, loss = model(x, y)

        # Backward & optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Wait for MPS operations to finish to measure accurate step time
        if device.type == "mps":
            torch.mps.synchronize()
        dt = (time.time() - t0) * 1000  # ms

        # Periodic evaluation
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model, dataset, batch_size, block_size, device)
            tokens_per_sec = (batch_size * block_size) / (dt / 1000) if dt > 0 else 0
            print(
                f"Step {step:4d}/{max_steps} | "
                f"Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | Step Time: {dt:.1f}ms ({tokens_per_sec:,.0f} tok/s)",
                flush=True
            )

            # Save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "step": step,
                    "val_loss": best_val_loss,
                    "total_params": total_params,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  --> Saved new best checkpoint to {checkpoint_path} (val_loss: {best_val_loss:.4f})", flush=True)

        # Periodic sample generation demo
        if (step > 0 and step % sample_interval == 0) or step == max_steps - 1:
            model.eval()
            prompt = "\nCITIZEN:\n" if dataset_name == "shakespeare" else "Once upon a time "
            prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            gen_tokens = model.generate(prompt_tokens, max_new_tokens=150, temperature=0.8, top_k=40)
            sample_text = tokenizer.decode(gen_tokens[0])
            print("\n" + "-" * 50, flush=True)
            print(f"[LIVE GENERATION AT STEP {step}]:", flush=True)
            print(sample_text, flush=True)
            print("-" * 50 + "\n", flush=True)
            model.train()

    total_time = time.time() - start_time
    print(f"\n Training finished in {total_time:.1f}s ({total_time / 60:.2f} minutes)!")
    print(f"Final best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoGPT 1M on Mac")
    parser.add_argument("--dataset", type=str, default="shakespeare", help="Dataset name (shakespeare, etc.)")
    parser.add_argument("--custom_dataset", type=str, default=None, help="Path to custom .txt file")
    parser.add_argument("--steps", type=int, default=1500, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Context length")
    parser.add_argument("--lr", type=float, default=8e-4, help="Maximum learning rate")
    parser.add_argument("--eval_interval", type=int, default=200, help="Eval interval")
    parser.add_argument("--checkpoint", type=str, default="model_1m.pt", help="Checkpoint output path")

    args = parser.parse_args()
    train(
        dataset_name=args.dataset,
        max_steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_lr=args.lr,
        eval_interval=args.eval_interval,
        sample_interval=args.eval_interval,
        checkpoint_path=args.checkpoint,
        custom_dataset_path=args.custom_dataset,
    )
