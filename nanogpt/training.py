"""Shared training loop: device pick, LR schedule, eval, checkpointing."""

import math
import os
import time
from typing import Callable, Optional

import torch

from nanogpt.config import GPTConfig
from nanogpt.data import TextDataset
from nanogpt.model import NanoGPT


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def cosine_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, block_size, device, iters=30) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_loop(
    model: NanoGPT,
    dataset: TextDataset,
    config: GPTConfig,
    *,
    device: torch.device,
    steps: int = 1500,
    batch_size: int = 32,
    max_lr: float = 8e-4,
    min_lr: Optional[float] = None,
    warmup: Optional[int] = None,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_interval: int = 250,
    checkpoint_path: str = "checkpoints/model.pt",
    on_sample: Optional[Callable[[int], None]] = None,
) -> float:
    """Train `model` in place. Returns the best validation loss and saves the best checkpoint."""
    block_size = config.block_size
    min_lr = max_lr / 10 if min_lr is None else min_lr
    warmup = max(1, min(100, steps // 10)) if warmup is None else warmup
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    total = model.count_parameters()
    print(f"Parameters: {total:,} ({total / 1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    best_val = float("inf")
    t0 = time.time()
    model.train()

    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, warmup, steps, max_lr, min_lr)

        x, y = dataset.get_batch("train", batch_size, block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        last = step == steps - 1
        if step % eval_interval == 0 or last:
            if device.type == "mps":
                torch.mps.synchronize()
            m = estimate_loss(model, dataset, batch_size, block_size, device)
            print(f"step {step:5d}/{steps} | train {m['train']:.3f} | val {m['val']:.3f} | lr {opt.param_groups[0]['lr']:.2e}")
            if m["val"] < best_val:
                best_val = m["val"]
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": config,
                     "step": step, "val_loss": best_val, "total_params": total},
                    checkpoint_path,
                )
                print(f"  saved {checkpoint_path} (val {best_val:.3f})")
            if on_sample and (step > 0 or last):
                model.eval()
                on_sample(step)
                model.train()

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s ({dt / 60:.2f} min). Best val loss {best_val:.3f} -> {checkpoint_path}")
    return best_val


def load_checkpoint(path: str, device: torch.device):
    """Return (model, checkpoint_dict) from a saved checkpoint."""
    if not os.path.exists(path):
        raise SystemExit(f"checkpoint '{path}' not found - train first")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NanoGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
