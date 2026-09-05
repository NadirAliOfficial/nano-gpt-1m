"""
Train the tiny instruction / Q&A model on the synthetic corpus.

    python qa_data.py                 # build qa_corpus.txt
    python train_qa.py --steps 4000   # train -> model_qa.pt
    python chat.py                    # ask it questions

It is a ~5M parameter byte-level GPT. It answers the templated question
types in qa_data.py and nothing else.
"""

import argparse
import math
import os
import time

import torch

from model import GPTConfig, NanoGPT
from tokenizer import ByteTokenizer
from dataset import download_or_get_dataset, TextDataset

PROMPT_Q = "### Question: "
PROMPT_A = "\n### Answer: "
END = "<|end|>"

DEMO_QUESTIONS = [
    "What is 17 plus 26?",
    "What is 90 minus 45?",
    "What is 7 times 8?",
    "What is the capital of Japan?",
    "What color is the sky?",
    "What is the opposite of hot?",
    "How many days are in a week?",
    "Who are you?",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple Silicon 'mps' backend.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Using CPU.")
    return torch.device("cpu")


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    r = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, block_size, device, iters=30):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def answer(model, tokenizer, question, device, max_new_tokens=48):
    prompt = f"{PROMPT_Q}{question}{PROMPT_A}"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    buf = bytearray()
    for tok in model.stream_generate(idx, max_new_tokens, temperature=0.6, top_k=20):
        buf.append(max(0, min(255, int(tok))))
        text = buf.decode("utf-8", errors="ignore")
        if END in text or "\n###" in text:
            break
    ans = buf.decode("utf-8", errors="ignore")
    for stop in (END, "\n###", "\n"):
        if stop in ans:
            ans = ans.split(stop)[0]
    return ans.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default="qa_corpus.txt")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval_interval", type=int, default=400)
    p.add_argument("--checkpoint", type=str, default="model_qa.pt")
    args = p.parse_args()

    if not os.path.exists(args.corpus):
        raise SystemExit(f"{args.corpus} not found. Run: python qa_data.py")

    device = get_device()
    tokenizer = ByteTokenizer()

    raw_text = download_or_get_dataset("qa", args.corpus)
    dataset = TextDataset(raw_text, tokenizer)

    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.0,
        bias=False,
    )
    model = NanoGPT(config).to(device)
    total = model.count_parameters()
    print(f"Model: {total:,} parameters ({total / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warmup = min(200, args.steps // 10)
    best_val = float("inf")
    t0 = time.time()

    for step in range(args.steps):
        lr = get_lr(step, warmup, args.steps, args.lr, args.lr / 10)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = dataset.get_batch("train", args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            if device.type == "mps":
                torch.mps.synchronize()
            losses = estimate_loss(model, dataset, args.batch_size, args.block_size, device)
            print(f"Step {step:5d}/{args.steps} | train {losses['train']:.3f} | val {losses['val']:.3f} | lr {lr:.2e}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": config,
                     "step": step, "val_loss": best_val, "total_params": total},
                    args.checkpoint,
                )
                print(f"  saved {args.checkpoint} (val {best_val:.3f})")

        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            print("  --- sample answers ---")
            for q in DEMO_QUESTIONS[:5]:
                print(f"  Q: {q}\n  A: {answer(model, tokenizer, q, device)}")
            model.train()

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s ({dt / 60:.2f} min). Best val loss {best_val:.3f}. Saved {args.checkpoint}")
    print("\nFinal check:")
    model.eval()
    for q in DEMO_QUESTIONS:
        print(f"Q: {q}\nA: {answer(model, tokenizer, q, device)}\n")


if __name__ == "__main__":
    main()
