"""Train NanoGPT-1M on tinyshakespeare.

    python -m nanogpt.cli.train_shakespeare --steps 1500 --batch-size 32
"""

import argparse
from dataclasses import replace

import torch

from nanogpt.config import SHAKESPEARE
from nanogpt.data import TextDataset, get_dataset
from nanogpt.model import NanoGPT
from nanogpt.sampling import complete
from nanogpt.tokenizer import ByteTokenizer
from nanogpt.training import get_device, train_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=SHAKESPEARE.block_size)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--dataset", default="shakespeare")
    p.add_argument("--corpus-path", default=None, help="use a custom .txt instead of tinyshakespeare")
    p.add_argument("--checkpoint", default="checkpoints/shakespeare.pt")
    args = p.parse_args()

    device = get_device()
    tokenizer = ByteTokenizer()
    dataset = TextDataset(get_dataset(args.dataset, args.corpus_path), tokenizer)

    config = replace(SHAKESPEARE, vocab_size=len(tokenizer), block_size=args.block_size)
    model = NanoGPT(config).to(device)

    def sample(step: int):
        prompt = "\nCITIZEN:\n"
        text = complete(model, tokenizer, prompt, max_new_tokens=150,
                        temperature=0.8, top_k=40, device=device)
        print(f"\n--- sample @ step {step} ---\n{text}\n" + "-" * 30)

    train_loop(
        model, dataset, config, device=device,
        steps=args.steps, batch_size=args.batch_size, max_lr=args.lr,
        warmup=100, eval_interval=args.eval_interval,
        checkpoint_path=args.checkpoint, on_sample=sample,
    )


if __name__ == "__main__":
    main()
