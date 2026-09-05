"""Train NanoQA on the synthetic instruction corpus.

    python -m nanogpt.cli.build_qa_corpus
    python -m nanogpt.cli.train_qa --steps 4000
"""

import argparse
import os
from dataclasses import replace

from nanogpt.config import QA
from nanogpt.data import TextDataset, get_dataset
from nanogpt.model import NanoGPT
from nanogpt.qa import DEMO_QUESTIONS
from nanogpt.sampling import answer
from nanogpt.tokenizer import ByteTokenizer
from nanogpt.training import get_device, train_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/qa_corpus.txt")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--block-size", type=int, default=QA.block_size)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval-interval", type=int, default=400)
    p.add_argument("--checkpoint", default="checkpoints/qa.pt")
    args = p.parse_args()

    if not os.path.exists(args.corpus):
        raise SystemExit(f"{args.corpus} not found - run: python -m nanogpt.cli.build_qa_corpus")

    device = get_device()
    tokenizer = ByteTokenizer()
    dataset = TextDataset(get_dataset(path=args.corpus), tokenizer)

    config = replace(QA, vocab_size=len(tokenizer), block_size=args.block_size)
    model = NanoGPT(config).to(device)

    def sample(step: int):
        print("  --- sample answers ---")
        for q in DEMO_QUESTIONS[:5]:
            print(f"  Q: {q}\n  A: {answer(model, tokenizer, q, device=device)}")

    train_loop(
        model, dataset, config, device=device,
        steps=args.steps, batch_size=args.batch_size, max_lr=args.lr,
        warmup=min(200, args.steps // 10), eval_interval=args.eval_interval,
        checkpoint_path=args.checkpoint, on_sample=sample,
    )

    print("\nFinal check:")
    model.eval()
    for q in DEMO_QUESTIONS:
        print(f"Q: {q}\nA: {answer(model, tokenizer, q, device=device)}\n")


if __name__ == "__main__":
    main()
