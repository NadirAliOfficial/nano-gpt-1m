"""Ask the trained NanoQA model questions.

    python -m nanogpt.cli.chat --question "What is 23 plus 19?"
    python -m nanogpt.cli.chat                 # interactive loop
"""

import argparse

import torch

from nanogpt.sampling import answer
from nanogpt.tokenizer import ByteTokenizer
from nanogpt.training import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/qa.pt")
    p.add_argument("--question", default=None)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} | {model.count_parameters():,} params | val loss {ckpt.get('val_loss', float('nan')):.3f}")
    tokenizer = ByteTokenizer()

    def ask(q):
        return answer(model, tokenizer, q, device=device, temperature=args.temp, top_k=args.top_k)

    if args.question is not None:
        print(ask(args.question))
        return

    print("\nAsk a question (/exit to quit). Try: 'capital of Italy?', '6 times 7?', 'opposite of up?'\n")
    while True:
        try:
            q = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not q:
            continue
        if q in ("/exit", "/quit"):
            return
        print("bot>", ask(q))


if __name__ == "__main__":
    main()
