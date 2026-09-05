"""Generate text with a trained NanoGPT-1M checkpoint.

    python -m nanogpt.cli.generate --prompt "ROMEO:" --tokens 200
    python -m nanogpt.cli.generate               # interactive REPL
"""

import argparse
import sys
import time

import torch

from nanogpt.sampling import stream_complete
from nanogpt.tokenizer import ByteTokenizer
from nanogpt.training import load_checkpoint


def _stream(model, tokenizer, prompt, device, **kw):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    for piece in stream_complete(model, tokenizer, prompt, device=device, **kw):
        sys.stdout.write(piece)
        sys.stdout.flush()
        time.sleep(0.005)
    sys.stdout.write("\n")


def repl(model, tokenizer, device):
    temperature, top_k, max_tokens = 0.8, 40, 250
    print("\nNanoGPT-1M REPL. Commands: /temp <f>, /topk <i>, /exit\n")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not prompt:
            continue
        if prompt in ("/exit", "/quit"):
            return
        if prompt.startswith("/temp "):
            temperature = float(prompt.split()[1]); print(f"temperature = {temperature}"); continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1]); print(f"top_k = {top_k}"); continue
        _stream(model, tokenizer, prompt, device,
                max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/shakespeare.pt")
    p.add_argument("--prompt", default=None)
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} | {model.count_parameters():,} params | val loss {ckpt.get('val_loss', float('nan')):.3f}")
    tokenizer = ByteTokenizer()

    if args.prompt is not None:
        _stream(model, tokenizer, args.prompt, device,
                max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k)
    else:
        repl(model, tokenizer, device)


if __name__ == "__main__":
    main()
