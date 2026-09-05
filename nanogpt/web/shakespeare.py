"""Web UI for NanoGPT-1M.

    python -m nanogpt.web.shakespeare --checkpoint checkpoints/shakespeare.pt
"""

import argparse

import torch

from nanogpt.sampling import complete
from nanogpt.tokenizer import ByteTokenizer
from nanogpt.training import load_checkpoint
from nanogpt.web.server import load_template, serve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/shakespeare.pt")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    tokenizer = ByteTokenizer()
    print(f"Loaded {args.checkpoint} ({model.count_parameters():,} params) on {device}")

    def api_generate(payload: dict) -> dict:
        max_tokens = int(payload.get("max_tokens", 200))
        text = complete(
            model, tokenizer, payload.get("prompt", "ROMEO:"),
            max_new_tokens=max_tokens,
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload.get("top_k", 40)),
            device=device,
        )
        return {"generated_text": text, "new_tokens": max_tokens}

    serve(port=args.port, html=load_template("shakespeare.html"),
          endpoints={"/api/generate": api_generate})


if __name__ == "__main__":
    main()
