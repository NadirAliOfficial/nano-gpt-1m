"""Web chat UI for NanoQA.

    python -m nanogpt.web.qa --checkpoint checkpoints/qa.pt
"""

import argparse

import torch

from nanogpt.sampling import answer
from nanogpt.tokenizer import ByteTokenizer
from nanogpt.training import load_checkpoint
from nanogpt.web.server import load_template, serve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/qa.pt")
    p.add_argument("--port", type=int, default=7861)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    tokenizer = ByteTokenizer()
    print(f"Loaded {args.checkpoint} ({model.count_parameters():,} params) on {device}")

    def api_ask(payload: dict) -> dict:
        question = str(payload.get("question", "")).strip()
        if not question:
            return {"error": "empty question"}
        return {"answer": answer(
            model, tokenizer, question, device=device,
            temperature=float(payload.get("temperature", 0.6)),
            top_k=int(payload.get("top_k", 20)),
        )}

    serve(port=args.port, html=load_template("qa.html"),
          endpoints={"/api/ask": api_ask})


if __name__ == "__main__":
    main()
