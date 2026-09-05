"""
Interactive Q&A with the tiny instruction model (model_qa.pt).

    python chat.py
    python chat.py --question "What is 12 plus 30?"

It only knows the templated question types from qa_data.py
(small arithmetic, a fixed fact base, opposites, word length, greetings).
"""

import argparse
import os
import sys

import torch

from model import GPTConfig, NanoGPT
from tokenizer import ByteTokenizer

PROMPT_Q = "### Question: "
PROMPT_A = "\n### Answer: "
END = "<|end|>"


def load(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"{checkpoint_path} not found. Train first: python train_qa.py")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NanoGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {checkpoint_path} | {model.count_parameters():,} params | val loss {ckpt.get('val_loss', float('nan')):.3f}")
    return model, ByteTokenizer()


@torch.no_grad()
def ask(model, tokenizer, question, device, temperature=0.6, top_k=20, max_new_tokens=64):
    prompt = f"{PROMPT_Q}{question}{PROMPT_A}"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    buf = bytearray()
    for tok in model.stream_generate(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        buf.append(max(0, min(255, int(tok))))
        text = buf.decode("utf-8", errors="ignore")
        if END in text or "\n###" in text or "\n\n" in text:
            break
    ans = buf.decode("utf-8", errors="ignore")
    for stop in (END, "\n###", "\n\n", "\n"):
        if stop in ans:
            ans = ans.split(stop)[0]
    return ans.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="model_qa.pt")
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load(args.checkpoint, device)

    if args.question is not None:
        print(ask(model, tokenizer, args.question, device, args.temp, args.top_k))
        return

    print("\nAsk a question (Ctrl+C or /exit to quit).")
    print("Try: 'What is 23 plus 19?', 'capital of Italy?', 'opposite of up?'\n")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Bye!")
            break
        print("Bot:", ask(model, tokenizer, q, device, args.temp, args.top_k))


if __name__ == "__main__":
    main()
