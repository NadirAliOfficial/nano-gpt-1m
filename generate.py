"""
Interactive Inference & Text Generation for NanoGPT-1M.

Run text completions directly from your trained 1M parameter model checkpoint.
Supports:
- Temperature scaling
- Top-k filtering
- Real-time streaming generation
- Interactive CLI chat loop
"""

import argparse
import os
import sys
import time
from typing import Tuple, Optional
import torch

from model import GPTConfig, NanoGPT
from tokenizer import ByteTokenizer


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[NanoGPT, ByteTokenizer]:
    """Load model weights and configuration from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        print("Please train the model first by running: python train.py")
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = NanoGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = ByteTokenizer()
    total_params = model.count_parameters()
    print(f"Model loaded successfully! ({total_params:,} parameters)")
    if "val_loss" in checkpoint:
        print(f"Trained step: {checkpoint.get('step', '?')} | Val loss: {checkpoint['val_loss']:.4f}")
    return model, tokenizer


def stream_complete(
    model: NanoGPT,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = torch.device("cpu"),
):
    """Stream model completion token by token to standard output."""
    encoded = tokenizer.encode(prompt)
    if len(encoded) == 0:
        encoded = [tokenizer.encode(" ")[0]]

    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    # Print initial prompt
    sys.stdout.write(prompt)
    sys.stdout.flush()

    # Stream generated tokens. Byte-level tokens: a non-ASCII character spans
    # several tokens, so buffer bytes and only emit complete UTF-8 sequences.
    buf = bytearray()
    for token_id in model.stream_generate(idx, max_new_tokens, temperature, top_k):
        buf.append(max(0, min(255, int(token_id))))
        try:
            char_out = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue  # incomplete multi-byte sequence, wait for more bytes
        buf.clear()
        sys.stdout.write(char_out)
        sys.stdout.flush()
        # Fast typing delay effect (optional, keep crisp)
        time.sleep(0.005)
    if buf:
        sys.stdout.write(buf.decode("utf-8", errors="replace"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def interactive_loop(model: NanoGPT, tokenizer: ByteTokenizer, device: torch.device):
    """Interactive loop allowing the user to type prompts continuously."""
    print("\n" + "=" * 60)
    print("🤖 NanoGPT-1M Interactive Playground")
    print("Type a prompt and press ENTER to generate completions.")
    print("Commands:")
    print("  /temp <float>  - Set temperature (e.g. /temp 0.7)")
    print("  /topk <int>    - Set top-k sampling (e.g. /topk 30)")
    print("  /exit or /quit - Exit")
    print("=" * 60 + "\n")

    temperature = 0.8
    top_k = 40
    max_tokens = 250

    while True:
        try:
            prompt = input("\n📝 Enter Prompt: ")
            if not prompt.strip():
                continue

            if prompt.strip().lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Goodbye!")
                break

            if prompt.strip().startswith("/temp "):
                try:
                    temperature = float(prompt.strip().split()[1])
                    print(f"Temperature set to {temperature}")
                except Exception:
                    print("Invalid temperature format. Example: /temp 0.8")
                continue

            if prompt.strip().startswith("/topk "):
                try:
                    top_k = int(prompt.strip().split()[1])
                    print(f"Top-k set to {top_k}")
                except Exception:
                    print("Invalid top-k format. Example: /topk 40")
                continue

            print(f"\n[Generating (temp={temperature}, top_k={top_k})]:")
            print("-" * 50)
            stream_complete(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with NanoGPT-1M")
    parser.add_argument("--checkpoint", type=str, default="model_1m.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for single-run generation")
    parser.add_argument("--tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (0.1 to 1.5)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)

    if args.prompt is not None:
        print(f"\nGenerating {args.tokens} tokens with prompt: '{args.prompt}'...\n")
        stream_complete(model, tokenizer, args.prompt, args.tokens, args.temp, args.top_k, device)
    else:
        interactive_loop(model, tokenizer, device)
