"""Text generation helpers shared by the CLIs and web UIs."""

from typing import Generator

import torch

from nanogpt.model import NanoGPT
from nanogpt.tokenizer import ByteTokenizer

# Instruction format used by the NanoQA corpus and chat.
PROMPT_Q = "### Question: "
PROMPT_A = "\n### Answer: "
END = "<|end|>"
_QA_STOPS = (END, "\n###", "\n\n", "\n")


def _encode_prompt(tokenizer: ByteTokenizer, prompt: str, device) -> torch.Tensor:
    ids = tokenizer.encode(prompt) or tokenizer.encode(" ")
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def complete(
    model: NanoGPT, tokenizer: ByteTokenizer, prompt: str, *,
    max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 40,
    device: torch.device = torch.device("cpu"),
) -> str:
    idx = _encode_prompt(tokenizer, prompt, device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0])


@torch.no_grad()
def stream_complete(
    model: NanoGPT, tokenizer: ByteTokenizer, prompt: str, *,
    max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 40,
    device: torch.device = torch.device("cpu"),
) -> Generator[str, None, None]:
    """Yield decoded text pieces, buffering bytes so multi-byte UTF-8 stays intact."""
    idx = _encode_prompt(tokenizer, prompt, device)
    buf = bytearray()
    for token_id in model.stream_generate(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        buf.append(max(0, min(255, int(token_id))))
        try:
            piece = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue
        buf.clear()
        yield piece
    if buf:
        yield buf.decode("utf-8", errors="replace")


@torch.no_grad()
def answer(
    model: NanoGPT, tokenizer: ByteTokenizer, question: str, *,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.6, top_k: int = 20, max_new_tokens: int = 64,
) -> str:
    """Ask the NanoQA model one templated question and return just the answer."""
    idx = _encode_prompt(tokenizer, f"{PROMPT_Q}{question}{PROMPT_A}", device)
    buf = bytearray()
    for token_id in model.stream_generate(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        buf.append(max(0, min(255, int(token_id))))
        text = buf.decode("utf-8", errors="ignore")
        if any(s in text for s in _QA_STOPS):
            break
    text = buf.decode("utf-8", errors="ignore")
    for stop in _QA_STOPS:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()
