# nano-gpt-1m

Two tiny GPT-style language models built from scratch in pure PyTorch, sharing one
modular codebase. Runs on Apple Silicon (MPS), CUDA, or CPU — no GPU cluster needed.

| Model | Params | Trained on | What it does |
|---|---|---|---|
| **NanoGPT-1M** | 1,071,648 | tinyshakespeare | continues text in a Shakespeare style |
| **NanoQA** | 4,823,552 | synthetic Q&A corpus | answers a fixed set of templated questions |

Both are byte-level (256-token vocab, no out-of-vocab), decoder-only, pre-LayerNorm
Transformers with weight-tied embeddings and GPT-2 style scaled-residual init.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Everything below is also exposed as console scripts (`nanoqa-train`, `nanogpt-generate`, …) —
see `[project.scripts]` in `pyproject.toml`.

---

## NanoGPT-1M — Shakespeare

```bash
python -m nanogpt.cli.summary --preset shakespeare      # parameter breakdown
python -m nanogpt.cli.train_shakespeare --steps 1500    # ~100s on MPS, val loss ~1.83
python -m nanogpt.cli.generate --prompt "ROMEO:" --tokens 200
python -m nanogpt.cli.generate                          # interactive REPL
python -m nanogpt.web.shakespeare                       # web UI at http://localhost:7860
```

Output has real words and correct speaker-turn formatting but no sentence-level
coherence. That is the ceiling at 1M parameters and byte-level tokens.

## NanoQA — question answering

```bash
python -m nanogpt.cli.build_qa_corpus --examples 60000  # -> data/qa_corpus.txt
python -m nanogpt.cli.train_qa --steps 4000             # ~6 min on MPS, val loss ~0.17
python -m nanogpt.cli.chat --question "What is 23 plus 19?"
python -m nanogpt.cli.chat                              # interactive loop
python -m nanogpt.web.qa                                # web chat UI at http://localhost:7861
```

The corpus (`nanogpt/qa/corpus.py`) is fully synthetic: addition/subtraction (0–99),
the times table (0–12), number words, ~50 fixed facts, 24 opposites, word-length
counting, and greetings. NanoQA answers **only** these template types — facts,
opposites, greetings and multiplication are reliable; multi-digit subtraction is
usually within ±2. Anything outside the templates produces confident nonsense.

Each example is formatted as:

```
### Question: What is 7 plus 5?
### Answer: 12
<|end|>
```

Inference prompts the model with `### Question: <q>\n### Answer: ` and stops at the
first `<|end|>` / newline marker.

---

## Project layout

```
nanogpt/
├── config.py        GPTConfig + SHAKESPEARE / QA presets
├── model.py         NanoGPT: attention, MLP, block, generate / stream_generate
├── tokenizer.py     ByteTokenizer (default) + CharTokenizer
├── data.py          dataset download/cache + TextDataset batch sampler
├── sampling.py      complete(), stream_complete(), answer() + the Q&A format
├── training.py      cosine_lr, estimate_loss, train_loop, load_checkpoint
├── qa/corpus.py     synthetic instruction corpus generator
├── cli/             python -m entrypoints (summary, train_*, generate, chat, build_qa_corpus)
└── web/             dependency-free HTTP UIs (server.py + shakespeare.py / qa.py + templates/)

tests/test_model.py  tokenizer round-trip, param counts, forward/backward, generation
checkpoints/         saved models (git-ignored, created by training)
data/                downloaded / generated corpora (git-ignored)
```

## Design notes

- **Byte-level tokenizer** keeps the vocab at 256, so almost all parameters go to
  attention and MLP rather than a large embedding table. `stream_complete` buffers
  bytes and only emits complete UTF-8 sequences.
- **One training loop** (`train_loop`) drives both models; the CLIs differ only in
  the config preset, the dataset, and an `on_sample` callback for live demos.
- **`scaled_dot_product_attention`** is used when available (PyTorch ≥ 2.0), with a
  manual masked-softmax fallback for older versions. Attention-weight dropout is
  disabled (the MPS SDPA kernel rejects it); the residual/MLP/embedding dropout
  layers still regularize.
- Checkpoints store the `GPTConfig`, so `load_checkpoint` reconstructs the exact
  model. `torch.load(..., weights_only=False)` is required for that.

## Tests

```bash
pip install pytest && pytest -q      # or: python tests/test_model.py
```
