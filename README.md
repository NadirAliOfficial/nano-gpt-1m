# NanoGPT-1M

A small GPT-style decoder Transformer built from scratch in pure PyTorch, tuned to run on Apple Silicon (MPS) or CPU. Two models share the same architecture code:

- **NanoGPT-1M** – ~1.07M params, byte-level, trained on tinyshakespeare for text continuation.
- **NanoQA** – ~4.9M params, byte-level, trained on a synthetic instruction corpus to answer a fixed set of templated questions (small arithmetic, a fact base, opposites, word length, greetings).

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch
```

## NanoGPT-1M (Shakespeare)

| | |
|---|---|
| Parameters | 1,071,648 |
| Layers / heads / d_model | 4 / 6 / 144 |
| Context | 256 tokens |
| Tokenizer | byte-level UTF-8 (vocab 256) |

```bash
python model.py                                    # layer-by-layer param breakdown
python train.py --steps 1500 --batch_size 32       # ~100s on MPS, val loss ~1.83
python generate.py --prompt "ROMEO:" --temp 0.8 --tokens 200
python generate.py                                 # interactive REPL
python app.py                                       # web UI at http://localhost:7860
```

Output is Shakespeare-flavoured: real words and correct speaker-turn formatting, no sentence-level coherence. That is the ceiling at 1M params.

## NanoQA (question answering)

| | |
|---|---|
| Parameters | ~4.9M |
| Layers / heads / d_model | 6 / 8 / 256 |
| Context | 128 tokens |
| Tokenizer | byte-level UTF-8 (vocab 256) |

```bash
python qa_data.py                                  # build qa_corpus.txt (~56k examples)
python train_qa.py --steps 4000                    # ~6 min on MPS, val loss ~0.17
python chat.py                                      # interactive Q&A
python chat.py --question "What is 23 plus 19?"
python chat_app.py                                  # web chat UI at http://localhost:7861
```

It only handles the templates in `qa_data.py`. Facts, opposites, greetings and the times table are reliable; multi-digit subtraction is usually within ±2. Anything outside the trained templates produces confident nonsense — expected at this size.

## Files

| File | Purpose |
|---|---|
| `model.py` | `GPTConfig` + `NanoGPT` (attention, MLP, blocks, generate/stream) |
| `tokenizer.py` | `ByteTokenizer` (and a `CharTokenizer` alternative) |
| `dataset.py` | dataset download/caching and batch sampling |
| `train.py` / `generate.py` / `app.py` | NanoGPT-1M train, CLI, web UI |
| `qa_data.py` / `train_qa.py` / `chat.py` / `chat_app.py` | NanoQA corpus, train, CLI, web UI |
| `test_model.py` | tokenizer + architecture + forward/backward sanity checks |

## Tests

```bash
python test_model.py
```
