"""Generate the synthetic NanoQA corpus.

    python -m nanogpt.cli.build_qa_corpus --examples 60000
"""

import argparse
import os

from nanogpt.qa import build
from nanogpt.sampling import END


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/qa_corpus.txt")
    p.add_argument("--examples", type=int, default=60000)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    text = build(args.examples, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.out}: {len(text):,} chars, {text.count(END):,} examples.")


if __name__ == "__main__":
    main()
