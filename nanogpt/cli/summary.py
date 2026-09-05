"""Print the layer-by-layer parameter breakdown for a preset.

    python -m nanogpt.cli.summary --preset qa
"""

import argparse

from nanogpt import config as presets
from nanogpt.model import NanoGPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["shakespeare", "qa"], default="shakespeare")
    args = p.parse_args()

    cfg = {"shakespeare": presets.SHAKESPEARE, "qa": presets.QA}[args.preset]
    model = NanoGPT(cfg)
    print(model.parameter_summary())


if __name__ == "__main__":
    main()
