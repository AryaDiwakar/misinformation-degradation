#!/usr/bin/env python3
"""Fix slide number placeholders to match slide positions.

Usage:
    python3 fix_numbers.py [path/to/deck.pptx]

The path defaults to the "Review 1" deck in the repository root. The deck is
written in place unless ``--output`` is given.
"""
import argparse
import os
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parent
DEFAULT_DECK = ROOT / "Review 1 - 22 07 2026.pptx"

slide_number_map = {
    0: 1,  # Title
    1: 2,  # Aim
    2: 3,  # Abstract
    3: 4,  # Methodology (new)
    4: 5,  # Models & Feature Engineering (new)
    5: 6,  # Implementation & Code Structure (new)
    6: 7,  # Literature Review (old slide 4)
    7: 8,  # Expected Results & Evaluation (new)
    8: 9,  # Research Gap (old slide 5)
    9: 10,  # Objectives (old slide 6)
    10: 11,  # Timeline & Future Work (new)
    11: 12,  # References (old slide 7)
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", nargs="?", default=str(DEFAULT_DECK))
    parser.add_argument("--output", default=None, help="Write to a different file")
    args = parser.parse_args()

    deck = Path(args.deck)
    if not deck.is_file():
        raise SystemExit(f"Deck not found: {deck}")

    prs = Presentation(str(deck))
    for slide_idx, slide in enumerate(prs.slides):
        expected_num = slide_number_map.get(slide_idx)
        if expected_num is None:
            continue
        for shape in slide.shapes:
            if shape.is_placeholder and shape.placeholder_format.idx == 8:
                shape.text = str(expected_num)

    out = Path(args.output) if args.output else deck
    prs.save(str(out))
    print(f"Saved slide numbers to {out}")


if __name__ == "__main__":
    main()
