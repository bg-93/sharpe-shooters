#!/usr/bin/env python3
"""Decomposition runner for candidate_day4plus.

Windows: early 100-300 | old 250-500 | oos 500-750, eval-exact fees.
Prints mean/day and score per window + min score.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting"))
sys.path.insert(0, str(HERE))

from leadlag_research import load_prices, simulate
from candidate_day4plus import Day4Plus

WINDOWS = [("early", 100, 300), ("old", 250, 500), ("oos", 500, 750)]


def run(label, cfg, prices):
    scores = []
    parts = []
    for name, a, b in WINDOWS:
        book = Day4Plus(**cfg)
        m, s, sc = simulate(prices, a, b, book.target_dollars)
        scores.append(sc)
        parts.append(f"{name}: {m:7.1f}/d {sc:8.2f}")
    print(f"{label:34s} " + " | ".join(parts) + f" | min={min(scores):8.2f}")
    return min(scores)


def main():
    prices = load_prices()

    configs = [
        ("FULL: MRgated+pairs+LL", dict()),
        ("FULL w/ ADAPTIVE pairs", dict(pairs_adaptive=True)),
        ("adaptive pairs only", dict(mr_on=False, ll_on=False,
                                     pairs_adaptive=True)),
        ("adaptive pairs + LL (cv2 arch)", dict(mr_on=False,
                                                pairs_adaptive=True)),
        ("full, MR ungated", dict(mr_gate=False)),
        ("full, MR demeaned+gated", dict(mr_demean=True)),
        ("full, MR demeaned ungated", dict(mr_demean=True, mr_gate=False)),
        ("full, no guards", dict(guards_on=False)),
        ("MR only, gated", dict(pairs_on=False, ll_on=False)),
        ("MR only, ungated", dict(pairs_on=False, ll_on=False, mr_gate=False)),
        ("MR only, demeaned ungated", dict(pairs_on=False, ll_on=False,
                                           mr_demean=True, mr_gate=False)),
        ("pairs+LL only (cnext arch)", dict(mr_on=False)),
        ("LL only", dict(mr_on=False, pairs_on=False)),
    ]
    for label, cfg in configs:
        run(label, cfg, prices)


if __name__ == "__main__":
    main()
