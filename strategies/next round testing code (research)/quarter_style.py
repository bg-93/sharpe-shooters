#!/usr/bin/env python3
"""Quartercode-style cross-sectional ensemble (research replica).

Reverse-engineered from quartercode.webp (top performer, ~60 lines):
  - ensemble pred = weighted sum of sleeve signals
  - SLOWBOOK: multi-horizon MR  f = -0.5*((lpx[-1]-lpx[-1-h1]) + (lpx[-1]-lpx[-1-h2]))
              with f[0] = f[1:].mean()  (ALGO = index of the other 50)
  - PAIRS: spread z pushed onto both legs (pred += pk * PAIRS_A)
  - BASKET: fade each name's excess return over ALgo (basket residual)
  - pred -= pred.mean()  -> cross-sectionally demeaned, ~dollar-neutral book
  - position = LIMITS * tanh(z / TEMP)

Stateless (pure function of price history) — no FSMs, no dead-band.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from teamName import PAIRS  # 15 frozen-gamma pairs from the live book

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-9

DEFAULT = {
    "SLOWBOOK_W": 1.0,
    "SLOWBOOK_H": (20, 60),
    "PAIRS_W": 1.0,
    "PAIR_ROLL": 60,
    "BASKET_W": 1.0,
    "BASKET_LB": 40,
    "TEMP": 1.0,
    "ALGO_LIMIT": 100000.0,  # 10000.0 = size ALGO like a stock
}


def target_dollars(prc, c=DEFAULT):
    nt = prc.shape[1]
    h1, h2 = c["SLOWBOOK_H"]
    need = max(h2, c["PAIR_ROLL"], c["BASKET_LB"]) + 2
    if nt < need:
        return np.zeros(N_INST)
    lpx = np.log(np.maximum(prc, 1e-12))
    pred = np.zeros(N_INST)

    if c["SLOWBOOK_W"]:
        f = -0.5 * ((lpx[:, -1] - lpx[:, -1 - h1])
                    + (lpx[:, -1] - lpx[:, -1 - h2]))
        f[0] = f[1:].mean()
        f = f - f.mean()
        pred += c["SLOWBOOK_W"] * f / (f.std() + EPS)

    if c["PAIRS_W"]:
        w = c["PAIR_ROLL"]
        pz = np.zeros(N_INST)
        for i, j, g in PAIRS:
            s = lpx[i, -w:] - g * lpx[j, -w:]
            z = (s[-1] - s.mean()) / (s.std() + EPS)
            pz[i] -= z          # rich leg expected to underperform
            pz[j] += g * z
        pz = pz - pz.mean()
        pred += c["PAIRS_W"] * pz / (pz.std() + EPS)

    if c["BASKET_W"]:
        lb = c["BASKET_LB"]
        r = lpx[:, -1] - lpx[:, -1 - lb]
        b = -(r - r[0])         # fade excess return over the index
        b[0] = b[1:].mean()
        b = b - b.mean()
        pred += c["BASKET_W"] * b / (b.std() + EPS)

    pred = pred - pred.mean()
    z = pred / (pred.std() + EPS)
    lim = LIMITS.copy()
    lim[0] = c["ALGO_LIMIT"]
    return lim * np.tanh(z / c["TEMP"])
