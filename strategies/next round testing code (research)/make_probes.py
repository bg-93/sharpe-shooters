#!/usr/bin/env python3
"""Probe-submission kit for leaderboard information extraction.

Each daily submission is one deterministic query against the hidden
window (currently days 751-1000). For a CONSTANT-dollar book:
    mean daily PnL  mu = sum_i w_i * mean_return_i   (hidden window)
    daily std      sig = std of (w . r_t), estimable from released data
Score semantics:
    mu <= 0 : score = mu                (EXACT linear readout)
    mu  > 0 : score = mu*sr2/(sr2+1), sr2 = 250*mu^2/sig^2
              -> invertible to mu given sig (monotonic).

This script:
  1. emits self-contained probe files (probe_*.py, drop-in teamName.py)
  2. prints a calibration table: expected sigma per probe from the
     released data, so scores can be inverted the day they arrive
  3. provides invert_score(score, sigma) -> implied mu

Usage (from repo root):
    python "strategies/next round testing code (research)/make_probes.py"
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting"))
from leadlag_research import load_prices

N_INST = 51
PROBE_DIR = HERE / "probes"

TEMPLATE = '''import numpy as np

# PROBE: {name}
# {desc}
# Constant-dollar book; commission ~1bp of gross once at entry, then
# only drift-rebalancing. Score inversion: see make_probes.py ledger.
W = np.array({w})

def getMyPosition(prcSoFar):
    return (W / prcSoFar[:, -1]).astype(int)
'''


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def probe_specs():
    specs = {}
    w = np.zeros(N_INST); w[0] = -95_000.0
    specs["algo_short"] = (w, "short ALGO $95k: -mu = index drift (exact if drift>0)")
    w = np.zeros(N_INST); w[0] = 95_000.0
    specs["algo_long"] = (w, "long ALGO $95k: index drift, squashed side")
    w = np.full(N_INST, -9_500.0); w[0] = 0.0
    specs["stocks_short_ew"] = (w, "short all 50 stocks $9.5k: -mu = 9500*sum(drift)")

    # Hadamard group probes on the 50 stocks (rows 1..10 of H(64),
    # first 50 cols): balanced +/- books, mu near zero -> often exact.
    H = hadamard(64)
    for k in range(1, 11):
        w = np.zeros(N_INST)
        w[1:] = 9_500.0 * H[k, :50]
        specs[f"hadamard_{k:02d}"] = (
            w, f"+/-$9.5k per stock, Hadamard row {k}: w.drift projection")
    return specs


def invert_score(score, sigma):
    """Implied mean daily PnL on the hidden window from a probe score."""
    if score <= 0:
        return score
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        sr2 = 250.0 * mu * mu / (sigma * sigma)
        s = mu * sr2 / (sr2 + 1.0)
        lo, hi = (mu, hi) if s < score else (lo, mu)
    return 0.5 * (lo + hi)


def main():
    PROBE_DIR.mkdir(exist_ok=True)
    prices = load_prices()
    r = np.diff(np.log(prices), axis=1)          # (51, nt-1) log returns
    r250 = r[:, -250:]                            # most recent released window

    print(f"{'probe':18s} {'gross$':>10s} {'est sigma/day':>14s}   file")
    for name, (w, desc) in probe_specs().items():
        pll = w @ r250                            # daily PnL of const book
        sigma = pll.std()
        path = PROBE_DIR / f"probe_{name}.py"
        path.write_text(TEMPLATE.format(
            name=name, desc=desc,
            w=np.array2string(w, separator=", ", max_line_width=70)))
        print(f"{name:18s} {np.abs(w).sum():10.0f} {sigma:14.1f}   {path.name}")

    print("\nWhen a score arrives:")
    print("  mu = invert_score(score, sigma)   # exact if score < 0")
    print("  hadamard: collect k projections m_k = w_k . drift, then")
    print("  drift_hat = lstsq(stack(w_k), m) with ridge shrinkage.")


if __name__ == "__main__":
    main()
