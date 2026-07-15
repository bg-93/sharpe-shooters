#!/usr/bin/env python3
"""Rank instruments for mean-reversion quality using the EV proxy
(sigma / cost) * (1 / half_life), computed on a training window only.

sigma      = std of the residual (price - rolling mean), i.e. typical
             dislocation size available to capture per round trip.
cost       = round-trip commission in dollars at that price level.
half_life  = AR(1) half-life of the residual, i.e. how fast dislocations
             revert (faster reversion = more round trips per period).

Then validates candidate CORE_ASSETS lists out-of-sample on the held-out
window by running the actual strategy with eval.py fee semantics.

Usage (from repo root):
    python backtesting/rank_assets.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import teamName as tn

TRAIN_END = 250          # train on days [0, 250)
ROLL_WIN = 30            # residual = price - 30d rolling mean (matches SLOW_LB)
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002


def load_prices():
    df = pd.read_csv(REPO_ROOT / "prices.txt", sep=r"\s+", header=0)
    return df.values.T.astype(float)


def ev_proxy(prices, start, end):
    """Per-asset EV proxy on the window [start, end)."""
    n_inst = prices.shape[0]
    comm = np.full(n_inst, DEFAULT_COMM)
    comm[0] = INST0_COMM

    rows = []
    for i in range(n_inst):
        p = prices[i, start:end]
        roll = pd.Series(p).rolling(ROLL_WIN).mean().values
        res = (p - roll)[ROLL_WIN:]
        sigma = res.std()

        # AR(1) half-life of the residual.
        r0, r1 = res[:-1], res[1:]
        rho = np.corrcoef(r0, r1)[0, 1]
        if 0 < rho < 1:
            half_life = np.log(0.5) / np.log(rho)
        else:
            half_life = np.inf

        cost = 2.0 * comm[i] * p.mean()  # round-trip commission in dollars
        ev = (sigma / cost) / half_life if np.isfinite(half_life) else 0.0
        rows.append((i, sigma, half_life, cost, ev))

    return pd.DataFrame(rows, columns=["inst", "sigma", "half_life", "cost", "ev"])


def run_segment(prices, start, end):
    """Run the actual strategy with eval.py fee semantics; return (mu, sd, score)."""
    n_inst, _ = prices.shape
    comm_rate = np.full(n_inst, DEFAULT_COMM)
    comm_rate[0] = INST0_COMM
    lim = np.full(n_inst, 10000.0)
    lim[0] = 100000.0

    tn.reset_state()
    cash = value = comm = 0.0
    cur_pos = np.zeros(n_inst)
    pll = []
    for t in range(start, end + 1):
        h = prices[:, :t]
        p = h[:, -1]
        if t < end:
            pos_lim = (lim / p).astype(int)
            new_pos = np.clip(tn.getMyPosition(h), -pos_lim, pos_lim).astype(int)
        else:
            new_pos = cur_pos.copy()
        d = new_pos - cur_pos
        cash -= p.dot(d) + comm
        comm = np.sum(p * np.abs(d) * comm_rate)
        cur_pos = new_pos
        today_pl = cash + cur_pos.dot(p) - value
        value = cash + cur_pos.dot(p)
        if t > start:
            pll.append(today_pl)

    pll = np.array(pll)
    mu, sd = pll.mean(), pll.std()
    sr2 = 250.0 * mu * mu / (sd * sd) if sd > 0 else 0.0
    score = mu * sr2 / (sr2 + 1.0) if mu > 0 else mu
    return mu, sd, score


def main():
    prices = load_prices()
    n_days = prices.shape[1]

    train = ev_proxy(prices, 0, TRAIN_END).sort_values("ev", ascending=False)
    print("==== EV proxy ranking (train days 0-%d) ====" % (TRAIN_END - 1))
    print(train.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    current = set(tn.CORE_ASSETS.tolist())
    ranked = train["inst"].tolist()

    candidates = {
        "current (20 names)": sorted(current),
        "top20 by EV": sorted(ranked[:20]),
        "top15 by EV": sorted(ranked[:15]),
        "top25 by EV": sorted(ranked[:25]),
        "current minus bottom-EV laggards": sorted(
            current - set(train.sort_values("ev").head(10)["inst"]) | {0}
        ),
    }

    print("\n==== OOS validation (days %d-%d, eval fee semantics) ====" % (TRAIN_END, n_days))
    original = tn.CORE_ASSETS.copy()
    for name, assets in candidates.items():
        tn.CORE_ASSETS = np.array(assets, dtype=int)
        mu, sd, score = run_segment(prices, TRAIN_END, n_days)
        print(f"{name:38s} n={len(assets):2d}  mean={mu:8.1f} std={sd:8.1f} score={score:8.2f}")
    tn.CORE_ASSETS = original


if __name__ == "__main__":
    main()
