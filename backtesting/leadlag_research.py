#!/usr/bin/env python3
"""Lead-lag research: predict next-day returns from all lagged returns.

Motivated by the 2024 Algothon winner (lead-lag algorithm, score 1560).
Our data shows real cross-lag structure: ridge r(t+1) ~ r(t) gives mean
OOS IC ~0.056 with 17/51 assets above IC 0.1 on a chronological split.

This script:
  1. Walk-forward ridge engine (refit every RETRAIN days on all history).
  2. Asset selection by *walk-forward* IC (past predictions vs realised),
     never in-sample fit (in-sample ICs are inflated and useless).
  3. Config grid over lambda / sizing / selection threshold.
  4. Eval-exact fee semantics (1bp, 0.2bp inst 0; $10k/$100k limits).
  5. Daily-PnL correlation vs the current teamName.py book.
  6. Combination testing: current book + sleeve at fractional caps,
     with/without pair-owned name exclusion.

Windows reported:
  honest OOS  : trade days 250-499 (tuning selects here, but the config
                plateau + all-window agreement is the guard)
  early       : trade days 100-299
  reversed    : time-reversed prices, trade last 250 of that series

Usage (from repo root):
    python backtesting/leadlag_research.py            # grid + combos
    python backtesting/leadlag_research.py --quick    # headline configs only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import teamName as tn

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
COMM = np.array([0.00002] + [0.0001] * 50)
EPS = 1e-12


def load_prices():
    df = pd.read_csv(REPO_ROOT / "prices.txt", sep=r"\s+", header=0)
    return df.values.T.astype(float)


def score_from_pll(pll):
    pll = np.asarray(pll, dtype=float)
    m, s = pll.mean(), pll.std()
    sr2 = 250.0 * m * m / (s * s) if s > 0 else 0.0
    score = m * sr2 / (sr2 + 1.0) if m > 0 else m
    return m, s, score


# ----------------------------------------------------------------------
# Lead-lag sleeve: walk-forward target-dollar generator
# ----------------------------------------------------------------------

class LeadLagModel:
    """Ridge r(t+1) ~ r(t), refit every `retrain` days, walk-forward IC
    tracked per asset for selection."""

    def __init__(self, lam=50.0, retrain=50, sizing="sign", scale=3.0,
                 sel_ic=None, ic_min_obs=60, cap_frac=1.0):
        self.lam = lam
        self.retrain = retrain
        self.sizing = sizing            # "sign" | "tanh" | "linear"
        self.scale = scale
        self.sel_ic = sel_ic            # walk-forward IC threshold or None
        self.ic_min_obs = ic_min_obs    # min WF obs before selection kicks in
        self.cap_frac = cap_frac
        self.reset()

    def reset(self):
        self.W = None
        self.mu = None
        self.sd = None
        self.resid_sd = None
        self.last_fit = -1
        self.wf_preds = []              # predictions made so far
        self.wf_reals = []              # realised returns they targeted
        self.pending = None             # prediction awaiting realisation

    def _fit(self, r):
        X = r[:, :-1].T
        Y = r[:, 1:].T
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd = np.where(self.sd > 1e-12, self.sd, 1.0)
        Xs = (X - self.mu) / self.sd
        self.W = np.linalg.solve(
            Xs.T @ Xs + self.lam * np.eye(X.shape[1]), Xs.T @ Y
        )
        self.resid_sd = np.maximum(Y.std(0), 1e-8)

    def wf_ic(self):
        """Walk-forward IC per asset from stored (pred, realised) pairs."""
        if len(self.wf_preds) < self.ic_min_obs:
            return None
        P = np.array(self.wf_preds)
        R = np.array(self.wf_reals)
        ics = np.zeros(N_INST)
        for j in range(N_INST):
            if P[:, j].std() > 1e-12 and R[:, j].std() > 1e-12:
                ics[j] = np.corrcoef(P[:, j], R[:, j])[0, 1]
        return ics

    def target_dollars(self, prc):
        """prc: (N_INST, nt) price history through today. Returns sleeve
        target dollars for the coming day."""
        nt = prc.shape[1]
        if nt < 120:  # need history to fit
            return np.zeros(N_INST)
        r = np.diff(np.log(np.maximum(prc, EPS)), axis=1)

        # record realisation of yesterday's prediction
        if self.pending is not None:
            self.wf_preds.append(self.pending)
            self.wf_reals.append(r[:, -1])
            self.pending = None

        if self.W is None or (nt - self.last_fit) >= self.retrain:
            self._fit(r)
            self.last_fit = nt

        x = (r[:, -1] - self.mu) / self.sd
        pred = x @ self.W
        self.pending = pred.copy()

        z = pred / self.resid_sd
        if self.sizing == "sign":
            raw = np.sign(z)
        elif self.sizing == "tanh":
            raw = np.tanh(self.scale * z)
        else:  # linear
            raw = np.clip(self.scale * z, -1.0, 1.0)

        mask = np.ones(N_INST)
        if self.sel_ic is not None:
            ics = self.wf_ic()
            if ics is not None:
                mask = (ics > self.sel_ic).astype(float)

        return LIMITS * self.cap_frac * raw * mask


# ----------------------------------------------------------------------
# Simulation harness (eval-exact fees), generic over a target-dollar fn
# ----------------------------------------------------------------------

def simulate(prices, start, end, get_target_dollars, collect_pll=False):
    """Trade days [start, end): position set at close t, PnL accrued t->t+1.
    Matches eval.py semantics (commission on traded value; positions
    clipped to per-instrument dollar limits)."""
    n = prices.shape[0]
    cash = 0.0
    value = 0.0
    cur_pos = np.zeros(n)
    comm_pending = 0.0
    pll = []
    for t in range(start, end + 1):
        h = prices[:, :t]
        p = h[:, -1]
        if t < end:
            tgt = get_target_dollars(h)
            pos_lim = (LIMITS / p).astype(int)
            new_pos = np.clip((tgt / p).astype(int), -pos_lim, pos_lim)
        else:
            new_pos = cur_pos.copy()
        d = new_pos - cur_pos
        cash -= p.dot(d) + comm_pending
        comm_pending = np.sum(p * np.abs(d) * COMM)
        cur_pos = new_pos
        today_pl = cash + cur_pos.dot(p) - value
        value = cash + cur_pos.dot(p)
        if t > start:
            pll.append(today_pl)
    pll = np.array(pll)
    m, s, sc = score_from_pll(pll)
    return (m, s, sc, pll) if collect_pll else (m, s, sc)


def team_targets(h):
    """Current book's target dollars (reconstructed from its share output)."""
    p = h[:, -1]
    shares = tn.getMyPosition(h)
    return shares * p


def run_windows(make_fn, prices, label, quick=False):
    """Run a target-dollar factory across validation windows."""
    rows = {}
    # honest OOS window
    fn = make_fn()
    m, s, sc = simulate(prices, 250, 500, fn)
    rows["oos250"] = (m, s, sc)
    if not quick:
        fn = make_fn()
        m, s, sc = simulate(prices, 100, 300, fn)
        rows["early"] = (m, s, sc)
        rev = prices[:, ::-1].copy()
        fn = make_fn()
        m, s, sc = simulate(rev, 250, 500, fn)
        rows["reversed"] = (m, s, sc)
    parts = [f"{k}: mean={v[0]:7.1f} std={v[1]:7.1f} score={v[2]:7.2f}"
             for k, v in rows.items()]
    print(f"{label:52s} | " + " | ".join(parts))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    prices = load_prices()

    # ------------------------------------------------------------------
    # 1. Standalone sleeve grid
    # ------------------------------------------------------------------
    print("==== Standalone lead-lag sleeve (honest walk-forward) ====")
    if args.quick:
        grid = [
            dict(lam=50, sizing="sign", scale=1.0, sel_ic=None),
            dict(lam=50, sizing="sign", scale=1.0, sel_ic=0.05),
        ]
    else:
        grid = []
        for lam in (20, 50, 150, 400):
            for sizing, scale in (("sign", 1.0), ("tanh", 3.0),
                                  ("tanh", 10.0), ("linear", 5.0)):
                for sel in (None, 0.0, 0.05):
                    grid.append(dict(lam=lam, sizing=sizing, scale=scale,
                                     sel_ic=sel))

    results = []
    for cfg in grid:
        def make(cfg=cfg):
            model = LeadLagModel(**cfg)
            return lambda h: model.target_dollars(h)
        label = (f"lam={cfg['lam']:<4} {cfg['sizing']:<6}"
                 f" scale={cfg['scale']:<4} selIC={str(cfg['sel_ic']):<5}")
        rows = run_windows(make, prices, label, quick=args.quick)
        results.append((cfg, rows))

    # pick best config by min(oos, early, reversed) score if available
    def robust_score(rows):
        return min(v[2] for v in rows.values())
    best_cfg, best_rows = max(results, key=lambda cr: robust_score(cr[1]))
    print(f"\nBest robust config: {best_cfg}  "
          f"(min-window score {robust_score(best_rows):.2f})")

    # ------------------------------------------------------------------
    # 2. PnL correlation vs current book (honest OOS window)
    # ------------------------------------------------------------------
    print("\n==== PnL correlation vs current book (days 250-499) ====")
    tn.reset_state()
    _, _, _, book_pll = simulate(prices, 250, 500, team_targets,
                                 collect_pll=True)
    model = LeadLagModel(**best_cfg)
    _, _, _, ll_pll = simulate(prices, 250, 500,
                               lambda h: model.target_dollars(h),
                               collect_pll=True)
    corr = np.corrcoef(book_pll, ll_pll)[0, 1]
    mb, sb, scb = score_from_pll(book_pll)
    print(f"current book: mean={mb:.1f} std={sb:.1f} score={scb:.2f}")
    print(f"PnL corr(book, sleeve) = {corr:.3f}")

    # ------------------------------------------------------------------
    # 3. Combination testing
    # ------------------------------------------------------------------
    print("\n==== Combinations: book + sleeve (clipped to limits) ====")
    pair_owned = np.array(tn.PAIR_OWNED, dtype=int)

    def make_combo(cap_frac, exclude_pairs):
        model = LeadLagModel(**{**best_cfg, "cap_frac": cap_frac})
        tn.reset_state()

        def fn(h):
            tgt = team_targets(h)
            sleeve = model.target_dollars(h)
            if exclude_pairs:
                sleeve[pair_owned] = 0.0
            total = tgt + sleeve
            return np.clip(total, -LIMITS, LIMITS)
        return fn

    for cap in (0.25, 0.5, 1.0):
        for excl in (False, True):
            label = f"book + sleeve cap={cap:<4} exclPairs={excl}"
            run_windows(lambda cap=cap, excl=excl: make_combo(cap, excl),
                        prices, label, quick=args.quick)

    print("\nBaseline book alone:")
    def make_book():
        tn.reset_state()
        return team_targets
    run_windows(make_book, prices, "book alone", quick=args.quick)


if __name__ == "__main__":
    main()
