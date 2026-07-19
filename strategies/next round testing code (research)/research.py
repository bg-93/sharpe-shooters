#!/usr/bin/env python3
"""Next-round research runner.

Tests the quartercode-style cross-sectional ensemble (quarter_style.py)
plus hybrids against the known baselines, on three windows with
eval-exact fees. Selection metric = MIN score across windows (the rule
that would have caught the MR-core overfit before the testing round).

Windows:  early 100-300 | old 250-500 | oos 500-750 (true OOS)
Baselines: Score-1k book (min 127.91), LL+pairs rebuild (min ~438).

Usage (from repo root):
    python "strategies/next round testing code (research)/research.py"
    ... --quick   (headline configs only)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting"))
sys.path.insert(0, str(HERE))

from leadlag_research import LeadLagModel, load_prices, simulate, team_targets
import quarter_style as qs
import teamName as tn

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
WINDOWS = [("early", 100, 300), ("old", 250, 500), ("oos", 500, 750)]

ORIG = dict(CORE_ASSETS=tn.CORE_ASSETS, ALGO_FADE_CAP=tn.ALGO_FADE_CAP,
            PAIR_OWNED=tn.PAIR_OWNED)


def restore_tn():
    for k, v in ORIG.items():
        setattr(tn, k, v)


def patch_llpairs():
    """LL+pairs rebuild: MR off, fade off, LL trades pair-owned names."""
    tn.CORE_ASSETS = np.array([], dtype=int)
    tn.ALGO_FADE_CAP = 0.0
    tn.PAIR_OWNED = []


def run(label, make_fn, prices):
    scores = []
    parts = []
    for name, a, b in WINDOWS:
        fn = make_fn()
        m, s, sc = simulate(prices, a, b, fn)
        scores.append(sc)
        parts.append(f"{name}={sc:7.2f}")
    print(f"{label:46s} " + "  ".join(parts) + f"  | min={min(scores):7.2f}")
    return min(scores)


# ---------------------------------------------------------------- factories

def make_quarter(cfg):
    def make():
        c = {**qs.DEFAULT, **cfg}
        return lambda h: qs.target_dollars(h, c)
    return make


def make_book():
    def make():
        restore_tn()
        tn.reset_state()
        return team_targets
    return make


def make_llpairs():
    def make():
        patch_llpairs()
        tn.reset_state()
        return team_targets
    return make


def make_hybrid_a(sb_w, cfg=None):
    """LL+pairs book + cross-sectional slowbook sleeve at weight sb_w."""
    c = {**qs.DEFAULT, "PAIRS_W": 0.0, "BASKET_W": 0.0, **(cfg or {})}

    def make():
        patch_llpairs()
        tn.reset_state()

        def fn(h):
            tgt = team_targets(h) + sb_w * qs.target_dollars(h, c)
            return np.clip(tgt, -LIMITS, LIMITS)
        return fn
    return make


class DemeanLL(LeadLagModel):
    """Lead-lag with cross-sectionally demeaned prediction z + tanh sizing."""

    def __init__(self, temp, **kw):
        self.temp = temp
        super().__init__(**kw)

    def target_dollars(self, prc):
        if prc.shape[1] < 120:
            return np.zeros(N_INST)
        super().target_dollars(prc)          # updates fit/WF state
        z = self.pending / self.resid_sd
        z = z - z.mean()
        z = z / (z.std() + 1e-9)
        mask = np.ones(N_INST)
        ics = self.wf_ic()
        if ics is not None:
            mask = (ics > 0.0).astype(float)
        return LIMITS * np.tanh(z / self.temp) * mask


def make_hybrid_b(temp):
    """Pairs (from book) + demeaned/tanh lead-lag replacing the sign LL."""
    def make():
        patch_llpairs()
        tn.reset_state()
        model = DemeanLL(temp, lam=400.0, retrain=50, sel_ic=None)
        tn._leadlag_target_dollars = lambda h: model.target_dollars(h)
        return team_targets
    return make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    prices = load_prices()

    print("== baselines ==")
    run("Score-1k book (live submission)", make_book(), prices)
    run("LL+pairs rebuild (MR/fade off, noexcl)", make_llpairs(), prices)
    restore_tn()

    print("\n== quartercode replica: defaults + ablations ==")
    run("quarter: all sleeves (default)", make_quarter({}), prices)
    run("quarter: slowbook only", make_quarter(
        {"PAIRS_W": 0.0, "BASKET_W": 0.0}), prices)
    run("quarter: pairs only", make_quarter(
        {"SLOWBOOK_W": 0.0, "BASKET_W": 0.0}), prices)
    run("quarter: basket only", make_quarter(
        {"SLOWBOOK_W": 0.0, "PAIRS_W": 0.0}), prices)
    run("quarter: no basket", make_quarter({"BASKET_W": 0.0}), prices)
    run("quarter: ALGO sized as stock", make_quarter(
        {"ALGO_LIMIT": 10000.0}), prices)

    if not args.quick:
        print("\n== quartercode grid: horizons x TEMP ==")
        for h in ((10, 40), (15, 50), (20, 60)):
            for t in (0.5, 1.0, 2.0):
                run(f"quarter: H={h} TEMP={t}",
                    make_quarter({"SLOWBOOK_H": h, "TEMP": t}), prices)

        print("\n== hybrid A: LL+pairs book + slowbook sleeve ==")
        for w in (0.25, 0.5, 1.0):
            run(f"hybrid A: LL+pairs + {w}x slowbook",
                make_hybrid_a(w), prices)

        print("\n== hybrid B: pairs + demeaned tanh lead-lag ==")
        for t in (0.5, 1.0, 2.0):
            run(f"hybrid B: demeanLL TEMP={t} + pairs",
                make_hybrid_b(t), prices)

    restore_tn()


if __name__ == "__main__":
    main()
