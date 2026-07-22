#!/usr/bin/env python3
"""Decomposition runner for candidate_hybrid.

Tests momentum standalone, pairs+LL standalone (cnext), and the
hybrid with PnL-gated blending on early/old/oos windows.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting"))
sys.path.insert(0, str(HERE))

from leadlag_research import load_prices, simulate

WINDOWS = [("early", 100, 300), ("old", 250, 500), ("oos", 500, 750)]


def make_dollar_adapter(get_pos_fn, reset_fn):
    """Wrap a getMyPosition-style function into a target_dollars function
    that simulate() expects."""
    def target_dollars(prices_so_far):
        pos = get_pos_fn(prices_so_far)
        p = prices_so_far[:, -1]
        return pos.astype(float) * p
    return target_dollars, reset_fn


def run(label, make_strat, prices):
    scores = []
    parts = []
    for name, a, b in WINDOWS:
        get_pos, reset = make_strat()
        reset()
        tgt_fn = lambda h, gp=get_pos: gp(h).astype(float) * h[:, -1]
        m, s, sc = simulate(prices, a, b, tgt_fn)
        scores.append(sc)
        parts.append(f"{name}: {m:7.1f}/d {sc:8.2f}")
    print(f"{label:40s} " + " | ".join(parts) + f" | min={min(scores):8.2f}")
    return min(scores)


def main():
    prices = load_prices()

    # ── Import strategies ────────────────────────────────────────────
    import candidate_hybrid as hybrid
    import candidate_next as cnext

    # Standalone momentum: hack hybrid to only run momentum
    def make_mom_only():
        hybrid.reset_state()
        def get_pos(prc):
            prc = np.asarray(prc, dtype=float)
            nt = prc.shape[1]
            if nt <= hybrid._prev_nt:
                hybrid.reset_state()
            hybrid._prev_nt = nt
            mom = hybrid._momentum_dollars(prc)
            mom = np.clip(mom, -hybrid.LIMITS, hybrid.LIMITS)
            return (mom / prc[:, -1]).astype(int)
        return get_pos, hybrid.reset_state

    # Standalone pairs+LL (cnext)
    def make_cnext():
        cnext.reset_state()
        return cnext.getMyPosition, cnext.reset_state

    # Full hybrid
    def make_hybrid():
        hybrid.reset_state()
        return hybrid.getMyPosition, hybrid.reset_state

    # Hybrid variants for parameter sweeps
    def make_hybrid_variant(**overrides):
        def factory():
            for k, v in overrides.items():
                setattr(hybrid, k, v)
            hybrid.reset_state()
            return hybrid.getMyPosition, hybrid.reset_state
        return factory

    configs = [
        ("pairs+LL only (cnext baseline)", make_cnext),
        ("momentum only", make_mom_only),
        ("HYBRID: pairs+LL -> momentum gate", make_hybrid),
    ]
    print("=" * 110)
    print("DECOMPOSITION")
    print("=" * 110)
    for label, factory in configs:
        run(label, factory, prices)

    # ── Gate sensitivity sweep ───────────────────────────────────────
    print("\n" + "=" * 110)
    print("GATE_WIN SWEEP (hybrid)")
    print("=" * 110)
    orig_gate = hybrid.GATE_WIN
    for gw in [30, 40, 60, 80, 100, 120]:
        hybrid.GATE_WIN = gw
        run(f"hybrid GATE_WIN={gw}", make_hybrid, prices)
    hybrid.GATE_WIN = orig_gate

    # ── Momentum parameter sweep ─────────────────────────────────────
    print("\n" + "=" * 110)
    print("MOMENTUM PARAM SWEEP (standalone)")
    print("=" * 110)
    orig_fast = hybrid.MOM_FAST
    orig_slow = hybrid.MOM_SLOW
    orig_temp = hybrid.MOM_TEMP
    for fast, slow in [(5, 30), (10, 60), (10, 40), (20, 60), (20, 100)]:
        hybrid.MOM_FAST = fast
        hybrid.MOM_SLOW = slow
        run(f"mom fast={fast} slow={slow}", make_mom_only, prices)
    hybrid.MOM_FAST = orig_fast
    hybrid.MOM_SLOW = orig_slow

    for temp in [0.3, 0.5, 0.7, 1.0]:
        hybrid.MOM_TEMP = temp
        run(f"mom temp={temp}", make_mom_only, prices)
    hybrid.MOM_TEMP = orig_temp

    # ── Blend sweep ──────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("BLEND_SMOOTH SWEEP (hybrid)")
    print("=" * 110)
    orig_blend = hybrid.BLEND_SMOOTH
    for bs in [10, 20, 30, 50, 80]:
        hybrid.BLEND_SMOOTH = bs
        run(f"hybrid BLEND_SMOOTH={bs}", make_hybrid, prices)
    hybrid.BLEND_SMOOTH = orig_blend


if __name__ == "__main__":
    main()
