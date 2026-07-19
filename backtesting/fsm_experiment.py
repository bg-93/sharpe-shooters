#!/usr/bin/env python3
"""Architecture test: pairs + MLR (lead-lag ridge) + explicit FSM.

Variants (each on eval/early/reversed windows):
  A. current book                (MR core + fade + pairs + LL sign-flip)
  B. trio: pairs + MLR + FSM     (MR core off, fade off)
  C. current book with FSM dead-zone on the MLR sleeve
  D. trio with FSM dead-zone on the MLR sleeve

FSM per name (3 states: -1 flat=0 +1) on z = pred / resid_sd:
  flat  -> long  if z >  z_in     flat  -> short if z < -z_in
  long  -> hold while z > -z_out; else exit (re-enter short if z < -z_in)
  short -> symmetric
z_in = z_out = 0 reproduces the current daily sign-flip exactly.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting"))

from leadlag_research import LeadLagModel, load_prices, simulate, team_targets
import teamName as tn

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)


class FSMLeadLag(LeadLagModel):
    """Lead-lag ridge with a per-name 3-state machine instead of daily
    sign-flipping."""

    def __init__(self, z_in, z_out, **kw):
        self.z_in = z_in
        self.z_out = z_out
        super().__init__(**kw)

    def reset(self):
        super().reset()
        self.state = np.zeros(N_INST)

    def target_dollars(self, prc):
        # run parent to update fit/WF state, then re-derive z + FSM
        nt = prc.shape[1]
        if nt < 120:
            return np.zeros(N_INST)
        super().target_dollars(prc)          # updates W, pending, WF lists
        z = self.pending / self.resid_sd     # today's prediction z

        mask = np.ones(N_INST)
        ics = self.wf_ic()
        if ics is not None:
            mask = (ics > 0.0).astype(float)

        s = self.state
        for j in range(N_INST):
            if s[j] == 0:
                if z[j] > self.z_in:
                    s[j] = 1
                elif z[j] < -self.z_in:
                    s[j] = -1
            elif s[j] == 1:
                if z[j] < -self.z_in:
                    s[j] = -1
                elif z[j] < -self.z_out:
                    s[j] = 0
            else:
                if z[j] > self.z_in:
                    s[j] = 1
                elif z[j] > self.z_out:
                    s[j] = 0

        tgt = LIMITS * s * mask
        tgt[tn.PAIR_OWNED] = 0.0
        return tgt


ORIG_LL = tn._leadlag_target_dollars
ORIG_CORE = tn.CORE_ASSETS
ORIG_FADE = tn.ALGO_FADE_CAP


def windows(label, setup):
    prices = load_prices()
    rev = prices[:, ::-1].copy()
    out = []
    for name, (px, a, b) in {"eval": (prices, 250, 500),
                             "early": (prices, 100, 300),
                             "rev": (rev, 250, 500)}.items():
        setup()
        tn.reset_state()
        m, s, sc = simulate(px, a, b, team_targets)
        out.append(f"{name}={sc:7.2f}")
    print(f"{label:44s} " + "  ".join(out))


def set_book(mr_on, fade_on, fsm=None):
    """Returns a setup callback configuring the book."""
    def setup():
        tn.CORE_ASSETS = ORIG_CORE if mr_on else np.array([], dtype=int)
        tn.ALGO_FADE_CAP = ORIG_FADE if fade_on else 0.0
        if fsm is None:
            tn._leadlag_target_dollars = ORIG_LL
        else:
            model = FSMLeadLag(z_in=fsm[0], z_out=fsm[1],
                               lam=400.0, retrain=50, sel_ic=None)
            tn._leadlag_target_dollars = lambda h: model.target_dollars(h)
    return setup


def main():
    print("== A. current book ==")
    windows("A: MR + fade + pairs + LL(sign)", set_book(True, True))

    print("== B. trio: pairs + MLR + (implicit) FSM ==")
    windows("B: pairs + LL(sign), MR off fade off", set_book(False, False))

    print("== C. current + FSM dead-zone on MLR ==")
    for zi, zo in ((0.1, 0.0), (0.2, 0.05), (0.3, 0.1)):
        windows(f"C: full book, FSM z_in={zi} z_out={zo}",
                set_book(True, True, fsm=(zi, zo)))

    print("== D. trio + FSM dead-zone on MLR ==")
    for zi, zo in ((0.1, 0.0), (0.2, 0.05), (0.3, 0.1)):
        windows(f"D: trio, FSM z_in={zi} z_out={zo}",
                set_book(False, False, fsm=(zi, zo)))

    # restore
    set_book(True, True)()


if __name__ == "__main__":
    main()
