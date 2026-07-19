#!/usr/bin/env python3
"""Walk-forward adaptive pairs sleeve.

No hardcoded pairs or gammas: every RESCAN days the sleeve re-runs the
same selection procedure that was honesty-validated on released data
(pairs picked on days 0-249 earned 81/day OOS on 250-499), using ONLY
history before the current day:

  1. prefilter: top candidates by |corr| of log prices (stocks only)
  2. gamma: OLS of lpx_i on lpx_j over the selection window
  3. spread AR(1) half-life < HL_MAX days
  4. z-reversion mini-backtest profitable in BOTH halves, SR >= SR_MIN
  5. rank by SR, keep at most MAX_PAIRS with at most 2 per name

Trading is identical to the live sleeve: 60d rolling z on the spread,
entry 1.5 / exit 0.5 hysteresis, $9k legs. On re-scan, pairs that drop
out of the selected set are flattened.
"""

import numpy as np

N_INST = 51
EPS = 1e-12

RESCAN = 50          # re-select pairs every 50 days
SEL_WIN = 10**9      # expanding window: selection uses all history
MIN_SEL_HIST = 250   # need this much history before trading at all
PREFILTER = 200      # candidate pairs kept by |corr| before slow checks
HL_MAX = 20.0
SR_MIN = 1.5
SR_KEEP = 1.0        # incumbency: sitting pairs only need this bar
MAX_PAIRS = 15
MAX_PER_NAME = 2

PAIR_LEG = 9_000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5


def _half_life(s):
    x, y = s[:-1] - s.mean(), s[1:] - s.mean()
    denom = (x * x).sum()
    if denom < EPS:
        return np.inf
    phi = (x * y).sum() / denom
    if not (0.0 < phi < 1.0):
        return np.inf
    return -np.log(2.0) / np.log(phi)


def _spread_sr_halves(s, roll=PAIR_ROLL, entry=PAIR_ENTRY, exit_=PAIR_EXIT):
    """Mini z-reversion backtest on a spread; returns (sr_full, ok_halves)."""
    n = len(s)
    if n < roll + 20:
        return -np.inf, False
    pos = 0
    pnl = np.zeros(n)
    mu = s[:roll].mean()
    sd = s[:roll].std() + EPS
    for t in range(roll, n - 1):
        win = s[t - roll:t]
        mu, sd = win.mean(), win.std() + EPS
        z = (s[t] - mu) / sd
        if pos == 0:
            if z > entry:
                pos = -1
            elif z < -entry:
                pos = 1
        elif pos == 1 and z > -exit_:
            pos = 0
        elif pos == -1 and z < exit_:
            pos = 0
        pnl[t + 1] = pos * (s[t + 1] - s[t])
    pnl = pnl[roll:]
    sd_p = pnl.std()
    if sd_p < EPS:
        return -np.inf, False
    sr = np.sqrt(250.0) * pnl.mean() / sd_p
    h = len(pnl) // 2
    ok = pnl[:h].sum() > 0 and pnl[h:].sum() > 0
    return sr, ok


def _score_pair(lpx, a, b):
    """Gamma + quality checks for one pair. Returns (sr, gamma) or None."""
    x, y = lpx[b, -SEL_WIN:], lpx[a, -SEL_WIN:]
    vx = x.var()
    if vx < EPS:
        return None
    g = np.cov(y, x)[0, 1] / vx                   # OLS hedge ratio
    if not (0.1 < g < 3.0):
        return None
    s = y - g * x
    if _half_life(s) > HL_MAX:
        return None
    sr, ok = _spread_sr_halves(s)
    if not ok:
        return None
    return sr, g


def select_pairs(lpx, incumbents=()):
    """Selection with incumbency hysteresis: pairs already in the book
    keep their seat while sr >= SR_KEEP; newcomers need sr >= SR_MIN."""
    L = lpx[1:, -SEL_WIN:]                       # stocks only
    n = L.shape[0]
    Ld = L - L.mean(axis=1, keepdims=True)
    norm = np.sqrt((Ld * Ld).sum(axis=1)) + EPS
    C = (Ld @ Ld.T) / np.outer(norm, norm)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-np.abs(C[iu]))[:PREFILTER]

    inc = set(incumbents)
    cands = {(iu[0][idx] + 1, iu[1][idx] + 1) for idx in order} | inc
    scored = []
    for a, b in cands:
        res = _score_pair(lpx, a, b)
        if res is None:
            continue
        sr, g = res
        keep_bar = SR_KEEP if (a, b) in inc else SR_MIN
        if sr >= keep_bar:
            # incumbents get a ranking bonus so marginal newcomers
            # cannot unseat a working pair
            rank = sr + (0.5 if (a, b) in inc else 0.0)
            scored.append((rank, a, b, g))

    scored.sort(reverse=True)
    used = {}
    chosen = []
    for _, a, b, g in scored:
        if used.get(a, 0) >= MAX_PER_NAME or used.get(b, 0) >= MAX_PER_NAME:
            continue
        chosen.append((a, b, round(g, 4)))
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
        if len(chosen) >= MAX_PAIRS:
            break
    return chosen


class AdaptivePairs:
    """Stateful walk-forward pairs sleeve. Call target_dollars daily."""

    def __init__(self):
        self.pairs = []
        self.pos = {}          # (i, j) -> -1/0/1
        self.last_scan = -1

    def target_dollars(self, prcSoFar):
        nt = prcSoFar.shape[1]
        tgt = np.zeros(N_INST)
        if nt < MIN_SEL_HIST:
            return tgt
        lpx = np.log(np.maximum(prcSoFar, EPS))

        if self.last_scan < 0 or (nt - self.last_scan) >= RESCAN:
            incumbents = tuple((i, j) for i, j, _ in self.pairs)
            self.pairs = select_pairs(lpx, incumbents)
            keys = {(i, j) for i, j, _ in self.pairs}
            self.pos = {k: v for k, v in self.pos.items() if k in keys}
            self.last_scan = nt

        for i, j, g in self.pairs:
            s = lpx[i] - g * lpx[j]
            win = s[-PAIR_ROLL - 1:-1]
            z = (s[-1] - win.mean()) / (win.std() + EPS)
            pos = self.pos.get((i, j), 0)
            if pos == 0:
                if z > PAIR_ENTRY:
                    pos = -1
                elif z < -PAIR_ENTRY:
                    pos = 1
            elif pos == 1 and z > -PAIR_EXIT:
                pos = 0
            elif pos == -1 and z < PAIR_EXIT:
                pos = 0
            self.pos[(i, j)] = pos
            if pos != 0:
                tgt[i] += pos * PAIR_LEG
                tgt[j] -= pos * g * PAIR_LEG
        return tgt
