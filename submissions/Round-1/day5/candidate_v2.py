#!/usr/bin/env python3
"""candidate_v2: adaptive pairs + cross-sectionally demeaned lead-lag.

Evolution of candidate_next.py (round-1 score 1420.84 on days 751-800):
NOTHING about the pairs is hardcoded any more. Every 50 days the sleeve
re-runs the pair-selection procedure on all history available at that
day (walk-forward, no lookahead):

  - prefilter top 200 stock pairs by |corr| of log prices
  - OLS hedge ratio, spread AR(1) half-life < 20d
  - z-reversion mini-backtest profitable in both halves, annSR >= 1.5
  - incumbency hysteresis: pairs already in the book keep their seat
    while annSR >= 1.0 and get a +0.5 ranking bonus, so marginal
    newcomers cannot unseat a working pair (kills selection churn)
  - max 15 pairs, max 2 per name; gammas refit at every re-scan

Lead-lag sleeve unchanged from candidate_next: online ridge lam=400
refit 50d, z = pred/resid_sd demeaned across names, tanh(z/0.35) at
limits. Tested (eval-exact fees):
  old 250-500: 465.63 | oos 500-750: 650.74  (frozen-pairs: 654.36 oos,
  but its old/early numbers are inflated by lookahead — the frozen set
  was picked on days 1-500). Fully walk-forward at frozen-level OOS.

Rejected en route: feature-ridge fusion of pairs/basket into the
predictor (oos 684 but old 323 — single-window winner, min-window
loser), gamma-only refresh of frozen ids (matches, but keeps hardcoded
pair identities).
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

# ---------------------------------------------------------------- pairs
RESCAN = 50
MIN_SEL_HIST = 250
PREFILTER = 200
HL_MAX = 20.0
SR_MIN = 1.5
SR_KEEP = 1.0
MAX_PAIRS = 15
MAX_PER_NAME = 2

PAIR_LEG = 9_000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5

# -------------------------------------------------------------- lead-lag
LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_TEMP = 0.35

_pairs = []
_pair_pos = {}
_pair_last_scan = -1
_ll_W = None
_ll_mu = None
_ll_sd = None
_ll_resid_sd = None
_ll_last_fit = -1
_prev_nt = -1


def reset_state():
    global _pairs, _pair_pos, _pair_last_scan
    global _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit, _prev_nt
    _pairs = []
    _pair_pos = {}
    _pair_last_scan = -1
    _ll_W = None
    _ll_mu = None
    _ll_sd = None
    _ll_resid_sd = None
    _ll_last_fit = -1
    _prev_nt = -1


def _half_life(s):
    x, y = s[:-1] - s.mean(), s[1:] - s.mean()
    denom = (x * x).sum()
    if denom < EPS:
        return np.inf
    phi = (x * y).sum() / denom
    if not (0.0 < phi < 1.0):
        return np.inf
    return -np.log(2.0) / np.log(phi)


def _spread_sr_halves(s):
    n = len(s)
    if n < PAIR_ROLL + 20:
        return -np.inf, False
    pos = 0
    pnl = np.zeros(n)
    for t in range(PAIR_ROLL, n - 1):
        win = s[t - PAIR_ROLL:t]
        z = (s[t] - win.mean()) / (win.std() + EPS)
        if pos == 0:
            if z > PAIR_ENTRY:
                pos = -1
            elif z < -PAIR_ENTRY:
                pos = 1
        elif pos == 1 and z > -PAIR_EXIT:
            pos = 0
        elif pos == -1 and z < PAIR_EXIT:
            pos = 0
        pnl[t + 1] = pos * (s[t + 1] - s[t])
    pnl = pnl[PAIR_ROLL:]
    sd = pnl.std()
    if sd < EPS:
        return -np.inf, False
    sr = np.sqrt(250.0) * pnl.mean() / sd
    h = len(pnl) // 2
    return sr, (pnl[:h].sum() > 0 and pnl[h:].sum() > 0)


def _score_pair(lpx, a, b):
    x, y = lpx[b], lpx[a]
    vx = x.var()
    if vx < EPS:
        return None
    g = np.cov(y, x)[0, 1] / vx
    if not (0.1 < g < 3.0):
        return None
    s = y - g * x
    if _half_life(s) > HL_MAX:
        return None
    sr, ok = _spread_sr_halves(s)
    if not ok:
        return None
    return sr, g


def _select_pairs(lpx, incumbents):
    L = lpx[1:]
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
        if sr >= (SR_KEEP if (a, b) in inc else SR_MIN):
            scored.append((sr + (0.5 if (a, b) in inc else 0.0), a, b, g))

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


def _pairs_target_dollars(lpx):
    global _pairs, _pair_pos, _pair_last_scan
    nt = lpx.shape[1]
    tgt = np.zeros(N_INST)
    if nt < MIN_SEL_HIST:
        return tgt

    if _pair_last_scan < 0 or (nt - _pair_last_scan) >= RESCAN:
        incumbents = tuple((i, j) for i, j, _ in _pairs)
        _pairs = _select_pairs(lpx, incumbents)
        keys = {(i, j) for i, j, _ in _pairs}
        _pair_pos = {k: v for k, v in _pair_pos.items() if k in keys}
        _pair_last_scan = nt

    for i, j, g in _pairs:
        s = lpx[i] - g * lpx[j]
        win = s[-PAIR_ROLL - 1:-1]
        z = (s[-1] - win.mean()) / (win.std() + EPS)
        pos = _pair_pos.get((i, j), 0)
        if pos == 0:
            if z > PAIR_ENTRY:
                pos = -1
            elif z < -PAIR_ENTRY:
                pos = 1
        elif pos == 1 and z > -PAIR_EXIT:
            pos = 0
        elif pos == -1 and z < PAIR_EXIT:
            pos = 0
        _pair_pos[(i, j)] = pos
        if pos != 0:
            tgt[i] += pos * PAIR_LEG
            tgt[j] -= pos * g * PAIR_LEG
    return tgt


def _leadlag_target_dollars(prcSoFar):
    global _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit
    nt = prcSoFar.shape[1]
    if nt < LL_MIN_HIST:
        return np.zeros(N_INST)
    r = np.diff(np.log(np.maximum(prcSoFar, EPS)), axis=1)

    if _ll_W is None or (nt - _ll_last_fit) >= LL_RETRAIN:
        X = r[:, :-1].T
        Y = r[:, 1:].T
        _ll_mu, _ll_sd = X.mean(0), X.std(0)
        _ll_sd = np.where(_ll_sd > 1e-12, _ll_sd, 1.0)
        Xs = (X - _ll_mu) / _ll_sd
        _ll_W = np.linalg.solve(
            Xs.T @ Xs + LL_LAM * np.eye(N_INST), Xs.T @ Y
        )
        _ll_resid_sd = np.maximum(Y.std(0), 1e-8)
        _ll_last_fit = nt

    pred = ((r[:, -1] - _ll_mu) / _ll_sd) @ _ll_W
    z = pred / _ll_resid_sd
    z = z - z.mean()
    z = z / (z.std() + 1e-9)
    return LIMITS * np.tanh(z / LL_TEMP)


def getMyPosition(prcSoFar):
    global _prev_nt
    nt = prcSoFar.shape[1]
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt

    lpx = np.log(np.maximum(prcSoFar, EPS))
    target_dollars = _pairs_target_dollars(lpx)
    target_dollars = target_dollars + _leadlag_target_dollars(prcSoFar)
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)
    return (target_dollars / prcSoFar[:, -1]).astype(int)
