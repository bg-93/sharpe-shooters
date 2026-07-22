#!/usr/bin/env python3
"""Hybrid strategy: pairs+LL with PnL-gated switch to momentum.

Concept: run the proven pairs+LL book (cnext architecture, 1420/day on
751-850). When its trailing paper PnL goes cold, fade into a momentum
sleeve that profits in trending regimes where pairs/MR bleed.

Sleeves:
  1. Pairs: 15 frozen pairs, $9k legs, 60d z, entry 1.5 / exit 0.5
  2. Lead-lag: online ridge, lam=400, refit every 50d, demeaned tanh
  3. Momentum: per-asset cross-sectional momentum (long winners, short
     losers), sized by tanh of normalised trailing return signal

Switching logic: track pairs+LL combined paper PnL over a trailing
window. When it's positive, run pairs+LL fully. When it turns negative,
blend in momentum proportionally. This is adaptive — no hardcoded date.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

# ── Pairs ────────────────────────────────────────────────────────────
PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836), (13, 45, 1.0132),
    (33, 40, 0.2577), (10, 46, 1.0331), (33, 42, 0.8358), (31, 43, 0.9692),
    (18, 28, 0.5642), (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)
PAIR_LEG = 9_000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5

# ── Lead-lag ─────────────────────────────────────────────────────────
LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_TEMP = 0.35

# ── Momentum ─────────────────────────────────────────────────────────
MOM_FAST = 10          # fast lookback (days)
MOM_SLOW = 60          # slow lookback (days)
MOM_TEMP = 0.50        # tanh temperature (lower = more aggressive)
MOM_SCALE = 1.0        # fraction of LIMITS to use
MOM_MIN_HIST = 65      # need MOM_SLOW + a few days

# ── Gate / blending ──────────────────────────────────────────────────
GATE_WIN = 60           # trailing window for pairs+LL paper PnL
GATE_MIN_OBS = 20       # blind period: run pairs+LL unconditionally
BLEND_SMOOTH = 30       # days over which blend transitions (sigmoid width)

# ── State ────────────────────────────────────────────────────────────
_pair_pos = [0] * len(PAIRS)
_ll_W = None
_ll_mu = None
_ll_sd = None
_ll_resid_sd = None
_ll_last_fit = -1
_prev_nt = -1
_prev_pairsll_dollars = None   # for paper PnL tracking
_pairsll_pnl = []              # daily paper PnL of pairs+LL sleeve


def reset_state():
    global _pair_pos, _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit
    global _prev_nt, _prev_pairsll_dollars, _pairsll_pnl
    _pair_pos = [0] * len(PAIRS)
    _ll_W = None
    _ll_mu = None
    _ll_sd = None
    _ll_resid_sd = None
    _ll_last_fit = -1
    _prev_nt = -1
    _prev_pairsll_dollars = None
    _pairsll_pnl = []


# ── Pairs sleeve ─────────────────────────────────────────────────────
def _pairs_dollars(log_all, nt):
    tgt = np.zeros(N_INST)
    if nt <= PAIR_ROLL + 1:
        return tgt
    for k, (i, j, g) in enumerate(PAIRS):
        s = log_all[i] - g * log_all[j]
        win = s[-PAIR_ROLL - 1:-1]
        z = (s[-1] - win.mean()) / (win.std() + EPS)
        pos = _pair_pos[k]
        if pos == 0:
            if z > PAIR_ENTRY:
                pos = -1
            elif z < -PAIR_ENTRY:
                pos = 1
        elif pos == 1 and z > -PAIR_EXIT:
            pos = 0
        elif pos == -1 and z < PAIR_EXIT:
            pos = 0
        _pair_pos[k] = pos
        if pos != 0:
            tgt[i] += pos * PAIR_LEG
            tgt[j] -= pos * g * PAIR_LEG
    return tgt


# ── Lead-lag sleeve ──────────────────────────────────────────────────
def _ll_dollars(prcSoFar):
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


# ── Momentum sleeve ─────────────────────────────────────────────────
def _momentum_dollars(prcSoFar):
    nt = prcSoFar.shape[1]
    if nt < max(MOM_MIN_HIST, MOM_SLOW + 2):
        return np.zeros(N_INST)

    # log returns
    lpx = np.log(np.maximum(prcSoFar, EPS))

    # fast momentum: trailing MOM_FAST-day return
    r_fast = lpx[:, -1] - lpx[:, -min(MOM_FAST, nt - 1) - 1]
    # slow momentum: trailing MOM_SLOW-day return
    r_slow = lpx[:, -1] - lpx[:, -min(MOM_SLOW, nt - 1) - 1]

    # combined signal: fast + slow (equal weight)
    sig = 0.5 * r_fast + 0.5 * r_slow

    # cross-sectional demeaning + standardisation
    sig = sig - sig.mean()
    sd = sig.std()
    if sd < 1e-9:
        return np.zeros(N_INST)
    sig = sig / sd

    # tanh sizing
    tgt = LIMITS * MOM_SCALE * np.tanh(sig / MOM_TEMP)
    return tgt


# ── Main ─────────────────────────────────────────────────────────────
def getMyPosition(prcSoFar):
    global _prev_nt, _prev_pairsll_dollars, _pairsll_pnl

    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nt = prcSoFar.shape[1]

    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt

    log_all = np.log(np.maximum(prcSoFar, EPS))

    # ── Compute pairs+LL target (always, for paper PnL tracking) ─────
    pairsll_dollars = _pairs_dollars(log_all, nt) + _ll_dollars(prcSoFar)

    # ── Track paper PnL of pairs+LL sleeve ───────────────────────────
    if _prev_pairsll_dollars is not None and nt >= 2:
        r = prcSoFar[:, -1] / prcSoFar[:, -2] - 1.0
        day_pnl = float(_prev_pairsll_dollars @ r)
        _pairsll_pnl.append(day_pnl)

    # ── Gate: decide blend ratio ─────────────────────────────────────
    n_obs = len(_pairsll_pnl)
    if n_obs < GATE_MIN_OBS:
        # blind period: trust pairs+LL fully
        alpha = 1.0
    else:
        # trailing PnL of pairs+LL
        trail = sum(_pairsll_pnl[-GATE_WIN:])
        # sigmoid blend: alpha=1 when trail >> 0, alpha->0 when trail << 0
        # scale trail by number of days for stability
        norm_trail = trail / (GATE_WIN * 100)  # normalise roughly
        alpha = 1.0 / (1.0 + np.exp(-BLEND_SMOOTH * norm_trail))

    # ── Compute momentum target ──────────────────────────────────────
    mom_dollars = _momentum_dollars(prcSoFar)

    # ── Blend ────────────────────────────────────────────────────────
    target_dollars = alpha * pairsll_dollars + (1.0 - alpha) * mom_dollars

    # ── Clip + store ─────────────────────────────────────────────────
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)
    _prev_pairsll_dollars = pairsll_dollars.copy()

    return (target_dollars / prcSoFar[:, -1]).astype(int)
