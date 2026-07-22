#!/usr/bin/env python3
"""Candidate FLIP: pairs+LL while hot, NEGATE the book when cold.

Thesis: days 851-1000 invert the patterns from local data. The pairs+LL
book (cnext architecture) prints +1431/day on 751-850, then bleeds.
When the gate detects the bleed, flip every position — profit from
the regime inversion instead of sitting in cash.

Gate: trailing 60-day paper PnL of the pairs+LL sleeve. When positive,
run pairs+LL normally. When negative (after 20-day blind period),
negate the entire book.

This is a GAMBLE. On local data the gate never fires (pairs+LL stays
hot on 100-750), so local backtest = cnext. The flip only activates
on the hidden regime where pairs+LL bleeds.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

# ── Pairs (frozen, identical to cnext) ───────────────────────────────
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

# ── Lead-lag (identical to cnext) ────────────────────────────────────
LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_TEMP = 0.35

# ── Gate ─────────────────────────────────────────────────────────────
GATE_WIN = 60
GATE_MIN_OBS = 20

# ── State ────────────────────────────────────────────────────────────
_pair_pos = [0] * len(PAIRS)
_ll_W = None
_ll_mu = None
_ll_sd = None
_ll_resid_sd = None
_ll_last_fit = -1
_prev_nt = -1
_prev_pairsll_dollars = None
_pairsll_pnl = []


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


def getMyPosition(prcSoFar):
    global _prev_nt, _prev_pairsll_dollars, _pairsll_pnl

    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nt = prcSoFar.shape[1]

    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt

    log_all = np.log(np.maximum(prcSoFar, EPS))

    # ── Compute pairs+LL book ────────────────────────────────────────
    pairsll_dollars = _pairs_dollars(log_all, nt) + _ll_dollars(prcSoFar)

    # ── Track paper PnL ──────────────────────────────────────────────
    if _prev_pairsll_dollars is not None and nt >= 2:
        r = prcSoFar[:, -1] / prcSoFar[:, -2] - 1.0
        day_pnl = float(_prev_pairsll_dollars @ r)
        _pairsll_pnl.append(day_pnl)

    # ── Gate: normal or FLIP ─────────────────────────────────────────
    n_obs = len(_pairsll_pnl)
    if n_obs < GATE_MIN_OBS:
        sign = 1.0  # blind period: trust pairs+LL
    else:
        trail = sum(_pairsll_pnl[-GATE_WIN:])
        sign = 1.0 if trail > 0 else -1.0  # THE FLIP

    target_dollars = sign * pairsll_dollars
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)

    _prev_pairsll_dollars = pairsll_dollars.copy()

    return (target_dollars / prcSoFar[:, -1]).astype(int)
