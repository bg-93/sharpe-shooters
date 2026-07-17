# PROBE: demeaned tanh lead-lag only (candidate minus pairs)
#!/usr/bin/env python3
"""Next-round candidate: pairs + cross-sectionally demeaned lead-lag.

Born from the quartercode study (see research.py): took the OOS-proven
LL+pairs book and applied the top performer's structural trick to the
lead-lag sleeve — demean predictions across names, size with tanh.

Sleeves:
  1. Pairs: 15 frozen-gamma pairs, $9k legs, 60d z, entry 1.5 / exit 0.5
     (identical to the live book).
  2. Lead-lag: online ridge r(t+1) ~ all r(t), lam=400, refit every 50d;
     z = pred/resid_sd, cross-sectionally demeaned and re-standardised;
     position = LIMITS * tanh(z / TEMP). No IC mask (tanh already
     down-weights weak names), no pair-owned exclusion.

Dropped vs live book: MR core (-86/day true OOS), ALGO fade (~0),
regime FSMs / dead-band (only served the MR core).

Window scores (eval-exact fees, min-window selection):
  early 100-300: 531.67 | old 250-500: 580.74 | oos 500-750: 654.36
  vs live Score-1k book min 127.70, LL+pairs sign/masked min 438.19.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836), (13, 45, 1.0132),
    (33, 40, 0.2577), (10, 46, 1.0331), (33, 42, 0.8358), (31, 43, 0.9692),
    (18, 28, 0.5642), (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)
PAIR_LEG = 0.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5

LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_TEMP = 0.35

_pair_pos = [0] * len(PAIRS)
_ll_W = None
_ll_mu = None
_ll_sd = None
_ll_resid_sd = None
_ll_last_fit = -1
_prev_nt = -1


def reset_state():
    global _pair_pos, _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit
    global _prev_nt
    _pair_pos = [0] * len(PAIRS)
    _ll_W = None
    _ll_mu = None
    _ll_sd = None
    _ll_resid_sd = None
    _ll_last_fit = -1
    _prev_nt = -1


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
    z = z - z.mean()                    # cross-sectional demeaning
    z = z / (z.std() + 1e-9)
    return LIMITS * np.tanh(z / LL_TEMP)


def getMyPosition(prcSoFar):
    global _prev_nt
    nt = prcSoFar.shape[1]
    if nt <= _prev_nt:                  # fresh simulation started
        reset_state()
    _prev_nt = nt

    target_dollars = np.zeros(N_INST)
    log_all = np.log(np.maximum(prcSoFar, EPS))

    # Pairs sleeve (hysteresis FSM per pair).
    if nt > PAIR_ROLL + 1:
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
                target_dollars[i] += pos * PAIR_LEG
                target_dollars[j] -= pos * g * PAIR_LEG

    target_dollars = target_dollars + _leadlag_target_dollars(prcSoFar)
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)
    return (target_dollars / prcSoFar[:, -1]).astype(int)
