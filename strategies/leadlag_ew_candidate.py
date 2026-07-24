#!/usr/bin/env python3
"""Lead-lag variant: gently exponentially weighted ridge + demeaned tanh."""

import numpy as np

N = 51
LAM = 400.0
DECAY = 0.9975
RETRAIN = 25
MIN_HIST = 120
TEMP = 0.35
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * 50)

_coef = _mu = _sd = _ysd = None
_last_fit = _prev_nt = -1


def reset_state():
    global _coef, _mu, _sd, _ysd, _last_fit, _prev_nt
    _coef = _mu = _sd = _ysd = None
    _last_fit = _prev_nt = -1


def getMyPosition(prcSoFar):
    global _coef, _mu, _sd, _ysd, _last_fit, _prev_nt
    nt = prcSoFar.shape[1]
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt
    if nt < MIN_HIST:
        return np.zeros(N, dtype=int)

    returns = np.diff(np.log(np.maximum(prcSoFar, EPS)), axis=1)
    if _coef is None or nt - _last_fit >= RETRAIN:
        x, y = returns[:, :-1].T, returns[:, 1:].T
        _mu, _sd = x.mean(0), x.std(0) + EPS
        z = (x - _mu) / _sd
        weights = DECAY ** np.arange(len(z) - 1, -1, -1)
        root_w = np.sqrt(weights)[:, None]
        zw, yw = z * root_w, y * root_w
        _coef = np.linalg.solve(
            zw.T @ zw + LAM * np.eye(N), zw.T @ yw
        )
        _ysd = y.std(0) + EPS
        _last_fit = nt

    prediction = ((returns[:, -1] - _mu) / _sd) @ _coef
    signal = prediction / _ysd
    signal -= signal.mean()
    signal /= signal.std() + EPS
    dollars = LIMITS * np.tanh(signal / TEMP)
    return (dollars / prcSoFar[:, -1]).astype(int)
