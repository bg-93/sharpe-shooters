#!/usr/bin/env python3
"""Lead-lag candidate: recursive ridge with demeaned tanh sizing."""

import numpy as np

N = 51
LAM = 400.0
FORGETTING = 0.9975
TEMP = 0.35
MIN_HIST = 120
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * 50)

_precision = _coef = _mu = _sd = _ysd = None
_prev_nt = -1


def reset_state():
    global _precision, _coef, _mu, _sd, _ysd, _prev_nt
    _precision = _coef = _mu = _sd = _ysd = None
    _prev_nt = -1


def getMyPosition(prcSoFar):
    global _precision, _coef, _mu, _sd, _ysd, _prev_nt
    nt = prcSoFar.shape[1]
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt
    if nt < MIN_HIST:
        return np.zeros(N, dtype=int)

    returns = np.diff(np.log(np.maximum(prcSoFar, EPS)), axis=1)
    if _coef is None:
        x, y = returns[:, :-1].T, returns[:, 1:].T
        _mu, _sd = x.mean(0), x.std(0) + EPS
        _ysd = y.std(0) + EPS
        z = (x - _mu) / _sd
        _precision = np.linalg.inv(z.T @ z + LAM * np.eye(N))
        _coef = _precision @ z.T @ y
    else:
        # Incorporate the single newly observed input/output pair.
        x = (returns[:, -2] - _mu) / _sd
        y = returns[:, -1]
        px = _precision @ x
        gain = px / (FORGETTING + x @ px)
        _coef += np.outer(gain, y - x @ _coef)
        _precision = (
            _precision - np.outer(gain, x @ _precision)
        ) / FORGETTING

    prediction = ((returns[:, -1] - _mu) / _sd) @ _coef
    signal = prediction / _ysd
    signal -= signal.mean()
    signal /= signal.std() + EPS
    dollars = LIMITS * np.tanh(signal / TEMP)
    return (dollars / prcSoFar[:, -1]).astype(int)
