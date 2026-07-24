#!/usr/bin/env python3
"""Dense ridge with confidence-preserving, non-saturating allocation."""

import numpy as np

N = 51
LAM = 400.0
RETRAIN = 50
MIN_HIST = 120
TEMP = 0.50
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * 50)

_coef = _mu = _sd = _ysd = None
_fixed_scale = None
_last_fit = _prev_nt = -1


def reset_state():
    global _coef, _mu, _sd, _ysd, _fixed_scale, _last_fit, _prev_nt
    _coef = _mu = _sd = _ysd = _fixed_scale = None
    _last_fit = _prev_nt = -1


def getMyPosition(prcSoFar):
    global _coef, _mu, _sd, _ysd, _fixed_scale, _last_fit, _prev_nt
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
        _ysd = y.std(0) + EPS
        z = (x - _mu) / _sd
        _coef = np.linalg.solve(
            z.T @ z + LAM * np.eye(N), z.T @ y
        )

        # Calibrate once per refit from the historical distribution. Unlike
        # daily normalization, this preserves whether today is weak or strong.
        fitted = (z @ _coef) / _ysd
        fitted -= fitted.mean(1, keepdims=True)
        _fixed_scale = np.median(fitted.std(1)) + EPS
        _last_fit = nt

    prediction = ((returns[:, -1] - _mu) / _sd) @ _coef
    signal = prediction / _ysd
    signal -= signal.mean()
    signal /= _fixed_scale

    dollars = LIMITS * np.tanh(signal / TEMP)
    return (dollars / prcSoFar[:, -1]).astype(int)
