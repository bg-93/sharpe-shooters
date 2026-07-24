#!/usr/bin/env python3
"""Persistent-mode canonical lead-lag for the 1,000-day release.

The broad dense model in the previous candidate became unreliable late in
the newly revealed window.  This version keeps only the three strongest
regularized canonical predictive modes and uses dense ridge as a small
stabilizer.  It is pair-free and uses expanding history: the signal is meant
to capture persistent directed dependence, not recent price momentum.
"""

import numpy as np

N = 51
STOCKS = 50
MIN_HISTORY = 120
RETRAIN = 100
RIDGE_LAMBDA = 400.0
CANONICAL_RANK = 3
CANONICAL_SHRINK = 0.50
DENSE_WEIGHT = 0.25
TEMPERATURE = 0.20
INCLUDE_DRIFT = True
ALGO_SIGNAL_SCALE = 1.0
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * STOCKS)

_ridge = _canonical = None
_x_mean = _x_sd = _y_mean = _y_sd = None
_algo_sd = None
_last_fit = _previous_nt = -1


def reset_state():
    global _ridge, _canonical, _x_mean, _x_sd, _y_mean, _y_sd
    global _algo_sd, _last_fit, _previous_nt
    _ridge = _canonical = None
    _x_mean = _x_sd = _y_mean = _y_sd = None
    _algo_sd = None
    _last_fit = _previous_nt = -1


def _matrix_root(matrix, inverse=False):
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    exponent = -0.5 if inverse else 0.5
    return (vectors * values ** exponent) @ vectors.T


def _fit(prices):
    global _ridge, _canonical, _x_mean, _x_sd, _y_mean, _y_sd
    global _algo_sd

    all_returns = prices[:, 1:] / prices[:, :-1] - 1.0
    stock_returns = all_returns[1:]
    x = stock_returns[:, :-1].T
    y = stock_returns[:, 1:].T

    _x_mean, _x_sd = x.mean(0), x.std(0) + EPS
    _y_mean, _y_sd = y.mean(0), y.std(0) + EPS
    xz = (x - _x_mean) / _x_sd
    yz = (y - _y_mean) / _y_sd

    _ridge = np.linalg.solve(
        xz.T @ xz + RIDGE_LAMBDA * np.eye(STOCKS),
        xz.T @ yz,
    )

    observations = len(xz)
    cov_x = xz.T @ xz / observations
    cov_y = yz.T @ yz / observations
    cov_x = (
        (1.0 - CANONICAL_SHRINK) * cov_x
        + CANONICAL_SHRINK * np.eye(STOCKS)
    )
    cov_y = (
        (1.0 - CANONICAL_SHRINK) * cov_y
        + CANONICAL_SHRINK * np.eye(STOCKS)
    )
    inverse_x = _matrix_root(cov_x, inverse=True)
    inverse_y = _matrix_root(cov_y, inverse=True)
    root_y = _matrix_root(cov_y)
    cross = xz.T @ yz / observations
    left, singular, right_t = np.linalg.svd(
        inverse_x @ cross @ inverse_y
    )
    rank = CANONICAL_RANK
    _canonical = (
        inverse_x
        @ left[:, :rank]
        @ np.diag(singular[:rank])
        @ right_t[:rank]
        @ root_y
    )
    _algo_sd = all_returns[0, 1:].std() + EPS


def _normalize(signal):
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


def _coherent_signal(stock_z, prices, include_drift):
    stock_prediction = stock_z * _y_sd
    if include_drift:
        stock_prediction = stock_prediction + _y_mean

    # ALGO is the exact normalized-level index of the 50 stocks.  Forecast
    # the independent stocks and derive, rather than separately fit, ALGO.
    normalized_levels = prices[1:, -1] / prices[1:, 0]
    index_weights = normalized_levels / normalized_levels.sum()
    algo_prediction = index_weights @ stock_prediction
    signal = np.r_[algo_prediction / _algo_sd, stock_z]
    signal[0] *= ALGO_SIGNAL_SCALE
    return _normalize(signal)


def getMyPosition(prcSoFar):
    global _last_fit, _previous_nt
    observations = prcSoFar.shape[1]
    if observations <= _previous_nt:
        reset_state()
    _previous_nt = observations
    if observations < MIN_HISTORY:
        return np.zeros(N, dtype=int)

    if _canonical is None or observations - _last_fit >= RETRAIN:
        _fit(prcSoFar)
        _last_fit = observations

    stock_returns = (
        prcSoFar[1:, 1:] / prcSoFar[1:, :-1] - 1.0
    )
    current = (stock_returns[:, -1] - _x_mean) / _x_sd
    canonical_signal = _coherent_signal(
        current @ _canonical,
        prcSoFar,
        include_drift=INCLUDE_DRIFT,
    )
    dense_signal = _coherent_signal(
        current @ _ridge,
        prcSoFar,
        include_drift=False,
    )
    signal = _normalize(
        canonical_signal + DENSE_WEIGHT * dense_signal
    )
    target_dollars = LIMITS * np.tanh(signal / TEMPERATURE)
    return (target_dollars / prcSoFar[:, -1]).astype(int)
