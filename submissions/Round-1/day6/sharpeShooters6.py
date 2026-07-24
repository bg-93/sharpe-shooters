#!/usr/bin/env python3
"""Coherent canonical lead-lag strategy.

Forecast the 50 stocks from their lagged simple returns using two views:
regularized canonical predictive modes and dense ridge. Derive ALGO's
forecast from its exact normalized-index identity, blend the two normalized
views, then use the proven cross-sectional tanh allocation.

ON THE SUBMISSION THIS YEILDED 763 score.
The day 2 submission strategy yeilded 500 on this so its the next best strategy rn.

"""

import numpy as np

N = 51
STOCKS = 50
MIN_HIST = 120
RETRAIN = 50
RIDGE_LAM = 400.0
CCA_RANK = 7
CCA_SHRINK = 0.25
DENSE_WEIGHT = 0.50
TEMP = 0.35
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * STOCKS)

_ridge = _canonical = None
_x_mean = _x_sd = _y_mean = _y_sd = None
_algo_sd = None
_last_fit = _prev_nt = -1


def reset_state():
    global _ridge, _canonical, _x_mean, _x_sd, _y_mean, _y_sd
    global _algo_sd, _last_fit, _prev_nt
    _ridge = _canonical = None
    _x_mean = _x_sd = _y_mean = _y_sd = None
    _algo_sd = None
    _last_fit = _prev_nt = -1


def _matrix_root(matrix, inverse=False):
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    power = -0.5 if inverse else 0.5
    return (vectors * values ** power) @ vectors.T


def _fit(prices):
    global _ridge, _canonical, _x_mean, _x_sd, _y_mean, _y_sd
    global _algo_sd

    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    stock_returns = returns[1:]
    x = stock_returns[:, :-1].T
    y = stock_returns[:, 1:].T

    _x_mean, _x_sd = x.mean(0), x.std(0) + EPS
    _y_mean, _y_sd = y.mean(0), y.std(0) + EPS
    xz = (x - _x_mean) / _x_sd
    yz = (y - _y_mean) / _y_sd

    # Broad dense view: weak information distributed across all stocks.
    _ridge = np.linalg.solve(
        xz.T @ xz + RIDGE_LAM * np.eye(STOCKS), xz.T @ yz
    )

    # Canonical view: whiten both sides and keep only predictive joint modes.
    n_obs = len(xz)
    cov_x = xz.T @ xz / n_obs
    cov_y = yz.T @ yz / n_obs
    cov_x = (1.0 - CCA_SHRINK) * cov_x + CCA_SHRINK * np.eye(STOCKS)
    cov_y = (1.0 - CCA_SHRINK) * cov_y + CCA_SHRINK * np.eye(STOCKS)
    inv_x = _matrix_root(cov_x, inverse=True)
    inv_y = _matrix_root(cov_y, inverse=True)
    root_y = _matrix_root(cov_y)
    cross = xz.T @ yz / n_obs
    left, singular, right_t = np.linalg.svd(inv_x @ cross @ inv_y)
    k = CCA_RANK
    _canonical = (
        inv_x
        @ left[:, :k]
        @ np.diag(singular[:k])
        @ right_t[:k]
        @ root_y
    )
    _algo_sd = returns[0, 1:].std() + EPS


def _normalize(signal):
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


def _coherent_signal(stock_z, prices, include_drift):
    stock_prediction = stock_z * _y_sd
    if include_drift:
        stock_prediction = stock_prediction + _y_mean

    # ALGO = 100 * mean(P_i / P_i,0), so its next simple return is
    # exactly the current normalized-level-weighted stock simple return.
    normalized_levels = prices[1:, -1] / prices[1:, 0]
    index_weights = normalized_levels / normalized_levels.sum()
    algo_prediction = index_weights @ stock_prediction
    return _normalize(np.r_[algo_prediction / _algo_sd, stock_z])


def getMyPosition(prcSoFar):
    global _last_fit, _prev_nt
    nt = prcSoFar.shape[1]
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt
    if nt < MIN_HIST:
        return np.zeros(N, dtype=int)

    if _ridge is None or nt - _last_fit >= RETRAIN:
        _fit(prcSoFar)
        _last_fit = nt

    stock_returns = prcSoFar[1:, 1:] / prcSoFar[1:, :-1] - 1.0
    x_now = (stock_returns[:, -1] - _x_mean) / _x_sd

    canonical_signal = _coherent_signal(
        x_now @ _canonical, prcSoFar, include_drift=True
    )
    dense_signal = _coherent_signal(
        x_now @ _ridge, prcSoFar, include_drift=False
    )
    signal = _normalize(canonical_signal + DENSE_WEIGHT * dense_signal)

    target_dollars = LIMITS * np.tanh(signal / TEMP)
    return (target_dollars / prcSoFar[:, -1]).astype(int)
