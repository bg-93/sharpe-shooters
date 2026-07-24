#!/usr/bin/env python3
"""Shock-regime canonical lead-lag for deployment after day 1,000.

Forecasts are learned from expanding one-day stock returns.  A regularized
rank-3 canonical map captures the persistent predictive modes.  Two
additional rank-3 maps are fit for quiet and broad-shock days, then softly
blended with the global map according to today's observable shock state.

The strategy is deliberately pair-free.  ALGO is derived from its exact
normalized-stock-index identity rather than forecast as an extra asset.
The 250-day refit interval deliberately freezes maps across one leaderboard
window: deployment fits on all 1,000 supplied days without learning from the
future test segment.
"""

import numpy as np

N = 51
STOCKS = 50
MIN_HISTORY = 180
MIN_STATE_OBS = 300
RETRAIN = 250
RIDGE_LAMBDA = 400.0
CANONICAL_RANK = 3
CANONICAL_SHRINK = 0.50
DENSE_WEIGHT = 0.25
REGIME_WEIGHT = 0.10
ALGO_POSITION_BOOST = 1.50
TEMPERATURE = 0.20
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * STOCKS)

_global_params = None
_state_params = None
_shock_cut = None
_algo_sd = None
_last_fit = _previous_nt = -1


def reset_state():
    global _global_params, _state_params, _shock_cut, _algo_sd
    global _last_fit, _previous_nt
    _global_params = _state_params = None
    _shock_cut = _algo_sd = None
    _last_fit = _previous_nt = -1


def _normalize(signal):
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


def _matrix_root(matrix, inverse=False):
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    exponent = -0.5 if inverse else 0.5
    return (vectors * values ** exponent) @ vectors.T


def _fit_map(x, y):
    x_mean, x_sd = x.mean(0), x.std(0) + EPS
    y_mean, y_sd = y.mean(0), y.std(0) + EPS
    xz = (x - x_mean) / x_sd
    yz = (y - y_mean) / y_sd

    dense = np.linalg.solve(
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
    canonical = (
        inverse_x
        @ left[:, :rank]
        @ np.diag(singular[:rank])
        @ right_t[:rank]
        @ root_y
    )
    return canonical, dense, x_mean, x_sd, y_mean, y_sd


def _fit(prices):
    global _global_params, _state_params, _shock_cut, _algo_sd

    all_returns = prices[:, 1:] / prices[:, :-1] - 1.0
    stock_returns = all_returns[1:]
    x = stock_returns[:, :-1].T
    y = stock_returns[:, 1:].T

    _global_params = _fit_map(x, y)
    _, _, x_mean, x_sd, _, _ = _global_params
    standardized = (x - x_mean) / x_sd
    shock_energy = np.sqrt(
        np.mean(standardized * standardized, axis=1)
    )
    _shock_cut = float(np.median(shock_energy))
    labels = shock_energy >= _shock_cut

    states = []
    for high_shock in (False, True):
        selected = labels == high_shock
        if selected.sum() < MIN_STATE_OBS:
            states.append(_global_params)
        else:
            states.append(_fit_map(x[selected], y[selected]))
    _state_params = states
    _algo_sd = all_returns[0, 1:].std() + EPS


def _coherent_signal(stock_z, stock_prediction, prices):
    normalized_levels = prices[1:, -1] / prices[1:, 0]
    index_weights = normalized_levels / normalized_levels.sum()
    algo_prediction = index_weights @ stock_prediction
    return _normalize(np.r_[algo_prediction / _algo_sd, stock_z])


def _forecast_signal(params, current_return, prices):
    canonical, dense, x_mean, x_sd, y_mean, y_sd = params
    current = (current_return - x_mean) / x_sd

    canonical_z = current @ canonical
    canonical_signal = _coherent_signal(
        canonical_z,
        canonical_z * y_sd + y_mean,
        prices,
    )
    dense_z = current @ dense
    dense_signal = _coherent_signal(
        dense_z,
        dense_z * y_sd,
        prices,
    )
    return _normalize(
        canonical_signal + DENSE_WEIGHT * dense_signal
    )


def getMyPosition(prcSoFar):
    global _last_fit, _previous_nt
    observations = prcSoFar.shape[1]
    if observations <= _previous_nt:
        reset_state()
    _previous_nt = observations
    if observations < MIN_HISTORY:
        return np.zeros(N, dtype=int)

    if (
        _global_params is None
        or observations - _last_fit >= RETRAIN
    ):
        _fit(prcSoFar)
        _last_fit = observations

    all_returns = prcSoFar[:, 1:] / prcSoFar[:, :-1] - 1.0
    current_return = all_returns[1:, -1]
    global_signal = _forecast_signal(
        _global_params, current_return, prcSoFar
    )

    # Use the global fit's exact standardization for both the historical
    # median threshold and today's label.
    x_mean, x_sd = _global_params[2], _global_params[3]
    current_z = (current_return - x_mean) / x_sd
    high_shock = int(
        np.sqrt(np.mean(current_z * current_z)) >= _shock_cut
    )
    state_signal = _forecast_signal(
        _state_params[high_shock], current_return, prcSoFar
    )
    signal = _normalize(
        global_signal + REGIME_WEIGHT * state_signal
    )
    target_dollars = LIMITS * np.tanh(signal / TEMPERATURE)
    target_dollars[0] *= ALGO_POSITION_BOOST
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)
    return (target_dollars / prcSoFar[:, -1]).astype(int)
