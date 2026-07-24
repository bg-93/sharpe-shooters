#!/usr/bin/env python3
"""Sample-size curriculum: precision network -> canonical lead-lag.

The genuinely new core is pair-free.  PAIR_SCALE controls an optional,
separable overlay of the existing frozen pairs.  Set it to 0.0 for the pure
new strategy or 1.0 for the highest released-data score.
"""

import numpy as np

N = 51
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * 50)

# Core settings.
PRECISION_POWER = 0.70
EIGEN_FLOOR = 0.05
PRECISION_RETRAIN = 25
VALIDATION = 80
CCA_RANK = 7
CCA_SHRINK = 0.25
RIDGE_LAM = 400.0
CCA_RETRAIN = 50
TEMP = 0.35
TRANSITION_START = 500
TRANSITION_END = 600

# Optional legacy overlay.  0.0 is completely pair-free.
PAIR_SCALE = 1.0
PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836),
    (13, 45, 1.0132), (33, 40, 0.2577), (10, 46, 1.0331),
    (33, 42, 0.8358), (31, 43, 0.9692), (18, 28, 0.5642),
    (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)
PAIR_ROLL, PAIR_ENTRY, PAIR_EXIT, PAIR_LEG = 60, 1.5, 0.5, 9000.0

_p_map = _p_xmu = _p_xsd = _p_ysd = None
_p_last_fit = _select_last = -1
_direct_algo = 1.0

_cca = _ridge = None
_c_xmu = _c_xsd = _c_ymu = _c_ysd = _algo_sd = None
_c_last_fit = -1

_pair_state = [0] * len(PAIRS)
_prev_nt = -1


def reset_state():
    global _p_map, _p_xmu, _p_xsd, _p_ysd, _p_last_fit
    global _select_last, _direct_algo
    global _cca, _ridge, _c_xmu, _c_xsd, _c_ymu, _c_ysd, _algo_sd
    global _c_last_fit, _pair_state, _prev_nt
    _p_map = _p_xmu = _p_xsd = _p_ysd = None
    _p_last_fit = _select_last = -1
    _direct_algo = 1.0
    _cca = _ridge = None
    _c_xmu = _c_xsd = _c_ymu = _c_ysd = _algo_sd = None
    _c_last_fit = -1
    _pair_state = [0] * len(PAIRS)
    _prev_nt = -1


def _standardize(x):
    mean, sd = x.mean(0), x.std(0) + EPS
    return (x - mean) / sd, mean, sd


def _fractional_map(zx, zy):
    covariance = zx.T @ zx / len(zx)
    cross = zx.T @ zy / len(zx)
    values, vectors = np.linalg.eigh(covariance)
    inverse = np.maximum(values, EIGEN_FLOOR) ** (-PRECISION_POWER)
    whitener = (vectors * inverse) @ vectors.T
    return whitener @ cross


def _fit_precision(prices):
    global _p_map, _p_xmu, _p_xsd, _p_ysd
    r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
    zx, _p_xmu, _p_xsd = _standardize(r[:-1])
    zy, _, _p_ysd = _standardize(r[1:])
    _p_map = _fractional_map(zx, zy)


def _select_algo(prices):
    """Held-out choice between direct ALGO alpha and a stock beta hedge."""
    global _direct_algo
    r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
    cut = len(r) - VALIDATION
    if cut < 60:
        _direct_algo = 1.0
        return
    zx, xmu, xsd = _standardize(r[:cut - 1])
    zy, _, _ = _standardize(r[1:cut])
    mapping = _fractional_map(zx, zy)
    predictions = ((r[cut - 1:-1] - xmu) / xsd) @ mapping
    predictions -= predictions.mean(1, keepdims=True)
    direct = LIMITS[0] * np.sign(predictions[:, 0])

    covariance = np.cov(r[:cut], rowvar=False)
    beta = covariance[:, 0] / (covariance[0, 0] + EPS)
    stock_dollars = LIMITS[1:] * np.sign(predictions[:, 1:])
    hedge = -(stock_dollars * beta[1:]).sum(1) / (beta[0] + EPS)
    hedge = np.clip(hedge, -LIMITS[0], LIMITS[0])
    realised = r[cut:, 0]
    _direct_algo = float(
        np.mean(direct * realised) >= np.mean(hedge * realised)
    )


def _precision_dollars(prices):
    global _p_last_fit, _select_last
    nt = prices.shape[1]
    if nt < 100:
        return np.zeros(N)
    if _p_map is None or nt - _p_last_fit >= PRECISION_RETRAIN:
        _fit_precision(prices)
        _p_last_fit = nt
    if nt - _select_last >= PRECISION_RETRAIN:
        _select_algo(prices)
        _select_last = nt

    now = np.log(np.maximum(prices[:, -1], EPS)
                 / np.maximum(prices[:, -2], EPS))
    prediction = (((now - _p_xmu) / _p_xsd) @ _p_map) * _p_ysd
    signal = prediction / _p_ysd
    signal -= signal.mean()
    dollars = LIMITS * np.sign(signal)

    r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
    covariance = np.cov(r, rowvar=False)
    beta = covariance[:, 0] / (covariance[0, 0] + EPS)
    hedge = -(dollars[1:] * beta[1:]).sum() / (beta[0] + EPS)
    hedge = np.clip(hedge, -LIMITS[0], LIMITS[0])
    dollars[0] = _direct_algo * dollars[0] + (1.0 - _direct_algo) * hedge
    return dollars


def _matrix_root(matrix, inverse=False):
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    return (vectors * values ** (-0.5 if inverse else 0.5)) @ vectors.T


def _fit_canonical(prices):
    global _cca, _ridge, _c_xmu, _c_xsd, _c_ymu, _c_ysd, _algo_sd
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    stock = returns[1:]
    x, y = stock[:, :-1].T, stock[:, 1:].T
    _c_xmu, _c_xsd = x.mean(0), x.std(0) + EPS
    _c_ymu, _c_ysd = y.mean(0), y.std(0) + EPS
    xz, yz = (x - _c_xmu) / _c_xsd, (y - _c_ymu) / _c_ysd
    _ridge = np.linalg.solve(
        xz.T @ xz + RIDGE_LAM * np.eye(50), xz.T @ yz
    )

    n = len(xz)
    cov_x, cov_y = xz.T @ xz / n, yz.T @ yz / n
    cov_x = (1 - CCA_SHRINK) * cov_x + CCA_SHRINK * np.eye(50)
    cov_y = (1 - CCA_SHRINK) * cov_y + CCA_SHRINK * np.eye(50)
    inv_x, inv_y = _matrix_root(cov_x, True), _matrix_root(cov_y, True)
    root_y = _matrix_root(cov_y)
    left, singular, right_t = np.linalg.svd(
        inv_x @ (xz.T @ yz / n) @ inv_y
    )
    k = CCA_RANK
    _cca = (
        inv_x @ left[:, :k] @ np.diag(singular[:k])
        @ right_t[:k] @ root_y
    )
    _algo_sd = returns[0, 1:].std() + EPS


def _normalize(signal):
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


def _coherent(stock_z, prices, include_mean):
    prediction = stock_z * _c_ysd
    if include_mean:
        prediction += _c_ymu
    levels = prices[1:, -1] / prices[1:, 0]
    algo = (levels / levels.sum()) @ prediction
    return _normalize(np.r_[algo / _algo_sd, stock_z])


def _canonical_dollars(prices):
    global _c_last_fit
    nt = prices.shape[1]
    if nt < 120:
        return np.zeros(N)
    if _cca is None or nt - _c_last_fit >= CCA_RETRAIN:
        _fit_canonical(prices)
        _c_last_fit = nt
    returns = prices[1:, 1:] / prices[1:, :-1] - 1.0
    now = (returns[:, -1] - _c_xmu) / _c_xsd
    cca_signal = _coherent(now @ _cca, prices, True)
    ridge_signal = _coherent(now @ _ridge, prices, False)
    signal = _normalize(cca_signal + 0.5 * ridge_signal)
    return LIMITS * np.tanh(signal / TEMP)


def _pair_dollars(prices):
    if PAIR_SCALE <= 0 or prices.shape[1] <= PAIR_ROLL + 1:
        return np.zeros(N)
    log_prices = np.log(np.maximum(prices, EPS))
    dollars = np.zeros(N)
    for k, (i, j, gamma) in enumerate(PAIRS):
        spread = log_prices[i] - gamma * log_prices[j]
        window = spread[-PAIR_ROLL - 1:-1]
        z = (spread[-1] - window.mean()) / (window.std() + EPS)
        state = _pair_state[k]
        if state == 0:
            state = -1 if z > PAIR_ENTRY else (1 if z < -PAIR_ENTRY else 0)
        elif state == 1 and z > -PAIR_EXIT:
            state = 0
        elif state == -1 and z < PAIR_EXIT:
            state = 0
        _pair_state[k] = state
        dollars[i] += PAIR_SCALE * state * PAIR_LEG
        dollars[j] -= PAIR_SCALE * state * gamma * PAIR_LEG
    return dollars


def getMyPosition(prcSoFar):
    global _prev_nt
    nt = prcSoFar.shape[1]
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt

    canonical_weight = np.clip(
        (nt - TRANSITION_START) / (TRANSITION_END - TRANSITION_START),
        0.0, 1.0,
    )
    core = (
        (1.0 - canonical_weight) * _precision_dollars(prcSoFar)
        + canonical_weight * _canonical_dollars(prcSoFar)
    )
    dollars = np.clip(core + _pair_dollars(prcSoFar), -LIMITS, LIMITS)
    return (dollars / prcSoFar[:, -1]).astype(int)
