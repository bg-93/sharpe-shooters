#!/usr/bin/env python3
"""Simple non-pairs candidate: rolling cross-sectional factor rotation.

Compress the 50 stocks into principal-component factor returns, regress the
next stock-return cross section on today's factors, and trade the forecast
with smooth dollar-neutral sizing.  ALGO is deliberately unused.
"""

import numpy as np

N_INST = 51
N_FACTORS = 35
LOOKBACK = 400
RIDGE = 20.0
TEMP = 0.8
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * 50)


def getMyPosition(prcSoFar):
    if prcSoFar.shape[1] < 101:
        return np.zeros(N_INST, dtype=int)

    returns = np.diff(
        np.log(np.maximum(prcSoFar[1:, -LOOKBACK - 1:], EPS)), axis=1
    )
    mean = returns.mean(axis=1, keepdims=True)
    scale = returns.std(axis=1, keepdims=True) + EPS
    standardized = (returns - mean) / scale

    loadings, _, _ = np.linalg.svd(standardized, full_matrices=False)
    loadings = loadings[:, :N_FACTORS]
    factors = loadings.T @ standardized

    x = factors[:, :-1].T
    y = standardized[:, 1:].T
    weights = np.linalg.solve(
        x.T @ x + RIDGE * np.eye(N_FACTORS), x.T @ y
    )
    forecast = factors[:, -1] @ weights
    forecast -= forecast.mean()
    forecast /= forecast.std() + EPS

    dollars = np.zeros(N_INST)
    dollars[1:] = LIMITS[1:] * np.tanh(forecast / TEMP)
    return (dollars / prcSoFar[:, -1]).astype(int)
