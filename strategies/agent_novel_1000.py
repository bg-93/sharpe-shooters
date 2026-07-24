#!/usr/bin/env python3
"""Leakage-safe 1,000-day strategy research.

The first selection window is days 500--750.  Candidate parameters are
chosen there and evaluated once on the newly revealed days 750--1000.
Everything in this file is walk-forward: a position for day ``t`` only uses
prices strictly available at the close of day ``t``.

This is intentionally a research runner rather than the live submission.
It explores simple alternatives to frozen pairs and the existing
dense/CCA lead-lag curriculum:

* rolling versus expanding lag maps;
* smooth online expert combinations;
* regime-conditioned ridge with shrinkage back to a global map;
* slow residual momentum and trend-breakout overlays;
* target reliability calibration.

Run from the repository root with::

    .venv/bin/python strategies/agent_novel_1000.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
N = 51
STOCKS = 50
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * STOCKS)
COMM = np.array([0.00002] + [0.0001] * STOCKS)


def load_prices(committed: bool = False) -> np.ndarray:
    if not committed:
        return np.loadtxt(ROOT / "prices.txt", skiprows=1).T
    # Useful when another concurrent research process is deliberately
    # perturbing the worktree copy for a stress test.
    raw = subprocess.check_output(
        ["git", "show", "HEAD:prices.txt"], cwd=ROOT
    )
    return np.loadtxt(io.BytesIO(raw), skiprows=1).T


def score_pl(pl: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(pl))
    std = float(np.std(pl))
    if mean <= 0.0 or std < EPS:
        return mean, std, mean
    sr2 = 250.0 * mean * mean / (std * std)
    return mean, std, mean * sr2 / (sr2 + 1.0)


def simulate(
    prices: np.ndarray,
    start: int,
    end: int,
    target: Callable[[np.ndarray], np.ndarray],
    collect: bool = False,
):
    """Official evaluator semantics for trade days ``[start, end)``."""
    current = np.zeros(N)
    cash = value = pending_fee = volume = 0.0
    pnl = []
    for t in range(start, end + 1):
        hist = prices[:, :t]
        price = hist[:, -1]
        if t < end:
            dollars = np.asarray(target(hist), dtype=float)
            max_shares = (LIMITS / price).astype(int)
            new = np.clip((dollars / price).astype(int),
                          -max_shares, max_shares)
        else:
            new = current.copy()
        change = new - current
        cash -= price @ change + pending_fee
        traded = price * np.abs(change)
        volume += traded.sum()
        pending_fee = traded @ COMM
        current = new
        new_value = cash + current @ price
        day_pl = new_value - value
        value = new_value
        if t > start:
            pnl.append(day_pl)
    pnl = np.asarray(pnl)
    result = (*score_pl(pnl), volume)
    return (*result, pnl) if collect else result


def simulate_by_asset(
    prices: np.ndarray,
    start: int,
    end: int,
    target: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return daily and aggregate PnL contributions per instrument."""
    current = np.zeros(N)
    pending_fee = np.zeros(N)
    rows = []
    for t in range(start, end):
        hist = prices[:, :t]
        price = hist[:, -1]
        dollars = np.asarray(target(hist), dtype=float)
        max_shares = (LIMITS / price).astype(int)
        new = np.clip((dollars / price).astype(int),
                      -max_shares, max_shares)
        change = new - current
        # Position selected at p[t-1] earns the p[t]-p[t-1] move.  The
        # evaluator charges the previous close's pending commission then.
        next_price = prices[:, t]
        rows.append(new * (next_price - price) - pending_fee)
        pending_fee = price * np.abs(change) * COMM
        current = new
    rows = np.asarray(rows)
    return rows, rows.sum(0)


def standardize(x: np.ndarray):
    mean = x.mean(0)
    sd = x.std(0) + EPS
    return (x - mean) / sd, mean, sd


def normalize(signal: np.ndarray) -> np.ndarray:
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


def coherent_stock_signal(
    stock_z: np.ndarray,
    stock_prediction: np.ndarray,
    prices: np.ndarray,
    algo_sd: float,
) -> np.ndarray:
    """Add the mechanically coherent ALGO forecast to a stock forecast."""
    levels = prices[1:, -1] / prices[1:, 0]
    weights = levels / levels.sum()
    algo_prediction = weights @ stock_prediction
    return normalize(np.r_[algo_prediction / (algo_sd + EPS), stock_z])


@dataclass(frozen=True)
class Config:
    lam: float = 400.0
    retrain: int = 25
    temperature: float = 0.35
    rolling: int | None = None
    recent_weight: float = 0.0
    recent_window: int = 300
    reliability_power: float = 0.0
    reliability_floor: float = 0.25
    regime: str = "none"
    regime_strength: float = 0.0
    regime_lam: float = 400.0
    momentum_weight: float = 0.0
    momentum_fast: int = 10
    momentum_slow: int = 60
    momentum_mode: str = "plain"


class NovelModel:
    """Dense lag-one ridge with optional simple, leakage-safe refinements."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.last_nt = -1
        self.last_fit = -1
        self.x_mean = self.x_sd = self.y_mean = self.y_sd = None
        self.global_map = self.recent_map = None
        self.regime_maps = None
        self.algo_sd = None
        self.reliability = None

    @staticmethod
    def _ridge(xz: np.ndarray, yz: np.ndarray, lam: float,
               prior: np.ndarray | None = None) -> np.ndarray:
        rhs = xz.T @ yz
        if prior is not None:
            rhs = rhs + lam * prior
        return np.linalg.solve(
            xz.T @ xz + lam * np.eye(xz.shape[1]), rhs
        )

    def _regime_values(self, full_returns: np.ndarray,
                       stock_x: np.ndarray) -> np.ndarray:
        mode = self.cfg.regime
        if mode == "market_sign":
            return (full_returns[0, :-1] >= 0.0).astype(int)
        if mode == "breadth":
            return (np.mean(stock_x > 0.0, axis=1) >= 0.5).astype(int)
        if mode == "dispersion":
            dispersion = stock_x.std(1)
            threshold = np.median(dispersion)
            return (dispersion >= threshold).astype(int)
        if mode == "market_size":
            magnitude = np.abs(full_returns[0, :-1])
            threshold = np.median(magnitude)
            return (magnitude >= threshold).astype(int)
        return np.zeros(len(stock_x), dtype=int)

    def _current_regime(self, full_returns: np.ndarray,
                        stock_return: np.ndarray) -> int:
        mode = self.cfg.regime
        if mode == "market_sign":
            return int(full_returns[0, -1] >= 0.0)
        if mode == "breadth":
            return int(np.mean(stock_return > 0.0) >= 0.5)
        if mode == "dispersion":
            historical = full_returns[1:, :-1].std(0)
            return int(stock_return.std() >= np.median(historical))
        if mode == "market_size":
            historical = np.abs(full_returns[0, :-1])
            return int(abs(full_returns[0, -1]) >= np.median(historical))
        return 0

    def _fit(self, prices: np.ndarray):
        returns = prices[:, 1:] / prices[:, :-1] - 1.0
        stocks = returns[1:]
        x_all, y_all = stocks[:, :-1].T, stocks[:, 1:].T
        if self.cfg.rolling is not None:
            keep = min(self.cfg.rolling, len(x_all))
            x, y = x_all[-keep:], y_all[-keep:]
        else:
            x, y = x_all, y_all
        xz, self.x_mean, self.x_sd = standardize(x)
        yz, self.y_mean, self.y_sd = standardize(y)
        self.global_map = self._ridge(xz, yz, self.cfg.lam)

        recent = min(self.cfg.recent_window, len(x_all))
        xr, yr = x_all[-recent:], y_all[-recent:]
        xrz = (xr - self.x_mean) / self.x_sd
        yrz = (yr - self.y_mean) / self.y_sd
        self.recent_map = self._ridge(xrz, yrz, self.cfg.lam)

        if self.cfg.regime != "none":
            # Regime maps are deviations shrunk toward the full-history map.
            regimes = self._regime_values(
                returns[:, -(len(x) + 1):], x
            )
            maps = []
            for state in (0, 1):
                take = regimes == state
                if take.sum() < 60:
                    maps.append(self.global_map)
                else:
                    maps.append(self._ridge(
                        xz[take], yz[take], self.cfg.regime_lam,
                        prior=self.global_map,
                    ))
            self.regime_maps = maps

        # In-fit forecast reliability is strongly upward biased in level, but
        # relative target differences can still be shrunk smoothly.
        fitted = xz @ self.global_map
        cov = np.mean((fitted - fitted.mean(0))
                      * (yz - yz.mean(0)), axis=0)
        pred_sd = fitted.std(0) + EPS
        rel = np.maximum(cov / pred_sd, 0.0)
        rel /= rel.mean() + EPS
        floor = self.cfg.reliability_floor
        self.reliability = floor + (1.0 - floor) * rel
        self.algo_sd = returns[0, 1:].std() + EPS

    def _momentum(self, prices: np.ndarray) -> np.ndarray:
        c = self.cfg
        if prices.shape[1] <= c.momentum_slow:
            return np.zeros(STOCKS)
        logp = np.log(np.maximum(prices[1:], EPS))
        fast = logp[:, -1] - logp[:, -1 - c.momentum_fast]
        slow = logp[:, -1] - logp[:, -1 - c.momentum_slow]
        ret = np.diff(logp[:, -c.momentum_slow - 1:], axis=1)
        vol = ret.std(1) + EPS
        trend = (0.5 * fast / np.sqrt(c.momentum_fast)
                 + 0.5 * slow / np.sqrt(c.momentum_slow)) / vol
        if c.momentum_mode == "residual":
            trend -= trend.mean()
        elif c.momentum_mode == "breakout":
            # Only express trends whose fast and slow estimates agree.
            trend *= np.sign(fast) == np.sign(slow)
        elif c.momentum_mode == "regime":
            # Momentum only when cross-sectional breadth and ALGO trend agree.
            algo = np.log(prices[0, -1] / prices[0, -1-c.momentum_slow])
            breadth = np.mean(slow > 0.0) - 0.5
            if algo * breadth <= 0.0:
                trend[:] = 0.0
        return normalize(trend)

    def target(self, prices: np.ndarray) -> np.ndarray:
        nt = prices.shape[1]
        if nt <= self.last_nt:
            self.reset()
        self.last_nt = nt
        if nt < 120:
            return np.zeros(N)
        if self.global_map is None or nt - self.last_fit >= self.cfg.retrain:
            self._fit(prices)
            self.last_fit = nt

        returns = prices[:, 1:] / prices[:, :-1] - 1.0
        now = (returns[1:, -1] - self.x_mean) / self.x_sd
        mapping = (
            (1.0 - self.cfg.recent_weight) * self.global_map
            + self.cfg.recent_weight * self.recent_map
        )
        stock_z = now @ mapping
        if self.regime_maps is not None:
            state = self._current_regime(returns, returns[1:, -1])
            regime_z = now @ self.regime_maps[state]
            weight = self.cfg.regime_strength
            stock_z = (1.0 - weight) * stock_z + weight * regime_z

        if self.cfg.reliability_power:
            stock_z *= self.reliability ** self.cfg.reliability_power

        stock_prediction = stock_z * self.y_sd + self.y_mean
        signal = coherent_stock_signal(
            stock_z, stock_prediction, prices, self.algo_sd
        )
        if self.cfg.momentum_weight:
            momentum = self._momentum(prices)
            momentum_prediction = momentum * self.y_sd
            mom_signal = coherent_stock_signal(
                momentum, momentum_prediction, prices, self.algo_sd
            )
            signal = normalize(
                signal + self.cfg.momentum_weight * mom_signal
            )
        return LIMITS * np.tanh(signal / self.cfg.temperature)


def make_model(cfg: Config):
    model = NovelModel(cfg)
    return model.target


@dataclass(frozen=True)
class CCAConfig:
    lag: int = 1
    rank: int = 7
    head_rank: int | None = None
    tail_weight: float = 1.0
    shrink: float = 0.25
    dense_weight: float = 0.5
    lam: float = 400.0
    retrain: int = 50
    temperature: float = 0.35
    train_window: int | None = None
    reliability_strength: float = 0.0
    reliability_window: int = 250
    reliability_floor: float = 0.35


def matrix_root(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    power = -0.5 if inverse else 0.5
    return (vectors * values ** power) @ vectors.T


def fit_cca_maps(x: np.ndarray, y: np.ndarray, cfg: CCAConfig):
    xz, xmean, xsd = standardize(x)
    yz, ymean, ysd = standardize(y)
    dense = np.linalg.solve(
        xz.T @ xz + cfg.lam * np.eye(STOCKS), xz.T @ yz
    )
    nobs = len(xz)
    covx = xz.T @ xz / nobs
    covy = yz.T @ yz / nobs
    covx = (1.0 - cfg.shrink) * covx + cfg.shrink * np.eye(STOCKS)
    covy = (1.0 - cfg.shrink) * covy + cfg.shrink * np.eye(STOCKS)
    invx, invy = matrix_root(covx, True), matrix_root(covy, True)
    rooty = matrix_root(covy)
    left, singular, right_t = np.linalg.svd(
        invx @ (xz.T @ yz / nobs) @ invy
    )
    k = min(cfg.rank, STOCKS)
    mode_weight = np.ones(k)
    if cfg.head_rank is not None:
        mode_weight[min(cfg.head_rank, k):] = cfg.tail_weight
    canonical = (
        invx @ left[:, :k] @ np.diag(singular[:k] * mode_weight)
        @ right_t[:k] @ rooty
    )
    return canonical, dense, xmean, xsd, ymean, ysd


class CanonicalModel:
    """Existing coherent CCA core plus optional honest target calibration."""

    def __init__(self, cfg: CCAConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.last_nt = self.last_fit = -1
        self.canonical = self.dense = None
        self.xmean = self.xsd = self.ymean = self.ysd = None
        self.algo_sd = None
        self.reliability = np.ones(STOCKS)

    def _crossfit_reliability(self, stocks: np.ndarray) -> np.ndarray:
        """Per-target edge from a strictly trailing train/validation split."""
        cfg = self.cfg
        lag = self.cfg.lag
        n_pairs = stocks.shape[1] - lag
        valid = min(cfg.reliability_window, max(0, n_pairs // 3))
        train_end = n_pairs - valid
        if valid < 80 or train_end < 120:
            return np.ones(STOCKS)
        x = stocks[:, :-lag].T
        y = stocks[:, lag:].T
        maps = fit_cca_maps(x[:train_end], y[:train_end], cfg)
        canonical, dense, xmean, xsd, _, ysd = maps
        xv = (x[train_end:] - xmean) / xsd
        pred = xv @ canonical
        dense_pred = xv @ dense
        # Normalize each view by day exactly as the portfolio layer does.
        pred -= pred.mean(1, keepdims=True)
        pred /= pred.std(1, keepdims=True) + EPS
        dense_pred -= dense_pred.mean(1, keepdims=True)
        dense_pred /= dense_pred.std(1, keepdims=True) + EPS
        signal = pred + cfg.dense_weight * dense_pred
        signal -= signal.mean(1, keepdims=True)
        signal /= signal.std(1, keepdims=True) + EPS
        position = np.tanh(signal / cfg.temperature)
        realised = y[train_end:] / (ysd + EPS)
        contribution = position * realised
        edge = contribution.mean(0)
        noise = contribution.std(0) / np.sqrt(len(contribution)) + EPS
        tstat = edge / noise
        # Smoothly suppress only targets with negative validated evidence;
        # positive estimates are not levered beyond their hard limits.
        floor = cfg.reliability_floor
        return floor + (1.0 - floor) / (1.0 + np.exp(-tstat))

    def _fit(self, prices: np.ndarray):
        returns = prices[:, 1:] / prices[:, :-1] - 1.0
        stocks = returns[1:]
        lag = self.cfg.lag
        x, y = stocks[:, :-lag].T, stocks[:, lag:].T
        if self.cfg.train_window is not None:
            keep = min(self.cfg.train_window, len(x))
            x, y = x[-keep:], y[-keep:]
        maps = fit_cca_maps(x, y, self.cfg)
        (self.canonical, self.dense, self.xmean, self.xsd,
         self.ymean, self.ysd) = maps
        self.algo_sd = returns[0, lag:].std() + EPS
        if self.cfg.reliability_strength:
            raw = self._crossfit_reliability(stocks)
            self.reliability = (
                1.0 + self.cfg.reliability_strength * (raw - 1.0)
            )
        else:
            self.reliability = np.ones(STOCKS)

    def _view(self, stock_z: np.ndarray, prices: np.ndarray,
              include_drift: bool) -> np.ndarray:
        stock_z = stock_z * self.reliability
        prediction = stock_z * self.ysd
        if include_drift:
            prediction += self.ymean
        return coherent_stock_signal(
            stock_z, prediction, prices, self.algo_sd
        )

    def target(self, prices: np.ndarray) -> np.ndarray:
        nt = prices.shape[1]
        if nt <= self.last_nt:
            self.reset()
        self.last_nt = nt
        if nt < 120:
            return np.zeros(N)
        if self.canonical is None or nt - self.last_fit >= self.cfg.retrain:
            self._fit(prices)
            self.last_fit = nt
        returns = prices[1:, 1:] / prices[1:, :-1] - 1.0
        now = (returns[:, -self.cfg.lag] - self.xmean) / self.xsd
        cca = self._view(now @ self.canonical, prices, True)
        dense = self._view(now @ self.dense, prices, False)
        signal = normalize(cca + self.cfg.dense_weight * dense)
        return LIMITS * np.tanh(signal / self.cfg.temperature)


def make_cca(cfg: CCAConfig):
    model = CanonicalModel(cfg)
    return model.target


FROZEN_PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836),
    (13, 45, 1.0132), (33, 40, 0.2577), (10, 46, 1.0331),
    (33, 42, 0.8358), (31, 43, 0.9692), (18, 28, 0.5642),
    (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)


class FrozenPairOverlay:
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.state = [0] * len(FROZEN_PAIRS)
        self.last_nt = -1

    def target(self, prices: np.ndarray) -> np.ndarray:
        nt = prices.shape[1]
        if nt <= self.last_nt:
            self.state = [0] * len(FROZEN_PAIRS)
        self.last_nt = nt
        dollars = np.zeros(N)
        if nt <= 61 or self.scale <= 0.0:
            return dollars
        logp = np.log(np.maximum(prices, EPS))
        for k, (i, j, gamma) in enumerate(FROZEN_PAIRS):
            spread = logp[i] - gamma * logp[j]
            window = spread[-61:-1]
            z = (spread[-1] - window.mean()) / (window.std() + EPS)
            state = self.state[k]
            if state == 0:
                state = -1 if z > 1.5 else (1 if z < -1.5 else 0)
            elif state == 1 and z > -0.5:
                state = 0
            elif state == -1 and z < 0.5:
                state = 0
            self.state[k] = state
            dollars[i] += self.scale * state * 9_000.0
            dollars[j] -= self.scale * state * gamma * 9_000.0
        return dollars


def make_cca_pairs(cfg: CCAConfig, pair_scale: float = 1.0):
    core = CanonicalModel(cfg)
    pairs = FrozenPairOverlay(pair_scale)

    def target(prices: np.ndarray):
        return np.clip(
            core.target(prices) + pairs.target(prices), -LIMITS, LIMITS
        )
    return target


@dataclass(frozen=True)
class TrendConfig:
    fast: int = 10
    slow: int = 40
    direction: float = 1.0
    temperature: float = 0.75
    regime: str = "all"


class SlowTrendModel:
    """Small multi-horizon trend/reversion probe with observable regimes."""

    def __init__(self, cfg: TrendConfig):
        self.cfg = cfg

    def target(self, prices: np.ndarray) -> np.ndarray:
        c = self.cfg
        if prices.shape[1] <= c.slow + 2:
            return np.zeros(N)
        logp = np.log(np.maximum(prices, EPS))
        returns = np.diff(logp[:, -c.slow-1:], axis=1)
        vol = returns.std(1) + EPS
        fast = (logp[:, -1] - logp[:, -1-c.fast]) / (
            vol * np.sqrt(c.fast)
        )
        slow = (logp[:, -1] - logp[:, -1-c.slow]) / (
            vol * np.sqrt(c.slow)
        )
        signal = c.direction * (fast + slow) * .5
        # ALGO is mechanically derived from stocks, so avoid making the
        # $100k leg an accidental ten-times directional trend bet.
        signal[0] = signal[1:].mean()
        if c.regime != "all":
            algo_trend = slow[0]
            breadth = np.mean(slow[1:] > 0.0) - .5
            active = {
                "up": algo_trend > 0.0,
                "down": algo_trend < 0.0,
                "aligned": algo_trend * breadth > 0.0,
                "disagree": algo_trend * breadth < 0.0,
                "strong": abs(algo_trend) > 1.0,
            }[c.regime]
            if not active:
                return np.zeros(N)
        signal = normalize(signal)
        return LIMITS * np.tanh(signal / c.temperature)


def make_trend(cfg: TrendConfig):
    return SlowTrendModel(cfg).target


def make_cca_trend(
    cca_cfg: CCAConfig,
    trend_cfg: TrendConfig,
    trend_weight: float,
    pair_scale: float = 0.0,
):
    core = CanonicalModel(cca_cfg)
    trend = SlowTrendModel(trend_cfg)
    pairs = FrozenPairOverlay(pair_scale)

    def target(prices: np.ndarray):
        return np.clip(
            core.target(prices)
            + trend_weight * trend.target(prices)
            + pairs.target(prices),
            -LIMITS, LIMITS,
        )
    return target


def make_cca_risk(
    cfg: CCAConfig,
    hedge_weight: float = 0.0,
    direct_scale: float = 1.0,
    pair_scale: float = 0.0,
    beta_window: int | None = None,
):
    """Canonical core with ALGO blended toward a stock-book beta hedge."""
    core = CanonicalModel(cfg)
    pairs = FrozenPairOverlay(pair_scale)

    def target(prices: np.ndarray):
        dollars = core.target(prices)
        direct = direct_scale * dollars[0]
        returns = np.diff(
            np.log(np.maximum(prices, EPS)), axis=1
        ).T
        if beta_window is not None:
            returns = returns[-beta_window:]
        covariance = np.cov(returns, rowvar=False)
        beta = covariance[:, 0] / (covariance[0, 0] + EPS)
        hedge = -np.sum(dollars[1:] * beta[1:]) / (beta[0] + EPS)
        dollars[0] = (
            (1.0 - hedge_weight) * direct
            + hedge_weight * np.clip(hedge, -LIMITS[0], LIMITS[0])
        )
        return np.clip(dollars + pairs.target(prices), -LIMITS, LIMITS)
    return target


def make_multi_cca(
    primary: CCAConfig,
    secondary: CCAConfig,
    secondary_weight: float,
    pair_scale: float = 0.0,
):
    first = CanonicalModel(primary)
    second = CanonicalModel(secondary)
    pairs = FrozenPairOverlay(pair_scale)

    def target(prices: np.ndarray):
        return np.clip(
            first.target(prices)
            + secondary_weight * second.target(prices)
            + pairs.target(prices),
            -LIMITS, LIMITS,
        )
    return target


@dataclass(frozen=True)
class AnalogConfig:
    components: int = 10
    neighbors: int = 50
    temperature: float = 0.5
    weighted: bool = True
    retrain: int = 25


class AnalogModel:
    """Nearest historical return regimes in a low-dimensional PCA space."""

    def __init__(self, cfg: AnalogConfig):
        self.cfg = cfg
        self.last_nt = self.last_fit = -1
        self.xmean = self.xsd = self.ymean = self.ysd = None
        self.basis = self.xfactor = self.yz = None
        self.algo_sd = None

    def _fit(self, prices: np.ndarray):
        returns = prices[:, 1:] / prices[:, :-1] - 1.0
        stocks = returns[1:]
        x, y = stocks[:, :-1].T, stocks[:, 1:].T
        xz, self.xmean, self.xsd = standardize(x)
        self.yz, self.ymean, self.ysd = standardize(y)
        _, _, right = np.linalg.svd(xz, full_matrices=False)
        self.basis = right[:self.cfg.components].T
        self.xfactor = xz @ self.basis
        self.xfactor /= (
            np.linalg.norm(self.xfactor, axis=1, keepdims=True) + EPS
        )
        self.algo_sd = returns[0, 1:].std() + EPS

    def target(self, prices: np.ndarray) -> np.ndarray:
        nt = prices.shape[1]
        if nt <= self.last_nt:
            self.__init__(self.cfg)
        self.last_nt = nt
        if nt < 120:
            return np.zeros(N)
        if self.basis is None or nt - self.last_fit >= self.cfg.retrain:
            self._fit(prices)
            self.last_fit = nt
        now_return = prices[1:, -1] / prices[1:, -2] - 1.0
        now = ((now_return - self.xmean) / self.xsd) @ self.basis
        now /= np.linalg.norm(now) + EPS
        similarity = self.xfactor @ now
        k = min(self.cfg.neighbors, len(similarity))
        indices = np.argpartition(similarity, -k)[-k:]
        if self.cfg.weighted:
            weight = np.maximum(similarity[indices], 0.0) ** 2
            weight /= weight.sum() + EPS
        else:
            weight = np.full(k, 1.0 / k)
        stock_z = weight @ self.yz[indices]
        prediction = stock_z * self.ysd + self.ymean
        signal = coherent_stock_signal(
            stock_z, prediction, prices, self.algo_sd
        )
        return LIMITS * np.tanh(signal / self.cfg.temperature)


def make_analog(cfg: AnalogConfig):
    return AnalogModel(cfg).target


def make_cca_analog(
    cca_cfg: CCAConfig,
    analog_cfg: AnalogConfig,
    analog_weight: float,
    pair_scale: float = 0.0,
):
    core = CanonicalModel(cca_cfg)
    analog = AnalogModel(analog_cfg)
    pairs = FrozenPairOverlay(pair_scale)

    def target(prices: np.ndarray):
        return np.clip(
            core.target(prices)
            + analog_weight * analog.target(prices)
            + pairs.target(prices),
            -LIMITS, LIMITS,
        )
    return target


@dataclass(frozen=True)
class RegimeCCAConfig:
    base: CCAConfig = CCAConfig(
        rank=7, head_rank=3, tail_weight=.5
    )
    regime: str = "trend20"
    weight: float = 0.5
    min_state_obs: int = 150
    lookback: int = 20


class RegimeCanonicalModel:
    """Two softly blended canonical maps keyed by an observable state."""

    def __init__(self, cfg: RegimeCCAConfig):
        self.cfg = cfg
        self.last_nt = self.last_fit = -1
        self.global_params = None
        self.state_params = None
        self.algo_sd = None
        self.dispersion_cut = None
        self.shock_cut = None

    def _labels(self, returns: np.ndarray, x: np.ndarray) -> np.ndarray:
        mode = self.cfg.regime
        if mode == "market_sign":
            return (returns[0, :-1] >= 0.0).astype(int)
        if mode == "breadth":
            return (np.mean(x > 0.0, axis=1) >= .5).astype(int)
        if mode == "dispersion":
            value = x.std(1)
            self.dispersion_cut = np.median(value)
            return (value >= self.dispersion_cut).astype(int)
        if mode == "shock":
            z, _, _ = standardize(x)
            value = np.sqrt(np.mean(z * z, axis=1))
            self.shock_cut = np.median(value)
            return (value >= self.shock_cut).astype(int)
        lookback = self.cfg.lookback
        if mode in ("trend20", "trend_breadth20", "alignment20"):
            algo = returns[0]
            stock = returns[1:]
            algo_trend = np.array([
                algo[max(0, t-lookback+1):t+1].sum()
                for t in range(len(x))
            ])
            breadth = np.array([
                np.mean(
                    stock[:, max(0, t-lookback+1):t+1].sum(1) > 0.0
                ) - .5
                for t in range(len(x))
            ])
            if mode == "trend20":
                return (algo_trend >= 0.0).astype(int)
            if mode == "trend_breadth20":
                return (breadth >= 0.0).astype(int)
            return (algo_trend * breadth >= 0.0).astype(int)
        return np.zeros(len(x), dtype=int)

    def _current_label(self, returns: np.ndarray) -> int:
        mode = self.cfg.regime
        stock = returns[1:, -1]
        if mode == "market_sign":
            return int(returns[0, -1] >= 0.0)
        if mode == "breadth":
            return int(np.mean(stock > 0.0) >= .5)
        if mode == "dispersion":
            return int(stock.std() >= self.dispersion_cut)
        if mode == "shock":
            # Approximate the fit-time standardized norm; the cut is close to
            # one, so target-vol standardization is enough.
            z = stock / (returns[1:].std(1) + EPS)
            return int(np.sqrt(np.mean(z * z)) >= self.shock_cut)
        lookback = self.cfg.lookback
        algo_trend = returns[0, -lookback:].sum()
        breadth = (
            np.mean(returns[1:, -lookback:].sum(1) > 0.0) - .5
        )
        if mode == "trend20":
            return int(algo_trend >= 0.0)
        if mode == "trend_breadth20":
            return int(breadth >= 0.0)
        if mode == "alignment20":
            return int(algo_trend * breadth >= 0.0)
        return 0

    def _fit(self, prices: np.ndarray):
        returns = prices[:, 1:] / prices[:, :-1] - 1.0
        stock = returns[1:]
        x, y = stock[:, :-1].T, stock[:, 1:].T
        self.global_params = fit_cca_maps(x, y, self.cfg.base)
        labels = self._labels(returns, x)
        states = []
        for state in (0, 1):
            take = labels == state
            if take.sum() < self.cfg.min_state_obs:
                states.append(self.global_params)
            else:
                states.append(fit_cca_maps(x[take], y[take], self.cfg.base))
        self.state_params = states
        self.algo_sd = returns[0, 1:].std() + EPS

    def _signal(
        self, params, now_return: np.ndarray, prices: np.ndarray
    ) -> np.ndarray:
        canonical, dense, xmean, xsd, ymean, ysd = params
        now = (now_return - xmean) / xsd
        cca_z = now @ canonical
        dense_z = now @ dense
        cca = coherent_stock_signal(
            cca_z, cca_z * ysd + ymean, prices, self.algo_sd
        )
        dense_signal = coherent_stock_signal(
            dense_z, dense_z * ysd, prices, self.algo_sd
        )
        return normalize(cca + self.cfg.base.dense_weight * dense_signal)

    def target(self, prices: np.ndarray) -> np.ndarray:
        nt = prices.shape[1]
        if nt <= self.last_nt:
            self.__init__(self.cfg)
        self.last_nt = nt
        if nt < 180:
            return np.zeros(N)
        if (self.global_params is None
                or nt - self.last_fit >= self.cfg.base.retrain):
            self._fit(prices)
            self.last_fit = nt
        returns = prices[:, 1:] / prices[:, :-1] - 1.0
        now = returns[1:, -1]
        global_signal = self._signal(
            self.global_params, now, prices
        )
        label = self._current_label(returns)
        state_signal = self._signal(
            self.state_params[label], now, prices
        )
        signal = normalize(
            global_signal + self.cfg.weight * state_signal
        )
        return LIMITS * np.tanh(
            signal / self.cfg.base.temperature
        )


def make_regime_cca(cfg: RegimeCCAConfig):
    return RegimeCanonicalModel(cfg).target


def make_regime_pairs(
    cfg: RegimeCCAConfig,
    pair_scale: float = 1.0,
    algo_scale: float = 1.0,
):
    core = RegimeCanonicalModel(cfg)
    pairs = FrozenPairOverlay(pair_scale)

    def target(prices: np.ndarray):
        core_dollars = core.target(prices)
        core_dollars[0] *= algo_scale
        return np.clip(
            core_dollars + pairs.target(prices), -LIMITS, LIMITS
        )
    return target


# Pre-OOS selection: this configuration was selected on days 250--500 and
# 500--750, then evaluated once on 750--1000.  The regime model does not
# activate until both states have 300 examples, avoiding small-state fits.
FINAL_CCA = CCAConfig(
    rank=7,
    head_rank=3,
    tail_weight=.35,
    shrink=.10,
    dense_weight=.25,
    temperature=.25,
)
FINAL_REGIME = RegimeCCAConfig(
    base=FINAL_CCA,
    regime="trend20",
    weight=.40,
    min_state_obs=300,
    lookback=20,
)


def make_final(pair_scale: float = 1.0):
    return make_regime_pairs(FINAL_REGIME, pair_scale)


# High-history deployment variant discovered after the rank-3 audit.  The
# shock split asks whether today's standardized 50-stock return vector has
# above- or below-median RMS magnitude.  It is a more stable conditioning
# variable than direct price momentum.
ULTRA_CCA = CCAConfig(
    rank=3,
    shrink=.50,
    dense_weight=.25,
    retrain=100,
    temperature=.20,
)
ULTRA_REGIME = RegimeCCAConfig(
    base=ULTRA_CCA,
    regime="shock",
    weight=.75,
    min_state_obs=300,
)


def make_ultra(pair_scale: float = 0.0, algo_scale: float = 1.5):
    return make_regime_pairs(
        ULTRA_REGIME, pair_scale=pair_scale, algo_scale=algo_scale
    )


# Conservative deployment version: fit once at the start of a 250-day
# evaluation and use only a 10% conditional view.  On 750--1000 this uses no
# test-window refit, making it the cleanest evidence for future deployment.
ROBUST_CCA = CCAConfig(
    rank=3,
    shrink=.50,
    dense_weight=.25,
    retrain=250,
    temperature=.20,
)
ROBUST_REGIME = RegimeCCAConfig(
    base=ROBUST_CCA,
    regime="shock",
    weight=.10,
    min_state_obs=300,
)


def make_ultra_robust(
    pair_scale: float = 0.0, algo_scale: float = 1.5
):
    return make_regime_pairs(
        ROBUST_REGIME, pair_scale=pair_scale, algo_scale=algo_scale
    )


def describe_returns(prices: np.ndarray):
    returns = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
    windows = ((0, 250), (250, 500), (500, 750), (750, 999))
    print("Return diagnostics")
    maps = []
    for a, b in windows:
        r = returns[a:b]
        xz, _, _ = standardize(r[:-1])
        yz, _, _ = standardize(r[1:])
        cross = xz.T @ yz / len(xz)
        maps.append(cross)
        own = np.diag(cross)
        singular = np.linalg.svd(cross, compute_uv=False)
        print(
            f"{a:3d}-{b:3d}: drift={r.mean(): .6f} "
            f"vol={r.std():.5f} own-lag={own.mean(): .4f} "
            f"lag-sv1={singular[0]:.3f} lag-sv7={singular[6]:.3f}"
        )
    print("lag-map correlations")
    for i in range(len(maps)):
        print(" ".join(
            f"{np.corrcoef(maps[i].ravel(), maps[j].ravel())[0,1]: .3f}"
            for j in range(len(maps))
        ))

    print("Univariate forward correlations by segment")
    for horizon in (1, 2, 5, 10, 20, 40, 60, 120):
        values = []
        for a, b in windows:
            r = returns[a:b, 1:]
            if len(r) <= horizon:
                values.append(np.nan)
                continue
            past = np.array([
                r[max(0, t-horizon):t].sum(0)
                for t in range(horizon, len(r))
            ])
            future = r[horizon:]
            values.append(np.corrcoef(past.ravel(), future.ravel())[0, 1])
        print(f"h={horizon:3d}: " + " ".join(f"{x: .4f}" for x in values))


def run_cfg(prices: np.ndarray, label: str, cfg: Config):
    rows = []
    for name, start, end in (
        ("old", 250, 500),
        ("validation", 500, 750),
        ("new_oos", 750, 1000),
    ):
        result = simulate(prices, start, end, make_model(cfg))
        rows.append((name, result))
    print(
        f"{label:43s} "
        + " ".join(
            f"{name}={r[2]:7.1f}({r[0]:.0f}/{r[1]:.0f})"
            for name, r in rows
        )
    )
    return rows


def run_cca_cfg(prices: np.ndarray, label: str, cfg: CCAConfig):
    rows = []
    for name, start, end in (
        ("old", 250, 500),
        ("validation", 500, 750),
        ("new_oos", 750, 1000),
    ):
        result = simulate(prices, start, end, make_cca(cfg))
        rows.append((name, result))
    print(
        f"{label:43s} "
        + " ".join(
            f"{name}={r[2]:7.1f}({r[0]:.0f}/{r[1]:.0f})"
            for name, r in rows
        )
    )
    return rows


def research(prices: np.ndarray):
    configs: list[tuple[str, Config]] = [
        ("dense expanding", Config()),
        ("rolling 250", Config(rolling=250)),
        ("rolling 400", Config(rolling=400)),
        ("global + recent250 .25",
         Config(recent_weight=.25, recent_window=250)),
        ("global + recent250 .50",
         Config(recent_weight=.50, recent_window=250)),
        ("global + recent400 .25",
         Config(recent_weight=.25, recent_window=400)),
    ]
    for regime in ("market_sign", "breadth", "dispersion", "market_size"):
        for strength in (.25, .5):
            configs.append((
                f"regime {regime} w={strength}",
                Config(regime=regime, regime_strength=strength),
            ))
    for power in (.25, .5, 1.0):
        configs.append((
            f"target reliability p={power}",
            Config(reliability_power=power),
        ))
    for mode in ("plain", "residual", "breakout", "regime"):
        for weight in (.10, .25, .50):
            configs.append((
                f"momentum {mode} w={weight}",
                Config(momentum_weight=weight, momentum_mode=mode),
            ))
    for label, cfg in configs:
        run_cfg(prices, label, cfg)


def cca_research(prices: np.ndarray):
    print("CCA baseline and robustness grid")
    for rank in (3, 5, 7, 10, 15, 20, 30, 50):
        run_cca_cfg(prices, f"CCA rank={rank}", CCAConfig(rank=rank))
    for shrink in (0.0, .1, .25, .5, .75):
        run_cca_cfg(
            prices, f"CCA shrink={shrink}",
            CCAConfig(shrink=shrink),
        )
    for weight in (0.0, .25, .5, .75, 1.0):
        run_cca_cfg(
            prices, f"CCA dense weight={weight}",
            CCAConfig(dense_weight=weight),
        )
    for window in (250, 400, 600):
        run_cca_cfg(
            prices, f"CCA rolling window={window}",
            CCAConfig(train_window=window),
        )
    for strength in (.25, .5, .75, 1.0):
        run_cca_cfg(
            prices, f"CCA calibrated strength={strength}",
            CCAConfig(reliability_strength=strength),
        )
    print("Spectrally tapered canonical maps")
    for total in (7, 10, 15):
        for tail in (.15, .25, .5, .75):
            run_cca_cfg(
                prices, f"CCA head3 total={total} tail={tail}",
                CCAConfig(rank=total, head_rank=3, tail_weight=tail),
            )


def headline(prices: np.ndarray):
    baseline = CCAConfig()
    tapered = FINAL_CCA
    rows = (
        ("existing coherent CCA", lambda: make_cca(baseline)),
        ("spectrally tapered CCA", lambda: make_cca(tapered)),
        ("taper + 20d trend regimes", lambda: make_regime_cca(FINAL_REGIME)),
        ("final + frozen pair overlay", lambda: make_final(1.0)),
        ("rank3 + shock regime, pair-free", lambda: make_ultra(0.0, 1.5)),
        ("ultra + pair diversifier", lambda: make_ultra(1.5, 1.5)),
        ("frozen robust ultra, pair-free",
         lambda: make_ultra_robust(0.0, 1.5)),
        ("frozen robust + pair diversifier",
         lambda: make_ultra_robust(1.5, 1.5)),
    )
    for label, factory in rows:
        values = []
        for start, end in ((250, 500), (500, 750), (750, 1000)):
            values.append(simulate(prices, start, end, factory())[:3])
        print(
            f"{label:32s} "
            + " ".join(
                f"{a}-{b}: {r[2]:7.1f} ({r[0]:.1f}/{r[1]:.1f})"
                for (a, b), r in zip(
                    ((250, 500), (500, 750), (750, 1000)), values
                )
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--cca", action="store_true")
    parser.add_argument("--git-head", action="store_true")
    parser.add_argument("--headline", action="store_true")
    args = parser.parse_args()
    prices = load_prices(committed=args.git_head)
    if args.diagnose or not (args.research or args.cca or args.headline):
        describe_returns(prices)
    if args.research:
        research(prices)
    if args.cca:
        cca_research(prices)
    if args.headline:
        headline(prices)


if __name__ == "__main__":
    main()
