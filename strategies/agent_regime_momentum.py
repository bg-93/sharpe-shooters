#!/usr/bin/env python3
"""Honest research harness for simple regime-momentum strategies.

The research protocol deliberately treats days [750, 1000) as an untouched
audit.  Develop with:

    .venv/bin/python strategies/agent_regime_momentum.py --phase develop

Only after freezing a short list, inspect the new window with:

    .venv/bin/python strategies/agent_regime_momentum.py --phase audit

All decisions use only prices visible on that day.  The simulator matches the
official evaluator's integer shares, limits, one-day-delayed commissions, and
score.  The module-level ``getMyPosition`` at the bottom is kept deliberately
small; its constants are filled from the robust development plateau rather
than from a single audit-window optimum.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
N = 51
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50)
COMM = np.array([0.00002] + [0.0001] * 50)
DEVELOPMENT_WINDOWS = (
    ("early", 100, 300),
    ("middle", 250, 500),
    ("late-dev", 500, 750),
)
AUDIT_WINDOWS = (
    ("audit", 750, 1000),
    ("audit-a", 750, 875),
    ("audit-b", 875, 1000),
)


def load_prices() -> np.ndarray:
    return pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T.astype(float)


def score_pl(pnl: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    sharpe = np.sqrt(250.0) * mean / std if std > EPS else 0.0
    if mean <= 0.0 or std < 1e-10:
        score = mean
    else:
        score = mean * sharpe * sharpe / (sharpe * sharpe + 1.0)
    return mean, std, sharpe, score


def simulate(
    prices: np.ndarray,
    start: int,
    end: int,
    target_fn: Callable[[np.ndarray], np.ndarray],
    collect: bool = False,
):
    """Replicate eval.py while accepting target dollars, not shares."""
    cash = 0.0
    value = 0.0
    current = np.zeros(N)
    pending_commission = 0.0
    pnl = []
    for t in range(start, end + 1):
        history = prices[:, :t]
        px = history[:, -1]
        if t < end:
            dollars = np.asarray(target_fn(history), dtype=float)
            pos_limit = (LIMITS / px).astype(int)
            new = np.clip(
                (dollars / px).astype(int), -pos_limit, pos_limit
            )
        else:
            new = current.copy()
        delta = new - current
        cash -= px @ delta + pending_commission
        pending_commission = float(np.sum(px * np.abs(delta) * COMM))
        current = new
        today = cash + current @ px - value
        value = cash + current @ px
        if t > start:
            pnl.append(today)
    pnl = np.asarray(pnl)
    result = score_pl(pnl)
    return (*result, pnl) if collect else result


def evaluate(
    label: str,
    factory: Callable[[], Callable[[np.ndarray], np.ndarray]],
    prices: np.ndarray,
    windows=DEVELOPMENT_WINDOWS,
    verbose: bool = True,
):
    rows = {}
    for name, start, end in windows:
        rows[name] = simulate(prices, start, end, factory())
    if verbose:
        text = " | ".join(
            f"{name} {m:7.1f}/{sd:7.1f}/{sc:5.2f}/{score:7.1f}"
            for name, (m, sd, sc, score) in rows.items()
        )
        minimum = min(row[3] for row in rows.values())
        print(f"{label:54s} | {text} | min {minimum:7.1f}")
    return rows


def _zscore_cross_section(signal: np.ndarray) -> np.ndarray:
    out = np.asarray(signal, float).copy()
    stock = out[1:]
    stock -= stock.mean()
    stock /= stock.std() + EPS
    out[0] = 0.0
    return out


def _rank_cross_section(signal: np.ndarray) -> np.ndarray:
    out = np.zeros(N)
    order = np.argsort(np.argsort(signal[1:])).astype(float)
    out[1:] = (order - order.mean()) / (order.max() / 2.0 + EPS)
    return out


def _trend_features(
    prices: np.ndarray,
    horizons: tuple[int, ...],
    vol_window: int,
):
    logp = np.log(np.maximum(prices, EPS))
    returns = np.diff(logp, axis=1)
    vol = returns[:, -vol_window:].std(axis=1) + EPS
    pieces = []
    for horizon in horizons:
        move = logp[:, -1] - logp[:, -horizon - 1]
        pieces.append(move / (vol * np.sqrt(horizon)))
    trend = np.mean(pieces, axis=0)
    longest = max(horizons)
    path = np.abs(returns[:, -longest:]).sum(axis=1) + EPS
    efficiency = np.abs(logp[:, -1] - logp[:, -longest - 1]) / path
    short_vol = returns[:, -min(20, vol_window):].std(axis=1) + EPS
    vol_ratio = short_vol / vol
    return trend, efficiency, vol_ratio


@dataclass(frozen=True)
class MomentumConfig:
    """A compact family spanning trend, cross-sectional momentum and gates."""

    horizons: tuple[int, ...] = (20,)
    vol_window: int = 100
    construction: str = "cross"
    temperature: float = 0.8
    gross: float = 1.0
    trend_threshold: float = 0.0
    efficiency_threshold: float = 0.0
    require_alignment: bool = False
    vol_gate: str = "none"
    algo: str = "zero"
    topk: int = 10


class RegimeMomentum:
    """Simple price momentum, activated only in explicitly trending regimes."""

    def __init__(self, config: MomentumConfig):
        self.c = config

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        need = max(max(self.c.horizons), self.c.vol_window) + 2
        if prices.shape[1] < need:
            return np.zeros(N)
        trend, efficiency, vol_ratio = _trend_features(
            prices, self.c.horizons, self.c.vol_window
        )

        if self.c.construction == "time":
            signal = trend.copy()
            signal[0] = 0.0
            signal[1:] /= np.std(signal[1:]) + EPS
        elif self.c.construction == "cross":
            signal = _zscore_cross_section(trend)
        elif self.c.construction == "rank":
            signal = _rank_cross_section(trend)
        elif self.c.construction == "topk":
            z = _zscore_cross_section(trend)
            signal = np.zeros(N)
            chosen = np.argpartition(np.abs(z[1:]), -self.c.topk)[
                -self.c.topk:
            ] + 1
            signal[chosen] = z[chosen]
        elif self.c.construction == "breakout":
            horizon = max(self.c.horizons)
            logp = np.log(np.maximum(prices, EPS))
            window = logp[:, -horizon - 1:-1]
            low, high = window.min(1), window.max(1)
            location = 2.0 * (logp[:, -1] - low) / (high - low + EPS) - 1.0
            signal = _zscore_cross_section(location)
        else:
            raise ValueError(self.c.construction)

        gate = np.ones(N)
        if self.c.trend_threshold > 0.0:
            gate *= np.abs(trend) >= self.c.trend_threshold
        if self.c.efficiency_threshold > 0.0:
            threshold = self.c.efficiency_threshold
            gate *= np.clip(
                (efficiency - threshold) / (1.0 - threshold + EPS),
                0.0,
                1.0,
            )
        if self.c.require_alignment and len(self.c.horizons) > 1:
            logp = np.log(np.maximum(prices, EPS))
            moves = np.stack([
                logp[:, -1] - logp[:, -h - 1] for h in self.c.horizons
            ])
            gate *= np.all(np.sign(moves) == np.sign(moves[0]), axis=0)
        if self.c.vol_gate == "calm":
            gate *= vol_ratio <= 1.0
        elif self.c.vol_gate == "hot":
            gate *= vol_ratio > 1.0

        signal *= gate
        if self.c.algo == "direct":
            signal[0] = trend[0] / (np.std(trend[1:]) + EPS)
        elif self.c.algo == "zero":
            signal[0] = 0.0
        else:
            raise ValueError(self.c.algo)
        return LIMITS * self.c.gross * np.tanh(
            signal / self.c.temperature
        )


@dataclass(frozen=True)
class AdaptiveConfig:
    horizon: int = 20
    vol_window: int = 100
    edge_window: int = 100
    edge_threshold: float = 0.0
    mode: str = "global"
    temperature: float = 0.8
    gross: float = 1.0


class AdaptiveMomentum:
    """Trade momentum only after its walk-forward realised edge turns positive.

    The gate is computed from past signal returns whose next-day outcomes are
    already observable.  No strategy PnL or future price is smuggled into the
    decision.
    """

    def __init__(self, config: AdaptiveConfig):
        self.c = config

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        h, ew = self.c.horizon, self.c.edge_window
        logp = np.log(np.maximum(prices, EPS))
        r = np.diff(logp, axis=1)
        if r.shape[1] < h + ew + 5:
            return np.zeros(N)

        end = r.shape[1] - 1
        starts = np.arange(max(h - 1, end - ew), end)
        past_signal = np.stack([
            (logp[:, e + 1] - logp[:, e + 1 - h])
            / ((r[:, max(0, e - self.c.vol_window + 1):e + 1].std(1)
                + EPS) * np.sqrt(h))
            for e in starts
        ])
        past_signal[:, 1:] -= past_signal[:, 1:].mean(1, keepdims=True)
        payoff = past_signal[:, 1:] * r[1:, starts + 1].T

        current, _, _ = _trend_features(
            prices, (h,), self.c.vol_window
        )
        signal = _zscore_cross_section(current)
        if self.c.mode == "global":
            daily = payoff.mean(1)
            edge_z = daily.mean() / (
                daily.std() / np.sqrt(len(daily)) + EPS
            )
            gate = float(edge_z > self.c.edge_threshold)
            signal *= gate
        elif self.c.mode == "asset":
            edge_z = payoff.mean(0) / (
                payoff.std(0) / np.sqrt(len(payoff)) + EPS
            )
            gate = np.clip(
                (edge_z - self.c.edge_threshold) / 2.0, 0.0, 1.0
            )
            signal[1:] *= gate
        elif self.c.mode == "signed-global":
            daily = payoff.mean(1)
            edge_z = daily.mean() / (
                daily.std() / np.sqrt(len(daily)) + EPS
            )
            signal *= np.sign(edge_z) * (abs(edge_z) > self.c.edge_threshold)
        else:
            raise ValueError(self.c.mode)
        return LIMITS * self.c.gross * np.tanh(
            signal / self.c.temperature
        )


@dataclass(frozen=True)
class SkillGateConfig:
    horizons: tuple[int, ...] = (5, 10, 20, 40, 60)
    vol_window: int = 100
    edge_window: int = 75
    edge_threshold: float = 1.0
    temperature: float = 0.8
    gross: float = 1.0


class SkillGatedMomentum:
    """Let only horizons with recently demonstrated continuation vote.

    Each horizon is a tiny expert.  Its walk-forward t-stat is measured from
    prior cross-sectional continuation payoffs.  Positive excess t-stat above
    ``edge_threshold`` becomes the expert weight; when no horizon has earned a
    positive weight the strategy is flat.  This is a direct regime detector:
    the regime is defined by demonstrated trend persistence, not a visual
    moving-average label.
    """

    def __init__(self, config: SkillGateConfig):
        self.c = config

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        logp = np.log(np.maximum(prices, EPS))
        r = np.diff(logp, axis=1)
        longest = max(self.c.horizons)
        if r.shape[1] < longest + self.c.edge_window + 5:
            return np.zeros(N)
        end = r.shape[1] - 1
        starts = np.arange(
            max(longest - 1, end - self.c.edge_window), end
        )
        combined = np.zeros(50)
        total_weight = 0.0
        stock_r = r[1:]
        current_vol = stock_r[:, -self.c.vol_window:].std(axis=1) + EPS
        csum = np.pad(
            np.cumsum(stock_r, axis=1), ((0, 0), (1, 0))
        )
        csum2 = np.pad(
            np.cumsum(stock_r * stock_r, axis=1), ((0, 0), (1, 0))
        )
        vol_start = np.maximum(0, starts + 1 - self.c.vol_window)
        count = starts + 1 - vol_start
        vol_sum = csum[:, starts + 1] - csum[:, vol_start]
        vol_sum2 = csum2[:, starts + 1] - csum2[:, vol_start]
        past_vol = np.sqrt(np.maximum(
            vol_sum2 / count - (vol_sum / count) ** 2, 0.0
        )) + EPS
        for horizon in self.c.horizons:
            current_move = (
                logp[1:, -1] - logp[1:, -horizon - 1]
            ) / (current_vol * np.sqrt(horizon))
            current_move -= current_move.mean()
            current_move /= current_move.std() + EPS

            move = (
                csum[:, starts + 1]
                - csum[:, starts + 1 - horizon]
            )
            past_signal = move / (past_vol * np.sqrt(horizon))
            past_signal -= past_signal.mean(axis=0, keepdims=True)
            past_signal /= past_signal.std(axis=0, keepdims=True) + EPS
            realised = stock_r[:, starts + 1] / past_vol
            factor_payoff = np.mean(past_signal * realised, axis=0)
            edge_z = factor_payoff.mean() / (
                factor_payoff.std() / np.sqrt(len(factor_payoff)) + EPS
            )
            weight = max(edge_z - self.c.edge_threshold, 0.0)
            combined += weight * current_move
            total_weight += weight

        if total_weight <= EPS:
            return np.zeros(N)
        signal = np.zeros(N)
        signal[1:] = combined / total_weight
        signal = _zscore_cross_section(signal)
        # Confidence grows smoothly instead of jumping from flat to full risk.
        confidence = np.tanh(total_weight)
        return (
            LIMITS * self.c.gross * confidence
            * np.tanh(signal / self.c.temperature)
        )


class DirectionalSkillGatedMomentum:
    """Skill-gated time-series momentum that retains the common direction.

    Cross-sectional momentum deliberately removes the average stock trend.
    That is correct for relative winner/loser rotation, but it cannot express
    a broad drift regime.  This sibling uses the same no-lookahead horizon
    skill test while retaining the common stock direction.  ALGO remains
    untouched so its 10x limit cannot dominate the result.
    """

    def __init__(self, config: SkillGateConfig):
        self.c = config

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        logp = np.log(np.maximum(prices, EPS))
        r = np.diff(logp, axis=1)
        longest = max(self.c.horizons)
        if r.shape[1] < longest + self.c.edge_window + 5:
            return np.zeros(N)
        end = r.shape[1] - 1
        endpoints = np.arange(
            max(longest - 1, end - self.c.edge_window), end
        )
        stock_r = r[1:]
        csum = np.pad(
            np.cumsum(stock_r, axis=1), ((0, 0), (1, 0))
        )
        csum2 = np.pad(
            np.cumsum(stock_r * stock_r, axis=1), ((0, 0), (1, 0))
        )
        vol_start = np.maximum(
            0, endpoints + 1 - self.c.vol_window
        )
        count = endpoints + 1 - vol_start
        vol_sum = csum[:, endpoints + 1] - csum[:, vol_start]
        vol_sum2 = csum2[:, endpoints + 1] - csum2[:, vol_start]
        past_vol = np.sqrt(np.maximum(
            vol_sum2 / count - (vol_sum / count) ** 2, 0.0
        )) + EPS
        current_vol = stock_r[:, -self.c.vol_window:].std(1) + EPS

        combined = np.zeros(50)
        total_weight = 0.0
        for horizon in self.c.horizons:
            current = (
                logp[1:, -1] - logp[1:, -horizon - 1]
            ) / (current_vol * np.sqrt(horizon))
            current /= current.std() + EPS

            move = (
                csum[:, endpoints + 1]
                - csum[:, endpoints + 1 - horizon]
            )
            past_signal = move / (past_vol * np.sqrt(horizon))
            past_signal /= past_signal.std(axis=0, keepdims=True) + EPS
            realised = stock_r[:, endpoints + 1] / past_vol
            factor_payoff = np.mean(past_signal * realised, axis=0)
            edge_z = factor_payoff.mean() / (
                factor_payoff.std() / np.sqrt(len(factor_payoff)) + EPS
            )
            weight = max(edge_z - self.c.edge_threshold, 0.0)
            combined += weight * current
            total_weight += weight

        if total_weight <= EPS:
            return np.zeros(N)
        signal = np.zeros(N)
        signal[1:] = combined / total_weight
        confidence = np.tanh(total_weight)
        return (
            LIMITS * self.c.gross * confidence
            * np.tanh(signal / self.c.temperature)
        )


@dataclass(frozen=True)
class LookupConfig:
    horizon: int = 20
    vol_window: int = 100
    train_window: int | None = None
    shrink: float = 100.0
    temperature: float = 0.7
    gross: float = 1.0
    use_efficiency: bool = False


class PooledRegimeLookup:
    """Learn whether strong/efficient trends continue, pooling all stocks.

    This is a tiny conditional-mean table rather than an unconstrained ML
    model.  It estimates next-day continuation separately for broad bins of
    absolute trend strength (and optionally path efficiency), then applies the
    learned continuation only when the bin's historical alpha is positive.
    """

    TREND_BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, np.inf])
    EFF_BINS = np.array([0.0, 0.2, 0.4, 1.01])

    def __init__(self, config: LookupConfig):
        self.c = config

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        h = self.c.horizon
        logp = np.log(np.maximum(prices, EPS))
        r = np.diff(logp, axis=1)
        nret = r.shape[1]
        if nret < h + 100:
            return np.zeros(N)

        first = h - 1
        if self.c.train_window is not None:
            first = max(first, nret - 1 - self.c.train_window)
        endpoints = np.arange(first, nret - 1)
        stock_r = r[1:]
        csum = np.pad(np.cumsum(stock_r, axis=1), ((0, 0), (1, 0)))
        csum2 = np.pad(
            np.cumsum(stock_r * stock_r, axis=1), ((0, 0), (1, 0))
        )
        cabs = np.pad(
            np.cumsum(np.abs(stock_r), axis=1), ((0, 0), (1, 0))
        )
        move = csum[:, endpoints + 1] - csum[:, endpoints + 1 - h]
        path = cabs[:, endpoints + 1] - cabs[:, endpoints + 1 - h]
        vol_start = np.maximum(0, endpoints + 1 - self.c.vol_window)
        count = endpoints + 1 - vol_start
        vol_sum = csum[:, endpoints + 1] - csum[:, vol_start]
        vol_sum2 = csum2[:, endpoints + 1] - csum2[:, vol_start]
        variance = np.maximum(
            vol_sum2 / count - (vol_sum / count) ** 2, 0.0
        )
        vol = np.sqrt(variance) + EPS
        trend_hist = (move / (vol * np.sqrt(h))).T
        eff_hist = (np.abs(move) / (path + EPS)).T
        outcome = (
            np.sign(move) * stock_r[:, endpoints + 1] / vol
        ).T

        current_trend, current_eff, _ = _trend_features(
            prices, (h,), self.c.vol_window
        )
        pred = np.zeros(50)
        trend_bin = np.digitize(
            np.abs(current_trend[1:]), self.TREND_BINS[1:-1]
        )
        eff_bin = np.digitize(
            current_eff[1:], self.EFF_BINS[1:-1]
        )
        for tb in range(len(self.TREND_BINS) - 1):
            eff_range = (
                range(len(self.EFF_BINS) - 1)
                if self.c.use_efficiency else range(1)
            )
            for eb in eff_range:
                mask = (
                    (np.abs(trend_hist) >= self.TREND_BINS[tb])
                    & (np.abs(trend_hist) < self.TREND_BINS[tb + 1])
                )
                current_mask = trend_bin == tb
                if self.c.use_efficiency:
                    mask &= (
                        (eff_hist >= self.EFF_BINS[eb])
                        & (eff_hist < self.EFF_BINS[eb + 1])
                    )
                    current_mask &= eff_bin == eb
                count = int(mask.sum())
                alpha = float(outcome[mask].sum() / (
                    count + self.c.shrink
                ))
                pred[current_mask] = max(alpha, 0.0)
        signal = np.zeros(N)
        signal[1:] = np.sign(current_trend[1:]) * pred
        signal = _zscore_cross_section(signal)
        return LIMITS * self.c.gross * np.tanh(
            signal / self.c.temperature
        )


class Blend:
    """Convex dollar blend of two books, preserving all position limits."""

    def __init__(self, left, right, weight: float):
        self.left, self.right, self.weight = left, right, weight

    def __call__(self, prices):
        dollars = (
            (1.0 - self.weight) * self.left(prices)
            + self.weight * self.right(prices)
        )
        return np.clip(dollars, -LIMITS, LIMITS)


class Additive:
    """Add a small independent sleeve without shrinking the baseline."""

    def __init__(self, left, right, scale: float):
        self.left, self.right, self.scale = left, right, scale

    def __call__(self, prices):
        dollars = self.left(prices) + self.scale * self.right(prices)
        return np.clip(dollars, -LIMITS, LIMITS)


def _load_baseline():
    path = ROOT / "strategies" / "unicorn_curriculum_candidate.py"
    spec = importlib.util.spec_from_file_location(
        f"_regime_baseline_{np.random.randint(1_000_000_000)}", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    def target(prices):
        return module.getMyPosition(prices) * prices[:, -1]

    return target


def _development_suite(prices):
    configs = []
    for horizon in (5, 10, 20, 40, 60, 100):
        for construction in ("time", "cross", "rank"):
            c = MomentumConfig(
                (horizon,), 100, construction, temperature=0.8
            )
            configs.append((f"{construction} momentum h={horizon}", c))
    for horizons in ((5, 20), (10, 40), (20, 60), (20, 60, 120)):
        for construction in ("cross", "rank"):
            configs.append((
                f"{construction} multi h={horizons}",
                MomentumConfig(
                    horizons, 120, construction, temperature=0.8
                ),
            ))
    for horizon in (20, 40, 60, 100):
        configs.append((
            f"breakout h={horizon}",
            MomentumConfig(
                (horizon,), 100, "breakout", temperature=0.8
            ),
        ))
    for h in (20, 40, 60):
        for threshold in (0.5, 1.0, 1.5):
            configs.append((
                f"strong trend h={h} z>{threshold}",
                MomentumConfig(
                    (h,), 100, "cross", 0.8,
                    trend_threshold=threshold,
                ),
            ))
        for threshold in (0.1, 0.2, 0.3, 0.4):
            configs.append((
                f"efficient trend h={h} e>{threshold}",
                MomentumConfig(
                    (h,), 100, "cross", 0.8,
                    efficiency_threshold=threshold,
                ),
            ))
    for horizons in ((10, 40), (20, 60), (20, 60, 120)):
        configs.append((
            f"aligned trend h={horizons}",
            MomentumConfig(
                horizons, 120, "cross", 0.8, require_alignment=True
            ),
        ))
    for h in (20, 40, 60):
        for regime in ("calm", "hot"):
            configs.append((
                f"{regime} trend h={h}",
                MomentumConfig(
                    (h,), 100, "cross", 0.8, vol_gate=regime
                ),
            ))

    ranked = []
    for label, config in configs:
        rows = evaluate(
            label, lambda c=config: RegimeMomentum(c), prices, verbose=False
        )
        scores = [row[3] for row in rows.values()]
        ranked.append((min(scores), np.mean(scores), label, config, rows))
    ranked.sort(reverse=True, key=lambda row: (row[0], row[1]))
    print("\nBest fixed momentum variants on development only")
    for _, _, label, _, rows in ranked[:20]:
        text = " | ".join(
            f"{name} {row[3]:7.1f}" for name, row in rows.items()
        )
        print(f"{label:54s} | {text}")

    print("\nWalk-forward adaptive gates")
    adaptive = []
    for h in (10, 20, 40, 60):
        for ew in (50, 100, 150, 250):
            for mode in ("global", "asset", "signed-global"):
                config = AdaptiveConfig(h, 100, ew, 0.0, mode, 0.8)
                label = f"adaptive {mode} h={h} edge={ew}"
                rows = evaluate(
                    label,
                    lambda c=config: AdaptiveMomentum(c),
                    prices,
                    verbose=False,
                )
                scores = [row[3] for row in rows.values()]
                adaptive.append(
                    (min(scores), np.mean(scores), label, config, rows)
                )
    adaptive.sort(reverse=True, key=lambda row: (row[0], row[1]))
    for _, _, label, _, rows in adaptive[:15]:
        text = " | ".join(
            f"{name} {row[3]:7.1f}" for name, row in rows.items()
        )
        print(f"{label:54s} | {text}")

    print("\nPooled regime lookup")
    lookups = []
    for h in (10, 20, 40, 60):
        for tw in (None, 250, 400):
            for shrink in (50.0, 100.0, 250.0):
                for efficiency in (False, True):
                    config = LookupConfig(
                        h, 100, tw, shrink, 0.8, 1.0, efficiency
                    )
                    label = (
                        f"lookup h={h} train={tw} shrink={shrink:g} "
                        f"eff={efficiency}"
                    )
                    rows = evaluate(
                        label,
                        lambda c=config: PooledRegimeLookup(c),
                        prices,
                        verbose=False,
                    )
                    scores = [row[3] for row in rows.values()]
                    lookups.append(
                        (min(scores), np.mean(scores), label, config, rows)
                    )
    lookups.sort(reverse=True, key=lambda row: (row[0], row[1]))
    for _, _, label, _, rows in lookups[:15]:
        text = " | ".join(
            f"{name} {row[3]:7.1f}" for name, row in rows.items()
        )
        print(f"{label:54s} | {text}")

    # Freeze candidates by development minimum; the audit phase below does
    # not perform another parameter search.
    shortlist = ranked[:3] + adaptive[:2] + lookups[:2]
    print("\nFrozen shortlist (copy these labels before running --phase audit)")
    for _, _, label, config, _ in shortlist:
        print(label, repr(config))

    print("\nOverlay plateau against the existing curriculum candidate")
    overlay_candidates = shortlist[:5]
    for _, _, label, config, _ in overlay_candidates:
        if isinstance(config, MomentumConfig):
            make = lambda c=config: RegimeMomentum(c)
        elif isinstance(config, AdaptiveConfig):
            make = lambda c=config: AdaptiveMomentum(c)
        else:
            make = lambda c=config: PooledRegimeLookup(c)
        for weight in (0.10, 0.20, 0.30, 0.40):
            evaluate(
                f"baseline + {weight:.2f} {label}",
                lambda f=make, w=weight: Blend(_load_baseline(), f(), w),
                prices,
            )


# Frozen after the development pass.  Do not select a different candidate
# based on --phase audit; audit is diagnosis, not another tuning set.
FROZEN = (
    (
        "skill-gated short (5,10,20), edge=60, z>2",
        lambda: SkillGatedMomentum(SkillGateConfig(
            (5, 10, 20), 100, 60, 2.0, 0.8, 1.0
        )),
    ),
    (
        "skill-gated long (20,40,60), edge=100, z>2",
        lambda: SkillGatedMomentum(SkillGateConfig(
            (20, 40, 60), 100, 100, 2.0, 0.8, 1.0
        )),
    ),
    (
        "signed adaptive h=60 edge=100",
        lambda: AdaptiveMomentum(AdaptiveConfig(
            60, 100, 100, 0.0, "signed-global", 0.8
        )),
    ),
    (
        "signed adaptive h=20 edge=150",
        lambda: AdaptiveMomentum(AdaptiveConfig(
            20, 100, 150, 0.0, "signed-global", 0.8
        )),
    ),
)


def _audit_suite(prices):
    print("Frozen standalone audit")
    for label, factory in FROZEN:
        evaluate(label, factory, prices, windows=AUDIT_WINDOWS)
    print("\nFrozen additive-overlay audit (scales chosen before seeing audit)")
    for label, factory in FROZEN:
        for weight in (0.05, 0.10, 0.20, 0.30):
            evaluate(
                f"baseline + {weight:.2f} {label}",
                lambda f=factory, w=weight: Additive(
                    _load_baseline(), f(), w
                ),
                prices,
                windows=AUDIT_WINDOWS,
            )
    evaluate(
        "existing curriculum baseline",
        _load_baseline,
        prices,
        windows=DEVELOPMENT_WINDOWS + AUDIT_WINDOWS,
    )


# Experimental deployment rule, retained for reproducibility rather than
# recommended as a standalone submission.  It was locked on days 0--750 and
# stayed flat on the untouched 750--1000 audit: the honest result is that the
# data did not confirm a broad momentum regime.
CANDIDATE_CONFIG = SkillGateConfig(
    horizons=(5, 10, 20),
    vol_window=100,
    edge_window=60,
    edge_threshold=2.0,
    temperature=0.8,
)
_candidate = SkillGatedMomentum(CANDIDATE_CONFIG)


def getMyPosition(prcSoFar):
    dollars = _candidate(prcSoFar)
    return (dollars / prcSoFar[:, -1]).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("develop", "audit"), default="develop"
    )
    args = parser.parse_args()
    prices = load_prices()
    if args.phase == "develop":
        _development_suite(prices[:, :750])
    else:
        if prices.shape[1] < 1000:
            raise SystemExit("audit requires all 1000 released days")
        _audit_suite(prices)


if __name__ == "__main__":
    main()
