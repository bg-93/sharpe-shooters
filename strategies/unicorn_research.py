#!/usr/bin/env python3
"""Search simple, non-pairs alternatives on chronological price windows.

The point of this file is breadth, not a giant optimiser.  Every model is
walk-forward, uses only history visible on the decision day, and has one
short economic/generator hypothesis behind it.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtesting"))
from leadlag_research import simulate

N = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12
WINDOWS = (("early", 100, 300), ("middle", 250, 500), ("late", 500, 750))


def prices():
    return pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T


def sized(pred, temp=1.0, algo=False):
    """Robust cross-sectional sizing shared by all experiments."""
    pred = np.asarray(pred, float).copy()
    pred[~np.isfinite(pred)] = 0.0
    if not algo:
        pred[0] = 0.0
    active = pred if algo else pred[1:]
    pred -= active.mean()
    scale = active.std()
    if scale < EPS:
        return np.zeros(N)
    return LIMITS * np.tanh(pred / (scale * temp))


class PhaseDrift:
    """Learn a separate cross-sectional drift for t modulo a short period."""

    def __init__(self, period=5, lookback=400, shrink=20.0):
        self.period, self.lookback, self.shrink = period, lookback, shrink

    def __call__(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1).T
        if len(r) < max(60, self.period * 8):
            return np.zeros(N)
        end = len(r)
        idx = np.arange(max(0, end - self.lookback), end)
        # At price day nt-1, the held return is return index nt-1.
        take = idx[idx % self.period == end % self.period]
        x = r[take]
        pred = x.sum(0) / (len(x) + self.shrink)
        return sized(pred, temp=0.8)


class ConditionalDrift:
    """Nonlinear lookup: own previous-return sign x market/dispersion state."""

    def __init__(self, lookback=400, threshold=0.0, mode="market"):
        self.lookback, self.threshold, self.mode = lookback, threshold, mode

    def __call__(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1).T
        if len(r) < 80:
            return np.zeros(N)
        x, y = r[:-1], r[1:]
        x, y = x[-self.lookback:], y[-self.lookback:]
        own = np.sign(x)
        if self.mode == "market":
            state = np.sign(x[:, 0])
            now_state = np.sign(r[-1, 0])
        else:
            disp = x[:, 1:].std(1)
            cut = np.median(disp)
            state = np.where(disp > cut, 1.0, -1.0)
            now_state = 1.0 if r[-1, 1:].std() > cut else -1.0
        now_own = np.sign(r[-1])
        pred = np.zeros(N)
        for j in range(1, N):
            mask = (own[:, j] == now_own[j]) & (state == now_state)
            if mask.sum() >= 15:
                pred[j] = y[mask, j].mean() * mask.sum() / (mask.sum() + 20)
        if self.threshold:
            pred[np.abs(r[-1]) < self.threshold * r[:, 1:].std(0)] = 0.0
        return sized(pred, temp=0.8)


class SparseMotifs:
    """Only unusually stable one-edge lead/lag motifs, not a dense ridge."""

    def __init__(self, lookback=400, top=12, agree=True):
        self.lookback, self.top, self.agree = lookback, top, agree

    def __call__(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        if r.shape[1] < 100:
            return np.zeros(N)
        z = r[:, -self.lookback:]
        x, y = z[:, :-1], z[:, 1:]
        h = x.shape[1] // 2

        def corr(a, b):
            a = a - a.mean(1, keepdims=True)
            b = b - b.mean(1, keepdims=True)
            return (a @ b.T) / (
                np.sqrt((a * a).sum(1))[:, None]
                * np.sqrt((b * b).sum(1))[None, :] + EPS
            )

        c = corr(x, y)
        if self.agree and h >= 40:
            c1, c2 = corr(x[:, :h], y[:, :h]), corr(x[:, h:], y[:, h:])
            c[(np.sign(c1) != np.sign(c2))] = 0.0
            c = np.sign(c) * np.minimum(np.abs(c1), np.abs(c2))
        np.fill_diagonal(c, 0.0)
        flat = np.argpartition(np.abs(c).ravel(), -self.top)[-self.top:]
        pred = np.zeros(N)
        vol = z.std(1) + EPS
        for k in flat:
            leader, target = np.unravel_index(k, c.shape)
            pred[target] += c[leader, target] * r[leader, -1] / vol[leader]
        return sized(pred, temp=0.7)


class FactorRotation:
    """Forecast cross-sectional rotation using only a few PCA factor lags."""

    def __init__(self, factors=5, lookback=400, lam=20.0, temp=0.8):
        self.factors, self.lookback, self.lam, self.temp = (
            factors, lookback, lam, temp
        )

    def __call__(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)[1:, -self.lookback:]
        if r.shape[1] < 100:
            return np.zeros(N)
        mu = r.mean(1, keepdims=True)
        sd = r.std(1, keepdims=True) + EPS
        q = (r - mu) / sd
        u, _, _ = np.linalg.svd(q, full_matrices=False)
        load = u[:, :self.factors]
        f = load.T @ q
        x, y = f[:, :-1].T, q[:, 1:].T
        w = np.linalg.solve(x.T @ x + self.lam * np.eye(self.factors), x.T @ y)
        pred = np.zeros(N)
        pred[1:] = f[:, -1] @ w
        return sized(pred, temp=self.temp)


class IndexTimer:
    """Use a handful of market breadth summaries to forecast only ALGO."""

    def __init__(self, lookback=400, lags=3, lam=20.0):
        self.lookback, self.lags, self.lam = lookback, lags, lam

    def __call__(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1).T[-self.lookback:]
        if len(r) < 100:
            return np.zeros(N)
        stocks = r[:, 1:]
        breadth = np.column_stack((
            r[:, 0],
            stocks.mean(1),
            np.median(stocks, axis=1),
            np.sign(stocks).mean(1),
            stocks.std(1),
        ))
        rows, target = [], []
        for t in range(self.lags, len(r) - 1):
            rows.append(breadth[t - self.lags + 1:t + 1].ravel())
            target.append(r[t + 1, 0])
        x, y = np.asarray(rows), np.asarray(target)
        mu, sd = x.mean(0), x.std(0) + EPS
        z = (x - mu) / sd
        w = np.linalg.solve(z.T @ z + self.lam * np.eye(z.shape[1]), z.T @ y)
        now = ((breadth[-self.lags:].ravel() - mu) / sd) @ w
        noise = y.std() + EPS
        out = np.zeros(N)
        out[0] = LIMITS[0] * np.tanh(now / (0.25 * noise))
        return out


class Ensemble:
    """Average target dollars from independent compact models."""

    def __init__(self, models):
        self.models = models

    def __call__(self, p):
        return np.clip(sum(model(p) for model in self.models) / len(self.models),
                       -LIMITS, LIMITS)


def evaluate(label, factory, p):
    vals = []
    bits = []
    for name, start, end in WINDOWS:
        m, sd, sc = simulate(p, start, end, factory())
        vals.append(sc)
        bits.append(f"{name} {m:7.1f}/{sd:7.1f}/{sc:7.1f}")
    print(f"{label:34s} | " + " | ".join(bits) + f" | min {min(vals):7.1f}")
    return vals


def main():
    p = prices()
    configs = [
        ("phase drift p=5", lambda: PhaseDrift(5)),
        ("phase drift p=10", lambda: PhaseDrift(10)),
        ("phase drift p=20", lambda: PhaseDrift(20)),
        ("conditional own x market", lambda: ConditionalDrift(mode="market")),
        ("conditional own x dispersion", lambda: ConditionalDrift(mode="disp")),
        ("sparse motifs top=6", lambda: SparseMotifs(top=6)),
        ("sparse motifs top=12", lambda: SparseMotifs(top=12)),
        ("sparse motifs top=24", lambda: SparseMotifs(top=24)),
        ("PCA rotation k=3", lambda: FactorRotation(3)),
        ("PCA rotation k=5", lambda: FactorRotation(5)),
        ("PCA rotation k=10", lambda: FactorRotation(10)),
        ("PCA k=10 lb=250 lam=5", lambda: FactorRotation(10, 250, 5)),
        ("PCA k=15 lb=250 lam=20", lambda: FactorRotation(15, 250, 20)),
        ("PCA k=20 lb=400 lam=20", lambda: FactorRotation(20, 400, 20)),
        ("PCA k=20 lb=250 lam=20", lambda: FactorRotation(20, 250, 20)),
        ("PCA k=20 lb=600 lam=20", lambda: FactorRotation(20, 600, 20)),
        ("PCA k=25 lb=400 lam=20", lambda: FactorRotation(25, 400, 20)),
        ("PCA k=30 lb=400 lam=20", lambda: FactorRotation(30, 400, 20)),
        ("PCA k=35 lb=400 lam=20", lambda: FactorRotation(35, 400, 20)),
        ("PCA k=40 lb=400 lam=20", lambda: FactorRotation(40, 400, 20)),
        ("PCA k=45 lb=400 lam=20", lambda: FactorRotation(45, 400, 20)),
        ("PCA k=50 lb=400 lam=20", lambda: FactorRotation(50, 400, 20)),
        ("PCA k=20 lam=5", lambda: FactorRotation(20, 400, 5)),
        ("PCA k=20 lam=100", lambda: FactorRotation(20, 400, 100)),
        ("PCA k=20 temp=.5", lambda: FactorRotation(20, 400, 20, .5)),
        ("PCA k=20 temp=1.2", lambda: FactorRotation(20, 400, 20, 1.2)),
        ("index timer lag=1", lambda: IndexTimer(lags=1)),
        ("index timer lag=3", lambda: IndexTimer(lags=3)),
        ("index timer lag=5", lambda: IndexTimer(lags=5)),
        ("ensemble sparse24 + PCA10", lambda: Ensemble(
            [SparseMotifs(top=24), FactorRotation(10)])),
        ("ensemble PCA3 + PCA10", lambda: Ensemble(
            [FactorRotation(3), FactorRotation(10)])),
        ("ensemble sparse24 + PCA3 + PCA10", lambda: Ensemble(
            [SparseMotifs(top=24), FactorRotation(3), FactorRotation(10)])),
    ]
    for label, factory in configs:
        evaluate(label, factory, p)


if __name__ == "__main__":
    main()
