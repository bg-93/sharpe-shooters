#!/usr/bin/env python3
"""Genuinely different signal families: structural drift and shock propagation."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backtesting"), str(ROOT / "strategies")]
from leadlag_research import load_prices, simulate
from leadlag_model_research import ForecastBook

N = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12
WINDOWS = (("early", 100, 300), ("middle", 250, 500), ("late", 500, 750))


def allocate(signal, temp=.5):
    signal = np.asarray(signal).copy()
    signal -= signal.mean()
    signal /= signal.std() + EPS
    return LIMITS * np.tanh(signal / temp)


class DriftBook:
    """Estimate persistent unconditional drift, with optional split stability."""

    def __init__(self, lookback=None, temp=.5, stable=False):
        self.lookback, self.temp, self.stable = lookback, temp, stable

    def __call__(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        if self.lookback:
            r = r[:, -self.lookback:]
        if r.shape[1] < 60:
            return np.zeros(N)
        signal = r.mean(1) / (r.std(1) + EPS)
        if self.stable:
            h = r.shape[1] // 2
            a = r[:, :h].mean(1) / (r[:, :h].std(1) + EPS)
            b = r[:, h:].mean(1) / (r[:, h:].std(1) + EPS)
            signal = np.where(np.sign(a) == np.sign(b),
                              np.sign(signal) * np.minimum(abs(a), abs(b)),
                              0.0)
        return allocate(signal, self.temp)


class ShockPropagation:
    """Trade learned lead-lag only following unusually large leader moves."""

    def __init__(self, threshold=1.5, top=20, temp=.5):
        self.threshold, self.top, self.temp = threshold, top, temp
        self.last_fit = -1
        self.edges = None

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        if self.edges is None or nt - self.last_fit >= 50:
            x, y = r[:, :-1], r[:, 1:]
            vol = x.std(1) + EPS
            scores = np.zeros((N, N))
            effects = np.zeros((N, N))
            for leader in range(N):
                z = x[leader] / vol[leader]
                mask = np.abs(z) >= self.threshold
                if mask.sum() < 15:
                    continue
                signed_y = y[:, mask] * np.sign(z[mask])
                effect = signed_y.mean(1) / (y.std(1) + EPS)
                # Shrink rare-event estimates.
                effect *= mask.sum() / (mask.sum() + 30.0)
                effects[leader] = effect
                scores[leader] = np.abs(effect)
            np.fill_diagonal(scores, 0.0)
            flat = np.argpartition(scores.ravel(), -self.top)[-self.top:]
            self.edges = [
                (*np.unravel_index(k, scores.shape),
                 effects[np.unravel_index(k, scores.shape)])
                for k in flat
            ]
            self.vol = vol
            self.last_fit = nt
        signal = np.zeros(N)
        for leader, target, effect in self.edges:
            z = r[leader, -1] / self.vol[leader]
            if abs(z) >= self.threshold:
                signal[target] += effect * np.sign(z)
        if signal.std() < EPS:
            return np.zeros(N)
        return allocate(signal, self.temp)


class ReliabilityWeightedRidge(ForecastBook):
    """Continuously weight targets by a trailing, strictly held-out IC."""

    def __init__(self, validation=60, floor=.25, power=1.0, **kwargs):
        super().__init__(**kwargs)
        self.validation = validation
        self.floor = floor
        self.power = power

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        if self.lookback:
            r = r[:, -self.lookback - 1:]
        x, y = r[:, :-1].T, r[:, 1:].T
        cut = max(40, len(x) - self.validation)
        xt, yt, xv, yv = x[:cut], y[:cut], x[cut:], y[cut:]
        mu, sd = xt.mean(0), xt.std(0) + EPS
        zt, zv = (xt - mu) / sd, (xv - mu) / sd
        w = np.linalg.solve(
            zt.T @ zt + self.lam * np.eye(N), zt.T @ yt
        )
        pv = zv @ w
        ic = np.zeros(N)
        for j in range(N):
            if pv[:, j].std() > EPS and yv[:, j].std() > EPS:
                ic[j] = np.corrcoef(pv[:, j], yv[:, j])[0, 1]
        positive = np.maximum(ic, 0.0)
        if positive.max() > EPS:
            positive = (positive / positive.max()) ** self.power
        self.reliability = self.floor + (1.0 - self.floor) * positive
        super()._fit(p)

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        pred = ((r[:, -1] - self.mu) / self.sd) @ self.coef
        signal = pred / self.ysd * self.reliability
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


class TargetCVRidge(ForecastBook):
    """Choose ridge shrinkage separately for each forecast target."""

    def __init__(self, lambdas=(20, 100, 400, 1000), validation=60, **kwargs):
        super().__init__(**kwargs)
        self.lambdas = lambdas
        self.validation = validation

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        x, y = r[:, :-1].T, r[:, 1:].T
        cut = max(40, len(x) - self.validation)
        mu, sd = x[:cut].mean(0), x[:cut].std(0) + EPS
        zt, zv = (x[:cut] - mu) / sd, (x[cut:] - mu) / sd
        losses, candidates = [], []
        for lam in self.lambdas:
            w = np.linalg.solve(
                zt.T @ zt + lam * np.eye(N), zt.T @ y[:cut]
            )
            pv = zv @ w
            losses.append(np.mean((pv - y[cut:]) ** 2, axis=0))
            candidates.append(lam)
        chosen = np.asarray(candidates)[np.argmin(losses, axis=0)]

        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        z = (x - self.mu) / self.sd
        self.ysd = y.std(0) + EPS
        self.coef = np.zeros((N, N))
        for lam in self.lambdas:
            cols = np.where(chosen == lam)[0]
            if len(cols):
                self.coef[:, cols] = np.linalg.solve(
                    z.T @ z + lam * np.eye(N), z.T @ y[:, cols]
                )


class CanonicalLeadLag(ForecastBook):
    """CCA-style predictive modes with regularized X/Y covariance whitening."""

    def __init__(self, rank=10, shrink=.25, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.shrink = shrink

    @staticmethod
    def _root(a, inverse=False):
        values, vectors = np.linalg.eigh(a)
        values = np.maximum(values, 1e-8)
        power = -0.5 if inverse else 0.5
        return (vectors * values ** power) @ vectors.T

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        if self.lookback:
            r = r[:, -self.lookback - 1:]
        x, y = r[:, :-1].T, r[:, 1:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        self.ymu, self.ysd = y.mean(0), y.std(0) + EPS
        xz, yz = (x - self.mu) / self.sd, (y - self.ymu) / self.ysd
        n = len(xz)
        cx, cy = xz.T @ xz / n, yz.T @ yz / n
        cx = (1 - self.shrink) * cx + self.shrink * np.eye(N)
        cy = (1 - self.shrink) * cy + self.shrink * np.eye(N)
        ix, iy = self._root(cx, True), self._root(cy, True)
        ry = self._root(cy, False)
        cross = xz.T @ yz / n
        u, singular, vt = np.linalg.svd(ix @ cross @ iy)
        k = self.rank
        self.coef = (
            ix @ u[:, :k] @ np.diag(singular[:k]) @ vt[:k] @ ry
        )

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        signal = ((r[:, -1] - self.mu) / self.sd) @ self.coef
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


class CoherentStockIndex:
    """Forecast stocks only; derive ALGO from its exact normalized-index identity."""

    def __init__(self, model="ridge", rank=7, shrink=.25, lam=400,
                 retrain=50, temp=.25, include_mean=False, lookback=None):
        self.model, self.rank, self.shrink = model, rank, shrink
        self.lam, self.retrain, self.temp = lam, retrain, temp
        self.include_mean = include_mean
        self.lookback = lookback
        self.last_fit = -1
        self.coef = None

    @staticmethod
    def _root(a, inverse=False):
        values, vectors = np.linalg.eigh(a)
        values = np.maximum(values, 1e-8)
        return (vectors * values ** (-.5 if inverse else .5)) @ vectors.T

    def _fit(self, p):
        # Simple returns respect the exact linear index construction.
        all_r = p[:, 1:] / p[:, :-1] - 1.0
        if self.lookback:
            all_r = all_r[:, -self.lookback - 1:]
        r = all_r[1:]
        x, y = r[:, :-1].T, r[:, 1:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        self.ymu, self.ysd = y.mean(0), y.std(0) + EPS
        xz, yz = (x - self.mu) / self.sd, (y - self.ymu) / self.ysd
        if self.model == "ridge":
            self.coef = np.linalg.solve(
                xz.T @ xz + self.lam * np.eye(50), xz.T @ yz
            )
        else:
            n = len(xz)
            cx, cy = xz.T @ xz / n, yz.T @ yz / n
            cx = (1 - self.shrink) * cx + self.shrink * np.eye(50)
            cy = (1 - self.shrink) * cy + self.shrink * np.eye(50)
            ix, iy = self._root(cx, True), self._root(cy, True)
            ry = self._root(cy, False)
            u, singular, vt = np.linalg.svd(ix @ (xz.T @ yz / n) @ iy)
            k = self.rank
            self.coef = (
                ix @ u[:, :k] @ np.diag(singular[:k]) @ vt[:k] @ ry
            )
        self.algo_sd = all_r[0, 1:].std() + EPS

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = p[1:, 1:] / p[1:, :-1] - 1.0
        stock_z = ((r[:, -1] - self.mu) / self.sd) @ self.coef
        stock_pred = stock_z * self.ysd
        if self.include_mean:
            stock_pred += self.ymu
        normalized = p[1:, -1] / p[1:, 0]
        index_weights = normalized / normalized.sum()
        algo_pred = index_weights @ stock_pred
        signal = np.r_[algo_pred / self.algo_sd, stock_z]
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


def evaluate(label, factory, prices):
    values, parts = [], []
    for name, start, end in WINDOWS:
        m, sd, score = simulate(prices, start, end, factory())
        values.append(score)
        parts.append(f"{name} {m:7.1f}/{sd:7.1f}/{score:7.1f}")
    print(f"{label:31s} | " + " | ".join(parts)
          + f" | min {min(values):7.1f}")


def main():
    p = load_prices()
    configs = [
        ("drift expanding t.5", lambda: DriftBook(temp=.5)),
        ("drift expanding t1", lambda: DriftBook(temp=1)),
        ("drift rolling100", lambda: DriftBook(lookback=100)),
        ("drift rolling250", lambda: DriftBook(lookback=250)),
        ("drift split-stable", lambda: DriftBook(stable=True)),
        ("shock propagation z1 top10", lambda: ShockPropagation(1, 10)),
        ("shock propagation z1 top30", lambda: ShockPropagation(1, 30)),
        ("shock propagation z1.5 top20", lambda: ShockPropagation(1.5, 20)),
        ("shock propagation z2 top20", lambda: ShockPropagation(2, 20)),
        ("reliability ridge floor0", lambda: ReliabilityWeightedRidge(
            lam=400, retrain=50, temp=.35, floor=0)),
        ("reliability ridge floor.25", lambda: ReliabilityWeightedRidge(
            lam=400, retrain=50, temp=.35, floor=.25)),
        ("reliability ridge floor.50", lambda: ReliabilityWeightedRidge(
            lam=400, retrain=50, temp=.35, floor=.5)),
        ("reliability ridge val100", lambda: ReliabilityWeightedRidge(
            lam=400, retrain=50, temp=.35, floor=.25, validation=100)),
        ("reliability ridge sqrt", lambda: ReliabilityWeightedRidge(
            lam=400, retrain=50, temp=.35, floor=.25, power=.5)),
        ("target-CV ridge val60", lambda: TargetCVRidge(
            retrain=50, temp=.35, validation=60)),
        ("target-CV ridge val100", lambda: TargetCVRidge(
            retrain=50, temp=.35, validation=100)),
        ("CCA leadlag rank5", lambda: CanonicalLeadLag(
            rank=5, shrink=.25, retrain=50, temp=.35)),
        ("CCA leadlag rank10", lambda: CanonicalLeadLag(
            rank=10, shrink=.25, retrain=50, temp=.35)),
        ("CCA leadlag rank7", lambda: CanonicalLeadLag(
            rank=7, shrink=.25, retrain=50, temp=.35)),
        ("CCA leadlag rank12", lambda: CanonicalLeadLag(
            rank=12, shrink=.25, retrain=50, temp=.35)),
        ("CCA leadlag rank20", lambda: CanonicalLeadLag(
            rank=20, shrink=.25, retrain=50, temp=.35)),
        ("CCA leadlag rank30", lambda: CanonicalLeadLag(
            rank=30, shrink=.25, retrain=50, temp=.35)),
        ("CCA leadlag rank40", lambda: CanonicalLeadLag(
            rank=40, shrink=.25, retrain=50, temp=.35)),
        ("CCA rank20 shrink.5", lambda: CanonicalLeadLag(
            rank=20, shrink=.5, retrain=50, temp=.35)),
        ("CCA rank10 shrink.10", lambda: CanonicalLeadLag(
            rank=10, shrink=.10, retrain=50, temp=.35)),
        ("CCA rank10 shrink.50", lambda: CanonicalLeadLag(
            rank=10, shrink=.50, retrain=50, temp=.35)),
        ("CCA rank10 shrink.75", lambda: CanonicalLeadLag(
            rank=10, shrink=.75, retrain=50, temp=.35)),
        ("CCA rank10 temp.50", lambda: CanonicalLeadLag(
            rank=10, shrink=.25, retrain=50, temp=.50)),
        ("coherent stock ridge", lambda: CoherentStockIndex(
            model="ridge", temp=.25)),
        ("coherent stock CCA5", lambda: CoherentStockIndex(
            model="cca", rank=5, shrink=.25, temp=.35)),
        ("coherent stock CCA7", lambda: CoherentStockIndex(
            model="cca", rank=7, shrink=.25, temp=.35)),
        ("coherent stock CCA10", lambda: CoherentStockIndex(
            model="cca", rank=10, shrink=.25, temp=.35)),
        ("coherent CCA7 + drift", lambda: CoherentStockIndex(
            model="cca", rank=7, shrink=.25, temp=.35, include_mean=True)),
    ]
    for label, factory in configs:
        evaluate(label, factory, p)


if __name__ == "__main__":
    main()
