#!/usr/bin/env python3
"""Risk-aware allocation around the existing dense lead-lag forecast."""

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


class RiskAwareRidge(ForecastBook):
    def __init__(self, mode="precision", cov_shrink=.5, factors=1, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.cov_shrink = cov_shrink
        self.factors = factors

    def _fit(self, p):
        super()._fit(p)
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1).T
        self.ret_cov = np.cov(r, rowvar=False)
        diag = np.diag(np.diag(self.ret_cov))
        self.shrunk_cov = (
            (1.0 - self.cov_shrink) * self.ret_cov
            + self.cov_shrink * diag
        )
        market = r[:, 0]
        self.beta = (r.T @ market) / (market @ market + EPS)
        sd = r.std(0) + EPS
        corr = self.ret_cov / np.outer(sd, sd)
        vals, vecs = np.linalg.eigh(corr)
        self.factor_load = vecs[:, np.argsort(vals)[-self.factors:]]

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        pred = ((r[:, -1] - self.mu) / self.sd) @ self.coef

        if self.mode == "precision":
            # Work in fractional-limit coordinates. This accounts for ALGO's
            # 10x larger dollar limit before the bounded tanh transform.
            signal = np.linalg.solve(self.shrunk_cov, pred) / LIMITS
        else:
            signal = pred / self.ysd

        signal -= signal.mean()
        signal /= signal.std() + EPS
        dollars = LIMITS * np.tanh(signal / self.temp)

        if self.mode == "betahedge":
            # Stock alpha book plus an explicit cheap ALGO market hedge.
            dollars[0] = np.clip(
                -np.dot(dollars[1:], self.beta[1:]) / self.beta[0],
                -LIMITS[0], LIMITS[0],
            )
        elif self.mode == "factorneutral":
            fractions = dollars / LIMITS
            fractions -= self.factor_load @ (self.factor_load.T @ fractions)
            fractions /= max(1.0, np.max(np.abs(fractions)))
            dollars = LIMITS * fractions
        return dollars


def evaluate(label, factory, prices):
    values, parts = [], []
    for name, start, end in WINDOWS:
        m, sd, score = simulate(prices, start, end, factory())
        values.append(score)
        parts.append(f"{name} {m:7.1f}/{sd:7.1f}/{score:7.1f}")
    print(f"{label:33s} | " + " | ".join(parts)
          + f" | min {min(values):7.1f}")


def main():
    p = load_prices()
    configs = [
        ("baseline", lambda: ForecastBook(lam=400, retrain=50, temp=.35)),
        ("precision shrink .10", lambda: RiskAwareRidge(
            cov_shrink=.10, lam=400, retrain=50, temp=.35)),
        ("precision shrink .25", lambda: RiskAwareRidge(
            cov_shrink=.25, lam=400, retrain=50, temp=.35)),
        ("precision shrink .50", lambda: RiskAwareRidge(
            cov_shrink=.50, lam=400, retrain=50, temp=.35)),
        ("precision shrink .75", lambda: RiskAwareRidge(
            cov_shrink=.75, lam=400, retrain=50, temp=.35)),
        ("precision diagonal", lambda: RiskAwareRidge(
            cov_shrink=1.0, lam=400, retrain=50, temp=.35)),
        ("explicit ALGO beta hedge", lambda: RiskAwareRidge(
            mode="betahedge", lam=400, retrain=50, temp=.35)),
        ("neutralize top factor", lambda: RiskAwareRidge(
            mode="factorneutral", factors=1, lam=400, retrain=50, temp=.35)),
        ("neutralize top 3 factors", lambda: RiskAwareRidge(
            mode="factorneutral", factors=3, lam=400, retrain=50, temp=.35)),
        ("neutralize top 5 factors", lambda: RiskAwareRidge(
            mode="factorneutral", factors=5, lam=400, retrain=50, temp=.35)),
    ]
    for label, factory in configs:
        evaluate(label, factory, p)


if __name__ == "__main__":
    main()
