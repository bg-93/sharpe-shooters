#!/usr/bin/env python3
"""Research overlays for the 1000-day release.

The newly revealed [750,1000) segment is treated as a one-time audit first.
All candidate rules are also reported on earlier chronological windows.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backtesting"), str(ROOT / "strategies")]
from leadlag_research import load_prices, simulate
from novel_signal_research import CoherentStockIndex

N = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12
WINDOWS = (
    ("early", 250, 500),
    ("old_oos", 500, 750),
    ("new_oos", 750, 1000),
)


def normalize(signal):
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


class TimeSeriesMomentum:
    """Volatility-scaled trend with optional breadth/strength regime gating."""

    def __init__(self, lookback=20, vol_window=60, temp=.5,
                 gate="none", threshold=0.0, topk=None):
        self.lookback, self.vol_window = lookback, vol_window
        self.temp, self.gate, self.threshold = temp, gate, threshold
        self.topk = topk

    def __call__(self, prices):
        nt = prices.shape[1]
        if nt <= max(self.lookback, self.vol_window) + 1:
            return np.zeros(N)
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1)
        trend = np.log(
            np.maximum(prices[:, -1], EPS)
            / np.maximum(prices[:, -1 - self.lookback], EPS)
        )
        vol = r[:, -self.vol_window:].std(1) * np.sqrt(self.lookback) + EPS
        z = trend / vol
        z = normalize(z)

        if self.topk:
            selected = np.argpartition(np.abs(z), -self.topk)[-self.topk:]
            mask = np.zeros(N)
            mask[selected] = 1.0
            z *= mask

        scale = 1.0
        if self.gate == "breadth":
            breadth = abs(np.mean(np.sign(trend[1:])))
            scale = np.clip(
                (breadth - self.threshold) / max(1 - self.threshold, EPS),
                0.0, 1.0
            )
        elif self.gate == "strength":
            strength = np.median(np.abs(trend[1:] / vol[1:]))
            scale = np.clip(strength / max(self.threshold, EPS), 0.0, 1.0)
        return scale * LIMITS * np.tanh(z / self.temp)


class DualHorizonMomentum:
    """Require short and long trends to agree; size by their weaker strength."""

    def __init__(self, fast=10, slow=50, vol_window=60, temp=.5):
        self.fast, self.slow = fast, slow
        self.vol_window, self.temp = vol_window, temp

    def __call__(self, prices):
        nt = prices.shape[1]
        if nt <= max(self.slow, self.vol_window) + 1:
            return np.zeros(N)
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1)
        vol = r[:, -self.vol_window:].std(1) + EPS
        fast = np.log(prices[:, -1] / prices[:, -1 - self.fast])
        slow = np.log(prices[:, -1] / prices[:, -1 - self.slow])
        a = fast / (vol * np.sqrt(self.fast))
        b = slow / (vol * np.sqrt(self.slow))
        signal = np.where(np.sign(a) == np.sign(b),
                          np.sign(a) * np.minimum(abs(a), abs(b)), 0.0)
        return LIMITS * np.tanh(normalize(signal) / self.temp)


class BreakoutMomentum:
    """Trade only statistically exceptional trends; stay flat otherwise."""

    def __init__(self, lookback=20, vol_window=60, threshold=2.0,
                 temp=.5, long_only=False):
        self.lookback, self.vol_window = lookback, vol_window
        self.threshold, self.temp, self.long_only = threshold, temp, long_only

    def __call__(self, prices):
        nt = prices.shape[1]
        if nt <= max(self.lookback, self.vol_window) + 1:
            return np.zeros(N)
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1)
        trend = np.log(prices[:, -1] / prices[:, -1 - self.lookback])
        scale = r[:, -self.vol_window:].std(1) * np.sqrt(self.lookback) + EPS
        strength = trend / scale
        excess = np.maximum(np.abs(strength) - self.threshold, 0.0)
        signal = np.sign(strength) * excess
        if self.long_only:
            signal = np.maximum(signal, 0.0)
        return LIMITS * np.tanh(signal / self.temp)


class OnlineCanonicalExperts:
    """Choose CCA/ridge blend by trailing realised paper PnL."""

    def __init__(self, rank=5, shrink=.5, memory=60, min_obs=30,
                 default_weight=.5):
        self.cca = CoherentStockIndex(
            model="cca", rank=rank, shrink=shrink, retrain=50,
            temp=.35, include_mean=True
        )
        self.ridge = CoherentStockIndex(
            model="ridge", lam=400, retrain=50, temp=.35
        )
        self.memory, self.min_obs = memory, min_obs
        self.default_weight = default_weight
        self.weights = (0.0, .25, .5, 1.0)
        self.history = {w: [] for w in self.weights}
        self.pending = None
        self.prev_nt = -1

    @staticmethod
    def _signal(dollars):
        fraction = np.clip(dollars / LIMITS, -.999999, .999999)
        return .35 * np.arctanh(fraction)

    def _targets(self, prices):
        ca = self._signal(self.cca(prices))
        ri = self._signal(self.ridge(prices))
        out = {}
        for weight in self.weights:
            z = normalize(ca + weight * ri)
            out[weight] = LIMITS * np.tanh(z / .35)
        return out

    def __call__(self, prices):
        nt = prices.shape[1]
        if nt <= self.prev_nt:
            self.__init__(
                rank=self.cca.rank, shrink=self.cca.shrink,
                memory=self.memory, min_obs=self.min_obs,
                default_weight=self.default_weight
            )
        self.prev_nt = nt
        if self.pending is not None:
            realised = prices[:, -1] / prices[:, -2] - 1.0
            for weight, dollars in self.pending.items():
                self.history[weight].append(float(dollars @ realised))
        self.pending = self._targets(prices)
        if len(self.history[0.0]) < self.min_obs:
            chosen = self.default_weight
        else:
            chosen = max(
                self.weights,
                key=lambda w: np.mean(self.history[w][-self.memory:])
            )
        return self.pending[chosen]


def evaluate(label, factory, prices):
    values, text = [], []
    for name, start, end in WINDOWS:
        mean, std, score = simulate(prices, start, end, factory())
        values.append(score)
        text.append(f"{name} {mean:7.1f}/{std:7.1f}/{score:7.1f}")
    print(f"{label:35s} | " + " | ".join(text)
          + f" | min {min(values):7.1f}")


def main():
    prices = load_prices()
    for lookback in (5, 10, 20, 40, 60, 120):
        evaluate(
            f"momentum lb={lookback}",
            lambda lb=lookback: TimeSeriesMomentum(lb, temp=.5),
            prices,
        )
    for fast, slow in ((5, 20), (10, 40), (10, 60), (20, 60), (20, 120)):
        evaluate(
            f"dual momentum {fast}/{slow}",
            lambda f=fast, s=slow: DualHorizonMomentum(f, s),
            prices,
        )
    for topk in (5, 10, 20):
        evaluate(
            f"momentum 20d top{topk}",
            lambda k=topk: TimeSeriesMomentum(20, temp=.5, topk=k),
            prices,
        )
    for lookback in (10, 20, 40):
        for threshold in (1.0, 1.5, 2.0, 2.5, 3.0):
            evaluate(
                f"breakout lb{lookback} z{threshold}",
                lambda lb=lookback, th=threshold: BreakoutMomentum(
                    lb, threshold=th, temp=.5
                ),
                prices,
            )
    for threshold in (1.5, 2.0, 2.5):
        evaluate(
            f"long breakout lb20 z{threshold}",
            lambda th=threshold: BreakoutMomentum(
                20, threshold=th, temp=.5, long_only=True
            ),
            prices,
        )
    for memory in (30, 60, 100):
        evaluate(
            f"online CCA/ridge experts m{memory}",
            lambda m=memory: OnlineCanonicalExperts(memory=m),
            prices,
        )


if __name__ == "__main__":
    main()
