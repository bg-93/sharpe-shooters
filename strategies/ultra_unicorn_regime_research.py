#!/usr/bin/env python3
"""Causal model-skill regime research for the 1,000-day release.

The regime variable is not price momentum.  It is the recent realised
incremental PnL of the dense-ridge view relative to a more stable canonical
lead-lag view.  On the first evaluator call, the state is reconstructed by
replaying only already-observed history, so the rule is deployable on day
1,001 without hidden state or look-ahead.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backtesting"), str(ROOT / "strategies")]
from leadlag_research import load_prices, simulate
from novel_signal_research import CoherentStockIndex

N = 51
EPS = 1e-12
LIMITS = np.array([100000.0] + [10000.0] * 50)
COMM = np.array([0.00002] + [0.0001] * 50)

PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836),
    (13, 45, 1.0132), (33, 40, 0.2577), (10, 46, 1.0331),
    (33, 42, 0.8358), (31, 43, 0.9692), (18, 28, 0.5642),
    (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)


def normalize(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal - signal.mean()
    return signal / (signal.std() + EPS)


class ModelSkillCanonical:
    """Canonical lead-lag with a causal dense-ridge skill gate."""

    def __init__(
        self,
        rank=5,
        shrink=.5,
        temperature=.35,
        memory=80,
        minimum=40,
        gate_speed=.5,
        default_gate=.5,
        core_scale=1.0,
        tail_gate=False,
        tail_rank=7,
        dense_weight=.25,
    ):
        self.rank = rank
        self.shrink = shrink
        self.temperature = temperature
        self.memory = memory
        self.minimum = minimum
        self.gate_speed = gate_speed
        self.default_gate = default_gate
        self.core_scale = core_scale
        self.tail_gate = tail_gate
        self.tail_rank = tail_rank
        self.dense_weight = dense_weight
        self._reset()

    def _reset(self):
        self.cca = CoherentStockIndex(
            model="cca", rank=self.rank, shrink=self.shrink,
            retrain=50, temp=self.temperature, include_mean=True,
        )
        self.cca_tail = None
        if self.tail_gate:
            self.cca_tail = CoherentStockIndex(
                model="cca", rank=self.tail_rank, shrink=self.shrink,
                retrain=50, temp=self.temperature, include_mean=True,
            )
        self.ridge = CoherentStockIndex(
            model="ridge", lam=400, retrain=50, temp=self.temperature,
        )
        alternative = 1.0 if self.tail_gate else .5
        self.pnl = {0.0: [], alternative: []}
        self.positions = {0.0: None, alternative: None}
        self.fees = {0.0: 0.0, alternative: 0.0}
        self.current_views = None
        self.previous_nt = -1
        self.last_gate = self.default_gate

    def _recover_signal(self, dollars):
        fraction = np.clip(dollars / LIMITS, -.999999, .999999)
        return self.temperature * np.arctanh(fraction)

    def _views(self, prices):
        cca = self._recover_signal(self.cca(prices))
        ridge = self._recover_signal(self.ridge(prices))
        if self.cca_tail is None:
            return cca, ridge
        tail = self._recover_signal(self.cca_tail(prices))
        return cca, ridge, tail

    def _dollars(self, views, expert_weight):
        cca, ridge = views[:2]
        if self.tail_gate:
            base = normalize(cca + self.dense_weight * ridge)
            full = normalize(views[2] + self.dense_weight * ridge)
            signal = normalize(
                (1.0 - expert_weight) * base + expert_weight * full
            )
        else:
            signal = normalize(cca + expert_weight * ridge)
        return (
            self.core_scale * LIMITS
            * np.tanh(signal / self.temperature)
        )

    def _record_then_position(self, prices, views):
        """Realise pending expert books, then stage their next positions."""
        nt = prices.shape[1]
        price = prices[:, -1]
        if self.positions[0.0] is not None:
            move = price - prices[:, -2]
            for weight in self.positions:
                realised = self.positions[weight] @ move - self.fees[weight]
                self.pnl[weight].append(float(realised))

        for weight in self.positions:
            dollars = self._dollars(views, weight)
            shares = np.clip(
                (dollars / price).astype(int),
                -(LIMITS / price).astype(int),
                (LIMITS / price).astype(int),
            )
            old = self.positions[weight]
            if old is None:
                old = np.zeros(N)
            self.fees[weight] = float(
                price @ (np.abs(shares - old) * COMM)
            )
            self.positions[weight] = shares

    def _backfill(self, prices):
        """Warm the gate from a strictly historical replay."""
        nt = prices.shape[1]
        start = max(120, nt - self.memory)
        for count in range(start, nt + 1):
            history = prices[:, :count]
            views = self._views(history)
            self._record_then_position(history, views)
        self.current_views = views

    def _gate(self):
        if len(self.pnl[0.0]) < self.minimum:
            return self.default_gate
        base = np.asarray(self.pnl[0.0][-self.memory:])
        alternative = 1.0 if self.tail_gate else .5
        other = np.asarray(self.pnl[alternative][-self.memory:])
        incremental = other - base
        error = incremental.std() / np.sqrt(len(incremental)) + EPS
        tstat = incremental.mean() / error
        # A smooth, bounded gate avoids a brittle all-or-nothing switch.
        return float(
            np.clip(.5 + self.gate_speed * tstat, 0.0, 1.0)
        )

    def target_dollars(self, prices):
        nt = prices.shape[1]
        if nt <= self.previous_nt:
            self._reset()
        if nt < 120:
            self.previous_nt = nt
            return np.zeros(N)
        if self.current_views is None:
            self._backfill(prices)
        else:
            self.current_views = self._views(prices)
            self._record_then_position(prices, self.current_views)
        self.previous_nt = nt
        self.last_gate = self._gate()
        expert_weight = (
            self.last_gate if self.tail_gate else .5 * self.last_gate
        )
        return self._dollars(self.current_views, expert_weight)


class PairOverlay:
    """Frozen diversifying pairs plus a supplied lead-lag core."""

    def __init__(self, core, scale=1.5, leg=9000.0):
        self.core = core
        self.scale = scale
        self.leg = leg
        self.states = [0] * len(PAIRS)
        self.previous_nt = -1

    def target_dollars(self, prices):
        nt = prices.shape[1]
        if nt <= self.previous_nt:
            self.states = [0] * len(PAIRS)
        self.previous_nt = nt
        target = getattr(self.core, "target_dollars", None)
        if target is None:
            target = self.core.target
        dollars = target(prices)
        if nt <= 61 or self.scale <= 0:
            return dollars
        log_prices = np.log(np.maximum(prices, EPS))
        for k, (left, right, gamma) in enumerate(PAIRS):
            spread = log_prices[left] - gamma * log_prices[right]
            history = spread[-61:-1]
            zscore = (
                spread[-1] - history.mean()
            ) / (history.std() + EPS)
            state = self.states[k]
            if state == 0:
                state = -1 if zscore > 1.5 else (1 if zscore < -1.5 else 0)
            elif state == 1 and zscore > -.5:
                state = 0
            elif state == -1 and zscore < .5:
                state = 0
            self.states[k] = state
            dollars[left] += self.scale * state * self.leg
            dollars[right] -= self.scale * state * gamma * self.leg
        return np.clip(dollars, -LIMITS, LIMITS)


def factory(memory=80, speed=.5, default=.5, pair_scale=0.0,
            tail_gate=False):
    core = ModelSkillCanonical(
        rank=3 if tail_gate else 5,
        memory=memory, gate_speed=speed, default_gate=default,
        tail_gate=tail_gate,
    )
    if pair_scale:
        return PairOverlay(core, pair_scale).target_dollars
    return core.target_dollars


def evaluate(prices, label, make):
    pieces, scores = [], []
    for name, start, end in (
        ("early", 100, 300),
        ("middle", 250, 500),
        ("old", 500, 750),
        ("new", 750, 1000),
    ):
        mean, std, score = simulate(prices, start, end, make())
        pieces.append(f"{name}={score:7.1f}({mean:.0f}/{std:.0f})")
        scores.append(score)
    print(f"{label:35s} " + " ".join(pieces)
          + f" min={min(scores):.1f}")


def main():
    prices = load_prices()
    for memory in (40, 60, 80, 120, 160):
        for speed in (.25, .5, 1.0):
            evaluate(
                prices,
                f"gate m{memory} speed{speed}",
                lambda m=memory, s=speed: factory(m, s, pair_scale=0),
            )
    for memory in (60, 80, 120):
        evaluate(
            prices,
            f"gate m{memory} + pairs1.5",
            lambda m=memory: factory(m, .5, pair_scale=1.5),
        )
    for memory in (40, 60, 80, 120, 160):
        for speed in (.25, .5, 1.0):
            evaluate(
                prices,
                f"tail gate m{memory} speed{speed}",
                lambda m=memory, s=speed: factory(
                    m, s, pair_scale=0, tail_gate=True
                ),
            )
    for memory in (40, 60, 80, 120, 160):
        evaluate(
            prices,
            f"tail gate m{memory} + pairs1.5",
            lambda m=memory: factory(
                m, .5, pair_scale=1.5, tail_gate=True
            ),
        )


if __name__ == "__main__":
    main()
