#!/usr/bin/env python3
"""Research genuinely different, compact predictors on the released prices.

This file intentionally avoids pairs, price-spread mean reversion, and the
submitted dense ridge -> tanh recipe.  It contains an eval-exact simulator and
several small forecasting families:

* lag-covariance graph propagation (no matrix regression solve),
* latent-factor innovation propagation,
* nearest historical cross-section ("analog") forecasting,
* online mixtures of lag-covariance experts,
* covariance-aware portfolio construction.

Validation is chronological and walk-forward:
    early  [100, 300), middle [250, 500), late [500, 750)

Best standalone (fractional precision + validated ALGO forecast/hedge expert):
    early   mean 534.3 / std 1890.4 / score 508.9
    middle  mean 645.6 / std 1667.0 / score 628.8
    late    mean 634.7 / std 2000.1 / score 610.4

Run from the repository root:
    .venv/bin/python strategies/agent_novel_models.py --suite quick
    .venv/bin/python strategies/agent_novel_models.py --suite full
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
N = 51
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50)
COMM = np.array([0.00002] + [0.0001] * 50)
WINDOWS = (
    ("early", 100, 300),
    ("middle", 250, 500),
    ("late", 500, 750),
)


def load_prices() -> np.ndarray:
    return pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T.astype(float)


def score_pl(pl: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(pl))
    std = float(np.std(pl))
    if mean <= 0.0 or std < 1e-10:
        return mean, std, mean
    sr2 = 250.0 * mean * mean / (std * std)
    return mean, std, mean * sr2 / (sr2 + 1.0)


def simulate(
    prices: np.ndarray,
    start: int,
    end: int,
    target_fn,
    collect: bool = False,
):
    """Match eval.py's position, fee, integer-share, and PnL semantics."""
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
            new = np.clip((dollars / px).astype(int), -pos_limit, pos_limit)
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
    stats = score_pl(pnl)
    return (*stats, pnl) if collect else stats


def evaluate(label: str, factory, prices: np.ndarray, verbose: bool = True):
    rows = {}
    for name, start, end in WINDOWS:
        rows[name] = simulate(prices, start, end, factory())
    minimum = min(row[2] for row in rows.values())
    if verbose:
        pieces = [
            f"{name} {m:7.1f}/{sd:7.1f}/{sc:7.1f}"
            for name, (m, sd, sc) in rows.items()
        ]
        print(f"{label:48s} | " + " | ".join(pieces)
              + f" | min {minimum:7.1f}")
    return rows


def _load_existing_candidate():
    path = (
        ROOT / "strategies" / "next round testing code (research)"
        / "candidate_next.py"
    )
    spec = importlib.util.spec_from_file_location("_existing_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _standardize(a: np.ndarray):
    mu = a.mean(axis=0)
    sd = a.std(axis=0) + EPS
    return (a - mu) / sd, mu, sd


def _rank_gaussianish(a: np.ndarray) -> np.ndarray:
    """Columnwise centred empirical ranks, scaled to unit variance."""
    order = np.argsort(a, axis=0)
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(len(a), dtype=float)
    for j in range(a.shape[1]):
        ranks[order[:, j], j] = rows
    ranks = ranks / max(len(a) - 1, 1) - 0.5
    return ranks / (ranks.std(axis=0) + EPS)


@dataclass
class Allocation:
    """Turn a forecast into target dollars without the submitted tanh layer."""

    kind: str = "linear"
    gross: float = 0.80
    topk: int = 15
    covariance_shrink: float | None = None
    steepness: float = 2.0
    rank_power: float = 0.25
    gate: float = 0.0

    def __call__(
        self,
        pred: np.ndarray,
        target_sd: np.ndarray,
        target_corr: np.ndarray | None = None,
    ) -> np.ndarray:
        signal = pred / (target_sd + EPS)
        signal -= signal.mean()

        if self.covariance_shrink is not None and target_corr is not None:
            alpha = self.covariance_shrink
            corr = (1.0 - alpha) * target_corr + alpha * np.eye(N)
            signal = np.linalg.solve(corr, signal)
            signal -= signal.mean()

        scale = np.std(signal) + EPS
        z = signal / scale
        if self.kind == "linear":
            raw = np.clip(z / 2.0, -1.0, 1.0)
        elif self.kind == "arctan":
            raw = (2.0 / np.pi) * np.arctan(self.steepness * z)
        elif self.kind == "softsign":
            raw = z / (1.0 + np.abs(z))
        elif self.kind == "sign":
            raw = np.sign(z)
        elif self.kind == "gated_sign":
            raw = np.sign(z) * (np.abs(z) >= self.gate)
        elif self.kind == "topk":
            raw = np.zeros(N)
            chosen = np.argpartition(np.abs(z), -self.topk)[-self.topk:]
            raw[chosen] = np.sign(z[chosen])
        elif self.kind == "rank":
            ranks = np.argsort(np.argsort(z)).astype(float)
            raw = (ranks - ranks.mean()) / (ranks.max() / 2.0 + EPS)
        elif self.kind == "rank_power":
            magnitude_order = np.argsort(np.argsort(np.abs(z))).astype(float)
            magnitude = (magnitude_order + 1.0) / len(z)
            raw = np.sign(z) * magnitude ** self.rank_power
        else:
            raise ValueError(f"unknown allocation {self.kind}")
        return LIMITS * self.gross * raw


class PrecisionPropagation:
    """A covariance-network conditional forecast with structured whitening.

    The lag graph is the cross-correlation between today's shocks and
    tomorrow's shocks.  Instead of fitting 51 target regressions, this model
    whitens graph inputs using a deliberately structured covariance estimate:
    fractional precision, diagonal shrinkage, or a thresholded covariance
    graph.  This exposes a useful continuum between raw shock propagation and
    unstable full inverse-covariance propagation.
    """

    def __init__(
        self,
        precision: str = "fractional",
        power: float = 0.5,
        power_slope: float = 0.0,
        eigen_floor: float = 0.08,
        shrink: float = 0.5,
        covariance_edge: float = 0.1,
        stock_predictors: bool = False,
        coherent_index: bool = False,
        validation_window: int | None = None,
        quality_keep: int | None = None,
        quality_threshold: float | None = None,
        lookback: int | None = None,
        retrain: int = 25,
        allocation: Allocation | None = None,
    ):
        self.precision = precision
        self.power = power
        self.power_slope = power_slope
        self.eigen_floor = eigen_floor
        self.shrink = shrink
        self.covariance_edge = covariance_edge
        self.stock_predictors = stock_predictors
        self.coherent_index = coherent_index
        self.validation_window = validation_window
        self.quality_keep = quality_keep
        self.quality_threshold = quality_threshold
        self.lookback = lookback
        self.retrain = retrain
        self.allocation = allocation or Allocation("rank", gross=1.0)
        self.last_fit = -1
        self.mapping = None

    def _mapping(self, zx: np.ndarray, zy: np.ndarray):
        covariance = zx.T @ zx / len(zx)
        cross = zx.T @ zy / len(zx)

        if self.precision == "fractional":
            eig, vec = np.linalg.eigh(covariance)
            # More history supports more aggressive precision whitening.
            # A zero slope gives the ordinary fixed fractional power.
            effective_power = np.clip(
                self.power + self.power_slope * len(zx), 0.0, 1.0
            )
            inv = np.maximum(eig, self.eigen_floor) ** (-effective_power)
            whitener = (vec * inv) @ vec.T
        elif self.precision == "shrink":
            regular = (
                (1.0 - self.shrink) * covariance
                + self.shrink * np.eye(covariance.shape[0])
            )
            whitener = np.linalg.inv(regular)
        elif self.precision == "threshold":
            sparse = covariance.copy()
            off_diag = ~np.eye(len(sparse), dtype=bool)
            sparse[off_diag & (np.abs(sparse) < self.covariance_edge)] = 0.0
            regular = (
                (1.0 - self.shrink) * sparse
                + self.shrink * np.eye(len(sparse))
            )
            whitener = np.linalg.inv(regular)
        else:
            whitener = np.eye(covariance.shape[0])
        return whitener @ cross

    def _fit(self, prices: np.ndarray):
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        if self.lookback:
            r = r[-(self.lookback + 1):]
        x = r[:-1, 1:] if self.stock_predictors else r[:-1]
        y = r[1:]

        self.quality_mask = np.ones(N)
        if self.validation_window and len(x) >= self.validation_window + 60:
            cut = len(x) - self.validation_window
            ztrain, train_mu, train_sd = _standardize(x[:cut])
            ytrain, train_ymu, train_ysd = _standardize(y[:cut])
            validation_mapping = self._mapping(ztrain, ytrain)
            zv = (x[cut:] - train_mu) / train_sd
            pv = zv @ validation_mapping
            yv = (y[cut:] - train_ymu) / train_ysd
            quality = np.zeros(N)
            for j in range(N):
                if pv[:, j].std() > EPS and yv[:, j].std() > EPS:
                    quality[j] = np.corrcoef(pv[:, j], yv[:, j])[0, 1]
            if self.quality_keep:
                chosen = np.argpartition(
                    quality, -self.quality_keep
                )[-self.quality_keep:]
                self.quality_mask[:] = 0.0
                self.quality_mask[chosen] = 1.0
            elif self.quality_threshold is not None:
                self.quality_mask = (
                    quality >= self.quality_threshold
                ).astype(float)

        zx, self.xmu, self.xsd = _standardize(x)
        zy, _, self.ysd = _standardize(y)
        self.mapping = self._mapping(zx, zy)
        self.target_sd = y.std(0) + EPS
        self.target_corr = np.corrcoef(y, rowvar=False)

    def raw_prediction(self, prices: np.ndarray):
        nt = prices.shape[1]
        if nt < 100:
            return np.zeros(N)
        if self.mapping is None or nt - self.last_fit >= self.retrain:
            self._fit(prices)
            self.last_fit = nt
        now = np.log(np.maximum(prices[:, -1], EPS)
                     / np.maximum(prices[:, -2], EPS))
        x = now[1:] if self.stock_predictors else now
        pred = (((x - self.xmu) / self.xsd) @ self.mapping) * self.ysd
        if self.coherent_index:
            normalized = prices[1:, -1] / prices[1:, 0]
            weights = normalized / (normalized.sum() + EPS)
            pred[0] = weights @ pred[1:]
        return pred * self.quality_mask

    def __call__(self, prices: np.ndarray):
        if prices.shape[1] < 100:
            return np.zeros(N)
        pred = self.raw_prediction(prices)
        return self.allocation(pred, self.target_sd, self.target_corr)


class PrecisionEnsemble:
    """Average several structural whitening views before taking direction."""

    def __init__(
        self,
        powers=(0.55, 0.7, 0.85),
        lookbacks=(None,),
        allocation: Allocation | None = None,
    ):
        self.models = [
            PrecisionPropagation(
                power=power,
                lookback=lookback,
                allocation=Allocation("linear"),
            )
            for power in powers
            for lookback in lookbacks
        ]
        self.allocation = allocation or Allocation("sign", gross=1.0)

    def __call__(self, prices: np.ndarray):
        if prices.shape[1] < 100:
            return np.zeros(N)
        normalized = []
        for model in self.models:
            pred = model.raw_prediction(prices)
            z = pred / model.target_sd
            z = (z - z.mean()) / (z.std() + EPS)
            normalized.append(z)
        pred = np.mean(normalized, axis=0)
        reference = self.models[0]
        return self.allocation(
            pred, np.ones(N), reference.target_corr
        )


class FactorHedgedPrecision:
    """Blend the graph's ALGO forecast with an explicit stock-book hedge."""

    def __init__(
        self,
        hedge_fraction: float = 1.0,
        hedge_kind: str = "beta",
        **model_kwargs,
    ):
        self.model = PrecisionPropagation(
            allocation=Allocation("sign", gross=1.0), **model_kwargs
        )
        self.hedge_fraction = hedge_fraction
        self.hedge_kind = hedge_kind

    def __call__(self, prices: np.ndarray):
        dollars = self.model(prices)
        returns = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        if len(returns) < 60:
            return dollars
        if self.hedge_kind == "dollar":
            hedge = -np.sum(dollars[1:])
        else:
            covariance = np.cov(returns, rowvar=False)
            beta = covariance[:, 0] / (covariance[0, 0] + EPS)
            hedge = -np.sum(dollars[1:] * beta[1:]) / (beta[0] + EPS)
        dollars[0] = np.clip(
            (1.0 - self.hedge_fraction) * dollars[0]
            + self.hedge_fraction * hedge,
            -LIMITS[0],
            LIMITS[0],
        )
        return dollars


class ValidatedAlgoChoice:
    """Walk-forward expert choice: forecast ALGO or hedge the stock book."""

    def __init__(
        self,
        validation: int = 60,
        soft: float = 0.0,
        power: float = 0.70,
        eigen_floor: float = 0.05,
        retrain: int = 25,
    ):
        self.validation = validation
        self.soft = soft
        self.model = PrecisionPropagation(
            power=power,
            eigen_floor=eigen_floor,
            retrain=retrain,
            allocation=Allocation("sign", gross=1.0),
        )
        self.last_select = -1
        self.direct_weight = 1.0

    def _select(self, prices: np.ndarray):
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        cut = len(r) - self.validation
        if cut < 60:
            self.direct_weight = 1.0
            return
        xtrain, ytrain = r[:cut - 1], r[1:cut]
        zx, xmu, xsd = _standardize(xtrain)
        zy, _, ysd = _standardize(ytrain)
        mapping = self.model._mapping(zx, zy)
        xv = (r[cut - 1:-1] - xmu) / xsd
        prediction = xv @ mapping
        signal = prediction
        signal -= signal.mean(axis=1, keepdims=True)
        direct = LIMITS[0] * np.sign(signal[:, 0])

        covariance = np.cov(r[:cut], rowvar=False)
        beta = covariance[:, 0] / (covariance[0, 0] + EPS)
        stock_dollars = LIMITS[1:] * np.sign(signal[:, 1:])
        hedge = -(stock_dollars * beta[1:]).sum(1) / (beta[0] + EPS)
        hedge = np.clip(hedge, -LIMITS[0], LIMITS[0])
        realised = r[cut:, 0]
        direct_edge = np.mean(direct * realised)
        hedge_edge = np.mean(hedge * realised)
        if self.soft > 0:
            gap = (direct_edge - hedge_edge) / (
                np.std((direct - hedge) * realised) / np.sqrt(len(realised))
                + EPS
            )
            self.direct_weight = 1.0 / (1.0 + np.exp(-gap / self.soft))
        else:
            self.direct_weight = float(direct_edge >= hedge_edge)

    def __call__(self, prices: np.ndarray):
        nt = prices.shape[1]
        if nt < 100:
            return np.zeros(N)
        if nt - self.last_select >= self.model.retrain:
            self._select(prices)
            self.last_select = nt
        dollars = self.model(prices)
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        covariance = np.cov(r, rowvar=False)
        beta = covariance[:, 0] / (covariance[0, 0] + EPS)
        hedge = -np.sum(dollars[1:] * beta[1:]) / (beta[0] + EPS)
        hedge = np.clip(hedge, -LIMITS[0], LIMITS[0])
        dollars[0] = (
            self.direct_weight * dollars[0]
            + (1.0 - self.direct_weight) * hedge
        )
        return dollars


class LagCovarianceNetwork:
    """Propagate current shocks through a lagged correlation network.

    Unlike ridge, this does not invert X'X or solve a coefficient regression.
    It treats each standardized return as a node shock and the lagged
    cross-correlation matrix as a directed weighted graph.
    """

    def __init__(
        self,
        lookback: int | None = None,
        retrain: int = 25,
        transform: str = "linear",
        edge_threshold: float = 0.0,
        stable_edges: bool = False,
        diffusion: float = 0.0,
        allocation: Allocation | None = None,
    ):
        self.lookback = lookback
        self.retrain = retrain
        self.transform = transform
        self.edge_threshold = edge_threshold
        self.stable_edges = stable_edges
        self.diffusion = diffusion
        self.allocation = allocation or Allocation()
        self.last_fit = -1
        self.graph = None

    def _xform_fit(self, x: np.ndarray):
        if self.transform == "linear":
            z, self.xmu, self.xsd = _standardize(x)
            return z
        if self.transform == "sign":
            self.xmu = np.zeros(x.shape[1])
            self.xsd = np.ones(x.shape[1])
            return np.sign(x)
        if self.transform == "rank":
            self.xmu, self.xsd = x.mean(0), x.std(0) + EPS
            return _rank_gaussianish(x)
        if self.transform == "clip":
            z, self.xmu, self.xsd = _standardize(x)
            return np.clip(z, -1.5, 1.5)
        raise ValueError(self.transform)

    def _xform_now(self, x: np.ndarray):
        if self.transform == "sign":
            return np.sign(x)
        z = (x - self.xmu) / self.xsd
        if self.transform == "clip":
            return np.clip(z, -1.5, 1.5)
        if self.transform == "rank":
            # Smooth out-of-sample approximation to centred empirical ranks.
            return np.clip(z, -2.5, 2.5)
        return z

    def _fit(self, prices: np.ndarray):
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        if self.lookback:
            r = r[-(self.lookback + 1):]
        x, y = r[:-1], r[1:]
        zx = self._xform_fit(x)
        zy, self.ymu, self.ysd = _standardize(y)
        self.target_sd = y.std(0) + EPS
        self.target_corr = np.corrcoef(y, rowvar=False)
        graph = zx.T @ zy / len(zx)

        if self.stable_edges and len(zx) >= 80:
            cut = len(zx) // 2
            c1 = zx[:cut].T @ zy[:cut] / cut
            c2 = zx[cut:].T @ zy[cut:] / (len(zx) - cut)
            graph *= (np.sign(c1) == np.sign(c2))

        if self.edge_threshold > 0:
            graph *= np.abs(graph) >= self.edge_threshold

        if self.diffusion:
            graph = graph + self.diffusion * (graph @ graph)
        self.graph = graph

    def raw_prediction(self, prices: np.ndarray):
        nt = prices.shape[1]
        if self.graph is None or nt - self.last_fit >= self.retrain:
            self._fit(prices)
            self.last_fit = nt
        r_now = np.log(np.maximum(prices[:, -1], EPS)
                       / np.maximum(prices[:, -2], EPS))
        return (self._xform_now(r_now) @ self.graph) * self.ysd

    def __call__(self, prices: np.ndarray):
        if prices.shape[1] < 100:
            return np.zeros(N)
        pred = self.raw_prediction(prices)
        return self.allocation(pred, self.target_sd, self.target_corr)


class FactorInnovationNetwork:
    """A small state-space approximation based on latent return innovations.

    PCA estimates the contemporaneous state.  The common state is stripped
    from both today's and tomorrow's standardized returns, and only the
    directed covariance of the remaining innovations is propagated.
    """

    def __init__(
        self,
        factors: int = 3,
        lookback: int | None = None,
        retrain: int = 25,
        include_factor_forecast: float = 0.0,
        allocation: Allocation | None = None,
    ):
        self.factors = factors
        self.lookback = lookback
        self.retrain = retrain
        self.include_factor_forecast = include_factor_forecast
        self.allocation = allocation or Allocation()
        self.last_fit = -1
        self.graph = None

    def _fit(self, prices: np.ndarray):
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        if self.lookback:
            r = r[-(self.lookback + 1):]
        x, y = r[:-1], r[1:]
        zx, self.xmu, self.xsd = _standardize(x)
        zy, self.ymu, self.ysd = _standardize(y)
        _, _, vt = np.linalg.svd(np.vstack((zx, zy)), full_matrices=False)
        self.load = vt[:self.factors].T
        projection = self.load @ self.load.T
        ix = zx - zx @ projection
        iy = zy - zy @ projection
        self.graph = ix.T @ iy / len(ix)
        fx, fy = zx @ self.load, zy @ self.load
        self.factor_graph = np.linalg.solve(
            fx.T @ fx + 20.0 * np.eye(self.factors), fx.T @ fy
        )
        self.target_sd = y.std(0) + EPS
        self.target_corr = np.corrcoef(y, rowvar=False)

    def __call__(self, prices: np.ndarray):
        nt = prices.shape[1]
        if nt < 100:
            return np.zeros(N)
        if self.graph is None or nt - self.last_fit >= self.retrain:
            self._fit(prices)
            self.last_fit = nt
        now = np.log(np.maximum(prices[:, -1], EPS)
                     / np.maximum(prices[:, -2], EPS))
        z = (now - self.xmu) / self.xsd
        factor = z @ self.load
        innovation = z - factor @ self.load.T
        pred_z = innovation @ self.graph
        if self.include_factor_forecast:
            pred_z += self.include_factor_forecast * (
                (factor @ self.factor_graph) @ self.load.T
            )
        pred = pred_z * self.ysd
        return self.allocation(pred, self.target_sd, self.target_corr)


class AnalogForecaster:
    """Kernel/nearest-neighbour forecast from historical market states."""

    def __init__(
        self,
        components: int = 10,
        neighbours: int = 25,
        weighting: str = "rank",
        lookback: int | None = None,
        retrain: int = 25,
        allocation: Allocation | None = None,
    ):
        self.components = components
        self.neighbours = neighbours
        self.weighting = weighting
        self.lookback = lookback
        self.retrain = retrain
        self.allocation = allocation or Allocation()
        self.last_fit = -1
        self.x = None

    def _fit(self, prices: np.ndarray):
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1).T
        if self.lookback:
            r = r[-(self.lookback + 1):]
        x, self.y = r[:-1], r[1:]
        z, self.xmu, self.xsd = _standardize(x)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        self.load = vt[:self.components].T
        self.x = z @ self.load
        self.x /= np.linalg.norm(self.x, axis=1, keepdims=True) + EPS
        self.target_sd = self.y.std(0) + EPS
        self.target_corr = np.corrcoef(self.y, rowvar=False)

    def __call__(self, prices: np.ndarray):
        nt = prices.shape[1]
        if nt < 100:
            return np.zeros(N)
        if self.x is None or nt - self.last_fit >= self.retrain:
            self._fit(prices)
            self.last_fit = nt
        now = np.log(np.maximum(prices[:, -1], EPS)
                     / np.maximum(prices[:, -2], EPS))
        q = ((now - self.xmu) / self.xsd) @ self.load
        q /= np.linalg.norm(q) + EPS
        similarity = self.x @ q
        k = min(self.neighbours, len(similarity))
        chosen = np.argpartition(similarity, -k)[-k:]
        if self.weighting == "uniform":
            weights = np.ones(k)
        elif self.weighting == "positive":
            weights = np.maximum(similarity[chosen], 0.0) ** 2
        else:
            order = np.argsort(np.argsort(similarity[chosen])).astype(float)
            weights = (order + 1.0) ** 2
        weights /= weights.sum() + EPS
        pred = weights @ self.y[chosen]
        return self.allocation(pred, self.target_sd, self.target_corr)


class OnlineNetworkExperts:
    """Combine short/medium/expanding graph forecasts by realised IC."""

    def __init__(
        self,
        memories=(100, 250, None),
        score_memory: int = 60,
        temperature: float = 8.0,
        allocation: Allocation | None = None,
    ):
        self.models = [
            LagCovarianceNetwork(
                lookback=m,
                retrain=25,
                allocation=Allocation(kind="linear"),
            )
            for m in memories
        ]
        self.score_memory = score_memory
        self.temperature = temperature
        self.allocation = allocation or Allocation()
        self.pred_history = [[] for _ in self.models]
        self.real_history = []
        self.pending = None
        self.last_nt = -1

    def __call__(self, prices: np.ndarray):
        nt = prices.shape[1]
        if nt <= self.last_nt:
            self.pred_history = [[] for _ in self.models]
            self.real_history = []
            self.pending = None
        self.last_nt = nt
        if nt < 100:
            return np.zeros(N)

        if self.pending is not None:
            realised = np.log(np.maximum(prices[:, -1], EPS)
                              / np.maximum(prices[:, -2], EPS))
            self.real_history.append(realised)
            for history, pred in zip(self.pred_history, self.pending):
                history.append(pred)

        predictions = [m.raw_prediction(prices) for m in self.models]
        self.pending = [p.copy() for p in predictions]
        if len(self.real_history) < 30:
            weights = np.ones(len(self.models)) / len(self.models)
        else:
            realised = np.asarray(self.real_history[-self.score_memory:])
            quality = []
            for history in self.pred_history:
                pred = np.asarray(history[-len(realised):])
                daily_ic = []
                for p, y in zip(pred, realised):
                    if p.std() > EPS and y.std() > EPS:
                        daily_ic.append(np.corrcoef(p, y)[0, 1])
                quality.append(np.mean(daily_ic) if daily_ic else 0.0)
            quality = np.asarray(quality)
            logits = self.temperature * (quality - quality.max())
            weights = np.exp(logits)
            weights /= weights.sum()

        pred = sum(w * p for w, p in zip(weights, predictions))
        target_sd = self.models[-1].target_sd
        target_corr = self.models[-1].target_corr
        return self.allocation(pred, target_sd, target_corr)


def make_network(**kwargs):
    return lambda: LagCovarianceNetwork(**kwargs)


def quick_suite(prices: np.ndarray):
    configs = [
        (
            "network expanding linear alloc",
            lambda: LagCovarianceNetwork(
                allocation=Allocation("linear", gross=0.8)
            ),
        ),
        (
            "network rolling250 linear alloc",
            lambda: LagCovarianceNetwork(
                lookback=250, allocation=Allocation("linear", gross=0.8)
            ),
        ),
        (
            "network stable edges |corr|>.04",
            lambda: LagCovarianceNetwork(
                stable_edges=True,
                edge_threshold=0.04,
                allocation=Allocation("linear", gross=0.8),
            ),
        ),
        (
            "innovation factors=3",
            lambda: FactorInnovationNetwork(
                factors=3, allocation=Allocation("linear", gross=0.8)
            ),
        ),
        (
            "analog PCA10 k25",
            lambda: AnalogForecaster(
                components=10,
                neighbours=25,
                allocation=Allocation("linear", gross=0.8),
            ),
        ),
        (
            "online graph experts",
            lambda: OnlineNetworkExperts(
                allocation=Allocation("linear", gross=0.8)
            ),
        ),
        (
            "network covariance-aware shrink=.7",
            lambda: LagCovarianceNetwork(
                allocation=Allocation(
                    "linear", gross=0.8, covariance_shrink=0.7
                )
            ),
        ),
    ]
    for label, factory in configs:
        evaluate(label, factory, prices)


def full_suite(prices: np.ndarray):
    print("== lag-covariance graph ==")
    for lookback in (None, 150, 250, 400):
        for transform in ("linear", "sign", "clip", "rank"):
            evaluate(
                f"graph lb={lookback} x={transform}",
                lambda lb=lookback, tr=transform: LagCovarianceNetwork(
                    lookback=lb,
                    transform=tr,
                    allocation=Allocation("linear", gross=0.8),
                ),
                prices,
            )

    print("\n== graph structure / propagation ==")
    for threshold in (0.02, 0.04, 0.06, 0.08):
        for stable in (False, True):
            evaluate(
                f"graph threshold={threshold} stable={stable}",
                lambda th=threshold, st=stable: LagCovarianceNetwork(
                    edge_threshold=th,
                    stable_edges=st,
                    allocation=Allocation("linear", gross=0.8),
                ),
                prices,
            )
    for diffusion in (-0.25, 0.25, 0.5):
        evaluate(
            f"graph diffusion={diffusion}",
            lambda d=diffusion: LagCovarianceNetwork(
                diffusion=d, allocation=Allocation("linear", gross=0.8)
            ),
            prices,
        )

    print("\n== state-space factor innovations ==")
    for factors in (1, 2, 3, 5, 8, 12):
        for factor_weight in (0.0, 0.25, 1.0):
            evaluate(
                f"innov k={factors} factorForecast={factor_weight}",
                lambda k=factors, fw=factor_weight: FactorInnovationNetwork(
                    factors=k,
                    include_factor_forecast=fw,
                    allocation=Allocation("linear", gross=0.8),
                ),
                prices,
            )

    print("\n== historical analogs ==")
    for components in (5, 10, 20, 35):
        for neighbours in (5, 15, 30, 60):
            evaluate(
                f"analog components={components} k={neighbours}",
                lambda c=components, k=neighbours: AnalogForecaster(
                    components=c,
                    neighbours=k,
                    allocation=Allocation("linear", gross=0.8),
                ),
                prices,
            )

    print("\n== allocation geometry on the graph forecast ==")
    for kind in ("linear", "softsign", "rank", "topk", "sign"):
        for gross in (0.5, 0.8, 1.0):
            evaluate(
                f"graph alloc={kind} gross={gross}",
                lambda kind=kind, gross=gross: LagCovarianceNetwork(
                    allocation=Allocation(kind, gross=gross, topk=15)
                ),
                prices,
            )
    for shrink in (0.3, 0.5, 0.7, 0.85, 0.95):
        evaluate(
            f"graph precision allocation shrink={shrink}",
            lambda sh=shrink: LagCovarianceNetwork(
                allocation=Allocation(
                    "linear", gross=0.8, covariance_shrink=sh
                )
            ),
            prices,
        )

    print("\n== adaptive graph-memory experts ==")
    for score_memory in (30, 60, 120):
        for temperature in (2.0, 8.0, 20.0):
            evaluate(
                f"experts memory={score_memory} temp={temperature}",
                lambda sm=score_memory, tp=temperature: OnlineNetworkExperts(
                    score_memory=sm,
                    temperature=tp,
                    allocation=Allocation("linear", gross=0.8),
                ),
                prices,
            )


def precision_suite(prices: np.ndarray):
    print("== fractional precision lag networks ==")
    for power in (0.0, 0.25, 0.5, 0.75, 1.0):
        for allocation in ("rank", "linear", "arctan", "sign"):
            evaluate(
                f"fractional power={power} alloc={allocation}",
                lambda pw=power, al=allocation: PrecisionPropagation(
                    precision="fractional",
                    power=pw,
                    allocation=Allocation(al, gross=1.0, steepness=2.0),
                ),
                prices,
            )

    print("\n== diagonal-shrunk precision lag networks ==")
    for shrink in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85):
        for stock_only in (False, True):
            evaluate(
                f"shrink={shrink} stockX={stock_only}",
                lambda sh=shrink, so=stock_only: PrecisionPropagation(
                    precision="shrink",
                    shrink=sh,
                    stock_predictors=so,
                    allocation=Allocation("rank", gross=1.0),
                ),
                prices,
            )

    print("\n== covariance graph precision ==")
    for edge in (0.05, 0.1, 0.15, 0.2):
        for shrink in (0.25, 0.5, 0.75):
            evaluate(
                f"threshold precision edge={edge} shrink={shrink}",
                lambda ed=edge, sh=shrink: PrecisionPropagation(
                    precision="threshold",
                    covariance_edge=ed,
                    shrink=sh,
                    allocation=Allocation("rank", gross=1.0),
                ),
                prices,
            )

    print("\n== exact-index forecast coherence constraint ==")
    for power in (0.25, 0.5, 0.75):
        for stocks in (False, True):
            for coherent in (False, True):
                evaluate(
                    f"power={power} stockX={stocks} coherent={coherent}",
                    lambda pw=power, st=stocks, co=coherent:
                        PrecisionPropagation(
                            precision="fractional",
                            power=pw,
                            stock_predictors=st,
                            coherent_index=co,
                            allocation=Allocation("rank", gross=1.0),
                        ),
                    prices,
                )


def tune_precision_suite(prices: np.ndarray):
    print("== fractional-power plateau, full sign allocation ==")
    for power in np.arange(0.50, 0.951, 0.05):
        evaluate(
            f"power={power:.2f}",
            lambda pw=float(power): PrecisionPropagation(
                power=pw, allocation=Allocation("sign", gross=1.0)
            ),
            prices,
        )

    print("\n== eigenvalue floor / history / refit ==")
    for floor in (0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35):
        evaluate(
            f"floor={floor}",
            lambda fl=floor: PrecisionPropagation(
                power=0.75,
                eigen_floor=fl,
                allocation=Allocation("sign", gross=1.0),
            ),
            prices,
        )
    for lookback in (150, 200, 250, 300, 400, 500, None):
        evaluate(
            f"lookback={lookback}",
            lambda lb=lookback: PrecisionPropagation(
                power=0.75,
                lookback=lb,
                allocation=Allocation("sign", gross=1.0),
            ),
            prices,
        )
    for retrain in (1, 5, 10, 25, 50, 100):
        evaluate(
            f"retrain={retrain}",
            lambda rt=retrain: PrecisionPropagation(
                power=0.75,
                retrain=rt,
                allocation=Allocation("sign", gross=1.0),
            ),
            prices,
        )

    print("\n== smooth rank-confidence continuum ==")
    for power in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        evaluate(
            f"rank magnitude power={power}",
            lambda pw=power: PrecisionPropagation(
                power=0.75,
                allocation=Allocation(
                    "rank_power", gross=1.0, rank_power=pw
                ),
            ),
            prices,
        )
    for gate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        evaluate(
            f"standardised confidence gate={gate}",
            lambda gt=gate: PrecisionPropagation(
                power=0.75,
                allocation=Allocation("gated_sign", gross=1.0, gate=gt),
            ),
            prices,
        )
    for topk in (20, 25, 30, 35, 40, 45, 50):
        evaluate(
            f"top forecasts k={topk}",
            lambda tk=topk: PrecisionPropagation(
                power=0.75,
                allocation=Allocation("topk", gross=1.0, topk=tk),
            ),
            prices,
        )

    print("\n== structural ALGO identity variants ==")
    for stocks in (False, True):
        for coherent in (False, True):
            evaluate(
                f"stock predictors={stocks} coherent index={coherent}",
                lambda st=stocks, co=coherent: PrecisionPropagation(
                    power=0.75,
                    stock_predictors=st,
                    coherent_index=co,
                    allocation=Allocation("sign", gross=1.0),
                ),
                prices,
            )


def adaptive_precision_suite(prices: np.ndarray):
    print("== sample-size-adaptive fractional precision ==")
    schedules = (
        (0.50, 0.0003),
        (0.50, 0.0004),
        (0.50, 0.0005),
        (0.55, 0.00025),
        (0.55, 0.00035),
        (0.55, 0.00045),
        (0.60, 0.0002),
        (0.60, 0.0003),
        (0.60, 0.0004),
        (0.65, 0.0002),
        (0.65, 0.0003),
    )
    for base, slope in schedules:
        evaluate(
            f"adaptive power={base}+{slope}n",
            lambda b=base, s=slope: PrecisionPropagation(
                power=b,
                power_slope=s,
                eigen_floor=0.05,
                allocation=Allocation("sign", gross=1.0),
            ),
            prices,
        )

    print("\n== structural precision ensembles ==")
    power_sets = (
        (0.55, 0.65, 0.75),
        (0.6, 0.7, 0.8),
        (0.65, 0.75, 0.85),
        (0.55, 0.7, 0.85),
        (0.5, 0.7, 0.9),
    )
    for powers in power_sets:
        evaluate(
            f"power ensemble {powers}",
            lambda ps=powers: PrecisionEnsemble(
                powers=ps, allocation=Allocation("sign", gross=1.0)
            ),
            prices,
        )
    for lookbacks in ((None, 400), (None, 500), (400, 500, None)):
        evaluate(
            f"history ensemble {lookbacks}",
            lambda lbs=lookbacks: PrecisionEnsemble(
                powers=(0.7,),
                lookbacks=lbs,
                allocation=Allocation("sign", gross=1.0),
            ),
            prices,
        )


def quality_suite(prices: np.ndarray):
    print("== chronological target-reliability filters ==")
    for validation in (40, 60, 80, 120):
        for keep in (25, 30, 35, 40, 45):
            evaluate(
                f"validation={validation} keep={keep}",
                lambda va=validation, ke=keep: PrecisionPropagation(
                    power=0.70,
                    eigen_floor=0.05,
                    validation_window=va,
                    quality_keep=ke,
                    allocation=Allocation("sign", gross=1.0),
                ),
                prices,
            )
    for validation in (40, 60, 80, 120):
        for threshold in (-0.05, 0.0, 0.025, 0.05):
            evaluate(
                f"validation={validation} quality>={threshold}",
                lambda va=validation, th=threshold: PrecisionPropagation(
                    power=0.70,
                    eigen_floor=0.05,
                    validation_window=va,
                    quality_threshold=th,
                    allocation=Allocation("sign", gross=1.0),
                ),
                prices,
            )


def hedge_suite(prices: np.ndarray):
    print("== explicit broad-factor hedge ==")
    evaluate(
        "unhedged power=.70",
        lambda: PrecisionPropagation(
            power=0.70,
            allocation=Allocation("sign", gross=1.0),
        ),
        prices,
    )
    for kind in ("beta", "dollar"):
        for fraction in (0.25, 0.5, 0.75, 1.0):
            evaluate(
                f"{kind} hedge fraction={fraction}",
                lambda k=kind, f=fraction: FactorHedgedPrecision(
                    power=0.70,
                    hedge_kind=k,
                    hedge_fraction=f,
                ),
                prices,
            )
    print("\n== walk-forward choice between ALGO forecast and hedge ==")
    for validation in (40, 60, 80, 120, 160):
        for soft in (0.0, 0.5, 1.0, 2.0):
            evaluate(
                f"ALGO expert validation={validation} soft={soft}",
                lambda va=validation, so=soft: ValidatedAlgoChoice(
                    validation=va, soft=so
                ),
                prices,
            )


def final_candidate_suite(prices: np.ndarray):
    print("== precision graph + walk-forward ALGO expert ==")
    for validation in (40, 80, 120):
        for power in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            evaluate(
                f"validation={validation} power={power}",
                lambda va=validation, pw=power: ValidatedAlgoChoice(
                    validation=va,
                    power=pw,
                    eigen_floor=0.05,
                ),
                prices,
            )
    print("\n== local structural sensitivity ==")
    for floor in (0.01, 0.03, 0.05, 0.08, 0.12):
        for retrain in (10, 25, 50):
            evaluate(
                f"floor={floor} retrain={retrain}",
                lambda fl=floor, rt=retrain: ValidatedAlgoChoice(
                    validation=80,
                    power=0.70,
                    eigen_floor=fl,
                    retrain=rt,
                ),
                prices,
            )


def comparison_suite(prices: np.ndarray):
    existing = _load_existing_candidate()

    def make_existing():
        existing.reset_state()
        return lambda h: existing.getMyPosition(h) * h[:, -1]

    def make_novel():
        return ValidatedAlgoChoice(
            validation=80, power=0.70, eigen_floor=0.05
        )

    evaluate("existing pairs+dense ridge candidate", make_existing, prices)
    evaluate("novel precision graph+ALGO expert", make_novel, prices)

    print("\n== PnL correlation and position-level blends ==")
    for name, start, end in WINDOWS:
        old = make_existing()
        novel = make_novel()
        _, _, _, old_pl = simulate(
            prices, start, end, old, collect=True
        )
        _, _, _, novel_pl = simulate(
            prices, start, end, novel, collect=True
        )
        print(
            f"{name:8s} pnl corr="
            f"{np.corrcoef(old_pl, novel_pl)[0, 1]:.3f}"
        )

    for novel_weight in (0.25, 0.5, 0.75):
        def convex_factory(weight=novel_weight):
            old = make_existing()
            novel = make_novel()
            return lambda h: (
                (1.0 - weight) * old(h) + weight * novel(h)
            )
        evaluate(
            f"convex blend novel weight={novel_weight}",
            convex_factory,
            prices,
        )

    for novel_scale in (
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0
    ):
        def additive_factory(scale=novel_scale):
            old = make_existing()
            novel = make_novel()
            return lambda h: np.clip(
                old(h) + scale * novel(h), -LIMITS, LIMITS
            )
        evaluate(
            f"additive clipped novel scale={novel_scale}",
            additive_factory,
            prices,
        )


# Submission-compatible entry point for the best genuinely novel standalone
# model.  The larger research CLI below is ignored when this module is imported.
_LIVE_MODEL = None
_LIVE_PREV_NT = -1


def reset_state():
    global _LIVE_MODEL, _LIVE_PREV_NT
    _LIVE_MODEL = ValidatedAlgoChoice(
        validation=80,
        power=0.70,
        eigen_floor=0.05,
        retrain=25,
    )
    _LIVE_PREV_NT = -1


def getMyPosition(prcSoFar):
    global _LIVE_MODEL, _LIVE_PREV_NT
    nt = prcSoFar.shape[1]
    if _LIVE_MODEL is None or nt <= _LIVE_PREV_NT:
        reset_state()
    _LIVE_PREV_NT = nt
    target_dollars = _LIVE_MODEL(prcSoFar)
    return (target_dollars / prcSoFar[:, -1]).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=(
            "quick", "full", "precision", "tune_precision",
            "adaptive_precision", "quality",
            "hedge",
            "candidate",
            "compare",
        ),
        default="quick",
    )
    args = parser.parse_args()
    prices = load_prices()
    if args.suite == "quick":
        quick_suite(prices)
    elif args.suite == "full":
        full_suite(prices)
    elif args.suite == "precision":
        precision_suite(prices)
    elif args.suite == "tune_precision":
        tune_precision_suite(prices)
    elif args.suite == "adaptive_precision":
        adaptive_precision_suite(prices)
    elif args.suite == "quality":
        quality_suite(prices)
    elif args.suite == "hedge":
        hedge_suite(prices)
    elif args.suite == "candidate":
        final_candidate_suite(prices)
    else:
        comparison_suite(prices)


if __name__ == "__main__":
    main()
