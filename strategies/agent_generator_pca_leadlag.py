#!/usr/bin/env python3
"""Adaptive spectral-filtered lead-lag research candidate.

The released returns have one almost-deterministic ALGO/index direction and
several weak, low-variance predictor directions.  Fitting the dense lag-one
map in all 51 directions spends parameters on those poorly estimated modes.
This model keeps the successful dense-ridge portfolio layer, but first removes
predictor PCs whose sample eigenvalue is below half the median eigenvalue.

The rule is sample-size adaptive: it retains 40 PCs with 120 observations,
44 around day 270, 48 around day 420, and 50 after roughly day 570.  It is
deliberately high-rank; retaining only 5--10 factors loses most of the
distributed lead-lag information.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

MIN_HISTORY = 120
RETRAIN = 50
EIGEN_CUTOFF = 0.50
RIDGE_LAMBDA = 300.0
TEMPERATURE = 0.35

PAIRS = (
    (7, 40, 0.7086),
    (25, 37, 0.9352),
    (1, 20, 0.9836),
    (13, 45, 1.0132),
    (33, 40, 0.2577),
    (10, 46, 1.0331),
    (33, 42, 0.8358),
    (31, 43, 0.9692),
    (18, 28, 0.5642),
    (41, 50, 0.4977),
    (8, 27, 1.0222),
    (18, 35, 0.8471),
    (37, 46, 0.4059),
    (36, 41, 0.9137),
    (35, 42, 0.9163),
)
PAIR_LEG = 9000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5


class AdaptiveSpectralLeadLag:
    """Walk-forward spectrally filtered predictor plus multivariate ridge."""

    def __init__(
        self,
        eigen_cutoff=EIGEN_CUTOFF,
        components=None,
        ridge_lambda=RIDGE_LAMBDA,
        retrain=RETRAIN,
        temperature=TEMPERATURE,
    ):
        self.eigen_cutoff = eigen_cutoff
        self.components = components
        self.ridge_lambda = ridge_lambda
        self.retrain = retrain
        self.temperature = temperature
        self.reset()

    def reset(self):
        self.loading = None
        self.coef = None
        self.x_mean = None
        self.x_std = None
        self.y_std = None
        self.last_fit = -1
        self.previous_nt = -1

    def _fit(self, prices):
        returns = np.diff(np.log(np.maximum(prices, EPS)), axis=1)
        x = returns[:, :-1].T
        y = returns[:, 1:].T

        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + EPS
        self.y_std = y.std(axis=0) + EPS
        x_standard = (x - self.x_mean) / self.x_std
        y_standard = y / self.y_std

        # Only the predictor space is compressed.  Every output instrument is
        # still forecast separately, preserving the distributed target edge.
        _, singular_values, vt = np.linalg.svd(
            x_standard, full_matrices=False
        )
        if self.components is None:
            eigenvalues = singular_values * singular_values
            keep = eigenvalues >= self.eigen_cutoff * np.median(eigenvalues)
            self.loading = vt[keep].T
        else:
            self.loading = vt[: self.components].T
        factors = x_standard @ self.loading
        self.coef = np.linalg.solve(
            factors.T @ factors
            + self.ridge_lambda * np.eye(self.loading.shape[1]),
            factors.T @ y_standard,
        )

    def target_dollars(self, prices):
        prices = np.asarray(prices, dtype=float)
        nt = prices.shape[1]
        if nt <= self.previous_nt:
            self.reset()
        self.previous_nt = nt

        if nt < MIN_HISTORY:
            return np.zeros(N_INST)

        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(prices)
            self.last_fit = nt

        returns = np.diff(np.log(np.maximum(prices, EPS)), axis=1)
        x_now = (returns[:, -1] - self.x_mean) / self.x_std
        signal = (x_now @ self.loading) @ self.coef
        signal -= signal.mean()
        signal /= signal.std() + EPS
        return LIMITS * np.tanh(signal / self.temperature)


# Backward-compatible research alias for earlier experiment imports.
NoiseFilteredLeadLag = AdaptiveSpectralLeadLag


_MODEL = AdaptiveSpectralLeadLag()


def reset_state():
    global _MODEL
    _MODEL = AdaptiveSpectralLeadLag()


def getMyPosition(prcSoFar):
    target = _MODEL.target_dollars(prcSoFar)
    return (target / prcSoFar[:, -1]).astype(int)


class FrozenPairsAdaptiveSpectral:
    """Existing frozen-pair FSM plus the adaptive spectral lead-lag sleeve."""

    def __init__(self, eigen_cutoff=EIGEN_CUTOFF):
        self.leadlag = AdaptiveSpectralLeadLag(eigen_cutoff=eigen_cutoff)
        self.reset()

    def reset(self):
        self.pair_positions = [0] * len(PAIRS)
        self.previous_nt = -1
        self.leadlag.reset()

    def target_dollars(self, prices):
        prices = np.asarray(prices, dtype=float)
        nt = prices.shape[1]
        if nt <= self.previous_nt:
            self.reset()
        self.previous_nt = nt

        target = np.zeros(N_INST)
        if nt > PAIR_ROLL + 1:
            log_prices = np.log(np.maximum(prices, EPS))
            for k, (left, right, gamma) in enumerate(PAIRS):
                spread = log_prices[left] - gamma * log_prices[right]
                history = spread[-PAIR_ROLL - 1 : -1]
                zscore = (
                    (spread[-1] - history.mean()) / (history.std() + EPS)
                )
                position = self.pair_positions[k]
                if position == 0:
                    if zscore > PAIR_ENTRY:
                        position = -1
                    elif zscore < -PAIR_ENTRY:
                        position = 1
                elif position == 1 and zscore > -PAIR_EXIT:
                    position = 0
                elif position == -1 and zscore < PAIR_EXIT:
                    position = 0
                self.pair_positions[k] = position
                if position:
                    target[left] += position * PAIR_LEG
                    target[right] -= position * gamma * PAIR_LEG

        target += self.leadlag.target_dollars(prices)
        return np.clip(target, -LIMITS, LIMITS)


_COMBINED_MODEL = FrozenPairsAdaptiveSpectral()


def reset_combined_state():
    global _COMBINED_MODEL
    _COMBINED_MODEL = FrozenPairsAdaptiveSpectral()


def getCombinedPosition(prcSoFar):
    target = _COMBINED_MODEL.target_dollars(prcSoFar)
    return (target / prcSoFar[:, -1]).astype(int)


def _research_main():
    """Reproduce the headline and non-overlapping validation tables."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backtesting"))
    from leadlag_research import load_prices, simulate

    prices = load_prices()
    headline = (
        ("early", 100, 300),
        ("middle", 250, 500),
        ("late", 500, 750),
    )
    chunks = (
        ("120-250", 120, 250),
        ("250-375", 250, 375),
        ("375-500", 375, 500),
        ("500-625", 500, 625),
        ("625-750", 625, 750),
    )

    configs = (
        ("dense ridge", None, 51, 400.0),
        ("fixed PCA-40", None, 40, 300.0),
        ("spectral .475", 0.475, None, 300.0),
        ("spectral .500", 0.500, None, 300.0),
        ("spectral .525", 0.525, None, 300.0),
    )
    for label, eigen_cutoff, components, ridge_lambda in configs:
        print(label)
        for window, start, end in headline:
            model = AdaptiveSpectralLeadLag(
                eigen_cutoff=eigen_cutoff,
                components=components,
                ridge_lambda=ridge_lambda,
            )
            mean, std, score = simulate(
                prices, start, end, model.target_dollars
            )
            print(
                f"  {window:8s} mean={mean:7.1f} "
                f"std={std:7.1f} score={score:7.1f}"
            )

    print("Adaptive spectral .500 non-overlapping chronological chunks")
    for window, start, end in chunks:
        model = AdaptiveSpectralLeadLag()
        mean, std, score = simulate(
            prices, start, end, model.target_dollars
        )
        print(
            f"  {window:8s} mean={mean:7.1f} "
            f"std={std:7.1f} score={score:7.1f}"
        )

    print("Frozen pairs + adaptive spectral .500")
    for window, start, end in headline + (("full", 1, 750),):
        model = FrozenPairsAdaptiveSpectral()
        mean, std, score = simulate(
            prices, start, end, model.target_dollars
        )
        print(
            f"  {window:8s} mean={mean:7.1f} "
            f"std={std:7.1f} score={score:7.1f}"
        )


if __name__ == "__main__":
    _research_main()
