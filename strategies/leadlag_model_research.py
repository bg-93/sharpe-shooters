#!/usr/bin/env python3
"""Compare forecasting engines under the successful demean+tanh portfolio layer.

Every model predicts next-day returns, refits walk-forward, and shares exactly
the same cross-sectional normalization and sizing.  This keeps the experiment
about forecast quality rather than portfolio construction.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtesting"))
from leadlag_research import simulate

N = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12
WINDOWS = (("early", 100, 300), ("middle", 250, 500), ("late", 500, 750))


def load_prices():
    return pd.read_csv(ROOT / "prices.txt", sep=r"\s+").values.T


class ForecastBook:
    def __init__(self, kind="ridge", lam=400.0, lags=1, lookback=None,
                 decay=None, rank=None, clip=None, temp=0.35, retrain=25,
                 stock_only=False):
        self.kind = kind
        self.lam = lam
        self.lags = lags
        self.lookback = lookback
        self.decay = decay
        self.rank = rank
        self.clip = clip
        self.temp = temp
        self.retrain = retrain
        self.stock_only = stock_only
        self.last_fit = -1
        self.coef = self.mu = self.sd = self.ysd = None
        self.load = None

    def _features(self, r):
        """Rows at t contain returns t, t-1, ... used to predict t+1."""
        rows = []
        for lag in range(self.lags):
            rows.append(r[:, self.lags - 1 - lag:r.shape[1] - 1 - lag].T)
        x = np.concatenate(rows, axis=1)
        if self.kind == "signed":
            x = np.concatenate((x, np.sign(x) * np.sqrt(np.abs(x))), axis=1)
        return x

    def _fit(self, p):
        first = 1 if self.stock_only else 0
        r = np.diff(np.log(np.maximum(p[first:], EPS)), axis=1)
        if self.lookback:
            r = r[:, -self.lookback - self.lags:]
        if self.clip:
            med = np.median(r, axis=1, keepdims=True)
            mad = np.median(np.abs(r - med), axis=1, keepdims=True) + EPS
            r = np.clip(r, med - self.clip * 1.4826 * mad,
                        med + self.clip * 1.4826 * mad)

        x = self._features(r)
        y = r[:, self.lags:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        z = (x - self.mu) / self.sd
        self.ysd = y.std(0) + EPS

        if self.kind == "reduced":
            # Supervised reduced-rank ridge: fit normally, then keep the
            # dominant output directions of its fitted-value matrix.
            base = np.linalg.solve(
                z.T @ z + self.lam * np.eye(z.shape[1]), z.T @ y
            )
            fitted = z @ base
            _, _, vt = np.linalg.svd(fitted, full_matrices=False)
            self.load = vt[:self.rank].T
            self.coef = base @ self.load
        else:
            if self.decay:
                w = self.decay ** np.arange(len(z) - 1, -1, -1)
                zw = z * np.sqrt(w[:, None])
                yw = y * np.sqrt(w[:, None])
                self.coef = np.linalg.solve(
                    zw.T @ zw + self.lam * np.eye(z.shape[1]), zw.T @ yw
                )
            else:
                self.coef = np.linalg.solve(
                    z.T @ z + self.lam * np.eye(z.shape[1]), z.T @ y
                )

    def __call__(self, p):
        nt = p.shape[1]
        need = max(120, self.lags + 40)
        if nt < need:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt

        first = 1 if self.stock_only else 0
        r = np.diff(np.log(np.maximum(p[first:], EPS)), axis=1)
        now = np.concatenate([r[:, -1 - k] for k in range(self.lags)])
        if self.kind == "signed":
            now = np.concatenate((now, np.sign(now) * np.sqrt(np.abs(now))))
        z = (now - self.mu) / self.sd
        if self.kind == "reduced":
            pred = (z @ self.coef) @ self.load.T
        else:
            pred = z @ self.coef
        signal = pred / self.ysd
        signal -= signal.mean()
        signal /= signal.std() + EPS

        dollars = np.zeros(N)
        dollars[first:] = LIMITS[first:] * np.tanh(signal / self.temp)
        return dollars


class AveragedBooks:
    """Average already-normalized books; all members retain identical sizing."""

    def __init__(self, books, weights=None):
        self.books = books
        self.weights = (np.ones(len(books)) / len(books) if weights is None
                        else np.asarray(weights) / np.sum(weights))

    def __call__(self, p):
        out = sum(w * book(p) for w, book in zip(self.weights, self.books))
        return np.clip(out, -LIMITS, LIMITS)


class SparseRidge(ForecastBook):
    """Dense ridge screening followed by a per-target sparse ridge refit."""

    def __init__(self, keep=10, **kwargs):
        super().__init__(**kwargs)
        self.keep = keep

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        if self.lookback:
            r = r[:, -self.lookback - self.lags:]
        x, y = self._features(r), r[:, self.lags:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        z = (x - self.mu) / self.sd
        self.ysd = y.std(0) + EPS
        screen = np.linalg.solve(
            z.T @ z + self.lam * np.eye(z.shape[1]), z.T @ y
        )
        self.coef = np.zeros_like(screen)
        for target in range(N):
            selected = np.argpartition(
                np.abs(screen[:, target]), -self.keep
            )[-self.keep:]
            zs = z[:, selected]
            self.coef[selected, target] = np.linalg.solve(
                zs.T @ zs + self.lam * np.eye(self.keep),
                zs.T @ y[:, target],
            )


class FixedScaleRidge(ForecastBook):
    """Preserve day-to-day confidence instead of forcing signal std to one."""

    def __init__(self, calibration="median", **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration

    def _fit(self, p):
        super()._fit(p)
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        x = self._features(r)
        z = (x - self.mu) / self.sd
        fitted = (z @ self.coef) / self.ysd
        fitted -= fitted.mean(1, keepdims=True)
        daily_strength = fitted.std(1)
        self.fixed_scale = (
            np.median(daily_strength)
            if self.calibration == "median"
            else np.sqrt(np.mean(daily_strength ** 2))
        ) + EPS

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        now = np.concatenate([r[:, -1 - k] for k in range(self.lags)])
        pred = ((now - self.mu) / self.sd) @ self.coef
        signal = pred / self.ysd
        signal -= signal.mean()
        signal /= self.fixed_scale
        return LIMITS * np.tanh(signal / self.temp)


class PCAFactors(ForecastBook):
    """Unsupervised PCA compression of predictors, then ridge to all targets."""

    def __init__(self, components=10, **kwargs):
        super().__init__(**kwargs)
        self.components = components

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        x, y = r[:, :-1].T, r[:, 1:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        z = (x - self.mu) / self.sd
        self.ysd = y.std(0) + EPS
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        self.load = vt[:self.components].T
        f = z @ self.load
        self.coef = np.linalg.solve(
            f.T @ f + self.lam * np.eye(self.components), f.T @ y
        )

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        pred = (((r[:, -1] - self.mu) / self.sd) @ self.load) @ self.coef
        signal = pred / self.ysd
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


class PLSFactors(ForecastBook):
    """Supervised partial-least-squares factors for the return map."""

    def __init__(self, components=5, **kwargs):
        super().__init__(**kwargs)
        self.components = components
        self.model = None

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        x, y = r[:, :-1].T, r[:, 1:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        self.ysd = y.std(0) + EPS
        self.model = PLSRegression(
            n_components=self.components, scale=False, max_iter=500
        )
        self.model.fit((x - self.mu) / self.sd, y / self.ysd)
        self.coef = True

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.model is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        signal = self.model.predict(
            ((r[:, -1] - self.mu) / self.sd)[None, :]
        )[0]
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


class RegimeRidge(ForecastBook):
    """Two ridge maps selected by lagged market direction or dispersion."""

    def __init__(self, regime="market", **kwargs):
        super().__init__(**kwargs)
        self.regime = regime

    def _states(self, x):
        if self.regime == "market":
            return x[:, 0] >= 0
        dispersion = x[:, 1:].std(1)
        return dispersion >= np.median(dispersion)

    def _fit(self, p):
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        x, y = r[:, :-1].T, r[:, 1:].T
        self.mu, self.sd = x.mean(0), x.std(0) + EPS
        z = (x - self.mu) / self.sd
        self.ysd = y.std(0) + EPS
        states = self._states(x)
        self.coef = []
        for state in (False, True):
            zs, ys = z[states == state], y[states == state]
            self.coef.append(np.linalg.solve(
                zs.T @ zs + self.lam * np.eye(N), zs.T @ ys
            ))
        if self.regime == "dispersion":
            self.cut = np.median(x[:, 1:].std(1))

    def __call__(self, p):
        nt = p.shape[1]
        if nt < 120:
            return np.zeros(N)
        if self.coef is None or nt - self.last_fit >= self.retrain:
            self._fit(p)
            self.last_fit = nt
        r = np.diff(np.log(np.maximum(p, EPS)), axis=1)
        now = r[:, -1]
        state = (now[0] >= 0 if self.regime == "market"
                 else now[1:].std() >= self.cut)
        pred = ((now - self.mu) / self.sd) @ self.coef[int(state)]
        signal = pred / self.ysd
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


class RecursiveRidge:
    """Exponentially forgetting recursive least squares, updated every day."""

    def __init__(self, lam=400.0, forgetting=.9975, temp=.35):
        self.lam, self.forgetting, self.temp = lam, forgetting, temp
        self.prev_nt = -1
        self.p = self.w = self.mu = self.sd = self.ysd = None

    def __call__(self, prices):
        nt = prices.shape[1]
        if nt <= self.prev_nt:
            self.p = self.w = None
        self.prev_nt = nt
        if nt < 120:
            return np.zeros(N)
        r = np.diff(np.log(np.maximum(prices, EPS)), axis=1)
        if self.w is None:
            x, y = r[:, :-1].T, r[:, 1:].T
            self.mu, self.sd = x.mean(0), x.std(0) + EPS
            self.ysd = y.std(0) + EPS
            z = (x - self.mu) / self.sd
            self.p = np.linalg.inv(z.T @ z + self.lam * np.eye(N))
            self.w = self.p @ z.T @ y
        else:
            x = (r[:, -2] - self.mu) / self.sd
            y = r[:, -1]
            px = self.p @ x
            gain = px / (self.forgetting + x @ px)
            self.w += np.outer(gain, y - x @ self.w)
            self.p = (self.p - np.outer(gain, x @ self.p)) / self.forgetting
        pred = ((r[:, -1] - self.mu) / self.sd) @ self.w
        signal = pred / self.ysd
        signal = (signal - signal.mean()) / (signal.std() + EPS)
        return LIMITS * np.tanh(signal / self.temp)


def evaluate(label, factory, p):
    vals, bits = [], []
    for name, start, end in WINDOWS:
        m, sd, sc = simulate(p, start, end, factory())
        vals.append(sc)
        bits.append(f"{name} {m:7.1f}/{sd:7.1f}/{sc:7.1f}")
    print(f"{label:37s} | " + " | ".join(bits) + f" | min {min(vals):7.1f}")
    return vals


def main():
    p = load_prices()
    configs = [
        ("baseline ridge l400 t.35",
         lambda: ForecastBook(lam=400, retrain=50)),
        ("dense ridge temp .50",
         lambda: ForecastBook(lam=400, retrain=50, temp=.50)),
        ("dense ridge temp .75",
         lambda: ForecastBook(lam=400, retrain=50, temp=.75)),
        ("dense ridge temp 1.00",
         lambda: ForecastBook(lam=400, retrain=50, temp=1.00)),
        ("dense ridge temp 1.50",
         lambda: ForecastBook(lam=400, retrain=50, temp=1.50)),
        ("dense ridge temp 2.00",
         lambda: ForecastBook(lam=400, retrain=50, temp=2.00)),
        ("fixed-scale dynamic temp .35",
         lambda: FixedScaleRidge(lam=400, retrain=50, temp=.35)),
        ("fixed-scale dynamic temp .50",
         lambda: FixedScaleRidge(lam=400, retrain=50, temp=.50)),
        ("fixed-scale dynamic temp .75",
         lambda: FixedScaleRidge(lam=400, retrain=50, temp=.75)),
        ("fixed-scale RMS temp .50",
         lambda: FixedScaleRidge(
             lam=400, retrain=50, temp=.50, calibration="rms")),
        ("ridge l100 t.35", lambda: ForecastBook(lam=100)),
        ("ridge l20 t.35", lambda: ForecastBook(lam=20)),
        ("ridge l800 t.35", lambda: ForecastBook(lam=800)),
        ("ridge stock-only l100", lambda: ForecastBook(
            lam=100, stock_only=True)),
        ("ridge rolling250 l100", lambda: ForecastBook(
            lam=100, lookback=250)),
        ("ridge rolling400 l100", lambda: ForecastBook(
            lam=100, lookback=400)),
        ("EW ridge decay .995", lambda: ForecastBook(
            lam=100, decay=.995)),
        ("EW ridge decay .9975 l400", lambda: ForecastBook(
            lam=400, decay=.9975)),
        ("EW ridge decay .999 l400", lambda: ForecastBook(
            lam=400, decay=.999)),
        ("EW ridge decay .99", lambda: ForecastBook(
            lam=100, decay=.99)),
        ("EW ridge decay .98", lambda: ForecastBook(
            lam=100, decay=.98)),
        ("robust clip 4MAD", lambda: ForecastBook(
            lam=100, clip=4)),
        ("robust clip 3MAD", lambda: ForecastBook(
            lam=100, clip=3)),
        ("ridge refit daily", lambda: ForecastBook(
            lam=400, retrain=1)),
        ("ridge refit 10d", lambda: ForecastBook(
            lam=400, retrain=10)),
        ("ridge refit 25d", lambda: ForecastBook(
            lam=400, retrain=25)),
        ("ridge refit 100d", lambda: ForecastBook(
            lam=400, retrain=100)),
        ("multi-lag 2 l400", lambda: ForecastBook(
            lam=400, lags=2)),
        ("multi-lag 3 l400", lambda: ForecastBook(
            lam=400, lags=3)),
        ("multi-lag 5 l800", lambda: ForecastBook(
            lam=800, lags=5)),
        ("signed-sqrt nonlinear l400", lambda: ForecastBook(
            kind="signed", lam=400)),
        ("reduced-rank 10", lambda: ForecastBook(
            kind="reduced", lam=100, rank=10)),
        ("reduced-rank 20", lambda: ForecastBook(
            kind="reduced", lam=100, rank=20)),
        ("reduced-rank 30", lambda: ForecastBook(
            kind="reduced", lam=100, rank=30)),
        ("reduced-rank 40", lambda: ForecastBook(
            kind="reduced", lam=100, rank=40)),
        ("blend base + EW995", lambda: AveragedBooks([
            ForecastBook(lam=400, retrain=50),
            ForecastBook(lam=100, decay=.995),
        ])),
        ("blend base + reduced30", lambda: AveragedBooks([
            ForecastBook(lam=400, retrain=50),
            ForecastBook(kind="reduced", lam=100, rank=30),
        ])),
        ("blend base + stock-only", lambda: AveragedBooks([
            ForecastBook(lam=400, retrain=50),
            ForecastBook(lam=100, stock_only=True),
        ])),
        ("blend base/EW/reduced30", lambda: AveragedBooks([
            ForecastBook(lam=400, retrain=50),
            ForecastBook(lam=100, decay=.995),
            ForecastBook(kind="reduced", lam=100, rank=30),
        ])),
        ("blend 2base + EW + stock", lambda: AveragedBooks([
            ForecastBook(lam=400, retrain=50),
            ForecastBook(lam=100, decay=.995),
            ForecastBook(lam=100, stock_only=True),
        ], weights=[2, 1, 1])),
        ("sparse ridge keep 3", lambda: SparseRidge(
            keep=3, lam=100)),
        ("sparse ridge keep 5", lambda: SparseRidge(
            keep=5, lam=100)),
        ("sparse ridge keep 10", lambda: SparseRidge(
            keep=10, lam=100)),
        ("sparse ridge keep 20", lambda: SparseRidge(
            keep=20, lam=100)),
        ("PCA predictor factors 5", lambda: PCAFactors(
            components=5, lam=20)),
        ("PCA predictor factors 10", lambda: PCAFactors(
            components=10, lam=20)),
        ("PCA predictor factors 20", lambda: PCAFactors(
            components=20, lam=20)),
        ("PLS factors 3", lambda: PLSFactors(components=3)),
        ("PLS factors 5", lambda: PLSFactors(components=5)),
        ("PLS factors 10", lambda: PLSFactors(components=10)),
        ("regime ridge market sign", lambda: RegimeRidge(
            regime="market", lam=400)),
        ("regime ridge dispersion", lambda: RegimeRidge(
            regime="dispersion", lam=400)),
        ("recursive ridge forget .999", lambda: RecursiveRidge(
            forgetting=.999)),
        ("recursive ridge forget .9975", lambda: RecursiveRidge(
            forgetting=.9975)),
        ("recursive ridge forget .995", lambda: RecursiveRidge(
            forgetting=.995)),
        ("recursive ridge .9975 l100", lambda: RecursiveRidge(
            lam=100, forgetting=.9975)),
        ("recursive ridge .9975 l800", lambda: RecursiveRidge(
            lam=800, forgetting=.9975)),
        ("blend base + recursive", lambda: AveragedBooks([
            ForecastBook(lam=400, retrain=50),
            RecursiveRidge(forgetting=.9975),
        ])),
        ("blend recursive + PLS10", lambda: AveragedBooks([
            RecursiveRidge(forgetting=.9975),
            PLSFactors(components=10),
        ])),
    ]
    for label, factory in configs:
        evaluate(label, factory, p)


if __name__ == "__main__":
    main()
