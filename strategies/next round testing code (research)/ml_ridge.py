#!/usr/bin/env python3
"""Feature-augmented walk-forward ridge ("ML" sleeve).

Learns per-name next-day return predictions from a richer feature set
than the plain lead-lag model:

  features(t) = [ r(t) all 51 names          (lead-lag block)
                , pair spread z's            (learned pairs weights)
                , 40d basket deviation       (learned basket weights) ]

The ridge is refit every RETRAIN days on ALL history before t (walk
forward — no lookahead). The pair set for the spread features is
re-selected at each refit by adaptive_pairs.select_pairs, so nothing
about the pairs is hardcoded. Positions use the OOS-proven construction:
z = pred / resid_sd, cross-sectionally demeaned, tanh(z/TEMP) sizing.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import adaptive_pairs as ap

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12


def _pair_z_series(lpx, pairs, roll=60):
    """Rolling z of each pair spread; shape (len(pairs), nt). First
    `roll` columns are zero (insufficient window)."""
    nt = lpx.shape[1]
    out = np.zeros((len(pairs), nt))
    for k, (a, b, g) in enumerate(pairs):
        s = lpx[a] - g * lpx[b]
        if nt <= roll:
            continue
        w = sliding_window_view(s, roll)          # (nt-roll+1, roll)
        mu = w.mean(axis=1)
        sd = w.std(axis=1) + EPS
        # z at col t uses window ending at t-1 (matches live sleeve)
        out[k, roll:] = (s[roll:] - mu[:-1]) / sd[:-1]
    return out


def _basket_dev_series(lpx, lb=40):
    """Each name's lb-day log return minus ALGO's; shape (N_INST, nt)."""
    nt = lpx.shape[1]
    out = np.zeros((N_INST, nt))
    if nt <= lb:
        return out
    mom = lpx[:, lb:] - lpx[:, :-lb]
    out[:, lb:] = mom - mom[0]
    return out


class FeatureRidge:
    def __init__(self, lam=400.0, retrain=50, temp=0.35,
                 use_pairs=True, use_basket=True, basket_lb=40,
                 min_hist=250):
        self.lam = lam
        self.retrain = retrain
        self.temp = temp
        self.use_pairs = use_pairs
        self.use_basket = use_basket
        self.basket_lb = basket_lb
        self.min_hist = min_hist
        self.W = None
        self.mu = None
        self.sd = None
        self.resid_sd = None
        self.pairs = []
        self.last_fit = -1

    def _feature_matrix(self, lpx, r):
        """Rows t index return-columns of r; row t is known at the close
        that realises r[:, t] and predicts r[:, t+1]."""
        blocks = [r.T]                             # (nt-1, 51): r(:, t)
        if self.use_pairs and self.pairs:
            pz = _pair_z_series(lpx, self.pairs)   # (P, nt)
            blocks.append(pz[:, 1:].T)             # align to close of r t
        if self.use_basket:
            bd = _basket_dev_series(lpx, self.basket_lb)
            blocks.append(bd[:, 1:].T)
        return np.hstack(blocks)

    def target_dollars(self, prc):
        nt = prc.shape[1]
        if nt < self.min_hist:
            return np.zeros(N_INST)
        lpx = np.log(np.maximum(prc, EPS))
        r = np.diff(lpx, axis=1)

        if self.W is None or (nt - self.last_fit) >= self.retrain:
            if self.use_pairs:
                incumbents = tuple((a, b) for a, b, _ in self.pairs)
                self.pairs = ap.select_pairs(lpx, incumbents)
            F = self._feature_matrix(lpx, r)       # (nt-1, K)
            X = F[:-1]                             # predicts next return
            Y = r[:, 1:].T
            self.mu, self.sd = X.mean(0), X.std(0)
            self.sd = np.where(self.sd > 1e-12, self.sd, 1.0)
            Xs = (X - self.mu) / self.sd
            K = Xs.shape[1]
            self.W = np.linalg.solve(
                Xs.T @ Xs + self.lam * np.eye(K), Xs.T @ Y)
            self.resid_sd = np.maximum(Y.std(0), 1e-8)
            self.last_fit = nt

        # today's feature row (last row of the matrix)
        F_last = self._feature_matrix(lpx[:, -80:], r[:, -79:])[-1]
        x = (F_last - self.mu) / self.sd
        pred = x @ self.W
        z = pred / self.resid_sd
        z = z - z.mean()
        z = z / (z.std() + 1e-9)
        return LIMITS * np.tanh(z / self.temp)
