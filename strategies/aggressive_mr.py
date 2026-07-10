"""Aggressive full-limit mean-reversion book (score-maximising variant).

Score = mean(PL) * SR^2/(SR^2+1). At SR ~2.5 the penalty factor is ~0.86,
so past that point mean PnL dominates the score. This strategy therefore
runs every instrument at (nearly) its full dollar limit in the direction of
a blended short/medium-horizon mean-reversion z-score:

  - signal_i = 0.70 * z8_i + 0.30 * z60_i, where z_k is today's price vs the
    mean of the prior k days, in units of that window's std dev.
  - position = limit * tanh(2 * signal) * 4 -> saturates to +/- full limit
    for any |signal| >~ 0.15, i.e. effectively sign(signal) sizing. The
    backtester clips to the exact per-instrument dollar limit.
  - ALGO (inst 0) is traded on the same signal with its $100k limit and
    0.2bp commission; its sleeve adds ~25% of total mean PnL, so it is NOT
    used as a hedge.

Validated on two 250-day segments (days 250-499 and 190-439):
  ~ score 436 / 375 with SR ~2.6 / 2.2. Params sit in the middle of a flat
  plateau (fast=8, slow 45-90, fw 0.70-0.75), not at the best single cell.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100_000.0] + [10_000.0] * 50)

FAST_LB = 8
SLOW_LB = 60
FAST_WEIGHT = 0.70
SIGNAL_SCALE = 2.0
OVERDRIVE = 4.0
MIN_HISTORY = SLOW_LB + 2


def reset_state():
    pass


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_inst, nt = prcSoFar.shape
    if n_inst != N_INST or nt < MIN_HISTORY:
        return np.zeros(n_inst, dtype=int)

    cur = np.maximum(prcSoFar[:, -1], 1.0)

    def zscore_to_past(lb):
        hist = prcSoFar[:, -lb - 1:-1]
        mu = hist.mean(axis=1)
        sig = hist.std(axis=1)
        sig = np.where(sig > 1e-8, sig, 1.0)
        return (mu - cur) / sig

    signal = FAST_WEIGHT * zscore_to_past(FAST_LB) \
        + (1.0 - FAST_WEIGHT) * zscore_to_past(SLOW_LB)

    target_dollars = LIMITS * np.tanh(SIGNAL_SCALE * signal) * OVERDRIVE
    return (target_dollars / cur).astype(int)
