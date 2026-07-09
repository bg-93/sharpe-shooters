import numpy as np

# Algothon 2026: simple robust mean-reversion strategy
# Submit this file renamed to your actual team name, e.g. SharpeShooters.py

N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

    cur = prcSoFar[:, -1]
    cur = np.where(cur > 0, cur, 1.0)

    # Need enough history for the slow signal.
    if nt < 31:
        return np.zeros(nInst, dtype=int)

    # Short/medium mean reversion. Exclude today's price from the average so
    # the signal is: "today is stretched versus recent history".
    def zscore_to_past(lb):
        hist = prcSoFar[:, -lb - 1:-1]
        mu = hist.mean(axis=1)
        sig = hist.std(axis=1)
        sig = np.where(sig > 1e-8, sig, 1.0)
        return (mu - cur) / sig

    z8 = zscore_to_past(8)
    z30 = zscore_to_past(30)

    # Blend fast edge with a slower stabiliser. Positive means long, negative short.
    signal = 0.75 * z8 + 0.25 * z30

    # Turn signal into dollar exposure. Multiplier > 1 lets strong signals reach caps,
    # but tanh prevents unlimited sensitivity to one extreme move.
    target_dollars = LIMITS * np.tanh(0.85 * signal) * 2.0

    # Extra ALGO scaling is already naturally handled by its larger $100k cap and
    # lower commission in the official evaluator, so no special-case logic needed.
    target_shares = target_dollars / cur
    return target_shares.astype(int)
