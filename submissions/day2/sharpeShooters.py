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
