import numpy as np

# Pure instrument-by-instrument Kalman mean-reversion strategy.

N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# `kf_top10` settings from local experiments.
KALMAN_Q = 1e-4
KALMAN_R = 4e-3
SIGNAL_SCALE = 0.9
TOP_K = 10


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    if nInst != N_INST or nt < 35:
        return np.zeros(nInst, dtype=int)

    log_prices = np.log(np.maximum(prcSoFar, EPS))
    cur = np.maximum(prcSoFar[:, -1], 1.0)
    target_dollars = np.zeros(nInst, dtype=float)

    for i in range(nInst):
        obs = log_prices[i]
        state = obs[0]
        variance = 1.0
        innovations = np.zeros(nt - 1, dtype=float)

        for t in range(1, nt):
            variance += KALMAN_Q
            innovation = obs[t] - state
            gain = variance / (variance + KALMAN_R)
            state += gain * innovation
            variance *= 1.0 - gain
            innovations[t - 1] = innovation

        recent = innovations[-30:]
        sigma = recent.std()
        if sigma < 1e-8:
            continue

        zscore = -(innovations[-1] / sigma)
        bounded_signal = np.tanh(SIGNAL_SCALE * zscore)
        target_dollars[i] = LIMITS[i] * bounded_signal

    strongest = np.argsort(np.abs(target_dollars[1:]))[-TOP_K:] + 1
    filtered = np.zeros_like(target_dollars)
    filtered[0] = target_dollars[0]
    filtered[strongest] = target_dollars[strongest]

    target_shares = filtered / cur
    return target_shares.astype(int)
