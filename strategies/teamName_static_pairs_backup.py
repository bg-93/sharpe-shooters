import numpy as np

'''
Previous live code kept commented out for reference.

import numpy as np


N_INST = 51
ALGO_IDX = 0
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50, dtype=float)

# Chosen from split testing with the exact local evaluator logic.
BETA_WINDOW = 55
ZSCORE_WINDOW = 40
ENTRY_Z = 1.75
EXIT_Z = 0.50
STOP_Z = 4.00
MAX_HOLD = 12
TOP_K = 8
STOCK_DOLLAR_FRACTION = 1.00
ALGO_HEDGE_CAP_FRACTION = 0.18
MIN_ABS_BETA = 0.05
MAX_ABS_BETA = 3.00
MIN_HISTORY = max(BETA_WINDOW + 2, ZSCORE_WINDOW + 2, 80)


class AlgoRelativeValueArb:
    ...


_LIVE_STRATEGY = AlgoRelativeValueArb()


def reset_state():
    _LIVE_STRATEGY.reset_state()


def getMyPosition(prcSoFar):
    return _LIVE_STRATEGY.get_position(prcSoFar)
'''

N_INST = 51
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50, dtype=float)

# Static pairs discovered offline from prices.txt and then frozen.
# Format: (left_instrument, right_instrument, beta, intercept)
PAIRS = [
    (0, 2, -0.1046, 5.0828),
    (0, 22, 0.1910, 3.5851),
    (6, 27, -0.9580, 7.0108),
    (14, 35, -0.5163, 6.7521),
    (21, 40, -0.2062, 5.7297),
]

LOOKBACK = 30
ENTRY_Z = 1.50
EXIT_Z = 0.50
STOP_Z = 4.00
MAX_HOLD = 15
TOP_K = 5
MIN_HISTORY = LOOKBACK + 2

_STATE = {
    "last_nt": 0,
    "direction": np.zeros(len(PAIRS), dtype=int),
    "hold_days": np.zeros(len(PAIRS), dtype=int),
}


def reset_state():
    _STATE["last_nt"] = 0
    _STATE["direction"].fill(0)
    _STATE["hold_days"].fill(0)


def getMyPosition(prcSoFar):
    prices = np.asarray(prcSoFar, dtype=float)
    n_inst, nt = prices.shape
    if n_inst != N_INST:
        reset_state()
        return np.zeros(n_inst, dtype=int)

    if nt < _STATE["last_nt"]:
        reset_state()
    _STATE["last_nt"] = nt

    if nt < MIN_HISTORY:
        return np.zeros(n_inst, dtype=int)

    log_prices = np.log(np.maximum(prcSoFar, EPS))
    desired_dollars = np.zeros(N_INST, dtype=float)
    ranked_signals = []

    for pair_idx, (left, right, beta, intercept) in enumerate(PAIRS):
        spread = log_prices[left] - (intercept + beta * log_prices[right])
        hist = spread[-LOOKBACK - 1 : -1]
        spread_std = float(np.std(hist))

        if spread_std < 1e-10:
            _STATE["direction"][pair_idx] = 0
            _STATE["hold_days"][pair_idx] = 0
            continue

        zscore = (float(spread[-1]) - float(np.mean(hist))) / spread_std
        direction = int(_STATE["direction"][pair_idx])
        hold_days = int(_STATE["hold_days"][pair_idx])

        if direction == 0:
            if zscore >= ENTRY_Z:
                direction, hold_days = -1, 1
            elif zscore <= -ENTRY_Z:
                direction, hold_days = 1, 1
        else:
            hold_days += 1
            if abs(zscore) <= EXIT_Z or abs(zscore) >= STOP_Z or hold_days >= MAX_HOLD:
                direction, hold_days = 0, 0

        _STATE["direction"][pair_idx] = direction
        _STATE["hold_days"][pair_idx] = hold_days

        if direction != 0:
            ranked_signals.append((abs(zscore), left, right, beta, direction))

    ranked_signals.sort(key=lambda row: row[0], reverse=True)

    for _, left, right, beta, direction in ranked_signals[:TOP_K]:
        pair_cap = min(LIMITS[left], LIMITS[right] / max(abs(beta), 1e-6))
        desired_dollars[left] += direction * pair_cap
        desired_dollars[right] -= direction * pair_cap * beta

    used = np.where(np.abs(desired_dollars) > 1e-8)[0]
    if used.size > 0:
        scale = float(
            np.min(LIMITS[used] / np.maximum(np.abs(desired_dollars[used]), 1e-8))
        )
        desired_dollars *= min(1.0, scale)

    current_prices = np.maximum(prices[:, -1], 1.0)
    return np.rint(desired_dollars / current_prices).astype(int)
