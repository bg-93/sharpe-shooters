import numpy as np


N_INST = 51
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50, dtype=float)

# Offline-discovered core pairs.
# Format: (left, right, beta, intercept, base_return_corr, base_spread_std)
INITIAL_ACTIVE_PAIRS = [
    (0, 2, -0.1046, 5.0828, 0.4332, 0.0638),
    (0, 22, 0.1910, 3.5851, 0.4264, 0.0550),
    (6, 27, -0.9580, 7.0108, 0.1296, 0.2030),
    (14, 35, -0.5163, 6.7521, 0.1728, 0.1326),
    (21, 40, -0.2062, 5.7297, 0.0715, 0.3191),
]

ALTERNATE_PAIRS = [
    (0, 10, 0.0739, 4.3128, 0.4529, 0.0633),
    (0, 26, 0.1591, 4.1706, 0.3363, 0.0615),
    (0, 9, 0.1984, 4.0479, 0.5218, 0.0447),
    (0, 12, 0.1417, 4.1439, 0.4800, 0.0596),
]

LOOKBACK = 30
ENTRY_Z = 1.50
EXIT_Z = 0.50
STOP_Z = 4.00
MAX_HOLD = 15
TOP_K = 5
MIN_HISTORY = LOOKBACK + 2

# Conservative refresh logic. We only rotate pairs if an active one looks badly
# broken and an unused alternate looks meaningfully better.
HEALTH_WINDOW = 60
REFRESH_CHECK_DAYS = 15
PAIR_FAIL_SHARPE = -4.0
CORR_BREAK_RATIO = 0.20
SPREAD_BLOWOUT_RATIO = 8.0
DRIFT_Z = 8.0
SWITCH_MARGIN = 10.0

_STATE = {
    "last_nt": 0,
    "last_refresh_nt": -1,
    "active_pairs": list(INITIAL_ACTIVE_PAIRS),
    "direction": {},
    "hold_days": {},
}


def reset_state():
    _STATE["last_nt"] = 0
    _STATE["last_refresh_nt"] = -1
    _STATE["active_pairs"] = list(INITIAL_ACTIVE_PAIRS)
    _STATE["direction"] = {}
    _STATE["hold_days"] = {}


def _spread(log_prices, pair):
    left, right, beta, intercept, _, _ = pair
    return log_prices[left] - (intercept + beta * log_prices[right])


def _spread_health(spread):
    if spread.size < LOOKBACK + 3:
        return -1e9

    position = 0
    hold_days = 0
    pnl = []
    for t in range(LOOKBACK, spread.size - 1):
        hist = spread[t - LOOKBACK : t]
        hist_std = float(np.std(hist))
        zscore = 0.0 if hist_std < 1e-10 else (spread[t] - float(np.mean(hist))) / hist_std

        if position == 0:
            if zscore >= ENTRY_Z:
                position = -1
                hold_days = 0
            elif zscore <= -ENTRY_Z:
                position = 1
                hold_days = 0
        else:
            hold_days += 1
            if abs(zscore) <= EXIT_Z or abs(zscore) >= STOP_Z or hold_days >= MAX_HOLD:
                position = 0
                hold_days = 0

        pnl.append(position * -(spread[t + 1] - spread[t]))

    pnl = np.asarray(pnl, dtype=float)
    pnl_std = float(np.std(pnl))
    if pnl.size == 0 or pnl_std < 1e-12:
        return -1e9
    return float(np.sqrt(250.0) * np.mean(pnl) / pnl_std)


def _pair_health(prcSoFar, pair):
    log_prices = np.log(np.maximum(prcSoFar, EPS))
    spread = _spread(log_prices, pair)
    recent = spread[-HEALTH_WINDOW:]
    recent_sr = _spread_health(recent)

    left, right, _, _, base_corr, base_spread_std = pair
    recent_corr = float(
        abs(
            np.corrcoef(
                np.diff(log_prices[left, -HEALTH_WINDOW:]),
                np.diff(log_prices[right, -HEALTH_WINDOW:]),
            )[0, 1]
        )
    )
    recent_std = float(np.std(recent))

    short_hist = spread[-21:-1]
    short_std = float(np.std(short_hist))
    current_z = 0.0 if short_std < 1e-10 else (float(spread[-1]) - float(np.mean(short_hist))) / short_std

    quality = recent_sr * (0.5 + 0.5 * recent_corr)
    unstable = (
        recent_sr < PAIR_FAIL_SHARPE
        or recent_corr < CORR_BREAK_RATIO * base_corr
        or recent_std > SPREAD_BLOWOUT_RATIO * base_spread_std
        or abs(current_z) > DRIFT_Z
    )
    return quality, unstable


def _maybe_refresh_active_pairs(prcSoFar):
    nt = prcSoFar.shape[1]
    if nt < HEALTH_WINDOW:
        return

    if (
        _STATE["last_refresh_nt"] >= 0
        and nt - _STATE["last_refresh_nt"] < REFRESH_CHECK_DAYS
    ):
        return

    active_scored = []
    for idx, pair in enumerate(_STATE["active_pairs"]):
        quality, unstable = _pair_health(prcSoFar, pair)
        active_scored.append((quality, unstable, idx, pair))
    active_scored.sort(key=lambda row: row[0])

    used_pairs = {(pair[0], pair[1]) for pair in _STATE["active_pairs"]}
    alternate_scored = []
    for pair in ALTERNATE_PAIRS:
        if (pair[0], pair[1]) in used_pairs:
            continue
        quality, unstable = _pair_health(prcSoFar, pair)
        alternate_scored.append((quality, unstable, pair))
    alternate_scored.sort(key=lambda row: row[0], reverse=True)

    if active_scored and alternate_scored:
        worst_quality, worst_unstable, worst_idx, worst_pair = active_scored[0]
        best_alt_quality, _, best_alt_pair = alternate_scored[0]
        if worst_unstable and best_alt_quality > worst_quality + SWITCH_MARGIN:
            _STATE["active_pairs"][worst_idx] = best_alt_pair
            removed_key = (worst_pair[0], worst_pair[1])
            _STATE["direction"].pop(removed_key, None)
            _STATE["hold_days"].pop(removed_key, None)

    _STATE["last_refresh_nt"] = nt


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

    _maybe_refresh_active_pairs(prices)

    log_prices = np.log(np.maximum(prices, EPS))
    desired_dollars = np.zeros(N_INST, dtype=float)
    ranked_signals = []

    for pair in _STATE["active_pairs"]:
        left, right, beta, intercept, _, _ = pair
        key = (left, right)
        spread = _spread(log_prices, pair)
        hist = spread[-LOOKBACK - 1 : -1]
        spread_std = float(np.std(hist))

        if spread_std < 1e-10:
            _STATE["direction"][key] = 0
            _STATE["hold_days"][key] = 0
            continue

        zscore = (float(spread[-1]) - float(np.mean(hist))) / spread_std
        direction = int(_STATE["direction"].get(key, 0))
        hold_days = int(_STATE["hold_days"].get(key, 0))

        if direction == 0:
            if zscore >= ENTRY_Z:
                direction, hold_days = -1, 1
            elif zscore <= -ENTRY_Z:
                direction, hold_days = 1, 1
        else:
            hold_days += 1
            if abs(zscore) <= EXIT_Z or abs(zscore) >= STOP_Z or hold_days >= MAX_HOLD:
                direction, hold_days = 0, 0

        _STATE["direction"][key] = direction
        _STATE["hold_days"][key] = hold_days

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
