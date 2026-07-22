#!/usr/bin/env python3
"""Adaptive pairs + cross-sectionally demeaned lead-lag strategy.

The strategy starts with the original 15 frozen pairs. Once 250 observations
are available, and every 50 observations thereafter, it re-runs the original
pair-discovery procedure using ONLY data available at that point:

1. Scan all 1,225 pairs among instruments 1..50.
2. Fit OLS on log prices: log(P_i) = alpha + gamma * log(P_j) + error.
3. Keep spreads whose AR(1)-implied half-life is below 20 days.
4. Backtest the same 60-day z-score FSM, including estimated fees.
5. Require positive PnL in both halves and full-period annualised Sharpe >= 1.5.
6. Rank robust candidates and select at most 15, with at most two pairs per name.

The selected identities and hedge ratios are frozen until the next refit.
No future observations are used in a refit. If fewer than 15 pairs qualify,
only the qualifying pairs are traded.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

DEFAULT_PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836),
    (13, 45, 1.0132), (33, 40, 0.2577), (10, 46, 1.0331),
    (33, 42, 0.8358), (31, 43, 0.9692), (18, 28, 0.5642),
    (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)

# Pair trading and discovery settings.
PAIR_LEG = 9_000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5
PAIR_FEE = 1.0e-4              # 1 bp per dollar of turnover
PAIR_MAX_HALFLIFE = 20.0
PAIR_MIN_SR = 1.5
PAIR_MAX_COUNT = 15
PAIR_MAX_PER_NAME = 2
PAIR_MIN_FIT = 250
PAIR_FIT_LOOKBACK = 250         # rolling adaptation window
PAIR_RETRAIN = 50
PAIR_MIN_TRADES = 2
PAIR_GAMMA_MIN = 0.05
PAIR_GAMMA_MAX = 3.0

# Lead-lag settings (unchanged from the supplied strategy).
LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_TEMP = 0.35

_active_pairs = list(DEFAULT_PAIRS)
_pair_pos = [0] * len(DEFAULT_PAIRS)
_pair_last_fit = -1

_ll_W = None
_ll_mu = None
_ll_sd = None
_ll_resid_sd = None
_ll_last_fit = -1
_prev_nt = -1


def reset_state():
    global _active_pairs, _pair_pos, _pair_last_fit
    global _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit, _prev_nt

    _active_pairs = list(DEFAULT_PAIRS)
    _pair_pos = [0] * len(_active_pairs)
    _pair_last_fit = -1

    _ll_W = None
    _ll_mu = None
    _ll_sd = None
    _ll_resid_sd = None
    _ll_last_fit = -1
    _prev_nt = -1


def _ols_gamma(x, y):
    """OLS slope in x = alpha + gamma*y + error."""
    yc = y - y.mean()
    denom = float(yc @ yc)
    if denom <= EPS:
        return None
    gamma = float((yc @ (x - x.mean())) / denom)
    if not np.isfinite(gamma):
        return None
    if gamma < PAIR_GAMMA_MIN or gamma > PAIR_GAMMA_MAX:
        return None
    return gamma


def _spread_half_life(spread):
    """AR(1) half-life from spread_t = c + phi*spread_(t-1) + error."""
    lag = spread[:-1]
    nxt = spread[1:]
    lc = lag - lag.mean()
    denom = float(lc @ lc)
    if denom <= EPS:
        return np.inf

    phi = float((lc @ (nxt - nxt.mean())) / denom)
    # Positive, stationary AR(1) spreads have a well-behaved reversion time.
    if not np.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
        return np.inf

    half_life = -np.log(2.0) / np.log(phi)
    return float(half_life) if np.isfinite(half_life) else np.inf


def _pair_backtest(pi, pj, spread, gamma):
    """Fee-adjusted backtest matching the live pair FSM and dollar sizing."""
    n = spread.size
    if n <= PAIR_ROLL + 2:
        return None

    pnl = np.zeros(n - 1)
    state = 0
    prev_qi = 0.0
    prev_qj = 0.0
    entries = 0

    for t in range(PAIR_ROLL, n - 1):
        win = spread[t - PAIR_ROLL:t]
        sd = float(win.std())
        if sd <= EPS:
            continue
        z = float((spread[t] - win.mean()) / sd)

        old_state = state
        if state == 0:
            if z > PAIR_ENTRY:
                state = -1
            elif z < -PAIR_ENTRY:
                state = 1
        elif state == 1 and z > -PAIR_EXIT:
            state = 0
        elif state == -1 and z < PAIR_EXIT:
            state = 0

        if old_state == 0 and state != 0:
            entries += 1

        target_i = state * PAIR_LEG
        target_j = -state * gamma * PAIR_LEG
        qi = target_i / max(float(pi[t]), EPS)
        qj = target_j / max(float(pj[t]), EPS)

        turnover = abs(qi - prev_qi) * pi[t] + abs(qj - prev_qj) * pj[t]
        trading_pnl = qi * (pi[t + 1] - pi[t]) + qj * (pj[t + 1] - pj[t])
        pnl[t] = trading_pnl - PAIR_FEE * turnover

        prev_qi, prev_qj = qi, qj

    if entries < PAIR_MIN_TRADES:
        return None

    live = pnl[PAIR_ROLL:]
    if live.size < 4:
        return None

    split = live.size // 2
    first = live[:split]
    second = live[split:]
    pnl_first = float(first.sum())
    pnl_second = float(second.sum())
    pnl_total = float(live.sum())

    sd = float(live.std(ddof=0))
    sr = float(np.sqrt(252.0) * live.mean() / sd) if sd > EPS else -np.inf

    def half_sr(x):
        sx = float(x.std(ddof=0))
        return float(np.sqrt(252.0) * x.mean() / sx) if sx > EPS else -np.inf

    sr_first = half_sr(first)
    sr_second = half_sr(second)

    return pnl_total, pnl_first, pnl_second, sr, sr_first, sr_second, entries


def _select_pairs(prcSoFar):
    """Walk-forward scan of all pairs using only currently observed history."""
    nt = prcSoFar.shape[1]
    start = max(0, nt - PAIR_FIT_LOOKBACK)
    prices = np.maximum(prcSoFar[:, start:nt], EPS)
    logs = np.log(prices)
    candidates = []

    for i in range(1, N_INST - 1):
        xi = logs[i]
        pi = prices[i]
        for j in range(i + 1, N_INST):
            gamma = _ols_gamma(xi, logs[j])
            if gamma is None:
                continue

            spread = xi - gamma * logs[j]
            half_life = _spread_half_life(spread)
            if half_life >= PAIR_MAX_HALFLIFE:
                continue

            result = _pair_backtest(pi, prices[j], spread, gamma)
            if result is None:
                continue

            total, first, second, sr, sr1, sr2, entries = result
            if first <= 0.0 or second <= 0.0 or sr < PAIR_MIN_SR:
                continue

            # Conservative robustness rank: reward full Sharpe and the weaker
            # half, then total net PnL. This prevents one half dominating.
            robust_sr = min(sr1, sr2)
            rank = (robust_sr, sr, total, -half_life, entries)
            candidates.append((rank, i, j, gamma))

    candidates.sort(key=lambda row: row[0], reverse=True)

    selected = []
    ownership = np.zeros(N_INST, dtype=int)
    for _, i, j, gamma in candidates:
        if ownership[i] >= PAIR_MAX_PER_NAME or ownership[j] >= PAIR_MAX_PER_NAME:
            continue
        selected.append((i, j, gamma))
        ownership[i] += 1
        ownership[j] += 1
        if len(selected) >= PAIR_MAX_COUNT:
            break

    return selected


def _maybe_refit_pairs(prcSoFar):
    global _active_pairs, _pair_pos, _pair_last_fit

    nt = prcSoFar.shape[1]
    if nt < PAIR_MIN_FIT:
        return
    if _pair_last_fit >= 0 and (nt - _pair_last_fit) < PAIR_RETRAIN:
        return

    selected = _select_pairs(prcSoFar)
    _active_pairs = selected
    # Reset the FSM because hedge ratios and spread definitions may have changed.
    _pair_pos = [0] * len(_active_pairs)
    _pair_last_fit = nt


def _pair_target_dollars(prcSoFar):
    _maybe_refit_pairs(prcSoFar)

    nt = prcSoFar.shape[1]
    target = np.zeros(N_INST)
    if nt <= PAIR_ROLL + 1:
        return target

    log_all = np.log(np.maximum(prcSoFar, EPS))
    for k, (i, j, gamma) in enumerate(_active_pairs):
        spread = log_all[i] - gamma * log_all[j]
        win = spread[-PAIR_ROLL - 1:-1]
        sd = float(win.std())
        if sd <= EPS:
            continue
        z = float((spread[-1] - win.mean()) / sd)

        state = _pair_pos[k]
        if state == 0:
            if z > PAIR_ENTRY:
                state = -1
            elif z < -PAIR_ENTRY:
                state = 1
        elif state == 1 and z > -PAIR_EXIT:
            state = 0
        elif state == -1 and z < PAIR_EXIT:
            state = 0

        _pair_pos[k] = state
        if state != 0:
            target[i] += state * PAIR_LEG
            target[j] -= state * gamma * PAIR_LEG

    return target


def _leadlag_target_dollars(prcSoFar):
    global _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit

    nt = prcSoFar.shape[1]
    if nt < LL_MIN_HIST:
        return np.zeros(N_INST)

    r = np.diff(np.log(np.maximum(prcSoFar, EPS)), axis=1)
    if _ll_W is None or (nt - _ll_last_fit) >= LL_RETRAIN:
        X = r[:, :-1].T
        Y = r[:, 1:].T
        _ll_mu, _ll_sd = X.mean(0), X.std(0)
        _ll_sd = np.where(_ll_sd > 1e-12, _ll_sd, 1.0)
        Xs = (X - _ll_mu) / _ll_sd
        _ll_W = np.linalg.solve(
            Xs.T @ Xs + LL_LAM * np.eye(N_INST), Xs.T @ Y
        )
        _ll_resid_sd = np.maximum(Y.std(0), 1e-8)
        _ll_last_fit = nt

    pred = ((r[:, -1] - _ll_mu) / _ll_sd) @ _ll_W
    z = pred / _ll_resid_sd
    z = z - z.mean()
    z = z / (z.std() + 1e-9)
    return LIMITS * np.tanh(z / LL_TEMP)


def getMyPosition(prcSoFar):
    global _prev_nt

    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nt = prcSoFar.shape[1]

    # Detect the start of a fresh simulation.
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt

    # Calculate the sleeves separately.
    pair_target = _pair_target_dollars(prcSoFar)
    ll_target = _leadlag_target_dollars(prcSoFar)

    # Reserve some position capacity for instruments used by pairs.
    pair_owned = np.abs(pair_target) > EPS
    ll_target[pair_owned] *= 0.75

    # Combine the sleeves and enforce instrument limits.
    target_dollars = pair_target + ll_target
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)

    # Convert dollar targets into integer share positions.
    current_prices = np.maximum(prcSoFar[:, -1], EPS)
    return (target_dollars / current_prices).astype(int)
