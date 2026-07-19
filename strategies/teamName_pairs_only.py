import numpy as np

# Algothon 2026: risk-controlled mean-reversion strategy
# ------------------------------------------------------
# Core idea:
#   1. Use the same 8d / 30d blended mean-reversion signal.
#   2. Only trade the instruments that showed the cleanest standalone
#      mean-reversion behaviour on the released 500-day sample.
#   3. Ignore weak z-score signals so we do not churn on noise.
#   4. Add mild emergency regime guards for volatility/trend shocks.
#
# This is designed to reduce StdDev/turnover versus trading all 51 names.

N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Signal settings
FAST_LB = 8
SLOW_LB = 30
FAST_WEIGHT = 0.75
SIGNAL_SCALE = 1.2
POSITION_MULT = 2.0
MIN_ABS_SIGNAL = 0.25

# Hysteresis: once a position is on, keep it while the signal stays on the
# same side above a lower exit threshold, instead of flattening the moment
# it dips under the entry threshold. Saves commission on in-and-out churn.
EXIT_ABS_SIGNAL = 0.20

# Dollar dead-band: skip retrades when the new target barely moves.
DEAD_BAND_FRAC = 0.15

# ALGO index-fade sleeve. ALGO is exactly the equal-weight normalized index
# of the other 50 names and mean-reverts at multi-week horizons (past-20/40d
# vs next-5d return corr is strongly negative). Fading its trailing move via
# ALGO itself is the cheapest expression: 0.2bp commission, $100k limit.
# Params chosen from the centre of an all-window-positive plateau
# (LB 40 / scale 2.5 / cap 30-60k) across three chronological segments.
ALGO_FADE_LB = 40
ALGO_FADE_VOL_WIN = 60
ALGO_FADE_SCALE = 2.5
ALGO_FADE_CAP = 60_000.0

# ------------------------------------------------------------------
# Pairs-trading sleeve.
# Pairs found by scanning all 1225 combinations: OLS hedge ratio, spread
# AR(1) half-life < 20d, profitable in BOTH halves of the sample, and
# full-sample SR >= 1.5, max 2 pairs per name. The selection *procedure*
# was validated honestly: pairs picked on days 0-249 alone earned
# 81/day at annSR 3.1 out-of-sample on days 250-499.
# Trades z-score reversion of spread = log(p_i) - gamma * log(p_j).
# ------------------------------------------------------------------
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
PAIR_LEG = 9_000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5
# Product ownership: names traded by the pairs sleeve are excluded from the
# main mean-reversion book so the two strategies never fight each other.
PAIR_OWNED = sorted({i for i, _, _ in PAIRS} | {j for _, j, _ in PAIRS})

# ------------------------------------------------------------------
# Lead-lag sleeve.
# The 2024 Algothon was won (score 1560/600 teams) with a lead-lag
# algorithm; our data shows the same structure: ridge-predicting
# r(t+1) from all r(t) earns walk-forward IC ~0.06 mean with 17/51
# names above 0.1. Sleeve: refit ridge W every 50 days on all history,
# full-tilt sign positions on names whose *walk-forward* IC (past live
# predictions vs realised returns, never in-sample fit) is positive.
# Validated: every lambda/sizing/selection combo was profitable on
# honest OOS (train<250/test 250-499), early and time-reversed windows.
# PnL corr with the rest of the book is -0.14. Pair-owned names are
# excluded so pair hedge ratios never get clipped.
# ------------------------------------------------------------------
LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_IC_MIN_OBS = 60
LL_SEL_IC = 0.0

# Persistent state across days (backtester calls reset_state between runs).
_prev_target_dollars = np.zeros(N_INST)
_prev_nt = -1
_pair_pos = [0] * len(PAIRS)

# Lead-lag sleeve state.
_ll_W = None
_ll_mu = None
_ll_sd = None
_ll_resid_sd = None
_ll_last_fit = -1
_ll_wf_preds = []   # walk-forward predictions already made
_ll_wf_reals = []   # realised returns those predictions targeted
_ll_pending = None  # yesterday's prediction awaiting realisation


def reset_state():
    global _prev_target_dollars, _prev_nt, _pair_pos
    global _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit
    global _ll_wf_preds, _ll_wf_reals, _ll_pending
    _prev_target_dollars = np.zeros(N_INST)
    _prev_nt = -1
    _pair_pos = [0] * len(PAIRS)
    _ll_W = None
    _ll_mu = None
    _ll_sd = None
    _ll_resid_sd = None
    _ll_last_fit = -1
    _ll_wf_preds = []
    _ll_wf_reals = []
    _ll_pending = None


def _leadlag_target_dollars(prcSoFar):
    """Lead-lag sleeve target dollars for the coming day."""
    global _ll_W, _ll_mu, _ll_sd, _ll_resid_sd, _ll_last_fit, _ll_pending

    nt = prcSoFar.shape[1]
    if nt < LL_MIN_HIST:
        return np.zeros(N_INST)

    r = np.diff(np.log(np.maximum(prcSoFar, EPS)), axis=1)

    # Record the realisation of yesterday's prediction (walk-forward IC).
    if _ll_pending is not None:
        _ll_wf_preds.append(_ll_pending)
        _ll_wf_reals.append(r[:, -1])
        _ll_pending = None

    # Refit the ridge map every LL_RETRAIN days on all available history.
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

    x = (r[:, -1] - _ll_mu) / _ll_sd
    pred = x @ _ll_W
    _ll_pending = pred.copy()

    # Walk-forward IC mask: only trade names whose live predictions have
    # actually worked so far.
    mask = np.ones(N_INST)
    if len(_ll_wf_preds) >= LL_IC_MIN_OBS:
        P = np.array(_ll_wf_preds)
        R = np.array(_ll_wf_reals)
        ics = np.zeros(N_INST)
        for j in range(N_INST):
            if P[:, j].std() > 1e-12 and R[:, j].std() > 1e-12:
                ics[j] = np.corrcoef(P[:, j], R[:, j])[0, 1]
        mask = (ics > LL_SEL_IC).astype(float)

    tgt = LIMITS * np.sign(pred) * mask
    tgt[PAIR_OWNED] = 0.0
    return tgt

# Selected from released data using standalone mean-reversion quality.
# ALGO is included, plus the strongest non-ALGO names from the 500-day sample.
CORE_ASSETS = np.array([
    0, 35, 40, 5, 37, 29, 22, 14, 10, 44,
    36, 21, 41, 13, 17, 16, 19, 39, 27, 32
], dtype=int)

# Mild regime guard settings. These only activate on genuinely abnormal moves.
RECENT_VOL_WIN = 10
BASE_VOL_WIN = 80
VOL_DANGER = 2.0
VOL_CUT = 0.65

TREND_WIN = 20
TREND_LOOKBACK = 60
TREND_DANGER = 2.5
TREND_CUT = 0.50


def getMyPosition(prcSoFar):
    global _prev_target_dollars, _prev_nt

    nInst, nt = prcSoFar.shape

    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

    # Detect a fresh run (history restarted) and reset state.
    if nt <= _prev_nt:
        reset_state()
    _prev_nt = nt

    # Need enough history for 30-day mean reversion.
    if nt < SLOW_LB + 1:
        return np.zeros(nInst, dtype=int)

    cur = np.maximum(prcSoFar[:, -1], 1.0)

    def zscore_to_past(lb):
        # Exclude today's price, so we compare today against the recent past.
        hist = prcSoFar[:, -lb - 1:-1]
        mu = hist.mean(axis=1)
        sig = hist.std(axis=1)
        sig = np.where(sig > 1e-8, sig, 1.0)
        return (mu - cur) / sig

    z_fast = zscore_to_past(FAST_LB)
    z_slow = zscore_to_past(SLOW_LB)

    # Positive signal = long, negative signal = short.
    signal = FAST_WEIGHT * z_fast + (1.0 - FAST_WEIGHT) * z_slow

    # Avoid weak/noisy signals that mostly add turnover.
    # Hysteresis: new positions require |signal| >= MIN_ABS_SIGNAL, but an
    # existing position on the same side survives down to EXIT_ABS_SIGNAL.
    holding_side = np.sign(_prev_target_dollars)
    keep = (
        (holding_side != 0)
        & (np.sign(signal) == holding_side)
        & (np.abs(signal) >= EXIT_ABS_SIGNAL)
    )
    active = (np.abs(signal) >= MIN_ABS_SIGNAL) | keep
    signal = np.where(active, signal, 0.0)

    # Start with full signal-based desired dollar exposures.
    target_dollars = LIMITS * np.tanh(SIGNAL_SCALE * signal) * POSITION_MULT

    # Trade only the selected assets, minus names owned by the pairs sleeve.
    mask = np.zeros(nInst, dtype=float)
    mask[CORE_ASSETS] = 1.0
    mask[PAIR_OWNED] = 0.0
    target_dollars *= mask

    # ------------------------------------------------------------
    # Emergency regime guards
    # ------------------------------------------------------------
    if nt > BASE_VOL_WIN + RECENT_VOL_WIN + 1:
        log_prices = np.log(np.maximum(prcSoFar, EPS))
        log_rets = np.diff(log_prices, axis=1)

        asset_scale = np.ones(nInst, dtype=float)
        global_scale = 1.0

        # 1. ALGO/index volatility shock guard.
        # If the index volatility suddenly doubles, reduce the whole book.
        recent_market_vol = log_rets[0, -RECENT_VOL_WIN:].std()
        base_market_vol = log_rets[0, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std()

        if base_market_vol > 1e-8:
            market_vol_ratio = recent_market_vol / base_market_vol
            if market_vol_ratio > VOL_DANGER:
                global_scale *= VOL_CUT

        # 2. Per-asset volatility shock guard.
        recent_asset_vol = log_rets[:, -RECENT_VOL_WIN:].std(axis=1)
        base_asset_vol = log_rets[:, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std(axis=1)
        vol_ratio = recent_asset_vol / np.where(base_asset_vol > 1e-8, base_asset_vol, 1.0)
        asset_scale *= np.where(vol_ratio > VOL_DANGER, VOL_CUT, 1.0)

        # 3. Trend guard.
        # Mean reversion loses when we fight a strong persistent trend, so cut
        # exposure when our signal points against a high-z trend.
        if nt > TREND_LOOKBACK + TREND_WIN:
            recent_trend = log_prices[:, -1] - log_prices[:, -TREND_WIN - 1]
            daily_vol = log_rets[:, -TREND_LOOKBACK:].std(axis=1)
            trend_z = recent_trend / np.where(
                daily_vol > 1e-8,
                daily_vol * np.sqrt(TREND_WIN),
                1.0
            )

            fighting_trend = signal * trend_z < 0
            asset_scale *= np.where(
                fighting_trend & (np.abs(trend_z) > TREND_DANGER),
                TREND_CUT,
                1.0
            )

        target_dollars *= asset_scale * global_scale

    # Pairs sleeve: z-score reversion on cointegrated spreads with frozen
    # hedge ratios. Hysteresis via separate entry/exit thresholds.
    log_all = np.log(np.maximum(prcSoFar, EPS))
    if nt > PAIR_ROLL + 1:
        for k, (i, j, g) in enumerate(PAIRS):
            s = log_all[i] - g * log_all[j]
            win = s[-PAIR_ROLL - 1:-1]
            z = (s[-1] - win.mean()) / (win.std() + EPS)
            pos = _pair_pos[k]
            if pos == 0:
                if z > PAIR_ENTRY:
                    pos = -1
                elif z < -PAIR_ENTRY:
                    pos = 1
            elif pos == 1 and z > -PAIR_EXIT:
                pos = 0
            elif pos == -1 and z < PAIR_EXIT:
                pos = 0
            _pair_pos[k] = pos
            if pos != 0:
                target_dollars[i] += pos * PAIR_LEG
                target_dollars[j] -= pos * g * PAIR_LEG

    # ALGO index-fade sleeve: fade ALGO's trailing 40d move, scaled by its
    # own volatility. Added on top of the main book's ALGO position; the
    # $100k instrument-0 limit is enforced by the evaluator's clip.
    if nt > ALGO_FADE_LB + ALGO_FADE_VOL_WIN + 1:
        lp0 = np.log(np.maximum(prcSoFar[0], EPS))
        fade_ret = lp0[-1] - lp0[-1 - ALGO_FADE_LB]
        fade_vol = np.diff(lp0[-(ALGO_FADE_VOL_WIN + 1):]).std()
        fade_z = fade_ret / max(fade_vol * np.sqrt(ALGO_FADE_LB), 1e-9)
        target_dollars[0] += -np.clip(fade_z / ALGO_FADE_SCALE, -1.0, 1.0) * ALGO_FADE_CAP

    # Dollar dead-band: if the target barely moved, keep yesterday's target
    # instead of paying commission on a marginal adjustment. Full exits
    # (target 0 from an active flatten) always go through.
    small_change = (
        (np.abs(target_dollars - _prev_target_dollars) < DEAD_BAND_FRAC * LIMITS)
        & (target_dollars != 0.0)
    )
    target_dollars = np.where(small_change, _prev_target_dollars, target_dollars)

    _prev_target_dollars = target_dollars.copy()

    # Lead-lag sleeve: added AFTER the dead-band (it needs daily
    # rebalancing; fees are ~1bp vs a ~0.06 daily IC edge) and after
    # _prev_target_dollars is stored, so the dead-band keeps operating
    # on the main book only. Total is clipped to per-instrument limits.
    target_dollars = target_dollars + _leadlag_target_dollars(prcSoFar)
    target_dollars = np.clip(target_dollars, -LIMITS, LIMITS)

    target_shares = target_dollars / cur
    return target_shares.astype(int)
