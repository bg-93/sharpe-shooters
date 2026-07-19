import numpy as np
'''
# Dynamic regime strategy:
# - Uses the same 8d / 30d blended mean-reversion signal as before.
# - For each instrument, detects whether recent price action is sideways or trending.
# - Sideways / weak trend  -> mean reversion: long below average, short above average.
# - Strong uptrend         -> momentum long.
# - Strong downtrend       -> momentum short.
#
# Submit this file renamed to your actual team name.

N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Mean-reversion signal settings
FAST_LB = 8
SLOW_LB = 30
FAST_WEIGHT = 0.75

# Trend/momentum regime settings
TREND_WIN = 20
TREND_VOL_WIN = 60
SIDEWAYS_Z = 0.75
TREND_Z = 1.25
CONSISTENCY_MIN = 0.58

# Position settings
MR_SCALE = 0.85
MOM_SCALE = 1.10
POSITION_MULT = 1.4
MIN_ABS_SIGNAL = 0.20

# Optional risk control: only trade strongest current opportunities.
# Set TOP_K = N_INST if you want to trade everything.
TOP_K = 20


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape

    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

    min_hist = max(SLOW_LB + 1, TREND_VOL_WIN + 1, TREND_WIN + 1)
    if nt < min_hist:
        return np.zeros(nInst, dtype=int)

    cur = np.maximum(prcSoFar[:, -1], 1.0)

    # ------------------------------------------------------------
    # 1. 8d / 30d mean-reversion signal
    # ------------------------------------------------------------
    def zscore_to_past(lb):
        hist = prcSoFar[:, -lb - 1:-1]  # exclude today's price
        mu = hist.mean(axis=1)
        sig = hist.std(axis=1)
        sig = np.where(sig > 1e-8, sig, 1.0)
        return (mu - cur) / sig

    z8 = zscore_to_past(FAST_LB)
    z30 = zscore_to_past(SLOW_LB)

    # Positive = long, negative = short
    mr_signal = FAST_WEIGHT * z8 + (1.0 - FAST_WEIGHT) * z30

    # ------------------------------------------------------------
    # 2. Trend / momentum detection using log prices
    # ------------------------------------------------------------
    log_prices = np.log(np.maximum(prcSoFar, EPS))
    log_rets = np.diff(log_prices, axis=1)

    recent_trend = log_prices[:, -1] - log_prices[:, -TREND_WIN - 1]

    trend_vol = log_rets[:, -TREND_VOL_WIN:].std(axis=1)
    trend_vol = np.where(trend_vol > 1e-8, trend_vol, 1.0)

    trend_z = recent_trend / (trend_vol * np.sqrt(TREND_WIN))

    recent_rets = log_rets[:, -TREND_WIN:]
    up_consistency = np.mean(recent_rets > 0, axis=1)
    down_consistency = np.mean(recent_rets < 0, axis=1)

    consistency = np.where(trend_z >= 0, up_consistency, down_consistency)

    # Momentum signal:
    # positive trend_z => buy
    # negative trend_z => sell/short
    momentum_signal = trend_z

    # ------------------------------------------------------------
    # 3. Regime switch per instrument
    # ------------------------------------------------------------
    sideways = np.abs(trend_z) < SIDEWAYS_Z

    trending_up = (
        (trend_z > TREND_Z)
        & (consistency >= CONSISTENCY_MIN)
    )

    trending_down = (
        (trend_z < -TREND_Z)
        & (consistency >= CONSISTENCY_MIN)
    )

    final_signal = np.zeros(nInst, dtype=float)

    # Sideways / non-directional regime:
    # use classic mean reversion.
    final_signal = np.where(sideways, mr_signal, final_signal)

    # Strong trend regime:
    # use momentum direction, not mean reversion.
    final_signal = np.where(trending_up, momentum_signal, final_signal)
    final_signal = np.where(trending_down, momentum_signal, final_signal)

    # Grey zone:
    # trend exists but is not consistent/strong enough.
    # Use smaller mean reversion rather than full exposure.
    grey_zone = ~(sideways | trending_up | trending_down)
    final_signal = np.where(grey_zone, 0.35 * mr_signal, final_signal)

    # Ignore weak/noisy signals
    final_signal = np.where(np.abs(final_signal) >= MIN_ABS_SIGNAL, final_signal, 0.0)

    # ------------------------------------------------------------
    # 4. Position sizing
    # ------------------------------------------------------------
    scale = np.where(trending_up | trending_down, MOM_SCALE, MR_SCALE)

    target_dollars = (
        LIMITS
        * np.tanh(scale * final_signal)
        * POSITION_MULT
    )

    # ------------------------------------------------------------
    # 5. Edge-aware Top-K selection
    # ------------------------------------------------------------
    confidence = np.ones(nInst, dtype=float)
    confidence = np.where(sideways, 0.75, confidence)
    confidence = np.where(trending_up | trending_down, 1.00, confidence)
    confidence = np.where(grey_zone, 0.35, confidence)

    selection_score = np.abs(final_signal) * confidence
    selection_score = np.where(np.abs(target_dollars) > 1e-6, selection_score, 0.0)

    active = np.where(selection_score > 0)[0]
    filtered = np.zeros_like(target_dollars)

    if len(active) > 0:
        if len(active) > TOP_K:
            strongest = active[np.argsort(selection_score[active])[-TOP_K:]]
        else:
            strongest = active

        filtered[strongest] = target_dollars[strongest]

    target_shares = filtered / cur
    return target_shares.astype(int)
'''
'''
N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Signal settings
FAST_LB = 8
SLOW_LB = 30
FAST_WEIGHT = 0.75
SIGNAL_SCALE = 1.2
POSITION_MULT = 1.7
MIN_ABS_SIGNAL = 0.25
TOP_K = 15

# Meta edge detector: decide whether MR or momentum has worked recently
EDGE_WIN = 45
FLIP_EDGE_Z = -0.15      # if recent MR edge is below this, flip to momentum
WEAK_EDGE_Z = 0.05       # if edge is weak, reduce size
WEAK_SCALE = 0.55

# Regime guards
RECENT_VOL_WIN = 10
BASE_VOL_WIN = 80
VOL_DANGER = 2.0
VOL_CUT = 0.70

TREND_WIN = 20
TREND_LOOKBACK = 60
TREND_DANGER = 2.5
TREND_CUT = 0.55


def _zscore_to_past_at(prices, day, lb):
    cur = prices[:, day]
    hist = prices[:, day - lb:day]
    mu = hist.mean(axis=1)
    sig = hist.std(axis=1)
    sig = np.where(sig > 1e-8, sig, 1.0)
    return (mu - cur) / sig


def _mr_signal_at(prices, day):
    z_fast = _zscore_to_past_at(prices, day, FAST_LB)
    z_slow = _zscore_to_past_at(prices, day, SLOW_LB)
    sig = FAST_WEIGHT * z_fast + (1.0 - FAST_WEIGHT) * z_slow
    sig = np.where(np.abs(sig) >= MIN_ABS_SIGNAL, sig, 0.0)
    return sig


def _current_mr_signal(prices):
    return _mr_signal_at(prices, prices.shape[1] - 1)


def _recent_mr_edge_z(prices, log_prices):
    """
    Portfolio-level realised edge of the MR rule over recent history.
    Positive => long losers / short winners has worked recently.
    Negative => the flipped momentum rule has worked recently.
    """
    nt = prices.shape[1]
    end_day = nt - 2
    start_day = max(SLOW_LB, end_day - EDGE_WIN + 1)

    if end_day < start_day + 5:
        return 0.0

    pseudo_pl = []
    for d in range(start_day, end_day + 1):
        sig = _mr_signal_at(prices, d)
        target = LIMITS * np.tanh(SIGNAL_SCALE * sig) * POSITION_MULT

        # Match the live strategy: only the strongest current opportunities get capital.
        strongest = np.argsort(np.abs(target))[-TOP_K:]
        filtered = np.zeros_like(target)
        filtered[strongest] = target[strongest]

        next_ret = log_prices[:, d + 1] - log_prices[:, d]
        pseudo_pl.append(np.sum(filtered * next_ret))

    pseudo_pl = np.asarray(pseudo_pl)
    sd = pseudo_pl.std()
    if sd < 1e-8:
        return 0.0
    return pseudo_pl.mean() / sd


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

    if nt < SLOW_LB + 5:
        return np.zeros(nInst, dtype=int)

    prices = np.maximum(prcSoFar, EPS)
    cur = np.maximum(prcSoFar[:, -1], 1.0)
    log_prices = np.log(prices)
    log_rets = np.diff(log_prices, axis=1)

    # 1. Base MR signal
    mr_signal = _current_mr_signal(prices)

    # 2. Meta switch: MR vs momentum
    edge_z = _recent_mr_edge_z(prices, log_prices)

    direction = 1.0
    meta_scale = 1.0

    if edge_z < FLIP_EDGE_Z:
        # Recent data says winners keep winning / losers keep losing.
        direction = -1.0
    elif abs(edge_z) < WEAK_EDGE_Z:
        # No strong evidence either way, so trade smaller.
        meta_scale = WEAK_SCALE

    signal = direction * mr_signal

    target_dollars = LIMITS * np.tanh(SIGNAL_SCALE * signal) * POSITION_MULT * meta_scale

    # 3. Regime guards
    if nt > BASE_VOL_WIN + RECENT_VOL_WIN + 1:
        asset_scale = np.ones(nInst, dtype=float)
        global_scale = 1.0

        # Index/ALGO volatility shock guard.
        recent_market_vol = log_rets[0, -RECENT_VOL_WIN:].std()
        base_market_vol = log_rets[0, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std()
        if base_market_vol > 1e-8:
            if recent_market_vol / base_market_vol > VOL_DANGER:
                global_scale *= VOL_CUT

        # Per-asset volatility shock guard.
        recent_asset_vol = log_rets[:, -RECENT_VOL_WIN:].std(axis=1)
        base_asset_vol = log_rets[:, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std(axis=1)
        vol_ratio = recent_asset_vol / np.where(base_asset_vol > 1e-8, base_asset_vol, 1.0)
        asset_scale *= np.where(vol_ratio > VOL_DANGER, VOL_CUT, 1.0)

        # Trend guard after the meta switch.
        if nt > TREND_LOOKBACK + TREND_WIN:
            recent_trend = log_prices[:, -1] - log_prices[:, -TREND_WIN - 1]
            daily_vol = log_rets[:, -TREND_LOOKBACK:].std(axis=1)
            trend_z = recent_trend / np.where(
                daily_vol > 1e-8,
                daily_vol * np.sqrt(TREND_WIN),
                1.0,
            )
            fighting_trend = signal * trend_z < 0
            asset_scale *= np.where(
                fighting_trend & (np.abs(trend_z) > TREND_DANGER),
                TREND_CUT,
                1.0,
            )

        target_dollars *= asset_scale * global_scale

    # 4. Current top-K allocation
    # ------------------------------------------------------------
    # Edge-adjusted Top-K allocation
    # ------------------------------------------------------------
    # Do NOT rank only by abs(target_dollars), because that just picks
    # the most extreme moves. Instead rank by:
    #
    #   signal strength × confidence × regime safety
    #
    # where:
    #   abs(signal)      = how strong the opportunity is
    #   confidence       = evidence that this signal type has worked recently
    #   regime_scale     = penalty for vol/trend danger
    # ------------------------------------------------------------

    # Example: if you have edge_z per asset
    # edge_z > 0  => mean reversion has worked recently
    # edge_z < 0  => momentum has worked recently
    # edge_z near 0 => no clear edge

    confidence = np.clip(np.abs(edge_z) / 2.0, 0.0, 1.0)

    # Remove weak/no-edge names completely
    confidence = np.where(np.abs(edge_z) < 0.25, 0.0, confidence)

    # Use regime_scale if you have it as an array.
    # If not, set
    regime_scale = np.ones(nInst)

    selection_score = (
        np.abs(signal)
        * confidence
        * regime_scale
    )

    # Do not select names with zero target
    selection_score = np.where(np.abs(target_dollars) > 1e-6, selection_score, 0.0)

    active = np.where(selection_score > 0)[0]

    filtered = np.zeros_like(target_dollars)

    if len(active) > 0:
        if len(active) > TOP_K:
            strongest = active[np.argsort(selection_score[active])[-TOP_K:]]
        else:
            strongest = active

        filtered[strongest] = target_dollars[strongest]

    target_dollars = filtered

    target_shares = filtered / cur
    return target_shares.astype(int)
'''

'''
N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Trend-channel settings
FAST_WIN = 35
SLOW_WIN = 100

# Trading settings
TOP_K = 5
MIN_ABS_SIGNAL = 0.35
SIGNAL_SCALE = 0.9
POSITION_MULT = 1.4

# Regime settings
RECENT_VOL_WIN = 10
BASE_VOL_WIN = 80
VOL_WARN = 1.5
VOL_DANGER = 2.2

TREND_WARN = 1.0
TREND_DANGER = 1.7


def weighted_trend_to_past(log_price, win):
    """
    Fit a weighted linear trend to the previous `win` days,
    excluding today. Then forecast today's trend value.

    Returns:
        fair_today: projected trend-line value for today
        resid_z: how far today's price is from trend
        slope_z: strength/direction of trend
        resid_std: residual volatility around trend
    """
    if len(log_price) < win + 1:
        return None

    y = log_price[-win-1:-1]
    cur = log_price[-1]

    x = np.arange(win, dtype=float)

    # More weight on recent data
    weights = 0.96 ** (win - 1 - x)
    weights = weights / np.sum(weights)

    x_bar = np.sum(weights * x)
    y_bar = np.sum(weights * y)

    denom = np.sum(weights * (x - x_bar) ** 2)
    if denom < EPS:
        return None

    slope = np.sum(weights * (x - x_bar) * (y - y_bar)) / denom
    intercept = y_bar - slope * x_bar

    fitted = intercept + slope * x
    residuals = y - fitted
    resid_std = residuals.std()

    if resid_std < 1e-8:
        return None

    # Forecast trend line to today's x = win
    fair_today = intercept + slope * win

    # Positive resid_z means price is above trend.
    resid_z = (cur - fair_today) / resid_std

    # Normalised trend strength.
    slope_z = slope * np.sqrt(win) / resid_std

    return fair_today, resid_z, slope_z, resid_std


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape

    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

    if nt < SLOW_WIN + 2:
        return np.zeros(nInst, dtype=int)

    cur_prices = np.maximum(prcSoFar[:, -1], 1.0)
    log_prices = np.log(np.maximum(prcSoFar, EPS))
    log_rets = np.diff(log_prices, axis=1)

    target_dollars = np.zeros(nInst, dtype=float)

    # ------------------------------------------------------------
    # Global market stress filter using ALGO/index
    # ------------------------------------------------------------
    global_scale = 1.0

    if nt > BASE_VOL_WIN + RECENT_VOL_WIN + 2:
        algo_rets = log_rets[0]

        recent_market_vol = algo_rets[-RECENT_VOL_WIN:].std()
        base_market_vol = algo_rets[-(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std()

        if base_market_vol > 1e-8:
            market_vol_ratio = recent_market_vol / base_market_vol

            if market_vol_ratio > VOL_DANGER:
                global_scale = 0.45
            elif market_vol_ratio > VOL_WARN:
                global_scale = 0.70

    # ------------------------------------------------------------
    # Per-instrument adaptive trend channel
    # ------------------------------------------------------------
    for i in range(nInst):
        lp = log_prices[i]
        rets = log_rets[i]

        fast = weighted_trend_to_past(lp, FAST_WIN)
        slow = weighted_trend_to_past(lp, SLOW_WIN)

        if fast is None or slow is None:
            continue

        _, fast_resid_z, fast_slope_z, _ = fast
        _, slow_resid_z, slow_slope_z, _ = slow

        # Volatility regime for this instrument
        recent_vol = rets[-RECENT_VOL_WIN:].std()
        base_vol = rets[-(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std()

        if base_vol > 1e-8:
            vol_ratio = recent_vol / base_vol
        else:
            vol_ratio = 1.0

        # --------------------------------------------------------
        # Recalibration logic
        # --------------------------------------------------------
        # If volatility spikes or fast trend disagrees with slow trend,
        # trust the fast trend more.
        trend_disagreement = np.sign(fast_slope_z) != np.sign(slow_slope_z)

        stress_regime = (
            vol_ratio > VOL_WARN
            or abs(slow_resid_z) > 2.5
            or (trend_disagreement and abs(fast_slope_z) > TREND_WARN)
        )

        if stress_regime:
            # Recalibrate quickly after regime shift.
            resid_z = fast_resid_z
            slope_z = fast_slope_z
        else:
            # Stable regime: blend short and long trend estimates.
            resid_z = 0.65 * fast_resid_z + 0.35 * slow_resid_z
            slope_z = 0.65 * fast_slope_z + 0.35 * slow_slope_z

        # --------------------------------------------------------
        # Trading signal
        # --------------------------------------------------------
        # Price below trend => resid_z negative => buy.
        # Price above trend => resid_z positive => short.
        signal = -resid_z

        if abs(signal) < MIN_ABS_SIGNAL:
            continue

        regime_scale = 1.0

        # Cut exposure when volatility is abnormal
        if vol_ratio > VOL_DANGER:
            regime_scale *= 0.40
        elif vol_ratio > VOL_WARN:
            regime_scale *= 0.70

        # --------------------------------------------------------
        # Trend guard
        # --------------------------------------------------------
        # If signal wants to buy while trend is strongly down, cut.
        # If signal wants to short while trend is strongly up, cut.
        fighting_trend = signal * slope_z < 0

        if fighting_trend and abs(slope_z) > TREND_DANGER:
            regime_scale *= 0.25
        elif fighting_trend and abs(slope_z) > TREND_WARN:
            regime_scale *= 0.60

        # If price is below an upward trend, buying is allowed.
        # If price is above a downward trend, shorting is allowed.
        bounded_signal = np.tanh(SIGNAL_SCALE * signal)

        target_dollars[i] = (
            LIMITS[i]
            * bounded_signal
            * POSITION_MULT
            * regime_scale
            * global_scale
        )

    # ------------------------------------------------------------
    # Only trade strongest live opportunities
    # ------------------------------------------------------------
    active = np.where(np.abs(target_dollars) > 1e-6)[0]

    if len(active) > TOP_K:
        strongest = active[np.argsort(np.abs(target_dollars[active]))[-TOP_K:]]
    else:
        strongest = active

    filtered = np.zeros_like(target_dollars)
    filtered[strongest] = target_dollars[strongest]

    target_shares = filtered / cur_prices
    return target_shares.astype(int)
'''
'''
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

# Strict dynamic allocation: only the top-k current opportunities get capital.
TOP_K = 5

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
    nInst, nt = prcSoFar.shape

    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

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
    signal = np.where(np.abs(signal) >= MIN_ABS_SIGNAL, signal, 0.0)

    # Start with full signal-based desired dollar exposures.
    target_dollars = LIMITS * np.tanh(SIGNAL_SCALE * signal) * POSITION_MULT

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

    # Only allocate to the current top-k signals instead of a fixed asset list.
    # Rank by post-filter dollar conviction so regime guards can demote stressed names.
    strongest = np.argsort(np.abs(target_dollars))[-TOP_K:]
    filtered = np.zeros_like(target_dollars)
    filtered[strongest] = target_dollars[strongest]
    target_dollars = filtered

    target_shares = target_dollars / cur
    return target_shares.astype(int)

'''
'''
N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Mean reversion settings
FAST_LB = 8
SLOW_LB = 30
SIGNAL_SCALE = 0.85
POSITION_MULT = 2.0

# Regime settings
RECENT_VOL_WIN = 10
BASE_VOL_WIN = 80
VOL_WARN = 1.5
VOL_DANGER = 2.0

TREND_WIN = 20
TREND_LOOKBACK = 60
TREND_Z_WARN = 1.2
TREND_Z_DANGER = 1.7


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape

    if nInst != N_INST:
        return np.zeros(nInst, dtype=int)

    cur = np.maximum(prcSoFar[:, -1], 1.0)

    if nt < 31:
        return np.zeros(nInst, dtype=int)

    log_prices = np.log(np.maximum(prcSoFar, EPS))
    log_rets = np.diff(log_prices, axis=1)

    # ------------------------------------------------------------
    # Simple moving-average mean reversion signal
    # ------------------------------------------------------------
    def zscore_to_past(lb):
        hist = prcSoFar[:, -lb - 1:-1]
        mu = hist.mean(axis=1)
        sig = hist.std(axis=1)
        sig = np.where(sig > 1e-8, sig, 1.0)
        return (mu - cur) / sig

    z8 = zscore_to_past(FAST_LB)
    z30 = zscore_to_past(SLOW_LB)

    # Positive signal = long, negative signal = short
    signal = 0.75 * z8 + 0.25 * z30

    # ------------------------------------------------------------
    # Global market volatility filter using ALGO/index
    # ------------------------------------------------------------
    global_scale = 1.0

    if nt > BASE_VOL_WIN + RECENT_VOL_WIN + 1:
        algo_rets = log_rets[0]

        recent_market_vol = algo_rets[-RECENT_VOL_WIN:].std()
        base_market_vol = algo_rets[-(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std()

        if base_market_vol > 1e-8:
            market_vol_ratio = recent_market_vol / base_market_vol

            if market_vol_ratio > VOL_DANGER:
                global_scale = 0.50
            elif market_vol_ratio > VOL_WARN:
                global_scale = 0.75

    # ------------------------------------------------------------
    # Per-instrument regime scaling
    # ------------------------------------------------------------
    regime_scale = np.ones(nInst)

    if nt > BASE_VOL_WIN + RECENT_VOL_WIN + 1:
        recent_vol = log_rets[:, -RECENT_VOL_WIN:].std(axis=1)
        base_vol = log_rets[:, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std(axis=1)

        vol_ratio = recent_vol / np.where(base_vol > 1e-8, base_vol, 1.0)

        regime_scale = np.where(vol_ratio > VOL_DANGER, regime_scale * 0.40, regime_scale)
        regime_scale = np.where(
            (vol_ratio > VOL_WARN) & (vol_ratio <= VOL_DANGER),
            regime_scale * 0.70,
            regime_scale
        )

    # ------------------------------------------------------------
    # Trend guard
    # Do not strongly fight persistent trends.
    # ------------------------------------------------------------
    if nt > TREND_LOOKBACK + TREND_WIN:
        recent_trend = log_prices[:, -1] - log_prices[:, -TREND_WIN - 1]
        daily_vol = log_rets[:, -TREND_LOOKBACK:].std(axis=1)

        trend_z = recent_trend / np.where(
            daily_vol > 1e-8,
            daily_vol * np.sqrt(TREND_WIN),
            1.0
        )

        # signal > 0 means long, signal < 0 means short.
        # trend_z < 0 means downtrend, trend_z > 0 means uptrend.
        # signal * trend_z < 0 means we are fighting the trend.
        fighting_trend = signal * trend_z < 0

        danger = fighting_trend & (np.abs(trend_z) > TREND_Z_DANGER)
        warn = fighting_trend & (np.abs(trend_z) > TREND_Z_WARN) & ~danger

        regime_scale = np.where(danger, regime_scale * 0.35, regime_scale)
        regime_scale = np.where(warn, regime_scale * 0.70, regime_scale)

    # ------------------------------------------------------------
    # Final target positions
    # ------------------------------------------------------------
    target_dollars = (
        LIMITS
        * np.tanh(SIGNAL_SCALE * signal)
        * POSITION_MULT
        * regime_scale
        * global_scale
    )

    target_shares = target_dollars / cur
    return target_shares.astype(int)
'''
'''
N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Kalman settings
KALMAN_Q = 1e-4
KALMAN_R = 4e-3

# Trading settings
SIGNAL_SCALE = 1.5
ALGO_SIGNAL_SCALE = 1.7
TOP_K = 20
MIN_ZSCORE = 0.5

# Regime detection settings
INNOV_RECENT_WIN = 20
INNOV_BASE_WIN = 80
BIAS_THRESHOLD = 0.60
VOL_RATIO_WARN = 1.4
VOL_RATIO_DANGER = 2.0

# Trend guard settings
TREND_WIN = 20
TREND_LOOKBACK = 60
TREND_Z_THRESHOLD = 1.7


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape

    if nInst != N_INST or nt < 35:
        return np.zeros(nInst, dtype=int)

    log_prices = np.log(np.maximum(prcSoFar, EPS))
    cur = np.maximum(prcSoFar[:, -1], 1.0)

    target_dollars = np.zeros(nInst, dtype=float)

    # ------------------------------------------------------------
    # Broad market stress filter using ALGO/index volatility.
    # Since ALGO is the index, use it to detect market-wide stress.
    # ------------------------------------------------------------
    global_scale = 1.0

    if nt > 90:
        algo_rets = np.diff(log_prices[0])

        recent_market_vol = algo_rets[-10:].std()
        normal_market_vol = algo_rets[-80:-10].std()

        if normal_market_vol > 1e-8:
            market_vol_ratio = recent_market_vol / normal_market_vol

            if market_vol_ratio > 2.0:
                global_scale = 0.50
            elif market_vol_ratio > 1.5:
                global_scale = 0.75

    # ------------------------------------------------------------
    # Per-instrument Kalman mean reversion
    # ------------------------------------------------------------
    for i in range(nInst):
        obs = log_prices[i]

        state = obs[0]
        variance = 1.0
        innovations = np.zeros(nt - 1, dtype=float)

        for t in range(1, nt):
            # Prediction uncertainty grows slightly each day
            variance += KALMAN_Q

            # Surprise: actual price minus predicted/fair price
            innovation = obs[t] - state

            # Kalman update
            gain = variance / (variance + KALMAN_R)
            state += gain * innovation
            variance *= 1.0 - gain

            innovations[t - 1] = innovation

        recent = innovations[-30:]
        sigma = recent.std()

        if sigma < 1e-8:
            continue

        # Mean reversion signal:
        # positive zscore => price below fair value => long
        # negative zscore => price above fair value => short
        zscore = -(innovations[-1] / sigma)

        if abs(zscore) < MIN_ZSCORE:
            continue

        # --------------------------------------------------------
        # Regime protection 1:
        # If recent innovations are persistently one-sided,
        # the model is being surprised in the same direction.
        # That suggests trend/regime shift, not temporary noise.
        # --------------------------------------------------------
        regime_scale = 1.0

        if nt > INNOV_BASE_WIN + INNOV_RECENT_WIN + 5:
            recent_innov = innovations[-INNOV_RECENT_WIN:]
            base_innov = innovations[-(INNOV_BASE_WIN + INNOV_RECENT_WIN):-INNOV_RECENT_WIN]

            recent_sigma = recent_innov.std()
            base_sigma = base_innov.std()

            # One-sided bias in innovations
            bias_strength = abs(recent_innov.mean()) / (recent_sigma + EPS)

            if bias_strength > 0.90:
                regime_scale *= 0.35
            elif bias_strength > BIAS_THRESHOLD:
                regime_scale *= 0.65

            # Innovation volatility spike
            if base_sigma > 1e-8:
                vol_ratio = recent_sigma / base_sigma

                if vol_ratio > VOL_RATIO_DANGER:
                    regime_scale *= 0.40
                elif vol_ratio > VOL_RATIO_WARN:
                    regime_scale *= 0.70

        # --------------------------------------------------------
        # Regime protection 2:
        # Trend guard.
        # Do not aggressively buy into a strong downtrend,
        # or short into a strong uptrend.
        # --------------------------------------------------------
        if nt > TREND_LOOKBACK + TREND_WIN:
            recent_trend = obs[-1] - obs[-TREND_WIN - 1]
            daily_vol = np.diff(obs[-TREND_LOOKBACK:]).std()

            if daily_vol > 1e-8:
                trend_z = recent_trend / (daily_vol * np.sqrt(TREND_WIN))

                # zscore > 0 means strategy wants to go long.
                # trend_z < 0 means asset is trending down.
                # So product < 0 means we are fighting the trend.
                fighting_trend = zscore * trend_z < 0

                if fighting_trend and abs(trend_z) > TREND_Z_THRESHOLD:
                    regime_scale *= 0.35
                elif fighting_trend and abs(trend_z) > 1.2:
                    regime_scale *= 0.70

        # --------------------------------------------------------
        # Convert signal into dollar position
        # --------------------------------------------------------
        if i == 0:
            bounded_signal = np.tanh(ALGO_SIGNAL_SCALE * zscore)
        else:
            bounded_signal = np.tanh(SIGNAL_SCALE * zscore)

        target_dollars[i] = (
            LIMITS[i]
            * bounded_signal
            * regime_scale
            * global_scale
        )

    # ------------------------------------------------------------
    # Trade ALGO plus top-K strongest non-ALGO opportunities
    # ------------------------------------------------------------
    strongest = np.argsort(np.abs(target_dollars[1:]))[-TOP_K:] + 1

    filtered = np.zeros_like(target_dollars)
    filtered[0] = target_dollars[0]
    filtered[strongest] = target_dollars[strongest]

    target_shares = filtered / cur
    return target_shares.astype(int)


'''
'''
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
    target_dollars = LIMITS * np.tanh(0.85 * signal)*2

    # Extra ALGO scaling is already naturally handled by its larger $100k cap and
    # lower commission in the official evaluator, so no special-case logic needed.
    target_shares = target_dollars / cur
    return target_shares.astype(int)


'''
'''
import numpy as np

N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
EPS = 1e-12

# Kalman settings
KALMAN_Q = 1e-4      # how fast the hidden fair value is allowed to move
KALMAN_R = 4e-3      # how noisy we think observed prices are

# Trading settings
SIGNAL_SCALE = 1.2
POSITION_MULT = 1.2
MIN_ZSCORE = 0.4


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape

    if nInst != N_INST or nt < 35:
        return np.zeros(nInst, dtype=int)

    cur_prices = np.maximum(prcSoFar[:, -1], 1.0)
    log_prices = np.log(np.maximum(prcSoFar, EPS))

    target_dollars = np.zeros(nInst, dtype=float)

    for i in range(nInst):
        obs = log_prices[i]

        # --------------------------------------------------------
        # Kalman filter estimates the smooth fair-value curve.
        # We update only up to yesterday, then compare today's
        # price to yesterday's fair-value estimate.
        # --------------------------------------------------------
        state = obs[0]
        variance = 1.0

        past_innovations = []

        for t in range(1, nt - 1):
            # Prediction step
            variance += KALMAN_Q

            # Surprise versus current fair-value estimate
            innovation = obs[t] - state
            past_innovations.append(innovation)

            # Update step
            gain = variance / (variance + KALMAN_R)
            state += gain * innovation
            variance *= (1.0 - gain)

        if len(past_innovations) < 20:
            continue

        past_innovations = np.array(past_innovations)

        # Today's deviation from the smoothed Kalman fair value
        today_innovation = obs[-1] - state

        sigma = past_innovations[-30:].std()
        if sigma < 1e-8:
            continue

        # Mean-reversion signal:
        # price above fair value => negative signal => short
        # price below fair value => positive signal => long
        zscore = -today_innovation / sigma

        if abs(zscore) < MIN_ZSCORE:
            continue

        bounded_signal = np.tanh(SIGNAL_SCALE * zscore)

        target_dollars[i] = (
            LIMITS[i]
            * bounded_signal
            * POSITION_MULT
        )

    target_shares = target_dollars / cur_prices
    return target_shares.astype(int)
'''

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
'''
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
'''
'''
# Final live version:
# - keeps the strong 8d/60d mean-reversion core from day 2
# - adds a very small basket-arbitrage overlay discovered and validated
#   causally on trailing data
# The overlay is intentionally conservative because split testing showed that
# basket residuals were helpful mainly as a sleeve, not as a standalone book.

import numpy as np

N_INST = 51
LIMITS = np.array([100_000.0] + [10_000.0] * 50)
FAST_LB = 8
SLOW_LB = 60
FAST_WEIGHT = 0.70
SIGNAL_SCALE = 2.0
OVERDRIVE = 4.0
MIN_HISTORY = SLOW_LB + 2
EPS = 1e-12

# Basket overlay settings
BASKET_POOL = 8
BASKET_SIZE = 4
BASKET_TRAIN = 180
BASKET_VAL = 40
BASKET_ALPHA = 5.0
BASKET_REFIT = 10
BASKET_ZWIN = 18
BASKET_ENTRY = 1.7
BASKET_EXIT = 0.5
BASKET_STOP = 4.2
BASKET_MAX_HOLD = 12
BASKET_MIN_CORR = 0.45
BASKET_MIN_VAL_SHARPE = 0.30
BASKET_TOP_ACTIVE = 1
BASKET_CAPACITY_FRACTION = 0.035
BASKET_MIN_HISTORY = BASKET_TRAIN + BASKET_VAL + BASKET_ZWIN + 5

_BASKET_STATE = {
    "last_refit_nt": -1,
    "models": [],
    "relation_state": {},
}


def reset_state():
    _BASKET_STATE["last_refit_nt"] = -1
    _BASKET_STATE["models"] = []
    _BASKET_STATE["relation_state"] = {}


def _base_target_dollars(prcSoFar):
    n_inst, nt = prcSoFar.shape
    if n_inst != N_INST or nt < MIN_HISTORY:
        return np.zeros(n_inst, dtype=float)

    cur = np.maximum(prcSoFar[:, -1], 1.0)

    def zscore_to_past(lb):
        hist = prcSoFar[:, -lb - 1:-1]
        mu = hist.mean(axis=1)
        sig = hist.std(axis=1)
        sig = np.where(sig > 1e-8, sig, 1.0)
        return (mu - cur) / sig

    signal = FAST_WEIGHT * zscore_to_past(FAST_LB) \
        + (1.0 - FAST_WEIGHT) * zscore_to_past(SLOW_LB)
    return LIMITS * np.tanh(SIGNAL_SCALE * signal) * OVERDRIVE


def _ridge_fit(x, y, alpha):
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    intercept = y_mean - float(x_mean @ beta)
    return intercept, beta


def _validation_spread_sharpe(spread):
    if spread.size <= BASKET_ZWIN + 1:
        return 0.0

    position = 0
    hold_days = 0
    daily_pl = []

    for t in range(BASKET_ZWIN, spread.size - 1):
        hist = spread[t - BASKET_ZWIN:t]
        hist_std = float(np.std(hist))
        if hist_std < 1e-10:
            zscore = 0.0
        else:
            zscore = (spread[t] - float(np.mean(hist))) / hist_std

        if position == 0:
            if zscore >= BASKET_ENTRY:
                position = -1
                hold_days = 0
            elif zscore <= -BASKET_ENTRY:
                position = 1
                hold_days = 0
        else:
            hold_days += 1
            if abs(zscore) <= BASKET_EXIT or abs(zscore) >= BASKET_STOP or hold_days >= BASKET_MAX_HOLD:
                position = 0
                hold_days = 0

        daily_pl.append(position * -(spread[t + 1] - spread[t]))

    pnl = np.asarray(daily_pl, dtype=float)
    pnl_std = float(np.std(pnl))
    if pnl.size == 0 or pnl_std < 1e-10:
        return 0.0
    return float(np.sqrt(250.0) * np.mean(pnl) / pnl_std)


def _discover_basket_models(prcSoFar):
    log_prices = np.log(np.maximum(prcSoFar[:, -(BASKET_TRAIN + BASKET_VAL + 1):], EPS))
    returns = np.diff(log_prices, axis=1)
    corr = np.nan_to_num(np.corrcoef(returns[:, :BASKET_TRAIN]), nan=0.0)
    models = []

    for target in range(1, N_INST):
        ordered = np.argsort(-np.abs(corr[target]))
        pool = [idx for idx in ordered if idx != target][:BASKET_POOL]
        if len(pool) < BASKET_SIZE:
            continue

        x_train_full = log_prices[pool, :BASKET_TRAIN].T
        y_train = log_prices[target, :BASKET_TRAIN]
        _, beta_full = _ridge_fit(x_train_full, y_train, BASKET_ALPHA)
        selected = np.argsort(np.abs(beta_full))[-BASKET_SIZE:]
        hedge_idx = np.asarray([pool[idx] for idx in selected], dtype=int)

        x_train = log_prices[hedge_idx, :BASKET_TRAIN].T
        intercept, beta = _ridge_fit(x_train, y_train, BASKET_ALPHA)

        x_val = log_prices[hedge_idx, BASKET_TRAIN:BASKET_TRAIN + BASKET_VAL].T
        y_val = log_prices[target, BASKET_TRAIN:BASKET_TRAIN + BASKET_VAL]
        pred_val = intercept + x_val @ beta
        spread_val = y_val - pred_val

        if spread_val.size <= BASKET_ZWIN + 1:
            continue

        y_val_diff = np.diff(y_val)
        pred_val_diff = np.diff(pred_val)
        if np.std(y_val_diff) < 1e-10 or np.std(pred_val_diff) < 1e-10:
            continue

        corr_val = float(np.corrcoef(y_val_diff, pred_val_diff)[0, 1])
        if corr_val < BASKET_MIN_CORR or float(np.std(spread_val)) < 0.01:
            continue

        val_sharpe = _validation_spread_sharpe(spread_val)
        if val_sharpe < BASKET_MIN_VAL_SHARPE:
            continue

        capacity = float(LIMITS[target])
        for hedge, coeff in zip(hedge_idx, beta):
            capacity = min(capacity, float(LIMITS[hedge]) / max(abs(float(coeff)), 1e-6))

        score = val_sharpe * corr_val
        models.append(
            (
                score,
                target,
                hedge_idx,
                float(intercept),
                np.asarray(beta, dtype=float),
                BASKET_CAPACITY_FRACTION * capacity,
            )
        )

    models.sort(key=lambda item: item[0], reverse=True)
    return models[:10]


def _basket_overlay_dollars(prcSoFar):
    _, nt = prcSoFar.shape
    if nt < BASKET_MIN_HISTORY:
        return np.zeros(N_INST, dtype=float)

    if (
        _BASKET_STATE["last_refit_nt"] < 0
        or nt - _BASKET_STATE["last_refit_nt"] >= BASKET_REFIT
    ):
        _BASKET_STATE["models"] = _discover_basket_models(prcSoFar)
        _BASKET_STATE["last_refit_nt"] = nt

    models = _BASKET_STATE["models"]
    if not models:
        return np.zeros(N_INST, dtype=float)

    log_prices = np.log(np.maximum(prcSoFar, EPS))
    desired = np.zeros(N_INST, dtype=float)
    ranked = []

    for score, target, hedge_idx, intercept, beta, capacity in models:
        spread = log_prices[target, -(BASKET_ZWIN + 2):] - (
            intercept + log_prices[hedge_idx, -(BASKET_ZWIN + 2):].T @ beta
        )
        hist = spread[:-1]
        hist_std = float(np.std(hist))
        if hist_std < 1e-10:
            continue

        zscore = (spread[-1] - float(np.mean(hist))) / hist_std
        direction, hold_days = _BASKET_STATE["relation_state"].get(target, (0, 0))

        if direction == 0:
            if zscore >= BASKET_ENTRY:
                direction = -1
                hold_days = 1
            elif zscore <= -BASKET_ENTRY:
                direction = 1
                hold_days = 1
        else:
            hold_days += 1
            if abs(zscore) <= BASKET_EXIT or abs(zscore) >= BASKET_STOP or hold_days >= BASKET_MAX_HOLD:
                direction = 0
                hold_days = 0

        _BASKET_STATE["relation_state"][target] = (direction, hold_days)

        if direction != 0:
            ranked.append((abs(zscore) * score, direction, capacity, target, hedge_idx, beta))

    ranked.sort(key=lambda item: item[0], reverse=True)

    for _, direction, capacity, target, hedge_idx, beta in ranked[:BASKET_TOP_ACTIVE]:
        leg_dollars = np.zeros(N_INST, dtype=float)
        leg_dollars[target] = direction * capacity
        leg_dollars[hedge_idx] = -direction * capacity * beta

        used = np.where(np.abs(leg_dollars) > 1e-8)[0]
        if used.size == 0:
            continue

        remaining = LIMITS[used] - np.abs(desired[used])
        leg_abs = np.abs(leg_dollars[used])
        scale = float(np.min(np.where(leg_abs > 1e-8, remaining / leg_abs, 1.0)))
        scale = min(1.0, max(0.0, scale))
        if scale < 0.20:
            continue

        desired += scale * leg_dollars

    return desired


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_inst, _ = prcSoFar.shape
    if n_inst != N_INST:
        return np.zeros(n_inst, dtype=int)

    cur = np.maximum(prcSoFar[:, -1], 1.0)
    base_dollars = _base_target_dollars(prcSoFar)
    overlay_dollars = _basket_overlay_dollars(prcSoFar)
    target_dollars = np.clip(base_dollars + overlay_dollars, -LIMITS, LIMITS)
    return (target_dollars / cur).astype(int)
'''
