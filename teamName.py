import numpy as np
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

    # Trade only the selected assets.
    mask = np.zeros(nInst, dtype=float)
    mask[CORE_ASSETS] = 1.0
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

    target_shares = target_dollars / cur
    return target_shares.astype(int)

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
