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
