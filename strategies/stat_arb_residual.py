"""Beta-neutral residual mean-reversion stat-arb.

Idea (standard quant-desk stat arb):
  1. Treat instrument 0 (ALGO) as the market factor.
  2. Estimate each stock's beta to ALGO on a rolling window of log returns.
  3. Compute residual (idiosyncratic) returns: e_i = r_i - beta_i * r_ALGO.
  4. Signal = z-score of the cumulative residual over a short horizon;
     fade it (mean reversion of idiosyncratic dislocations).
  5. Size positions inverse to residual vol so every name contributes
     roughly equal risk -> smooth PnL -> high Sharpe.
  6. Demean dollar targets across stocks (dollar-neutral stock book) and
     hedge the remaining beta exposure with ALGO (cheap commission, big limit).
  7. Turnover buffer: only rebalance a name when its target drifts far from
     the held position, keeping commission drag low.

Optimises for Sharpe, not raw PnL.
"""

import numpy as np

N_INST = 51
EPS = 1e-12

# --- parameters (chosen from the centre of a robust plateau across two
# non-overlapping validation segments, not the single best backtest cell) ---
BETA_WINDOW = 120        # days for beta estimation
VOL_WINDOW = 30          # days for residual vol estimation
REV_HORIZONS = (20, 60)  # ensemble of cumulative-residual lookbacks
Z_CAP = 2.5              # clamp extreme z-scores
GROSS_STOCK = 90_000.0   # total gross dollar budget across the stock book
PER_NAME_CAP = 9_000.0   # stay inside the $10k per-stock limit with margin
REBALANCE_BAND = 0.40    # skip trades smaller than this fraction of target
MIN_HISTORY = BETA_WINDOW + max(REV_HORIZONS) + 5

_prev_pos = np.zeros(N_INST)


def reset_state():
    global _prev_pos
    _prev_pos = np.zeros(N_INST)


def getMyPosition(prcSoFar):
    global _prev_pos
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_inst, nt = prcSoFar.shape
    if nt < MIN_HISTORY:
        _prev_pos = np.zeros(n_inst)
        return np.zeros(n_inst, dtype=int)

    cur = np.maximum(prcSoFar[:, -1], EPS)
    log_p = np.log(np.maximum(prcSoFar, EPS))
    rets = np.diff(log_p, axis=1)               # (n_inst, nt-1)

    win = rets[:, -BETA_WINDOW:]
    mkt = win[0]                                # ALGO returns
    mkt_c = mkt - mkt.mean()
    var_m = float(mkt_c @ mkt_c) + EPS

    stock = win[1:]                             # (50, W)
    stock_c = stock - stock.mean(axis=1, keepdims=True)
    betas = (stock_c @ mkt_c) / var_m           # (50,)
    betas = np.clip(betas, -3.0, 3.0)

    # residual returns over the beta window
    resid = stock - betas[:, None] * mkt[None, :]   # (50, W)

    resid_vol = resid[:, -VOL_WINDOW:].std(axis=1)
    resid_vol = np.maximum(resid_vol, 1e-5)

    # z-score of the cumulative residual move, averaged over an ensemble of
    # horizons (single horizons were fragile across validation segments)
    z = np.zeros(n_inst - 1)
    for h in REV_HORIZONS:
        cum = resid[:, -h:].sum(axis=1)
        z += np.clip(cum / (resid_vol * np.sqrt(h)), -Z_CAP, Z_CAP)
    z /= len(REV_HORIZONS)

    # fade the dislocation, risk-parity weights via inverse residual vol
    raw = -z / resid_vol
    # dollar-neutral stock book
    raw = raw - raw.mean()
    gross = np.sum(np.abs(raw))
    if gross < EPS:
        return _prev_pos.astype(int)
    dollars = raw * (GROSS_STOCK / gross)
    dollars = np.clip(dollars, -PER_NAME_CAP, PER_NAME_CAP)

    # hedge residual beta exposure with ALGO
    beta_dollars = float(np.sum(betas * dollars))
    hedge = np.clip(-beta_dollars, -95_000.0, 95_000.0)

    target_dollars = np.empty(n_inst)
    target_dollars[0] = hedge
    target_dollars[1:] = dollars

    target = target_dollars / cur

    # turnover buffer: keep old position unless target moved materially
    held_dollars = _prev_pos * cur
    drift = np.abs(target_dollars - held_dollars)
    band = REBALANCE_BAND * np.maximum(np.abs(target_dollars), 1_000.0)
    keep = drift < band
    keep[0] = False  # ALGO hedge is cheap to trade; always track it
    target[keep] = _prev_pos[keep]

    pos = target.astype(int)
    _prev_pos = pos.astype(float)
    return pos
