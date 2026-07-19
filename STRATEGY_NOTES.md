# Strategy Notes

Local experiments on the provided `prices.txt` were run with the official `eval.py` scoring logic, so these findings are useful for iteration but still in-sample and not guaranteed to hold on hidden data.

## ALGO-Aware Ideas Tested

- `Synthetic index spread`: build `mean(normalised prices of assets 1-50)` and compare it to ALGO.
- `Hedge with ALGO`: offset stock-book market exposure with ALGO.
- `Residual mean reversion`: trade stock moves relative to ALGO rather than raw moves.
- `Beta-neutral residuals`: estimate each stock's beta to ALGO, then trade the residual.
- `ALGO forecast`: use basket-vs-ALGO divergence as an ALGO timing sleeve.

## What Helped

- The best local improvement came from **keeping the existing raw mean-reversion core** and adding a **small residual overlay**.
- The overlay uses beta-adjusted residual returns versus ALGO and only boosts the **top 3 strongest idiosyncratic mean-reversion names** each day.
- This preserved most of the base strategy's PnL while slightly improving the mean/std tradeoff in local tests.

## What Did Not Help Much

- Replacing the whole strategy with only `top 3` trades reduced volatility, but cut mean PnL too hard.
- Pure residual-only or heavy beta-neutral books were cleaner statistically, but weaker economically on the local sample.
- Synthetic-index and ALGO-prediction sleeves were directionally interesting, but only added small gains relative to the simpler residual overlay.

## Prosperity-4 README Ideas (2026-07-12)

Tested ideas borrowed from `IMC-Prosperity-4-main-strat/README.md`:

- **Hysteresis + dollar dead-band (adopted)**: enter at |z| >= 0.25, hold an
  existing position down to |z| >= 0.20 on the same side, and skip retrades
  smaller than 15% of the instrument limit. Eval score 373.15 -> 377.99;
  survived three chronological windows (early/mid/late). Gains are modest
  because fees are only 1bp, but turnover dropped too.
- **EV-proxy asset ranking (rejected)**: ranked all 51 names by
  `(sigma/cost) / half_life` on days 0-249 (`backtesting/rank_assets.py`).
  The top-20 EV list scored 275.96 OOS on days 250-500 vs 377.99 for the
  existing hand-picked CORE_ASSETS, so the current list stays. The proxy
  captures reversion speed vs cost but not predictability.
- **Per-asset PnL decomposition (adopted as tooling)**: `backtest.py
  --per-asset` splits PnL into carry vs commission per instrument. On the
  full sample all 20 core names are net-positive; on the last 250 days only
  17 and 39 were slightly negative (kept: removing them hurt the OOS score
  via diversification loss).
- **Chronological OOS validation (process rule)**: any retuned parameter must
  survive train-early/test-late splits, not just the full sample.

## ALGO Identity + Index-Fade Sleeve (2026-07-12)

- **ALGO solved exactly**: ALGO(t) = mean_i(p_i(t)/p_i(0)) * 100, contemporaneous.
  Residual is price-rounding noise (sigma 0.003, max 0.008) — no index arb, no
  lag, no reversal. See `visualisations/eda_baskets.py` for the EDA.
- **Regime facts**: stock mean reversion only pays in ALGO-down 20d regimes
  (corr -0.075 vs +0.013 in up regimes); ALGO itself mean-reverts (past-20d vs
  next-5d corr -0.16). No momentum at any horizon; downtrend is not
  stock-persistent (1st/2nd-half drift corr -0.03).
- **ALGO 40d-fade sleeve (adopted)**: fade ALGO's trailing 40d vol-scaled move,
  cap $60k, on top of the main book. All-window gains: early 252.9->265.2,
  mid 364.1->371.1, eval 378.0->378.7. Chosen from a plateau (cap 30-60k all
  positive; 100k flips eval negative).
- **Exact-residual MR sleeve (rejected)**: residual_i = normalized price minus
  index, faded z-score. Positive standalone early but hurt the combined book
  on the eval window (354 vs 378) — overlaps with the existing MR core.

## OOS Post-Mortem: Days 501-750 Leaderboard (2026-07-12)

Submitted book scored mean -45.89/day, std 1090 on the hidden window
(days 501-750, continuation of our series; see Scoring & Evaluation PDF).

- **The loss is within noise of zero edge**: SE of a 250-day mean at std 1090
  is ~69, so t = -0.67. Synthetic no-edge simulations of our book produced
  -84/-83/-225 purely from noise. Conclusion: our edge was ~0 OOS, not
  negative; local 410/day (t=3.4) is inflated by in-sample selection.
- **Time-reversed stress test passes** (+124 score) - the strategy survives
  up-regimes; what died OOS was mean reversion itself, not trend direction.
- **Kill-switch rejected**: detecting edge death at our SNR needs ~560 days
  of evidence (>2x the scoring window). A rolling-PnL de-risking overlay
  cost ~17 points in every good window and was inconsistent on no-edge data.
- **All-51-name book rejected**: sigma +60% (1850->3000) with mostly lower
  mean across all windows incl. reversed. The core-20 names genuinely have
  lower vol and cleaner reversion; not pure selection bias.
- **Plan**: when days 501-750 are released ahead of the General Round,
  first re-run the per-asset decomposition and CORE_ASSETS/parameter
  validation on that fresh data before changing anything else.

## Pairs-Trading Sleeve (2026-07-12)

Motivated by leaderboard evidence (a pairs-trading team scoring 350 OOS)
that pair relationships persist on hidden days.

- **Scan**: all 1225 pairs; OLS hedge ratio on log prices, AR(1) spread
  half-life < 20d, z-score reversion backtest (entry 1.5 / exit 0.5,
  rolling 60d) with fees.
- **Honest procedure validation**: pairs selected on days 0-249 ONLY earned
  81/day at annSR 3.1 (score 73 standalone) on untouched days 250-499.
  52% of screened pairs were profitable in both halves vs ~25% by chance.
- **Final selection** (full sample, both-half profitable, SR >= 1.5, max 2
  pairs per name): 15 pairs, $9k legs, hardcoded in teamName.py with frozen
  gammas. Product ownership: the 22 pair-owned names are excluded from the
  main MR book (combining with overlap scored 436; ownership 440 with lower
  std).
- **Result**: eval score 378.67 -> 529.57 (annSharpe 3.44 -> 5.26, std down).
  All windows improved: early 265->470, mid 371->531, reversed 124->344.
- **Caveat**: the 529 is partly in-sample (final pairs use all 500 days);
  the honest OOS estimate of the sleeve's added value is ~+60 score from
  the train-only-selection experiment.

## Lead-Lag Sleeve (2026-07-12)

Motivated by research into past Algothon winners: the 2024 winner (600
teams) used a lead-lag algorithm scoring 1560 — structure the generator
apparently reuses.

- **Generator fingerprint**: returns are synthetic Gaussian (excess
  kurtosis ~0, no vol clustering, no own-autocorrelation, prices 2dp).
  Not disguised real stocks; linear cross-structure is the whole game.
- **Lead-lag confirmed in our data**: ridge r(t+1) ~ all r(t) earns mean
  walk-forward IC 0.056, 17/51 names above 0.1 (top 0.22) on a strict
  chronological split. Lag-1 pair scan: 3 pairs above 4-sigma vs 0.3
  expected (9->15, 9->5, 4->40). No signal at lags 2-3 above threshold.
- **Sleeve design** (`backtesting/leadlag_research.py`): ridge W refit
  every 50d on all history (lam=400), full-tilt sign positions, names
  masked by *walk-forward* IC > 0 (live predictions vs realisations,
  never in-sample fit — in-sample ICs are inflated and select nothing).
  Pair-owned names excluded to protect pair hedge ratios from clipping.
- **Robustness**: all 48 grid configs positive on honest OOS
  (train<250/test 250-499), early AND time-reversed windows. PnL corr
  with the existing book: -0.14.
- **Result**: eval 529.57 -> 617.25 (annSharpe 5.76); early 427 -> 507;
  reversed 344 -> 688. Sleeve added after the dead-band (needs daily
  rebalance; fees ~1bp are negligible vs the IC).

## Architecture Ablations (2026-07-15, `backtesting/fsm_experiment.py`)

- **ALGO fade sizing**: fade is worth ~+4.5 eval points at cap $60k; caps of
  $100k or bigger scales do NOT help (edge is 20-40d reversion, ~6-12
  independent bets/250d — more size adds variance, not signal). ALGO's
  $100k limit is already heavily monetised by the lead-lag sleeve
  ($992/day dollar-vol vs $212 for a maxed stock). Giving the fade
  priority over lead-lag on ALGO trades small early-window gains for a
  688->628 reversed-window loss. Keep cap $60k.
- **Pure "pairs + MLR + FSM" book (rejected)**: dropping the MR core and
  fade costs 147 eval points (617 -> 470) and 55 early points. The MR
  core earns its keep alongside the lead-lag sleeve.
- **FSM dead-zone on the lead-lag sleeve (rejected)**: entry/exit bands
  on the prediction z (0.1-0.3) strictly hurt (eval 617 -> 601/545/540).
  The ridge forecast refreshes daily and fees are 1bp, so holding
  yesterday's state means holding a stale forecast. Daily sign-flipping
  is optimal; FSMs belong on the slow signals (pairs 1.5/0.5, hysteresis
  0.25/0.20) where they already are.

## Regime FSM (2026-07-15, adopted)

Converted the three emergency regime guards (market vol shock, per-asset
vol shock, trend) from stateless daily conditions to finite state
machines: enter stress at DANGER (vol ratio 2.0 / trend z 2.5), exit
only below the lower EXIT threshold (1.5). Locally near-neutral
(eval 617.25 -> 618.68, early -3, reversed -10 — noise) because the
guards rarely fire on the released days; the point is defensive: after
a real shock on hidden data, the stateless guards re-risked the next
day, the FSM stays cut until vol genuinely normalises.

## Current Implementation

- Raw 8-day and 30-day mean reversion still drives the main target book.
- A lead-lag sleeve (online ridge, walk-forward IC mask, sign sizing at
  full limits, pair-owned names excluded) is added after the dead-band.
- Hysteresis (entry 0.25 / exit 0.20) and a 15%-of-limit dead-band cut churn.
- A beta-adjusted residual signal versus ALGO is computed for assets `1-50`.
- Only the top 3 residual dislocations receive an extra sizing boost, capped by each instrument's normal limit.
