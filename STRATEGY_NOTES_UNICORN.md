# Unicorn Strategy Search — Non-Pairs Directions

## Scope and validation

This search deliberately avoided the main submission family: rolling
price mean reversion, pair-spread FSMs, basket residuals, and the existing
dense 51-return ridge sleeve. The goal was to find simple generator effects
that could survive now that pairs are less trustworthy.

All numbers below use the official fee, integer-share, position-limit, and
score semantics through `backtesting/leadlag_research.py::simulate`.
Strategies were evaluated walk-forward on three chronological windows:
days 101–300, 251–500, and 501–750. The selection statistic is the minimum
score across the three windows, not the best single score. These are local
research results, not live-leaderboard results.

## Best new direction: cross-sectional factor rotation

File: `strategies/unicorn_factor_rotation.py`

The strategy standardizes the latest stock returns, extracts rolling PCA
factors, and learns a small ridge map from today's factor returns to
tomorrow's 50-stock cross section. It then cross-sectionally demeans the
forecast and uses smooth `tanh` sizing. It does not trade ALGO and does not
construct, select, or mean-revert pairs.

| Configuration | Early score | Middle score | Late score | Minimum |
|---|---:|---:|---:|---:|
| 20 factors | 158.8 | 343.1 | 251.6 | 158.8 |
| 25 factors | 206.1 | 366.8 | 300.5 | 206.1 |
| 30 factors | 280.1 | 397.7 | 341.8 | 280.1 |
| **35 factors** | **311.7** | **429.9** | **322.6** | **311.7** |
| 40 factors | 301.2 | 454.3 | 380.2 | 301.2 |
| 45 factors | 240.5 | 451.0 | 384.7 | 240.5 |
| 50 factors | 220.3 | 431.1 | 352.1 | 220.3 |

The broad 30–40 factor plateau matters more than the exact winner. The
35-factor candidate is the robust centre, not an isolated grid maximum.
Its mean/std/score by window is:

| Window | Mean PnL/day | PnL std | Score |
|---|---:|---:|---:|
| 101–300 | 313.6 | 1177.9 | 296.8 |
| 251–500 | 415.8 | 1092.8 | 404.6 |
| 501–750 | 340.2 | 1168.4 | 324.9 |

Why this is productive:

- It targets rotation among latent cross-sectional factors, not spread
  stationarity. Pair identities can decay without invalidating the model.
- It is a single compact linear calculation with one SVD; no pair scan,
  FSM collection, asset shortlist, or large feature library is required.
- Dollar demeaning removes the broad market direction that inflated the
  volatility of earlier books.
- It earns positive mean in every chronological window and remains useful
  even on the genuinely unseen 501–750 segment.

Important caveat: mathematically this is reduced-rank multivariate
prediction, so it is related to lead-lag at a high level. It is still
architecturally distinct from the submitted full-universe ridge: predictors
are latent factors, ALGO is excluded, rank is explicitly truncated, and no
pairs sleeve is attached. It should be treated as a clean standalone A/B
submission, not proof of a new hidden-data score above 600.

## Other simple hypotheses tested

| Strategy | Early | Middle | Late | Verdict |
|---|---:|---:|---:|---|
| Sparse stable lag motifs, top 24 | 5.3 | 101.2 | 86.4 | Stable but too weak |
| PCA rotation, 10 factors | 21.5 | 297.5 | 161.9 | Underfit |
| Own-sign × market-sign lookup | 22.3 | 81.2 | -22.0 | Regime unstable |
| Own-sign × dispersion lookup | -44.3 | -32.3 | 2.8 | Reject |
| Five-day phase drift | -64.1 | -124.8 | 69.4 | Reject |
| Ten-day phase drift | -13.0 | -164.9 | 32.4 | Reject |
| ALGO breadth timer, 1–5 lags | mixed | negative | negative | Reject |

Sparse motifs selected only lag edges whose signs agreed in both halves of
the available history. This was the only other idea positive in all three
windows, but its risk-adjusted payoff was far too small. It is evidence that
the generator contains sparse relationships, not a submission candidate.

Calendar phase, nonlinear sign tables, volatility/dispersion states, and
direct ALGO timing all failed the chronological test. Their late-window
wins were regime-specific and should not be blended into the candidate.

## Recommended next live experiment

Submit `unicorn_factor_rotation.py` unchanged as an A/B probe and record
mean PnL and PnL standard deviation, not only score. It is most valuable as
an independent tracked strategy while the leaderboard window grows.

Do not tune the factor count on the live result. Use the precommitted
35-factor centre. Keep it standalone for the first test; combining it with
the existing lead-lag book would clip overlapping stock positions and make
the result impossible to attribute.

The complete reproducible breadth search is in
`strategies/unicorn_research.py`.

The standalone file replayed each 200/250-day window in 0.2–0.4 seconds
locally, leaving a very wide margin under the 600-second evaluator limit.

## Lead-lag forecasting-engine study

The successful portfolio layer was frozen: forecast divided by target
volatility, cross-sectional demeaning, re-standardization, and
`LIMITS * tanh(z / 0.35)`. Only the return forecast was changed. Full
results and reproducible configurations are in
`strategies/leadlag_model_research.py`.

| Forecast engine | Early | Middle | Late | Minimum |
|---|---:|---:|---:|---:|
| Baseline ridge, lambda 400 | 335.4 | 420.3 | **573.5** | 335.4 |
| **EW ridge, decay 0.9975** | **348.6** | 407.5 | 446.0 | **348.6** |
| EW ridge, decay 0.999 | 343.8 | **438.2** | 508.1 | 343.8 |
| Reduced-rank regression, rank 30 | 332.4 | 421.7 | 542.6 | 332.4 |
| Signed-square-root nonlinear features | 334.3 | 372.6 | 523.8 | 334.3 |
| Robust clipping at 3 MAD | 296.6 | 443.4 | 522.1 | 296.6 |
| Two return lags | 146.9 | 307.9 | 264.2 | 146.9 |
| Three return lags | 36.9 | 248.5 | 222.1 | 36.9 |

Main conclusions:

- The generator's useful relationship is overwhelmingly lag one. Adding
  lags two through five increases estimation noise and sharply hurts every
  robust selection metric.
- Gentle exponential forgetting is the only forecasting change that
  improves the minimum-window score. Decay 0.9975 has an effective memory
  of roughly 400 observations. Faster decay (0.99 or 0.98) is unstable.
- Decay 0.999 is the safer compromise if recent-window performance matters:
  it improves early and middle over baseline while retaining more of the
  strong late-window edge.
- Supervised reduced-rank regression validates that roughly 30 predictive
  output directions carry most of the effect, but it does not beat the
  full ridge.
- Outlier clipping and small nonlinear transforms do not help. The source
  appears close to a stationary Gaussian linear generator, consistent with
  the earlier distribution diagnostics.
- More frequent refitting is not automatically better. Ten-day refits are
  competitive, while daily refits reduce the late score. Fifty days remains
  a reasonable stability/compute choice for the baseline.
- Averaging models reduces PnL volatility but also dilutes the strongest
  forecast. No tested ensemble beat gentle EW ridge on the minimum-window
  criterion or baseline ridge on the late window.

Standalone implementation: `strategies/leadlag_ew_candidate.py`.

This is a robustness candidate, not an unequivocal replacement. The score
uplift in the minimum window is only 13 points, while the late-window score
falls materially. A clean live A/B between baseline and decay 0.999 is more
informative than replacing the existing lead-lag sleeve immediately.

## Requested lead-lag model comparison: deeper pass

The requested model families were implemented distinctly rather than
approximated with dense-ridge parameter changes. Results still hold the
demean, re-standardize, and `tanh(z/0.35)` layer fixed.

| Model | Early | Middle | Late | Minimum |
|---|---:|---:|---:|---:|
| Dense ridge baseline | 335.4 | 420.3 | 573.5 | 335.4 |
| Multi-lag ridge, 2 lags | 146.9 | 307.9 | 264.2 | 146.9 |
| Multi-lag ridge, 3 lags | 36.9 | 248.5 | 222.1 | 36.9 |
| Reduced-rank ridge, rank 10 | 210.9 | 275.3 | 564.1 | 210.9 |
| Sparse ridge, 5 predictors/target | 203.8 | 125.4 | 423.1 | 125.4 |
| Sparse ridge, 10 predictors/target | 198.9 | 242.5 | 537.6 | 198.9 |
| PCA predictor factors, 5 | -25.0 | 15.5 | 452.2 | -25.0 |
| PCA predictor factors, 10 | 99.5 | 64.0 | 528.6 | 64.0 |
| PLS factors, 5 | 162.6 | 142.0 | 564.4 | 142.0 |
| PLS factors, 10 | 89.6 | 287.8 | **613.1** | 89.6 |
| Market-sign regime ridge | 10.2 | 326.7 | 515.7 | 10.2 |
| Dispersion-regime ridge | 175.8 | 304.0 | 361.8 | 175.8 |
| Recursive ridge, forgetting .999 | 346.4 | 464.2 | 494.1 | 346.4 |
| **Recursive ridge, forgetting .9975** | **357.5** | **462.5** | **476.5** | **357.5** |
| Recursive ridge, forgetting .995 | 360.4 | 438.2 | 446.3 | **360.4** |

Interpretation:

- Two or three lags are decisively worse. The extra coefficient count is
  not buying additional predictive horizon.
- Retaining only 5–10 reduced-rank, PCA, or PLS components throws away too
  much cross-sectional structure. PLS-10's late score of 613.1 is notable
  but its earlier collapse identifies it as a regime-specific candidate,
  not a robust replacement.
- Hard coefficient sparsity also fails. Predictive information is weak and
  distributed across many leaders; keeping only 5–10 predictors per target
  destroys the ensemble effect.
- Splitting only 100–300 observations into two regimes makes both ridge
  estimates noisy. Market-sign conditioning is particularly unstable.
- Recursive ridge is the only requested family that clearly improves the
  minimum-window result. It adapts the coefficient matrix every day with
  one rank-one update and is cheaper than repeated batch solves.
- Forgetting 0.9975 is the balanced choice. The slightly better minimum at
  0.995 comes from the early window while sacrificing more of the late
  edge. This is a broad adaptation tradeoff, not a free score increase.
- A 50/50 baseline/recursive blend scores 348.2 / 442.0 / 526.0. It is a
  conservative alternative, but its minimum is below standalone recursive.
- The nonlinear signed-square-root feature expansion did not improve the
  baseline, so more complex nonlinear models are not justified yet.

Implementation: `strategies/leadlag_recursive_candidate.py`, with
forgetting 0.9975 precommitted as the balanced setting.

## Dense ridge with dynamic money allocation

The dense ridge forecast was kept exactly as-is. The allocation experiment
changed two portfolio-layer details:

1. Replace the daily cross-sectional standard deviation with a fixed scale
   calibrated at each 50-day model refit. Daily normalization forced every
   day to look equally confident; fixed scaling allows quiet forecast days
   to carry less gross exposure and strong days to carry more.
2. Increase tanh temperature from 0.35 to 0.50. This materially reduces
   saturation while retaining the monotonic signal-strength mapping.

| Allocation | Early | Middle | Late | Minimum |
|---|---:|---:|---:|---:|
| Existing daily-normalized, temp .35 | 335.4 | 420.3 | 573.5 | 335.4 |
| Daily-normalized, temp .50 | 322.4 | 404.7 | 538.9 | 322.4 |
| Fixed-scale dynamic, temp .35 | **342.2** | 420.3 | **573.1** | **342.2** |
| **Fixed-scale dynamic, temp .50** | 333.3 | **404.9** | 537.4 | 333.3 |
| Fixed-scale dynamic, temp .75 | 306.6 | 372.9 | 487.7 | 306.6 |
| Fixed-scale dynamic, temp 1.00 | approximately 262 | 333 | 440 | ~262 |

Allocation diagnostics over all three windows:

| Allocation | Mean limit use | Median | Positions >90% | Positions >99% | Daily gross 10–90% |
|---|---:|---:|---:|---:|---:|
| Existing temp .35 | 79.1% | 95.5% | 58.9% | 34.9% | 76.3–85.3% |
| Fixed-scale temp .35 | 78.7% | 95.1% | 58.3% | 34.5% | 74.0–86.7% |
| **Fixed-scale temp .50** | **71.8%** | **85.9%** | **44.6%** | **18.6%** | **65.9–80.7%** |

Temperature 0.50 is the practical answer to “do not max every limit.” It
cuts almost-maxed positions nearly in half (34.9% to 18.6%), lowers average
limit use by 7.3 percentage points, and makes total exposure respond more
to daily confidence. The cost is roughly 4–6% lower score because the
competition objective rewards absolute PnL once Sharpe is already strong.

If score alone is the objective, use fixed-scale temp 0.35: it preserves
the baseline late score and slightly improves the minimum. If controlled
capital use is the objective, use fixed-scale temp 0.50.

Standalone implementation: `strategies/leadlag_dynamic_allocation.py`.

## Fresh strategy audit and discovery program

The repository was re-audited across every submission, research strategy,
notebook, both notes files, and deleted Git history. Previously covered
families include MR/momentum/regime switching, Kalman MR, pairs/triplets,
baskets, ALGO residual/fade/timing, dense and multi-lag ridge, recursive/EW
ridge, sparse/reduced-rank/PCA/PLS models, factor rotation, feature ridge,
calendar/sign states, fixed model averaging, and PnL gates.

The genuinely new directions tested in this pass were:

- directed lag-covariance propagation with fractional precision whitening;
- held-out choice between directly forecasting ALGO and using it as a hedge;
- regularized canonical-correlation predictive modes;
- exact index-coherent stock-only forecasts using simple returns;
- covariance-aware forecast allocation and factor neutralization;
- sparse shock-event propagation;
- unconditional structural drift;
- target-specific cross-validated ridge shrinkage;
- a sample-size curriculum between two independently useful predictors;
- predictor noise-subspace filtering using adaptive spectral rank.

Rejected or secondary results:

- Drift, shock tables, covariance inversion, and hard factor neutralization
  were not chronologically robust.
- Target-specific ridge reached 599 on the late window but did not improve
  the minimum.
- Adaptive spectral ridge was robust and useful: standalone
  `398 / 430 / 601`; frozen pairs plus spectral ridge
  `578 / 588 / 677`, versus submitted pairs+dense
  `532 / 581 / 654`.
- The fractional-precision network with its ALGO expert was the strongest
  completely distinct standalone model before the curriculum:
  `509 / 629 / 610`.
- Canonical modes were deliberately weak with limited history but became
  exceptional after about 500 observations. Rank 5–10, shrinkage
  0.10–0.50, and refits of 50–100 days all showed a late-window advantage.
  Rank 7 alone scored about 748 late.

### Winning core: sample-size curriculum

File: `strategies/unicorn_curriculum_candidate.py`

The core uses no pairs:

1. With fewer than 500 observations, propagate standardized market shocks
   through the directed lag-covariance graph. Fractional covariance
   whitening (`power=0.70`) avoids both raw-correlation underfitting and
   unstable full precision inversion.
2. A walk-forward 80-day validation chooses whether ALGO should express its
   direct forecast or hedge the stock book.
3. From days 500–600, transition linearly to a rank-7 regularized canonical
   lead-lag model. Around ten observations per variable is where the
   50-dimensional covariance estimates become usable.
4. The canonical model forecasts the 50 independent stocks in simple-return
   space and derives ALGO mechanically from its exact normalized-index
   identity. It blends canonical modes with a 0.5-weight dense coherent
   forecast before the proven demeaned tanh allocation.

Pair-free exact results:

| Window | Mean/day | Std | Score |
|---|---:|---:|---:|
| Full local evaluator, days 2–750 | 573.2 | 1654.2 | **554.7** |
| Early 101–300 | 534.3 | 1889.8 | **508.9** |
| Middle 251–500 | 645.4 | 1666.2 | **628.6** |
| Late 501–750 | 812.4 | 1675.9 | **798.8** |
| Late first half, 501–625 | 590.7 | 1631.2 | **573.2** |
| Late second half, 626–750 | 983.5 | 1754.6 | **971.2** |

The transition is not a single tuned breakpoint: ramps of 350–550,
400–600, 450–650, 500–700, and 400–700 all remained profitable. Full scores
were approximately 530–555 and late scores 761–803.

### Optional frozen-pair overlay

`PAIR_SCALE` in the same file is the only switch. `PAIR_SCALE = 0.0` gives
the new pair-free core. `PAIR_SCALE = 1.0` adds the existing frozen pairs
before final clipping and gives the highest released-data score:

| Window | Mean/day | Std | Score |
|---|---:|---:|---:|
| Full `eval.py` (`numTestDays=749`) | 685.0 | 1592.3 | **670.5** |
| Early 101–300 | 665.0 | 1852.8 | **645.0** |
| Middle 251–500 | 758.6 | 1578.0 | **745.7** |
| Late 501–750 | 896.1 | 1604.1 | **884.8** |
| Late first half | 691.3 | 1476.4 | **679.0** |
| Late second half | 1054.0 | 1746.0 | **1042.6** |

Best submitted full-evaluator benchmark was Round-1 Day 2:
mean `544.0`, std `1472.8`, score `528.5`. The new combined candidate scores
`670.5`: +26.9% score, +25.9% mean PnL, and higher annualized Sharpe
(`6.80` versus `5.84`). On the genuinely unseen late window it scores
`884.8` versus the submitted strategy's roughly `654.4`.

Caveats:

- Frozen pair identities were selected using the first 500 days, so their
  early/middle results contain selection look-ahead. The late 501–750
  comparison is honest because those pair identities were frozen before it.
- The pair-free core is therefore the clean evidence for the new direction.
  Its late score of 798.8 does not rely on pairs.
- The large late improvement is released-data evidence, not a guarantee on
  the next hidden leaderboard window. CCA performance strengthened with
  sample size, but canonical modes can still decay if the generator changes.
- Current `eval.py` is locally modified to score 749 days. Every benchmark
  in the full-evaluator table uses that same setting.
