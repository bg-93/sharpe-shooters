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

## 1,000-day release: shock-regime ultra-unicorn

The newly supplied days 751–1000 explain the approximately 700 leaderboard
result.  On those exact 250 days, the submitted pair-free rank-7
canonical/dense strategy produced mean PnL `778.46`, standard deviation
`1747.42`, annualized Sharpe `7.04`, and score **763.08**.  Its canonical
component remained useful, but the dense ridge component nearly stopped
working during days 901–1000.  `eval.py` now uses its current 250-day test
window; the earlier 749-day setting documented above was specific to the
previous release.

### Momentum hypothesis: rejected

Regime momentum was developed only on data through day 750 and then locked
before its audit on days 751–1000.  It did not validate:

| Frozen momentum rule | New-window score |
|---|---:|
| Multi-horizon 2-sigma short skill gate | 0.0 |
| Multi-horizon 2-sigma long skill gate | 0.0 |
| Signed adaptive, 60-day horizon | 8.2 |
| Signed adaptive, 20-day horizon | 0.9 |

Every active momentum overlay reduced the `797.3` curriculum-plus-pairs
baseline; the least harmful scored `790.3`.  Ordinary 5–120 day momentum,
dual-horizon confirmation, and breakouts were also negative or
regime-specific on earlier windows.  The useful meaning of "regime" here is
therefore **conditional lead-lag structure**, not recent price direction.

Reproduction: `strategies/agent_regime_momentum.py`.

### Final pair-free algorithm

Files:

- submission: `teamName.py`
- identical preserved candidate:
  `strategies/ultra_unicorn_regime_candidate.py`
- broad research runner: `strategies/agent_novel_1000.py`

The algorithm remains compact:

1. Convert the 50 independent stocks to one-day simple returns.  ALGO is not
   treated as a 51st independent series.
2. Fit a regularized canonical lead-lag map from today's standardized stock
   returns to tomorrow's.  Retain only the strongest three predictive
   canonical modes; a `0.25` dense-ridge view is only a stabilizer.
3. Label each historical predictor day by the RMS magnitude of its
   50-dimensional standardized return shock, above or below the historical
   median.  Fit one additional rank-3 map per state, requiring at least 300
   observations in each.
4. Each day, blend the global signal with only `0.10` of the map matching the
   currently observable quiet/shock state.  This is a gentle conditioner,
   not a hard regime switch.
5. Derive ALGO's forecast mechanically from its exact normalized-stock-index
   identity.  Its coherent dollar position receives a `1.5` boost before
   clipping because its limit is ten times larger and its commission is five
   times lower.
6. Allocate with the proven demeaned `tanh(signal / 0.20)` rule.
7. Refit every 250 observations.  Starting with the supplied 1,000 days
   therefore fits once at deployment and freezes coefficients through the
   expected next 250-day leaderboard window; only the daily observable state
   changes.

No pairs, hardcoded instrument identities, rolling price trends, or online
leaderboard-window fitting are used.

### Exact evaluator results

All rows use integer shares, official dollar limits, and delayed commission
semantics.  The final row fits at day 750 and performs **zero coefficient
refits** during days 751–1000:

| Strategy on days 751–1000 | Mean/day | Std | Ann. Sharpe | Score |
|---|---:|---:|---:|---:|
| Previous `teamName.py` | 778.46 | 1747.42 | 7.04 | **763.08** |
| Previous curriculum + pairs | 811.24 | 1652.79 | 7.76 | **797.99** |
| Frozen rank-3 base, no regime | 1038.10 | 1883.27 | 8.72 | **1024.61** |
| **Frozen shock-regime candidate** | **1050.80** | **1894.85** | **8.77** | **1037.31** |

The final candidate improves score by `274.23`, or **35.9%**, over the
previous submitted strategy.  Its mean fee was `48.53` per day and official
`eval.py` dollar volume was `136,439,558`.

The new-window 50-day score sequence was:

`1333.5 / 1092.0 / 1269.7 / 171.1 / 1297.9`.

Every block remained profitable.  The weak days 901–950 interval was not
eliminated, but it no longer caused the dense-ridge collapse seen in the
previous book.  The two sequential 125-day half scores were `1235.1` and
`837.6`.

Nearby choices form a useful plateau:

- regime weights `0.00 / 0.10 / 0.25` scored approximately
  `1024.6 / 1037.3 / 1032.2`;
- ALGO position boosts `1.00 / 1.25 / 1.50 / 2.00` scored approximately
  `1022.4 / 1034.4 / 1037.3 / 1035.9`;
- rank-3 shrinkage from `0.10` through `0.60` and dense weights from `0.10`
  through `0.50` remained strongly profitable in the adaptive ablations.

### Why pairs were omitted

The frozen pairs surprisingly remained positive in all five new 50-day
blocks and were weakly correlated with the canonical core.  Adding them can
raise the released-window score further, but their identities were selected
on earlier released data and their economic stability is less convincing.
The final `teamName.py` therefore takes the cleaner result: **1037.31 with no
pairs at all**.

### Interpretation and caveat

The evidence points to a persistent low-dimensional directed dependency:
the first three canonical modes strengthened as sample size increased,
whereas modes 4–7 and the broad dense map were less reliable.  Shock
conditioning adds only about 13 score points to the strict frozen base; the
real alpha is the stable rank-3 lead-lag structure, not an elaborate regime
classifier.

Days 751–1000 were initially a clean audit for the previous candidate, but
the new rank and shock architecture were investigated after that audit was
opened.  They are now legitimate training data for deployment after day
1,000, not a second untouched test set.  The frozen-coefficient replay,
positive subwindows, and parameter plateaus reduce overfit risk, but only
the future leaderboard days can confirm the approximately 1,000 score.
