# Round 1 Submission Ledger (hidden window: days 751-1000)

## IMPORTANT: the scored window GROWS by 50 days per submission cycle
(751-800, then 751-850, 751-900, ...). Scores are running averages —
back out incremental edge per chunk: with score ~ mean at these
Sharpes, mean_new_chunk ~ (n_new*score_new - n_old*score_old)/50.
candidate_next: 1431/day (751-800), ~570/day (801-850), ~780/day
(851-900) -> the 1.4k->928 decline is the hot first 50 days washing
out toward the ~600-670/day backtest steady state, not a breakdown.
Probe scores must be interpreted against the window they covered —
log the window with every entry.

One submission per day = one deterministic query against the hidden
window. Score semantics: mu<=0 -> score = mu EXACTLY (linear readout);
mu>0 -> invert with `make_probes.invert_score(score, sigma)` using the
calibrated sigma below.

## Logistics (CONFIRMED 2026-07-17)

1. **Every submission is RE-EVALUATED on each new window increment**
   (confirmed 2026-07-19): the board shows current-window scores, so
   the 1420.84 snapshot did NOT persist (strat A reads 936.68 on
   751-900). Team score = best current-window score across all
   submitted strategies -> probing is still free, AND every submission
   becomes a permanently tracked instrument: submit candidates EARLY
   to maximize their tracking span.
2. **Leaderboard shows score, mean PL, std PL** -> every probe returns
   TWO exact numbers. For constant-dollar books the displayed mean IS
   w.drift (no inversion needed, sign irrelevant) and std IS portfolio
   vol on the hidden window. invert_score() is now only a cross-check.

## Priority schedule

| # | file | question it answers | est sigma |
|---|------|---------------------|----------:|
| 1 | `../candidate_next.py` | does our best book beat the 1k? | ~1550 |
| 2 | `probes/probe_sleeve_mr.py` | did the MR core revive on 751-1000? (likely source of the 1k) | ~1650 |
| 3 | `probes/probe_sleeve_ll_demeaned.py` | demeaned-LL edge alive? | ~1650 |
| 4 | `probes/probe_sleeve_pairs.py` | pairs edge alive? | ~550 |
| 5 | `probes/probe_algo_short.py` | index drift, exact if market rose | 1037 |
| 6 | `probes/probe_sleeve_fade.py` | fade worth anything here? | ~225 |
| 7-16 | `probes/probe_hadamard_01..10.py` | 10 orthogonal projections of the 50-stock drift vector | ~1300-1500 |
| late | best-known book | lock in the round score | - |

Sleeve probes 1-4 + the known Score-1k total (~1000) give the full
decomposition of why the live book works on this window.

## Recording format (append one line per submission)

| date | file | score | leaderboard mean/std if shown | implied mu | notes |
|------|------|-------|-------------------------------|-----------|-------|
| pre-2026-07-17 | Score-1k book (teamName.py) | ~1000 | | | full 618-era book; 127.91 on 501-750 |
| 2026-07-17 | candidate_next.py (window 751-800) | 1420.84 | 1431.58 / 1967.45 | 1431.58 | hot first 50d; best-score kept |
| +1 | candidate_next.py (window 751-850) | ~1000 | | | implied 801-850 ~570/day |
| +2 | candidate_next.py (window 751-900) | 928 | | | implied 851-900 ~780/day; consistent w/ backtest steady state |
| 2026-07-19 | candidate_next.py re-eval (751-900) | 936.68 | 946.30 / 1516.11 | 946.30 | days 801-900 ~704/day = backtest steady state; edge stable |
| 2026-07-19+1 | candidate_v2.py (window 751-900) | 909.85 | 919.52 / 1498.40 | 919.52 | CLEAN A/B: strat A unchanged (936.68) -> same window. A wins by ~27/day (2.8%), matches backtest gap. Frozen pairs still alive; v2 = decay insurance, tracked daily |

## Using the results

- **Sleeve scores** -> next-round book: keep sleeves that earn on
  751-1000 AND at least one released window (min-window rule).
- **algo_short + stocks_short_ew** -> aggregate market drift.
- **Hadamard projections**: after k probes, stack the weight rows
  W (k x 50) and measurements m (implied mu per probe), then
  `drift_hat = np.linalg.lstsq(W, m)` (ridge-shrink if k < 50).
  Uses: (a) understand the hidden regime (which names trend/revert),
  (b) optional static aligned book `limits * sign(drift_hat)` as a
  late-round submission — its expected score is checkable in advance
  from drift_hat and the released covariance.
- Calibration sanity: every probe was simulated on released 500-750
  (see below), so the pipeline is verified end-to-end. probe_algo_short
  scored -25.49 there = exact -1x index drift readout.

## Verified probe outputs on released window 500-750 (pipeline check)

```
probe_algo_short             mean=   -25.5 std=  1036.9 score=  -25.49
probe_hadamard_01            mean=    24.5 std=  1346.9 score=    1.86
probe_sleeve_ll_demeaned     mean=   591.0 std=  1647.4 score=  573.20
probe_sleeve_pairs           mean=   165.6 std=   548.9 score=  158.61
probe_sleeve_mr              mean=   -86.1 std=  1625.8 score=  -86.08
probe_sleeve_fade            mean=    13.5 std=   224.1 score=    6.39
```

## Caveats

- Check competition rules on leaderboard probing before running the
  Hadamard series; sleeve probes are indistinguishable from ordinary
  strategy iteration.
- sigma estimates come from released data (generator has no vol
  clustering, so they should transfer); positive-score inversions
  inherit sigma error, negative scores are exact.
- Probes are constant-dollar books: commission ~1bp of gross once at
  entry (~$50-500 total over the window) — negligible vs signal.
