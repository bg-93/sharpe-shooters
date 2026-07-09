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

## Current Implementation

- Raw 8-day and 30-day mean reversion still drives the main target book.
- A beta-adjusted residual signal versus ALGO is computed for assets `1-50`.
- Only the top 3 residual dislocations receive an extra sizing boost, capped by each instrument's normal limit.
