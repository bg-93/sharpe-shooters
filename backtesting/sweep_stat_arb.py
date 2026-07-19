"""Quick parameter sweep for strategies/stat_arb_residual.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtesting"))
sys.path.insert(0, str(ROOT / "strategies"))

import backtest as bt
import stat_arb_residual as strat

prices = bt.load_prices(ROOT / "prices.txt")
n_days = prices.shape[1]

results = []
for rev in [1, 3, 5, 10, 20]:
    for beta_w in [60, 120]:
        for band in [0.0, 0.2, 0.4]:
            strat.REV_HORIZON = rev
            strat.BETA_WINDOW = beta_w
            strat.MIN_HISTORY = beta_w + rev + 5
            strat.REBALANCE_BAND = band
            strat.reset_state()
            r = bt.backtest(prices, strat, n_days - 250, n_days - 1,
                            0.0001, 10000.0, False)
            results.append((rev, beta_w, band, r["mean_pl"], r["ann_sharpe"]))

results.sort(key=lambda x: -x[4])
print(f"{'rev':>4} {'betaW':>6} {'band':>5} {'meanPL':>9} {'sharpe':>7}")
for rev, bw, band, mu, sr in results:
    print(f"{rev:>4} {bw:>6} {band:>5.1f} {mu:>9.2f} {sr:>7.2f}")
