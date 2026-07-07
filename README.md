# Sharpe Shooters

This repository contains the Sharpe Shooters trading strategy code and a local backtesting script for `prices.txt`.

## Setup

From inside `sharpe-shooters`, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If a teammate is using Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

After setup, they can run the backtester with:

```bash
python3 backtest.py --strategy main.py --prices prices.txt
```

## Backtester

The local backtester lives in [backtest.py](/Users/aditya/Documents/university_resource_files/sig_algothon_26/sharpe-shooters/backtest.py). It loads a Python strategy file containing a global `getMyPosition(prcSoFar)` function, simulates trading day by day, and reports the same main score used in evaluation:

`mean(PL) - 0.1 * StdDev(PL)`

### Assumptions used by the backtester

- Commission rate defaults to `0.0010` or 10 bps.
- Per-instrument position limit defaults to `$10,000`.
- Positions are clipped at trade time based on the latest price.
- The strategy is called once per day with all history up to that day.
- Trades are applied as changes from the previous position at the most recent price.

## How To Run

From inside `sharpe-shooters`:

```bash
python3 backtest.py --strategy main.py --prices prices.txt
```

From the repo root:

```bash
python3 sharpe-shooters/backtest.py \
  --strategy sharpe-shooters/main.py \
  --prices sharpe-shooters/prices.txt
```

## Common Examples

Backtest the full dataset:

```bash
python3 backtest.py --strategy main.py --prices prices.txt
```

Backtest a specific segment:

```bash
python3 backtest.py --strategy main.py --prices prices.txt --start-day 700 --end-day 999
```

Backtest the last 200 days:

```bash
python3 backtest.py --strategy main.py --prices prices.txt --num-test-days 200
```

Print day-by-day logs:

```bash
python3 backtest.py --strategy main.py --prices prices.txt --daily-log
```

Override commission or position limits:

```bash
python3 backtest.py \
  --strategy main.py \
  --prices prices.txt \
  --commission-rate 0.0010 \
  --position-limit 10000
```

## Command-Line Arguments

- `--strategy`: Path to the Python file that defines `getMyPosition`. Default is `main.py`.
- `--prices`: Path to the whitespace-separated price matrix. Default is `prices.txt`.
- `--start-day`: Zero-based first day of the evaluation segment.
- `--end-day`: Zero-based last day of the evaluation segment.
- `--num-test-days`: Use the last `N` days instead of manually specifying `start-day`.
- `--commission-rate`: Commission as a decimal fraction of traded dollar volume.
- `--position-limit`: Per-instrument dollar position cap applied at trade time.
- `--daily-log`: Print per-day PnL, traded volume, and exposure.

## Output Statistics

The backtester prints:

- `mean(PL)`
- `StdDev(PL)`
- `Score = mean(PL) - 0.1 * StdDev(PL)`
- `annSharpe(PL)`
- `Final value`
- `Cumulative PL`
- `Total dollar volume`
- `Return on volume`
- `Max drawdown`
- `Win rate`
- `Runtime`

## Strategy Interface

Strategies must be contained in a Python file, usually `main.py`, with a global function:

```python
def getMyPosition(prcSoFar):
    ...
```

Where:

- `prcSoFar` is a NumPy array of shape `nInst x nt`
- `nInst` is the number of instruments
- `nt` is the number of days observed so far
- the function returns a NumPy vector of integer desired positions

If the strategy file defines a `reset_state()` helper, the backtester will call it before starting a fresh run.
