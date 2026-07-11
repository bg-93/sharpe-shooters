#!/usr/bin/env python
"""ALGO relative-value arbitrage research script and live strategy.

Instrument 0 behaves almost like a fixed-share index of the other 50 names.
The direct ETF-vs-basket spread was too small after transaction costs, so the
live strategy trades stock-vs-ALGO residual dislocations instead:

- estimate each stock's recent beta to ALGO from trailing log returns
- build a beta-adjusted relative spread versus ALGO
- z-score that spread with trailing data only
- trade the strongest mean-reversion gaps, hedged with ALGO

The file is self-contained:
- `getMyPosition(prcSoFar)` is evaluator-compatible
- `python strategies/basket_arb.py --prices prices.txt` runs split backtests
- optional plots can be written with `--plot-dir`
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


N_INST = 51
ALGO_IDX = 0
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50, dtype=float)
COMM_RATES = np.array([0.00002] + [0.0001] * 50, dtype=float)

# Live strategy parameters chosen from split testing.
BETA_WINDOW = 55
ZSCORE_WINDOW = 40
ENTRY_Z = 1.75
EXIT_Z = 0.50
STOP_Z = 4.00
MAX_HOLD = 12
TOP_K = 8
STOCK_DOLLAR_FRACTION = 1.00
ALGO_HEDGE_CAP_FRACTION = 0.18
MIN_ABS_BETA = 0.05
MAX_ABS_BETA = 3.00
MIN_HISTORY = max(BETA_WINDOW + 2, ZSCORE_WINDOW + 2, 80)


def official_score(mu: float, sigma: float, param: float = 1.0) -> float:
    if mu <= 0.0 or sigma < 1e-10:
        return mu
    sharpe = np.sqrt(250.0) * mu / sigma
    frac = sharpe * sharpe / (sharpe * sharpe + param * param)
    return mu * frac


def safe_log(prices: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(prices, EPS))


def load_prices(path: str | Path) -> tuple[np.ndarray, list[str]]:
    import pandas as pd

    frame = pd.read_csv(path, sep=r"\s+")
    return frame.to_numpy(dtype=float).T, frame.columns.tolist()


def synthetic_algo_from_others(prices: np.ndarray) -> np.ndarray:
    """Rebuild ALGO from first-day fixed share weights on instruments 1..50."""
    if prices.shape[0] != N_INST:
        raise ValueError(f"Expected {N_INST} instruments, got {prices.shape[0]}")

    first_day = np.maximum(prices[1:, 0], EPS)
    weights = 100.0 / ((N_INST - 1) * first_day)
    return weights @ prices[1:]


def algo_replication_diagnostics(prices: np.ndarray) -> dict[str, float]:
    algo = prices[ALGO_IDX]
    synth = synthetic_algo_from_others(prices)
    error = algo - synth
    return {
        "corr": float(np.corrcoef(algo, synth)[0, 1]),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


class AlgoRelativeValueArb:
    """Stateful evaluator-compatible stock-vs-ALGO arbitrage strategy.

    The evaluator calls `getMyPosition` on expanding histories. We keep a small
    cache for efficiency, and rebuild from scratch if the history length jumps
    backwards or skips ahead unexpectedly.
    """

    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self.last_nt = 0
        self.direction = np.zeros(N_INST, dtype=int)
        self.hold_days = np.zeros(N_INST, dtype=int)
        self.cached_position = np.zeros(N_INST, dtype=int)
        self.last_snapshot: dict[str, np.ndarray | list[int]] = {}

    def _signal_snapshot(self, prices: np.ndarray) -> dict[str, np.ndarray] | None:
        n_inst, nt = prices.shape
        if n_inst != N_INST or nt < MIN_HISTORY:
            return None

        log_prices = safe_log(prices)
        log_returns = np.diff(log_prices, axis=1)
        algo_returns = log_returns[ALGO_IDX, -BETA_WINDOW:]
        algo_var = float(np.dot(algo_returns, algo_returns))
        if algo_var < 1e-12:
            return None

        stock_returns = log_returns[1:, -BETA_WINDOW:]
        betas = (stock_returns @ algo_returns) / (algo_var + EPS)

        # A rolling-beta-adjusted relative log price versus ALGO.
        relative_series = (
            (log_prices[1:] - log_prices[1:, [0]])
            - betas[:, None] * (log_prices[ALGO_IDX] - log_prices[ALGO_IDX, 0])
        )

        history = relative_series[:, -ZSCORE_WINDOW - 1 : -1]
        spread_mean = history.mean(axis=1)
        spread_std = history.std(axis=1)
        zscores = np.where(
            spread_std > 1e-10,
            (relative_series[:, -1] - spread_mean) / spread_std,
            np.nan,
        )

        algo_std = float(np.std(algo_returns))
        if algo_std > 1e-12:
            stock_std = np.std(stock_returns, axis=1)
            corr = (stock_returns @ algo_returns) / (
                BETA_WINDOW * np.maximum(stock_std * algo_std, 1e-12)
            )
        else:
            corr = np.zeros(N_INST - 1, dtype=float)

        return {
            "betas": betas,
            "corr": corr,
            "relative_series": relative_series,
            "spread_std": spread_std,
            "zscores": zscores,
        }

    def _step(self, prices: np.ndarray) -> np.ndarray:
        snapshot = self._signal_snapshot(prices)
        if snapshot is None:
            self.direction[:] = 0
            self.hold_days[:] = 0
            self.last_snapshot = {}
            return np.zeros(N_INST, dtype=int)

        betas = snapshot["betas"]
        zscores = snapshot["zscores"]
        spread_std = snapshot["spread_std"]
        corr = snapshot["corr"]

        desired_dollars = np.zeros(N_INST, dtype=float)
        proposals: list[tuple[float, int, float, int]] = []

        for local_idx, inst in enumerate(range(1, N_INST)):
            beta = float(betas[local_idx])
            zscore = float(zscores[local_idx])
            valid = (
                np.isfinite(zscore)
                and MIN_ABS_BETA <= abs(beta) <= MAX_ABS_BETA
                and float(spread_std[local_idx]) > 1e-10
            )

            if not valid:
                self.direction[inst] = 0
                self.hold_days[inst] = 0
                continue

            direction = int(self.direction[inst])
            hold = int(self.hold_days[inst])

            if direction == 0:
                if zscore >= ENTRY_Z:
                    direction, hold = -1, 1
                elif zscore <= -ENTRY_Z:
                    direction, hold = 1, 1
            else:
                hold += 1
                if (
                    abs(zscore) <= EXIT_Z
                    or abs(zscore) >= STOP_Z
                    or hold >= MAX_HOLD
                ):
                    direction, hold = 0, 0

            self.direction[inst] = direction
            self.hold_days[inst] = hold

            if direction != 0:
                proposals.append((abs(zscore), inst, beta, direction))

        proposals.sort(key=lambda row: row[0], reverse=True)
        active_relationships: list[int] = []

        for _, inst, beta, direction in proposals[:TOP_K]:
            stock_dollars = min(
                LIMITS[inst] * STOCK_DOLLAR_FRACTION,
                LIMITS[ALGO_IDX] * ALGO_HEDGE_CAP_FRACTION / max(abs(beta), 1e-6),
            )
            desired_dollars[inst] += direction * stock_dollars
            desired_dollars[ALGO_IDX] -= direction * stock_dollars * beta
            active_relationships.append(inst)

        used = np.where(np.abs(desired_dollars) > 1e-8)[0]
        if used.size > 0:
            scale = float(
                np.min(LIMITS[used] / np.maximum(np.abs(desired_dollars[used]), 1e-8))
            )
            desired_dollars *= min(1.0, scale)

        current_prices = np.maximum(prices[:, -1], 1.0)
        position = np.rint(desired_dollars / current_prices).astype(int)
        self.last_snapshot = {
            "betas": betas.copy(),
            "corr": corr.copy(),
            "zscores": zscores.copy(),
            "spread_std": spread_std.copy(),
            "active_instruments": np.asarray(active_relationships, dtype=int),
            "desired_dollars": desired_dollars.copy(),
            "relative_series": snapshot["relative_series"].copy(),
        }
        return position

    def _rebuild_to_current_day(self, prices: np.ndarray) -> np.ndarray:
        self.reset_state()
        n_inst, nt = prices.shape
        if n_inst != N_INST:
            return np.zeros(n_inst, dtype=int)
        if nt < MIN_HISTORY:
            self.last_nt = nt
            self.cached_position = np.zeros(n_inst, dtype=int)
            return self.cached_position.copy()

        for day in range(MIN_HISTORY, nt + 1):
            self.cached_position = self._step(prices[:, :day])
            self.last_nt = day
        return self.cached_position.copy()

    def get_position(self, prcSoFar: np.ndarray) -> np.ndarray:
        prices = np.asarray(prcSoFar, dtype=float)
        if prices.ndim != 2:
            raise ValueError("prcSoFar must be a 2D array")

        n_inst, nt = prices.shape
        if n_inst != N_INST:
            self.reset_state()
            return np.zeros(n_inst, dtype=int)

        if nt < MIN_HISTORY:
            if nt < self.last_nt:
                self.reset_state()
            self.last_nt = nt
            self.cached_position = np.zeros(n_inst, dtype=int)
            return self.cached_position.copy()

        if nt == self.last_nt:
            return self.cached_position.copy()

        if nt == self.last_nt + 1:
            self.cached_position = self._step(prices)
            self.last_nt = nt
            return self.cached_position.copy()

        return self._rebuild_to_current_day(prices)


LIVE_STRATEGY = AlgoRelativeValueArb()


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    return LIVE_STRATEGY.get_position(prcSoFar)


def reset_state() -> None:
    LIVE_STRATEGY.reset_state()


def backtest_window(
    prices: np.ndarray,
    strategy: AlgoRelativeValueArb,
    start_day: int,
    end_day: int,
) -> dict:
    """Exact evaluator-style backtest on a chosen window."""
    n_inst, n_days = prices.shape
    if not (0 < start_day < end_day <= n_days):
        raise ValueError("Invalid backtest window")

    strategy.reset_state()

    cash = 0.0
    current_position = np.zeros(n_inst, dtype=int)
    value = 0.0
    pending_commission = 0.0
    total_volume = 0.0
    total_costs = 0.0
    trade_count = 0

    prev_scored_prices = None
    per_inst_pnl = np.zeros(n_inst, dtype=float)
    hold_counters = np.zeros(n_inst, dtype=int)
    completed_holds: list[int] = []

    daily_pl: list[float] = []
    daily_value: list[float] = []
    gross_exposure: list[float] = []
    day_numbers: list[int] = []

    for t in range(start_day, end_day + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]

        if prev_scored_prices is not None:
            per_inst_pnl += current_position * (current_prices - prev_scored_prices)

        if t < end_day:
            desired = strategy.get_position(history)
            limits = (LIMITS / np.maximum(current_prices, EPS)).astype(int)
            new_position = np.clip(desired, -limits, limits).astype(int)
        else:
            new_position = current_position.copy()

        delta = new_position - current_position
        traded_dollars = current_prices * np.abs(delta)
        day_cost = float(np.sum(traded_dollars * COMM_RATES))
        total_volume += float(np.sum(traded_dollars))
        total_costs += day_cost
        trade_count += int(np.count_nonzero(delta))
        per_inst_pnl -= traded_dollars * COMM_RATES

        cash -= float(current_prices.dot(delta)) + pending_commission
        pending_commission = day_cost

        entered = (current_position == 0) & (new_position != 0)
        exited = (current_position != 0) & (new_position == 0)
        hold_counters[current_position != 0] += 1
        completed_holds.extend(hold_counters[exited].astype(int).tolist())
        hold_counters[entered] = 1
        hold_counters[exited] = 0
        hold_counters[(current_position == 0) & (new_position == 0)] = 0

        current_position = new_position
        portfolio_dollars = current_position * current_prices
        previous_value = value
        value = cash + float(np.sum(portfolio_dollars))
        today_pl = value - previous_value

        if t > start_day:
            daily_pl.append(today_pl)
            daily_value.append(value)
            gross_exposure.append(float(np.sum(np.abs(portfolio_dollars))))
            day_numbers.append(t)
            prev_scored_prices = current_prices.copy()
        else:
            prev_scored_prices = current_prices.copy()

    pl = np.asarray(daily_pl, dtype=float)
    mean_pl = float(np.mean(pl)) if pl.size else 0.0
    std_pl = float(np.std(pl)) if pl.size else 0.0
    ann_sharpe = float(np.sqrt(250.0) * mean_pl / std_pl) if std_pl > 1e-12 else 0.0
    score = float(official_score(mean_pl, std_pl))
    value_series = np.asarray(daily_value, dtype=float)
    running_max = np.maximum.accumulate(value_series) if value_series.size else np.zeros(0)
    max_drawdown = float(np.min(value_series - running_max)) if value_series.size else 0.0
    win_rate = float(np.mean(pl > 0.0)) if pl.size else 0.0
    avg_hold = (
        float(np.mean(np.asarray(completed_holds, dtype=float)))
        if completed_holds
        else 0.0
    )

    return {
        "summary": {
            "start_day": start_day,
            "end_day": end_day - 1,
            "mean_pl": mean_pl,
            "std_pl": std_pl,
            "score": score,
            "ann_sharpe": ann_sharpe,
            "final_value": float(value_series[-1]) if value_series.size else 0.0,
            "max_drawdown": max_drawdown,
            "transaction_costs": total_costs,
            "total_dollar_volume": total_volume,
            "trade_count": float(trade_count),
            "average_holding_period": avg_hold,
            "win_rate": win_rate,
        },
        "days": np.asarray(day_numbers, dtype=int),
        "daily_pl": pl,
        "portfolio_value": value_series,
        "gross_exposure": np.asarray(gross_exposure, dtype=float),
        "per_instrument_pnl": per_inst_pnl,
    }


def default_eval_windows(n_days: int) -> dict[str, tuple[int, int]]:
    return {
        "validation": (250, 375),
        "holdout": (375, n_days),
        "official250": (n_days - 250, n_days),
    }


def print_window_summary(label: str, result: dict) -> None:
    summary = result["summary"]
    print(
        f"{label:12s} score={summary['score']:8.2f} "
        f"mean={summary['mean_pl']:8.2f} std={summary['std_pl']:8.2f} "
        f"sharpe={summary['ann_sharpe']:5.2f} maxDD={summary['max_drawdown']:9.2f} "
        f"costs={summary['transaction_costs']:9.2f} trades={summary['trade_count']:6.0f}"
    )


def _prepare_matplotlib():
    mplconfigdir = Path(tempfile.gettempdir()) / "sharpe_shooters_mpl"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_algo_replication(
    prices: np.ndarray,
    names: list[str],
    output_path: Path,
) -> None:
    plt = _prepare_matplotlib()
    algo = prices[ALGO_IDX]
    synth = synthetic_algo_from_others(prices)
    spread = algo - synth
    days = np.arange(prices.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(days, algo, label=names[ALGO_IDX], lw=2.0, color="#1d3557")
    axes[0].plot(days, synth, label="Synthetic ALGO", lw=1.7, color="#e76f51")
    axes[0].set_ylabel("Price")
    axes[0].set_title("ALGO vs fixed-share replica from instruments 1..50")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(days, spread, color="#2a9d8f", lw=1.6)
    axes[1].axhline(0.0, color="black", lw=1.0, alpha=0.6)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("ALGO - synthetic")
    axes[1].set_title("Replication error")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_curve(result: dict, output_path: Path, title: str) -> None:
    plt = _prepare_matplotlib()
    days = result["days"]
    values = result["portfolio_value"]
    daily_pl = result["daily_pl"]
    cumulative_pl = np.cumsum(daily_pl)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(days, cumulative_pl, color="#1d3557", lw=1.8)
    axes[0].set_ylabel("Cumulative PnL")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    axes[1].plot(days, values, color="#e76f51", lw=1.8)
    axes[1].set_xlabel("Eval day")
    axes[1].set_ylabel("Portfolio value")
    axes[1].set_title("Portfolio value path")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_relative_value_examples(
    prices: np.ndarray,
    names: list[str],
    strategy: AlgoRelativeValueArb,
    output_path: Path,
    lookback: int = 180,
) -> None:
    plt = _prepare_matplotlib()
    strategy._rebuild_to_current_day(prices)
    snapshot = strategy.last_snapshot
    if not snapshot:
        return

    zscores = np.asarray(snapshot["zscores"], dtype=float)
    betas = np.asarray(snapshot["betas"], dtype=float)
    relative_series = np.asarray(snapshot["relative_series"], dtype=float)
    active = np.argsort(np.abs(zscores))[-4:]
    active = active[np.isfinite(zscores[active])]
    if active.size == 0:
        return

    fig, axes = plt.subplots(active.size, 1, figsize=(12, 3.0 * active.size), sharex=True)
    if active.size == 1:
        axes = [axes]

    start = max(0, prices.shape[1] - lookback)
    days = np.arange(start, prices.shape[1])

    for ax, idx in zip(axes, active[::-1]):
        inst = int(idx + 1)
        series = relative_series[idx, start:]
        hist = relative_series[idx, -ZSCORE_WINDOW - 1 : -1]
        spread_mean = float(np.mean(hist))
        spread_std = float(np.std(hist))
        upper = spread_mean + ENTRY_Z * spread_std
        lower = spread_mean - ENTRY_Z * spread_std

        ax.plot(days, series, lw=1.8, color="#264653")
        ax.axhline(spread_mean, color="black", lw=1.0, alpha=0.6)
        ax.axhline(upper, color="#e76f51", lw=1.0, ls="--")
        ax.axhline(lower, color="#e76f51", lw=1.0, ls="--")
        ax.set_ylabel(names[inst])
        ax.set_title(
            f"{names[inst]} vs ALGO | beta={betas[idx]:+.2f} | z={zscores[idx]:+.2f}"
        )
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_research(
    prices: np.ndarray,
    names: list[str],
    plot_dir: Path | None = None,
) -> dict[str, dict]:
    strategy = AlgoRelativeValueArb()
    windows = default_eval_windows(prices.shape[1])
    results = {
        label: backtest_window(prices, strategy, start_day, end_day)
        for label, (start_day, end_day) in windows.items()
    }

    for label, result in results.items():
        print_window_summary(label, result)

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_algo_replication(prices, names, plot_dir / "algo_replication.png")
        plot_backtest_curve(
            results["official250"],
            plot_dir / "official250_backtest.png",
            "Official 250-day backtest",
        )
        plot_relative_value_examples(
            prices,
            names,
            strategy,
            plot_dir / "relative_value_examples.png",
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="prices.txt", help="Path to prices.txt")
    parser.add_argument(
        "--plot-dir",
        default="visualisations/algo_relative_value",
        help="Directory for matplotlib outputs",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip writing diagnostic plots",
    )
    args = parser.parse_args()

    prices, names = load_prices(args.prices)
    print(f"Loaded {prices.shape[0]} instruments for {prices.shape[1]} days")
    diag = algo_replication_diagnostics(prices)
    print(
        "ALGO replication "
        f"corr={diag['corr']:.9f} rmse={diag['rmse']:.6f} "
        f"mae={diag['mae']:.6f} max_abs_error={diag['max_abs_error']:.6f}"
    )
    run_research(
        prices,
        names,
        None if args.skip_plots else Path(args.plot_dir),
    )


if __name__ == "__main__":
    main()
