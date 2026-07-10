#!/usr/bin/env python3

import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COMMISSION_RATE = 0.0010
DEFAULT_POSITION_LIMIT = 10000.0


def load_prices(path):
    df = pd.read_csv(path, sep=r"\s+", header=0)
    return df.values.T.astype(float)


def load_strategy_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("strategy_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "getMyPosition"):
        raise AttributeError(f"{path} does not define getMyPosition")
    if hasattr(module, "reset_state"):
        module.reset_state()
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest a getMyPosition strategy on a segment of prices.txt."
    )
    parser.add_argument(
        "--strategy",
        default="main.py",
        help="Path to the strategy file containing getM yPosition.",
    )
    parser.add_argument(
        "--prices",
        default="prices.txt",
        help="Path to the whitespace-separated price file.",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=None,
        help="Zero-based first day of the evaluation segment.",
    )
    parser.add_argument(
        "--end-day",
        type=int,
        default=None,
        help="Zero-based last day of the evaluation segment.",
    )
    parser.add_argument(
        "--num-test-days",
        type=int,
        default=None,
        help="If provided, evaluate on the last N days instead of setting start-day manually.",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=DEFAULT_COMMISSION_RATE,
        help="Commission as a decimal fraction of dollar volume traded.",
    )
    parser.add_argument(
        "--position-limit",
        type=float,
        default=DEFAULT_POSITION_LIMIT,
        help="Per-instrument dollar position limit applied at trade time.",
    )
    parser.add_argument(
        "--daily-log",
        action="store_true",
        help="Print per-day PnL and exposure details.",
    )
    return parser.parse_args()


def resolve_segment(num_days, start_day, end_day, num_test_days):
    if num_test_days is not None:
        if num_test_days < 2 or num_test_days > num_days:
            raise ValueError("num-test-days must be between 2 and the number of available days.")
        start_day = num_days - num_test_days
        end_day = num_days - 1
    else:
        if start_day is None:
            start_day = 0
        if end_day is None:
            end_day = num_days - 1

    if start_day < 0 or end_day >= num_days or start_day >= end_day:
        raise ValueError("Invalid evaluation segment. Ensure 0 <= start-day < end-day < num_days.")

    return start_day, end_day


def backtest(prices, strategy_module, start_day, end_day, commission_rate, position_limit, daily_log):
    n_inst, n_days = prices.shape
    cash = 0.0
    value = 0.0
    cur_pos = np.zeros(n_inst, dtype=int)
    total_volume = 0.0
    daily_pl = []
    daily_records = []

    loop_start = time.perf_counter()

    for day in range(start_day, end_day + 1):
        prc_so_far = prices[:, : day + 1]
        cur_prices = prc_so_far[:, -1]
        requested_pos = np.array(cur_pos, copy=True)
        clipped_pos = np.array(cur_pos, copy=True)
        commission = 0.0
        day_volume = 0.0

        if day < end_day:
            requested = np.asarray(strategy_module.getMyPosition(prc_so_far), dtype=float).reshape(-1)
            if requested.shape[0] != n_inst:
                raise ValueError(
                    f"Strategy returned {requested.shape[0]} positions for {n_inst} instruments on day {day}."
                )
            requested_pos = requested.astype(int)
            pos_limits = np.floor(position_limit / np.maximum(cur_prices, 1e-12)).astype(int)
            clipped_pos = np.clip(requested_pos, -pos_limits, pos_limits)
            delta_pos = clipped_pos - cur_pos
            traded_dollars = cur_prices * np.abs(delta_pos)
            day_volume = float(np.sum(traded_dollars))
            commission = day_volume * commission_rate
            total_volume += day_volume
            cash -= float(cur_prices.dot(delta_pos)) + commission
        else:
            delta_pos = np.zeros(n_inst, dtype=int)

        cur_pos = clipped_pos.astype(int)
        gross_exposure = float(np.sum(np.abs(cur_pos) * cur_prices))
        net_exposure = float(cur_pos.dot(cur_prices))
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if day > start_day:
            daily_pl.append(today_pl)

        daily_records.append(
            {
                "day": day,
                "value": value,
                "today_pl": today_pl,
                "cash": cash,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "day_volume": day_volume,
                "commission": commission,
                "active_positions": int(np.count_nonzero(cur_pos)),
                "requested_turnover_shares": int(np.sum(np.abs(requested_pos - cur_pos))) if day < end_day else 0,
            }
        )

        if daily_log and day > start_day:
            print(
                "Day {day:4d} | value {value:10.2f} | PL {pl:9.2f} | traded {vol:10.0f} | "
                "gross {gross:10.0f} | net {net:10.0f} | active {active:2d}".format(
                    day=day,
                    value=value,
                    pl=today_pl,
                    vol=day_volume,
                    gross=gross_exposure,
                    net=net_exposure,
                    active=np.count_nonzero(cur_pos),
                )
            )

    runtime_seconds = time.perf_counter() - loop_start
    pll = np.asarray(daily_pl, dtype=float)
    if pll.size == 0:
        raise ValueError("The selected segment is too short to produce evaluation statistics.")

    mean_pl = float(np.mean(pll))
    std_pl = float(np.std(pll))
    score = mean_pl - 0.1 * std_pl
    ann_sharpe = 0.0 if std_pl == 0 else float(np.sqrt(252.0) * mean_pl / std_pl)
    ret_on_volume = 0.0 if total_volume == 0 else float(value / total_volume)
    cumulative_pl = float(np.sum(pll))
    running_pl = np.cumsum(pll)
    running_peak = np.maximum.accumulate(running_pl)
    max_drawdown = float(np.max(running_peak - running_pl))
    win_rate = float(np.mean(pll > 0))

    return {
        "segment_start_day": start_day,
        "segment_end_day": end_day,
        "evaluation_days": pll.size,
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "score": score,
        "ann_sharpe": ann_sharpe,
        "total_volume": float(total_volume),
        "return_on_volume": ret_on_volume,
        "final_value": float(value),
        "cumulative_pl": cumulative_pl,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "runtime_seconds": runtime_seconds,
        "daily_records": daily_records,
    }


def print_summary(result, strategy_path, prices_path, commission_rate, position_limit):
    print("==== Backtest Summary ====")
    print(f"Strategy: {strategy_path}")
    print(f"Prices: {prices_path}")
    print(
        f"Segment: days {result['segment_start_day']} to {result['segment_end_day']} "
        f"({result['evaluation_days']} scored days)"
    )
    print(f"Commission rate: {commission_rate:.4f}")
    print(f"Per-instrument dollar limit: {position_limit:.0f}")
    print("-----")
    print(f"mean(PL): {result['mean_pl']:.4f}")
    print(f"StdDev(PL): {result['std_pl']:.4f}")
    print(f"Score = mean(PL) - 0.1 * StdDev(PL): {result['score']:.4f}")
    print(f"annSharpe(PL): {result['ann_sharpe']:.4f}")
    print(f"Final value: {result['final_value']:.4f}")
    print(f"Cumulative PL: {result['cumulative_pl']:.4f}")
    print(f"Total dollar volume: {result['total_volume']:.0f}")
    print(f"Return on volume: {result['return_on_volume']:.6f}")
    print(f"Max drawdown: {result['max_drawdown']:.4f}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Runtime: {result['runtime_seconds']:.3f}s")


def main():
    args = parse_args()
    strategy_path = Path(args.strategy).resolve()
    prices_path = Path(args.prices).resolve()

    prices = load_prices(prices_path)
    start_day, end_day = resolve_segment(
        prices.shape[1],
        args.start_day,
        args.end_day,
        args.num_test_days,
    )
    strategy_module = load_strategy_module(strategy_path)

    result = backtest(
        prices=prices,
        strategy_module=strategy_module,
        start_day=start_day,
        end_day=end_day,
        commission_rate=args.commission_rate,
        position_limit=args.position_limit,
        daily_log=args.daily_log,
    )
    print_summary(
        result=result,
        strategy_path=strategy_path,
        prices_path=prices_path,
        commission_rate=args.commission_rate,
        position_limit=args.position_limit,
    )


if __name__ == "__main__":
    main()
