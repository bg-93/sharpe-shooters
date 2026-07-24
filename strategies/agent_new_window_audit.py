#!/usr/bin/env python3
"""Audit the previous unicorn candidate on the newly revealed 250 days.

This script intentionally leaves the candidate unchanged.  It reproduces the
official evaluator's integer positions, dollar-limit clipping, and one-day
commission timing, then reports standalone model ablations and PnL
correlations.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "prices.txt"
CANDIDATE_PATH = ROOT / "strategies" / "unicorn_curriculum_candidate.py"
TEAM_PATH = ROOT / "teamName.py"

DEFAULT_LIMITS = np.array([100_000.0] + [10_000.0] * 50)
COMMISSION = np.array([0.00002] + [0.0001] * 50)


def load_module(name: str, path: Path = CANDIDATE_PATH) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0 or std < 1e-10:
        return mean
    sharpe = np.sqrt(250.0) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def metrics(pnl: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    sharpe = np.sqrt(250.0) * mean / std if std > 0 else 0.0
    return mean, std, sharpe, score(pnl)


def official_pnl(
    prices: np.ndarray,
    position: Callable[[np.ndarray], np.ndarray],
    start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return net PnL, gross PnL, and daily commissions for start+1..nt."""
    n_inst, nt = prices.shape
    current = np.zeros(n_inst)
    previous_prices: np.ndarray | None = None
    pending_commission = 0.0
    net_pnl: list[float] = []
    gross_pnl: list[float] = []
    commissions: list[float] = []

    for t in range(start, nt):
        current_prices = prices[:, t - 1]
        target_raw = position(prices[:, :t])
        limits = (DEFAULT_LIMITS / current_prices).astype(int)
        target = np.clip(target_raw, -limits, limits).astype(int)

        if previous_prices is not None:
            gross = float(current @ (current_prices - previous_prices))
            gross_pnl.append(gross)
            net_pnl.append(gross - pending_commission)
            commissions.append(pending_commission)

        turnover = current_prices * np.abs(target - current)
        pending_commission = float(turnover @ COMMISSION)
        current = target
        previous_prices = current_prices

    final_prices = prices[:, -1]
    gross = float(current @ (final_prices - previous_prices))
    gross_pnl.append(gross)
    net_pnl.append(gross - pending_commission)
    commissions.append(pending_commission)

    return np.asarray(net_pnl), np.asarray(gross_pnl), np.asarray(commissions)


def exact_getter(module: ModuleType, pair_scale: float) -> Callable:
    module.PAIR_SCALE = pair_scale
    module.reset_state()
    return module.getMyPosition


def pairs_getter(module: ModuleType) -> Callable:
    module.PAIR_SCALE = 1.0
    module.reset_state()

    def get_position(prices: np.ndarray) -> np.ndarray:
        dollars = np.clip(module._pair_dollars(prices), -module.LIMITS, module.LIMITS)
        return (dollars / prices[:, -1]).astype(int)

    return get_position


def canonical_ablation_getter(module: ModuleType, kind: str) -> Callable:
    """Use the candidate's exact fit/coherence code with one forecast branch."""
    if kind not in {"cca", "ridge"}:
        raise ValueError(kind)
    module.PAIR_SCALE = 0.0
    module.reset_state()

    def selected_dollars(prices: np.ndarray) -> np.ndarray:
        nt = prices.shape[1]
        if nt < 120:
            return np.zeros(module.N)
        if module._cca is None or nt - module._c_last_fit >= module.CCA_RETRAIN:
            module._fit_canonical(prices)
            module._c_last_fit = nt
        returns = prices[1:, 1:] / prices[1:, :-1] - 1.0
        now = (returns[:, -1] - module._c_xmu) / module._c_xsd
        if kind == "cca":
            signal = module._coherent(now @ module._cca, prices, True)
        else:
            signal = module._coherent(now @ module._ridge, prices, False)
        return module.LIMITS * np.tanh(signal / module.TEMP)

    module._canonical_dollars = selected_dollars
    return module.getMyPosition


def print_table(
    title: str,
    names: list[str],
    series: dict[str, np.ndarray],
    first: int,
    last: int,
) -> None:
    print(f"\n{title} (evaluator days {first}-{last})")
    print(f"{'variant':<18} {'mean':>10} {'std':>10} {'sharpe':>9} {'score':>10}")
    for name in names:
        mean, std, sharpe, value = metrics(series[name])
        print(f"{name:<18} {mean:>10.2f} {std:>10.2f} {sharpe:>9.3f} {value:>10.2f}")


def print_chunks(name: str, pnl: np.ndarray, first_day: int) -> None:
    print(f"\n{name}: 50-day chunks")
    print(f"{'days':<14} {'mean':>10} {'std':>10} {'sharpe':>9} {'score':>10}")
    for offset in range(0, len(pnl), 50):
        part = pnl[offset:offset + 50]
        lo = first_day + offset
        hi = lo + len(part) - 1
        mean, std, sharpe, value = metrics(part)
        print(f"{lo}-{hi:<8} {mean:>10.2f} {std:>10.2f} {sharpe:>9.3f} {value:>10.2f}")

    print(f"\n{name}: 100-day chunks")
    print(f"{'days':<14} {'mean':>10} {'std':>10} {'sharpe':>9} {'score':>10}")
    for offset in range(0, len(pnl), 100):
        part = pnl[offset:offset + 100]
        lo = first_day + offset
        hi = lo + len(part) - 1
        mean, std, sharpe, value = metrics(part)
        print(f"{lo}-{hi:<8} {mean:>10.2f} {std:>10.2f} {sharpe:>9.3f} {value:>10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=750)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(PRICES_PATH, sep=r"\s+")
    prices = frame.to_numpy(dtype=float).T
    if args.end is not None:
        prices = prices[:, :args.end]
    if not 1 < args.start < prices.shape[1]:
        raise ValueError(f"--start must be between 2 and {prices.shape[1] - 1}")

    getters = {
        "teamName": load_module("audit_team", TEAM_PATH).getMyPosition,
        "full": exact_getter(load_module("audit_full"), 1.0),
        "core": exact_getter(load_module("audit_core"), 0.0),
        "pairs": pairs_getter(load_module("audit_pairs")),
        "cca": canonical_ablation_getter(load_module("audit_cca"), "cca"),
        "ridge": canonical_ablation_getter(load_module("audit_ridge"), "ridge"),
    }

    net: dict[str, np.ndarray] = {}
    gross: dict[str, np.ndarray] = {}
    commissions: dict[str, np.ndarray] = {}
    for name, getter in getters.items():
        net[name], gross[name], commissions[name] = official_pnl(
            prices, getter, args.start
        )

    names = list(getters)
    first_day = args.start + 1
    last_day = prices.shape[1]
    print(f"Loaded {prices.shape[0]} instruments x {prices.shape[1]} observations")
    print_table("Net performance", names, net, first_day, last_day)

    print("\nMean daily commission")
    for name in names:
        print(f"{name:<18} {commissions[name].mean():>10.2f}")

    print_chunks("full", net["full"], first_day)
    print_chunks("core", net["core"], first_day)
    print_chunks("pairs", net["pairs"], first_day)
    print_chunks("cca", net["cca"], first_day)
    print_chunks("ridge", net["ridge"], first_day)

    print("\nNet daily-PnL correlation")
    correlation = np.corrcoef(np.vstack([net[name] for name in names]))
    print(f"{'':<12}" + "".join(f"{name:>10}" for name in names))
    for i, name in enumerate(names):
        print(f"{name:<12}" + "".join(f"{x:>10.4f}" for x in correlation[i]))

    synthetic = net["core"] + net["pairs"]
    residual = net["full"] - synthetic
    marginal_pair = net["full"] - net["core"]
    marginal_mean, marginal_std, marginal_sharpe, marginal_score = metrics(
        marginal_pair
    )
    print("\nNon-additivity diagnostics")
    print(f"corr(full, core+pairs): {np.corrcoef(net['full'], synthetic)[0, 1]:.6f}")
    print(f"mean(full-core-pairs): {residual.mean():.2f}")
    print(f"std(full-core-pairs):  {residual.std():.2f}")
    print(f"score(core+pairs):      {score(synthetic):.2f}")
    print(
        "actual pair marginal:   "
        f"mean={marginal_mean:.2f}, std={marginal_std:.2f}, "
        f"sharpe={marginal_sharpe:.3f}, score={marginal_score:.2f}"
    )
    print(
        "corr(core, marginal):   "
        f"{np.corrcoef(net['core'], marginal_pair)[0, 1]:.6f}"
    )


if __name__ == "__main__":
    main()
