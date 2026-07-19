import argparse
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import Callable

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "sharpe_shooters_mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_INST = 51
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50)
COMM_RATES = np.array([0.00002] + [0.0001] * 50)


def official_score(mu: float, sigma: float, param: float = 1.0) -> float:
    if mu <= 0.0 or sigma < 1e-10:
        return mu
    sharpe = np.sqrt(250.0) * mu / sigma
    frac = sharpe * sharpe / (sharpe * sharpe + param * param)
    return mu * frac


def load_prices(path: str) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep=r"\s+")
    return frame.to_numpy(dtype=float).T, frame.columns.tolist()


def safe_log_prices(prices: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(prices, EPS))


def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(safe_log_prices(prices), axis=1)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    reg = alpha * np.eye(gram.shape[0])
    beta = np.linalg.solve(gram + reg, xc.T @ yc)
    intercept = y_mean - float(x_mean @ beta)
    return intercept, beta


def correlation_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rolling_sum(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series.copy()
    if series.size < window:
        return np.zeros(0, dtype=float)
    csum = np.cumsum(np.r_[0.0, series])
    return csum[window:] - csum[:-window]


def ar1_half_life(spread: np.ndarray) -> tuple[float, float]:
    if spread.size < 12:
        return np.nan, np.nan
    x = spread[:-1]
    y = spread[1:]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return np.nan, np.nan
    phi = float(np.polyfit(x, y, 1)[0])
    if 0.0 < phi < 1.0:
        half_life = float(-np.log(2.0) / np.log(phi))
    else:
        half_life = np.nan
    return phi, half_life


def simple_spread_backtest(
    spread: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold: int,
    lookback: int,
) -> dict[str, float]:
    if spread.size <= max(lookback, 4):
        return {
            "mean_pl": 0.0,
            "std_pl": 0.0,
            "ann_sharpe": 0.0,
            "trade_count": 0.0,
            "avg_holding_period": 0.0,
            "win_rate": 0.0,
        }

    position = 0
    hold_days = 0
    trades = 0
    holding_periods: list[int] = []
    trade_pnls: list[float] = []
    open_trade_pl = 0.0
    daily_pl: list[float] = []

    for t in range(lookback, spread.size - 1):
        hist = spread[t - lookback : t]
        hist_std = float(np.std(hist))
        if hist_std < 1e-10:
            zscore = 0.0
        else:
            zscore = (spread[t] - float(np.mean(hist))) / hist_std

        if position == 0:
            if zscore >= entry_z:
                position = -1
                hold_days = 0
                open_trade_pl = 0.0
                trades += 1
            elif zscore <= -entry_z:
                position = 1
                hold_days = 0
                open_trade_pl = 0.0
                trades += 1
        else:
            hold_days += 1
            if abs(zscore) <= exit_z or abs(zscore) >= stop_z or hold_days >= max_hold:
                holding_periods.append(hold_days)
                trade_pnls.append(open_trade_pl)
                position = 0
                hold_days = 0
                open_trade_pl = 0.0

        pnl = position * -(spread[t + 1] - spread[t])
        open_trade_pl += pnl
        daily_pl.append(pnl)

    if position != 0 and hold_days > 0:
        holding_periods.append(hold_days)
        trade_pnls.append(open_trade_pl)

    pnl_arr = np.asarray(daily_pl, dtype=float)
    mean_pl = float(np.mean(pnl_arr)) if pnl_arr.size else 0.0
    std_pl = float(np.std(pnl_arr)) if pnl_arr.size else 0.0
    ann_sharpe = float(np.sqrt(250.0) * mean_pl / std_pl) if std_pl > 1e-12 else 0.0
    wins = float(np.mean(np.asarray(trade_pnls, dtype=float) > 0.0)) if trade_pnls else 0.0
    avg_hold = float(np.mean(np.asarray(holding_periods, dtype=float))) if holding_periods else 0.0
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "ann_sharpe": ann_sharpe,
        "trade_count": float(trades),
        "avg_holding_period": avg_hold,
        "win_rate": wins,
    }


@dataclass
class BasketArbConfig:
    method: str = "sparse"
    train_window: int = 180
    validation_window: int = 40
    candidate_pool: int = 8
    basket_size: int = 4
    ridge_alpha: float = 5.0
    refit_frequency: int = 20
    spread_window: int = 6
    zscore_window: int = 26
    entry_z: float = 1.75
    exit_z: float = 0.60
    stop_z: float = 4.50
    max_hold: int = 15
    max_active_relationships: int = 6
    min_validation_r2: float = 0.12
    min_validation_corr: float = 0.35
    min_quality_score: float = 0.16
    max_stability: float = 1.50
    min_signal_days: int = 50
    exclude_algo_as_target: bool = True
    trade_notional_fraction: float = 0.75
    min_scale_to_trade: float = 0.20

    @property
    def min_history(self) -> int:
        signal_buffer = self.spread_window + self.zscore_window + 4
        return self.train_window + self.validation_window + signal_buffer


@dataclass
class BasketModel:
    target: int
    hedge_idx: np.ndarray
    intercept: float
    beta: np.ndarray
    validation_r2: float
    test_corr_proxy: float
    stability: float
    spread_half_life: float
    validation_sharpe: float
    quality_score: float
    gross_capacity: float
    relationship_name: str
    hedge_names: list[str]


@dataclass
class RelationshipState:
    direction: int = 0
    hold_days: int = 0


@dataclass
class StrategyDiagnostics:
    relationship_names: list[str] = field(default_factory=list)
    current_signal_rows: list[dict] = field(default_factory=list)
    active_relationships: list[str] = field(default_factory=list)
    selected_models: list[BasketModel] = field(default_factory=list)


def build_relationship_name(target_name: str, hedge_names: list[str]) -> str:
    return f"{target_name} vs ({', '.join(hedge_names)})"


def candidate_indices(corr_row: np.ndarray, target: int, pool_size: int) -> np.ndarray:
    ordered = np.argsort(-np.abs(corr_row))
    filtered = [idx for idx in ordered if idx != target]
    return np.asarray(filtered[:pool_size], dtype=int)


def fit_equal_weight_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    corr_signs: np.ndarray,
) -> tuple[float, np.ndarray]:
    beta = corr_signs / max(corr_signs.size, 1)
    intercept = float(train_y.mean() - train_x.mean(axis=0) @ beta)
    return intercept, beta


def select_sparse_coefficients(beta: np.ndarray, max_non_zero: int) -> np.ndarray:
    keep = np.argsort(np.abs(beta))[-max_non_zero:]
    keep = keep[np.abs(beta[keep]) > 1e-10]
    return np.sort(keep)


def spread_from_model(target_returns: np.ndarray, hedge_returns: np.ndarray, intercept: float, beta: np.ndarray) -> np.ndarray:
    residual = target_returns - (intercept + hedge_returns @ beta)
    return rolling_sum(residual, 6)


def model_validation_score(
    validation_r2: float,
    validation_corr: float,
    validation_sharpe: float,
    stability: float,
    half_life: float,
) -> float:
    sharpe_component = np.clip(validation_sharpe, 0.0, 3.0) / 3.0
    corr_component = np.clip(validation_corr, 0.0, 1.0)
    r2_component = np.clip(validation_r2, -0.5, 1.0)
    stability_component = np.clip(1.0 - stability / 2.0, 0.0, 1.0)
    if np.isfinite(half_life):
        half_life_component = 1.0 - min(abs(half_life - 12.0), 24.0) / 24.0
    else:
        half_life_component = 0.0
    return float(
        0.34 * r2_component
        + 0.22 * corr_component
        + 0.22 * sharpe_component
        + 0.14 * stability_component
        + 0.08 * half_life_component
    )


def fit_basket_model(
    method: str,
    candidate_pool: np.ndarray,
    returns: np.ndarray,
    names: list[str],
    target: int,
    config: BasketArbConfig,
) -> BasketModel | None:
    train_len = config.train_window
    val_len = config.validation_window
    target_rets = returns[target]
    train_y = target_rets[:train_len]
    val_y = target_rets[train_len : train_len + val_len]
    train_x_full = returns[candidate_pool, :train_len].T
    val_x_full = returns[candidate_pool, train_len : train_len + val_len].T

    if method == "equal":
        k = min(config.basket_size, candidate_pool.size)
        hedge_idx = candidate_pool[:k]
        corr_signs = np.sign([
            correlation_safe(returns[idx, :train_len], train_y) for idx in hedge_idx
        ])
        corr_signs = np.where(corr_signs == 0.0, 1.0, corr_signs)
        intercept, beta = fit_equal_weight_model(
            returns[hedge_idx, :train_len].T,
            train_y,
            corr_signs,
        )
    else:
        intercept, beta_full = ridge_fit(train_x_full, train_y, config.ridge_alpha)
        if method == "ridge":
            selected_local = np.arange(min(config.basket_size, beta_full.size))
            ordered = np.argsort(-np.abs(beta_full))
            selected_local = np.sort(ordered[: config.basket_size])
        elif method == "sparse":
            selected_local = select_sparse_coefficients(beta_full, config.basket_size)
        else:
            raise ValueError(f"Unsupported method: {method}")
        if selected_local.size == 0:
            return None
        hedge_idx = candidate_pool[selected_local]
        intercept, beta = ridge_fit(returns[hedge_idx, :train_len].T, train_y, config.ridge_alpha)

    train_x = returns[hedge_idx, :train_len].T
    val_x = returns[hedge_idx, train_len : train_len + val_len].T
    pred_val = intercept + val_x @ beta
    denom = float(np.sum((val_y - val_y.mean()) ** 2))
    if denom < 1e-12:
        validation_r2 = 0.0
    else:
        validation_r2 = 1.0 - float(np.sum((val_y - pred_val) ** 2)) / denom
    validation_corr = correlation_safe(val_y, pred_val)

    spread_val = rolling_sum(val_y - pred_val, config.spread_window)
    phi, half_life = ar1_half_life(spread_val)
    spread_stats = simple_spread_backtest(
        spread_val,
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        stop_z=config.stop_z,
        max_hold=config.max_hold,
        lookback=min(config.zscore_window, max(config.spread_window + 2, spread_val.size - 2)),
    )
    validation_sharpe = float(spread_stats["ann_sharpe"])

    half = train_len // 2
    beta_left = ridge_fit(train_x[:half], train_y[:half], config.ridge_alpha)[1]
    beta_right = ridge_fit(train_x[half:], train_y[half:], config.ridge_alpha)[1]
    stability = float(np.linalg.norm(beta_left - beta_right) / (np.linalg.norm(beta) + 1e-9))

    quality = model_validation_score(
        validation_r2=validation_r2,
        validation_corr=validation_corr,
        validation_sharpe=validation_sharpe,
        stability=stability,
        half_life=half_life,
    )

    if validation_r2 < config.min_validation_r2:
        return None
    if validation_corr < config.min_validation_corr:
        return None
    if stability > config.max_stability:
        return None
    if quality < config.min_quality_score:
        return None

    gross_capacity = float(LIMITS[target])
    for hedge, coeff in zip(hedge_idx, beta):
        gross_capacity = min(gross_capacity, float(LIMITS[hedge]) / max(abs(float(coeff)), 1e-6))
    gross_capacity *= config.trade_notional_fraction

    hedge_names = [names[idx] for idx in hedge_idx]
    relationship_name = build_relationship_name(names[target], hedge_names)
    return BasketModel(
        target=target,
        hedge_idx=np.asarray(hedge_idx, dtype=int),
        intercept=float(intercept),
        beta=np.asarray(beta, dtype=float),
        validation_r2=float(validation_r2),
        test_corr_proxy=float(validation_corr),
        stability=float(stability),
        spread_half_life=float(half_life) if np.isfinite(half_life) else np.nan,
        validation_sharpe=validation_sharpe,
        quality_score=quality,
        gross_capacity=gross_capacity,
        relationship_name=relationship_name,
        hedge_names=hedge_names,
    )


class BasketArbStrategy:
    def __init__(self, names: list[str], config: BasketArbConfig):
        self.names = names
        self.config = copy.deepcopy(config)
        self.last_refit_nt = -1
        self.cached_models: list[BasketModel] = []
        self.relation_states: dict[int, RelationshipState] = {}
        self.diagnostics = StrategyDiagnostics()

    def reset_state(self) -> None:
        self.last_refit_nt = -1
        self.cached_models = []
        self.relation_states = {}
        self.diagnostics = StrategyDiagnostics()

    def _discover_models(self, prices: np.ndarray) -> list[BasketModel]:
        config = self.config
        total_returns = config.train_window + config.validation_window
        hist_prices = prices[:, -(total_returns + 1) :]
        returns = log_returns(hist_prices)
        corr = np.nan_to_num(np.corrcoef(returns[:, : config.train_window]), nan=0.0)

        models: list[BasketModel] = []
        for target in range(prices.shape[0]):
            if config.exclude_algo_as_target and target == 0:
                continue
            pool = candidate_indices(corr[target], target, config.candidate_pool)
            if pool.size < config.basket_size:
                continue
            model = fit_basket_model(
                method=config.method,
                candidate_pool=pool,
                returns=returns,
                names=self.names,
                target=target,
                config=config,
            )
            if model is not None:
                models.append(model)

        models.sort(key=lambda item: (item.quality_score, item.validation_sharpe, item.validation_r2), reverse=True)
        return models

    def _maybe_refit(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if nt < self.config.min_history:
            self.cached_models = []
            return
        if self.last_refit_nt < 0 or nt - self.last_refit_nt >= self.config.refit_frequency:
            self.cached_models = self._discover_models(prices)
            self.last_refit_nt = nt

    def _relationship_signal(self, prices: np.ndarray, model: BasketModel) -> tuple[float, float]:
        config = self.config
        hist_len = max(config.min_signal_days, config.zscore_window + config.spread_window + 4)
        price_slice = prices[:, -hist_len:]
        returns = log_returns(price_slice)
        target_returns = returns[model.target]
        hedge_returns = returns[model.hedge_idx].T
        residual = target_returns - (model.intercept + hedge_returns @ model.beta)
        spread = rolling_sum(residual, config.spread_window)
        if spread.size <= config.zscore_window + 1:
            return np.nan, 0.0
        history = spread[-config.zscore_window - 1 : -1]
        current = float(spread[-1])
        hist_std = float(np.std(history))
        if hist_std < 1e-10:
            return np.nan, 0.0
        zscore = (current - float(np.mean(history))) / hist_std
        recent_corr = correlation_safe(
            target_returns[-config.zscore_window :],
            (model.intercept + hedge_returns @ model.beta)[-config.zscore_window :],
        )
        return float(zscore), recent_corr

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        prices = np.asarray(prcSoFar, dtype=float)
        n_inst, nt = prices.shape
        if n_inst != N_INST or nt < self.config.min_history:
            return np.zeros(n_inst, dtype=int)

        current_prices = np.maximum(prices[:, -1], 1.0)
        self._maybe_refit(prices)
        if not self.cached_models:
            return np.zeros(n_inst, dtype=int)

        desired_dollars = np.zeros(n_inst, dtype=float)
        signal_rows: list[dict] = []
        proposals: list[tuple[float, BasketModel, int, float]] = []

        for model in self.cached_models:
            zscore, recent_corr = self._relationship_signal(prices, model)
            state = self.relation_states.setdefault(model.target, RelationshipState())
            if not np.isfinite(zscore) or recent_corr < 0.10:
                state.direction = 0
                state.hold_days = 0
                continue

            if state.direction == 0:
                if zscore >= self.config.entry_z:
                    state.direction = -1
                    state.hold_days = 1
                elif zscore <= -self.config.entry_z:
                    state.direction = 1
                    state.hold_days = 1
            else:
                state.hold_days += 1
                if abs(zscore) <= self.config.exit_z or abs(zscore) >= self.config.stop_z or state.hold_days >= self.config.max_hold:
                    state.direction = 0
                    state.hold_days = 0

            signal_rows.append(
                {
                    "target": model.target,
                    "relationship": model.relationship_name,
                    "zscore": float(zscore),
                    "recent_corr": float(recent_corr),
                    "quality": model.quality_score,
                    "active_direction": state.direction,
                    "hold_days": state.hold_days,
                }
            )

            if state.direction == 0:
                continue

            strength = min(1.0, max(0.0, (abs(zscore) - self.config.exit_z) / (self.config.stop_z - self.config.exit_z)))
            strength = max(0.25, strength)
            rank = abs(zscore) * model.quality_score * max(recent_corr, 0.0)
            proposals.append((rank, model, state.direction, strength))

        proposals.sort(key=lambda item: item[0], reverse=True)
        active_relationships: list[str] = []

        for _, model, direction, strength in proposals[: self.config.max_active_relationships]:
            leg_dollars = np.zeros(n_inst, dtype=float)
            leg_dollars[model.target] = direction * model.gross_capacity * strength
            leg_dollars[model.hedge_idx] = -direction * model.gross_capacity * strength * model.beta
            used = np.where(np.abs(leg_dollars) > 1e-8)[0]
            if used.size == 0:
                continue
            remaining = LIMITS[used] - np.abs(desired_dollars[used])
            raw_abs = np.abs(leg_dollars[used])
            scale = float(np.min(np.where(raw_abs > 1e-8, remaining / raw_abs, 1.0)))
            scale = min(1.0, max(0.0, scale))
            if scale < self.config.min_scale_to_trade:
                continue
            desired_dollars += scale * leg_dollars
            active_relationships.append(model.relationship_name)

        self.diagnostics = StrategyDiagnostics(
            relationship_names=[model.relationship_name for model in self.cached_models],
            current_signal_rows=signal_rows,
            active_relationships=active_relationships,
            selected_models=self.cached_models[:],
        )
        return (desired_dollars / current_prices).astype(int)


def no_trade_strategy(_prices: np.ndarray) -> np.ndarray:
    return np.zeros(N_INST, dtype=int)


def make_strategy_callable(strategy: BasketArbStrategy) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[], None]]:
    return strategy.getMyPosition, strategy.reset_state


def backtest_window(
    prices: np.ndarray,
    names: list[str],
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None,
    start_day: int,
    end_day: int,
) -> dict:
    if reset_fn is not None:
        reset_fn()

    n_inst, n_days = prices.shape
    if not (0 < start_day < end_day <= n_days):
        raise ValueError("Invalid evaluation window")

    cash = 0.0
    current_position = np.zeros(n_inst, dtype=int)
    value = 0.0
    pending_commission = 0.0
    total_volume = 0.0
    total_costs = 0.0

    scored_pl: list[float] = []
    daily_rows: list[dict] = []
    instrument_pnl = np.zeros(n_inst, dtype=float)
    instrument_volume = np.zeros(n_inst, dtype=float)
    current_open_days = np.zeros(n_inst, dtype=int)
    completed_holds: list[int] = []
    trade_entries = 0

    for t in range(start_day, end_day + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]

        if t < end_day:
            desired = np.asarray(strategy_fn(history), dtype=float).reshape(-1)
            limits = (LIMITS / np.maximum(current_prices, 1e-12)).astype(int)
            new_position = np.clip(desired, -limits, limits).astype(int)
        else:
            new_position = current_position.copy()

        delta = new_position - current_position
        traded_dollars = current_prices * np.abs(delta)
        day_cost = float(np.sum(traded_dollars * COMM_RATES))
        total_volume += float(np.sum(traded_dollars))
        total_costs += day_cost

        cash -= float(current_prices.dot(delta)) + pending_commission
        pending_commission = day_cost

        mark_to_market = new_position * current_prices
        previous_value = value
        value = cash + float(np.sum(mark_to_market))
        today_pl = value - previous_value

        if t > start_day:
            scored_pl.append(today_pl)

        instrument_pnl += current_position * (current_prices - prices[:, t - 2]) if t > start_day else 0.0
        instrument_pnl -= traded_dollars * COMM_RATES
        instrument_volume += traded_dollars

        entered = (current_position == 0) & (new_position != 0)
        exited = (current_position != 0) & (new_position == 0)
        trade_entries += int(np.count_nonzero(entered))
        current_open_days[current_position != 0] += 1
        completed_holds.extend(current_open_days[exited].astype(int).tolist())
        current_open_days[entered] = 1
        current_open_days[exited] = 0
        current_open_days[(current_position == 0) & (new_position == 0)] = 0

        gross_exposure = float(np.sum(np.abs(mark_to_market)))
        daily_rows.append(
            {
                "eval_day": t,
                "price_day_index": t - 1,
                "portfolio_value": value,
                "daily_pl": today_pl,
                "gross_exposure": gross_exposure,
                "net_exposure": float(np.sum(mark_to_market)),
                "dollar_traded": float(np.sum(traded_dollars)),
                "transaction_cost": day_cost,
                "active_positions": int(np.count_nonzero(new_position)),
            }
        )

        current_position = new_position

    day_df = pd.DataFrame(daily_rows)
    pl = np.asarray(scored_pl, dtype=float)
    mean_pl = float(np.mean(pl)) if pl.size else 0.0
    std_pl = float(np.std(pl)) if pl.size else 0.0
    ann_sharpe = float(np.sqrt(250.0) * mean_pl / std_pl) if std_pl > 1e-12 else 0.0
    score = official_score(mean_pl, std_pl)
    cum_value = float(day_df["portfolio_value"].iloc[-1]) if not day_df.empty else 0.0
    drawdown = day_df["portfolio_value"] - day_df["portfolio_value"].cummax()
    max_drawdown = float(drawdown.min()) if not day_df.empty else 0.0
    win_rate = float(np.mean(pl > 0.0)) if pl.size else 0.0
    turnover = float(day_df["dollar_traded"].sum() / max(day_df["gross_exposure"].mean(), 1.0)) if not day_df.empty else 0.0
    avg_hold = float(np.mean(np.asarray(completed_holds, dtype=float))) if completed_holds else 0.0

    instrument_df = pd.DataFrame(
        {
            "instrument": names,
            "total_pl": instrument_pnl,
            "dollar_traded": instrument_volume,
            "avg_abs_limit_usage": 0.0,
        }
    ).sort_values("total_pl", ascending=False)

    return {
        "summary": {
            "mean_pl": mean_pl,
            "std_pl": std_pl,
            "score": float(score),
            "ann_sharpe": ann_sharpe,
            "final_value": cum_value,
            "max_drawdown": max_drawdown,
            "turnover": turnover,
            "transaction_costs": total_costs,
            "win_rate": win_rate,
            "number_of_trades": float(trade_entries),
            "average_holding_period": avg_hold,
            "total_dollar_volume": total_volume,
            "start_day": start_day,
            "end_day": end_day - 1,
        },
        "daily": day_df,
        "instrument": instrument_df,
    }


def print_window_summary(label: str, result: dict) -> None:
    summary = result["summary"]
    print(
        f"{label:14s} score={summary['score']:8.2f} mean={summary['mean_pl']:8.2f} "
        f"std={summary['std_pl']:8.2f} sharpe={summary['ann_sharpe']:5.2f} "
        f"maxDD={summary['max_drawdown']:9.2f} turnover={summary['turnover']:7.2f} "
        f"costs={summary['transaction_costs']:10.2f} trades={summary['number_of_trades']:6.0f}"
    )


def default_basket_only_kwargs() -> dict:
    return {
        "train_window": 180,
        "validation_window": 30,
        "min_validation_r2": -0.03,
        "min_validation_corr": 0.40,
        "min_quality_score": 0.26,
        "max_stability": 1.0,
        "spread_window": 6,
        "zscore_window": 24,
        "entry_z": 1.7,
        "exit_z": 0.5,
        "stop_z": 4.2,
        "max_active_relationships": 6,
        "refit_frequency": 20,
    }


def default_eval_windows(n_days: int) -> dict[str, tuple[int, int]]:
    return {
        "validation": (250, 375),
        "holdout": (375, n_days),
        "official250": (n_days - 250, n_days),
    }


def evaluate_windows(
    prices: np.ndarray,
    names: list[str],
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None,
    windows: dict[str, tuple[int, int]],
) -> dict[str, dict]:
    return {
        label: backtest_window(
            prices=prices,
            names=names,
            strategy_fn=strategy_fn,
            reset_fn=reset_fn,
            start_day=start_day,
            end_day=end_day,
        )
        for label, (start_day, end_day) in windows.items()
    }


def load_strategy_namespace(strategy_path: str | Path) -> tuple[dict, Callable[[np.ndarray], np.ndarray], Callable[[], None] | None]:
    strategy_path = Path(strategy_path).resolve()
    module_name = f"_loaded_strategy_{strategy_path.stem}_{abs(hash(str(strategy_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = module.__dict__
    strategy_fn = namespace["getMyPosition"]
    reset_fn = namespace.get("reset_state")
    if not callable(reset_fn):
        reset_fn = None
    return namespace, strategy_fn, reset_fn


def label_for_method(label: str) -> str:
    mapping = {
        "no_trade": "No Trade",
        "equal_weight": "Equal Weight Basket",
        "ridge": "Ridge Basket",
        "sparse": "Sparse Basket",
        "core_baseline": "Core Mean Reversion",
        "final_hybrid": "Final Hybrid",
    }
    return mapping.get(label, label.replace("_", " ").title())


def extract_live_basket_models(
    prices: np.ndarray,
    strategy_path: str | Path,
    as_of_day: int | None = None,
) -> tuple[list[tuple], dict]:
    namespace, _, _ = load_strategy_namespace(strategy_path)
    discover_fn = namespace.get("_discover_basket_models")
    if not callable(discover_fn):
        return [], namespace
    if as_of_day is None:
        as_of_day = max(250, int(namespace.get("BASKET_MIN_HISTORY", prices.shape[1])))
    as_of_day = max(1, min(as_of_day, prices.shape[1]))
    models = discover_fn(prices[:, :as_of_day])
    return models, namespace


def live_model_series(
    prices: np.ndarray,
    model: tuple,
    zscore_window: int,
) -> dict[str, np.ndarray | int | float]:
    score, target, hedge_idx, intercept, beta, capacity = model
    hedge_idx = np.asarray(hedge_idx, dtype=int)
    beta = np.asarray(beta, dtype=float)
    log_prices = safe_log_prices(prices)
    target_log = log_prices[target]
    synthetic_log = float(intercept) + log_prices[hedge_idx].T @ beta
    spread = target_log - synthetic_log
    zscore = np.full(spread.shape, np.nan, dtype=float)
    for t in range(zscore_window, spread.size):
        hist = spread[t - zscore_window : t]
        hist_std = float(np.std(hist))
        if hist_std > 1e-10:
            zscore[t] = (spread[t] - float(np.mean(hist))) / hist_std
    return {
        "score": float(score),
        "target": int(target),
        "hedge_idx": hedge_idx,
        "beta": beta,
        "capacity": float(capacity),
        "target_price": prices[target],
        "synthetic_price": np.exp(synthetic_log),
        "spread": spread,
        "zscore": zscore,
    }


def plot_live_basket_overview(
    prices: np.ndarray,
    names: list[str],
    models: list[tuple],
    output_path: Path,
    title_suffix: str,
) -> None:
    if not models:
        return

    top_models = models[: min(8, len(models))]
    scores = [float(model[0]) for model in top_models]
    y_pos = np.arange(len(top_models))

    fig, ax = plt.subplots(figsize=(15, 1.2 * len(top_models) + 2.5))
    bars = ax.barh(y_pos, scores, color="#2a6f97", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[int(model[1])] for model in top_models])
    ax.invert_yaxis()
    ax.set_xlabel("Validation signal score")
    ax.set_ylabel("Target instrument")
    ax.set_title(f"Top live basket relationships {title_suffix}")
    ax.grid(axis="x", alpha=0.25)

    score_pad = max(scores) * 0.03 if scores else 0.1
    ax.set_xlim(0.0, max(scores) * 1.85 if scores else 1.0)
    for bar, model in zip(bars, top_models):
        _, _, hedge_idx, _, beta, capacity = model
        hedge_desc = ", ".join(
            f"{names[int(idx)]} {float(weight):+.2f}" for idx, weight in zip(hedge_idx, beta)
        )
        ax.text(
            float(bar.get_width()) + score_pad,
            float(bar.get_y()) + bar.get_height() / 2.0,
            f"{hedge_desc} | cap ${float(capacity):.0f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_live_basket_spreads(
    prices: np.ndarray,
    names: list[str],
    models: list[tuple],
    zscore_window: int,
    entry_z: float,
    output_path: Path,
    as_of_day: int,
) -> None:
    if not models:
        return

    top_models = models[: min(3, len(models))]
    fig, axes = plt.subplots(len(top_models), 2, figsize=(16, 4.5 * len(top_models)), squeeze=False)
    start_idx = max(0, as_of_day - 120)
    x = np.arange(start_idx, prices.shape[1])

    for row, model in enumerate(top_models):
        series = live_model_series(prices, model, zscore_window)
        target_name = names[int(series["target"])]
        hedge_desc = ", ".join(
            f"{names[int(idx)]} {float(weight):+.2f}"
            for idx, weight in zip(series["hedge_idx"], series["beta"])
        )

        target_price = np.asarray(series["target_price"])[start_idx:]
        synthetic_price = np.asarray(series["synthetic_price"])[start_idx:]
        target_norm = 100.0 * target_price / target_price[0]
        synthetic_norm = 100.0 * synthetic_price / synthetic_price[0]

        ax_left = axes[row, 0]
        ax_left.plot(x, target_norm, color="#15616d", linewidth=1.8, label=target_name)
        ax_left.plot(x, synthetic_norm, color="#ff7d00", linewidth=1.6, linestyle="--", label="Synthetic basket")
        ax_left.axvline(as_of_day - 1, color="#6c757d", linestyle=":", linewidth=1.2)
        ax_left.set_title(f"{target_name} vs synthetic basket")
        ax_left.set_ylabel("Normalised price (start = 100)")
        ax_left.grid(alpha=0.25)
        ax_left.legend(loc="upper left", fontsize=9)
        ax_left.text(
            0.01,
            0.04,
            hedge_desc,
            transform=ax_left.transAxes,
            fontsize=8.8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        ax_right = axes[row, 1]
        spread = np.asarray(series["spread"])[start_idx:]
        zscore = np.asarray(series["zscore"])[start_idx:]
        ax_right.plot(x, spread, color="#003049", linewidth=1.6, label="Log-price spread")
        ax_right.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax_right.axvline(as_of_day - 1, color="#6c757d", linestyle=":", linewidth=1.2)
        ax_right.set_title(f"{target_name} spread and z-score")
        ax_right.set_ylabel("Spread")
        ax_right.grid(alpha=0.25)

        z_ax = ax_right.twinx()
        z_ax.plot(x, zscore, color="#d62828", linewidth=1.1, alpha=0.9, label="Z-score")
        z_ax.axhline(entry_z, color="#d62828", linewidth=0.9, linestyle="--")
        z_ax.axhline(-entry_z, color="#d62828", linewidth=0.9, linestyle="--")
        z_ax.set_ylabel("Z-score")
        z_ax.set_ylim(-4.8, 4.8)

        if row == len(top_models) - 1:
            ax_left.set_xlabel("Day")
            ax_right.set_xlabel("Day")

    fig.suptitle("Example live baskets and their spreads", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(
    results_by_label: dict[str, dict[str, dict]],
    output_path: Path,
) -> None:
    if not results_by_label:
        return

    labels = list(results_by_label.keys())
    display_labels = [label_for_method(label) for label in labels]
    colors = {
        "no_trade": "#adb5bd",
        "equal_weight": "#8ecae6",
        "ridge": "#ffb703",
        "sparse": "#fb8500",
        "core_baseline": "#219ebc",
        "final_hybrid": "#023047",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 11), height_ratios=[1.3, 1.0])

    ax_curve = axes[0]
    for label in labels:
        daily = results_by_label[label]["official250"]["daily"]
        ax_curve.plot(
            daily["price_day_index"],
            daily["portfolio_value"],
            linewidth=2.0 if label in {"core_baseline", "final_hybrid"} else 1.6,
            alpha=0.95,
            label=label_for_method(label),
            color=colors.get(label),
        )
    ax_curve.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_curve.set_title("Official 250-day equity curves")
    ax_curve.set_ylabel("Portfolio value")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper left", fontsize=9)

    ax_bar = axes[1]
    windows = ["validation", "holdout", "official250"]
    width = 0.24
    base_x = np.arange(len(labels))
    offsets = [-width, 0.0, width]
    window_colors = ["#577590", "#43aa8b", "#f94144"]
    for offset, window, color in zip(offsets, windows, window_colors):
        values = [results_by_label[label][window]["summary"]["score"] for label in labels]
        ax_bar.bar(base_x + offset, values, width=width, label=window, color=color, alpha=0.9)
    ax_bar.axhline(0.0, color="black", linewidth=0.8)
    ax_bar.set_xticks(base_x)
    ax_bar.set_xticklabels(display_labels, rotation=18, ha="right")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Score comparison across validation, holdout, and official 250-day windows")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_strategy_diagnostics(
    official_result: dict,
    output_path: Path,
    title: str,
) -> None:
    daily = official_result["daily"]
    instrument = official_result["instrument"].copy()
    summary = official_result["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(daily["price_day_index"], daily["portfolio_value"], color="#1d3557", linewidth=1.9)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Equity curve")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.98,
        (
            f"Score {summary['score']:.2f}\n"
            f"Mean PL {summary['mean_pl']:.1f}\n"
            f"PL std {summary['std_pl']:.1f}\n"
            f"Sharpe {summary['ann_sharpe']:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
    )

    ax = axes[0, 1]
    pnl = daily["daily_pl"]
    colors = np.where(pnl >= 0.0, "#2a9d8f", "#e63946")
    ax.bar(daily["price_day_index"], pnl, color=colors, width=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Daily PnL")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    ax.plot(daily["price_day_index"], daily["gross_exposure"], color="#6a4c93", linewidth=1.8, label="Gross exposure")
    ax.set_title("Capital deployment and trading")
    ax.set_xlabel("Day")
    ax.set_ylabel("Gross exposure")
    ax.grid(alpha=0.25)
    trade_ax = ax.twinx()
    trade_ax.bar(
        daily["price_day_index"],
        daily["dollar_traded"],
        color="#f4a261",
        alpha=0.30,
        width=0.9,
        label="Dollar traded",
    )
    trade_ax.set_ylabel("Dollar traded")

    ax = axes[1, 1]
    instrument = instrument.reindex(np.argsort(np.abs(instrument["total_pl"].to_numpy()))[::-1]).head(12)
    colors = np.where(instrument["total_pl"] >= 0.0, "#2a9d8f", "#e76f51")
    ax.barh(instrument["instrument"], instrument["total_pl"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Top instrument PnL contributions")
    ax.set_xlabel("Total PnL")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_matplotlib_report(
    price_path: str,
    output_dir: str,
    team_strategy_path: str,
    baseline_strategy_path: str | None,
) -> list[Path]:
    prices, names = load_prices(price_path)
    n_days = prices.shape[1]
    windows = default_eval_windows(n_days)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_by_label: dict[str, dict[str, dict]] = {}
    basket_only_kwargs = default_basket_only_kwargs()
    experiments = [
        ("no_trade", None, no_trade_strategy, None),
        ("equal_weight", BasketArbConfig(method="equal", **basket_only_kwargs), None, None),
        ("ridge", BasketArbConfig(method="ridge", **basket_only_kwargs), None, None),
        ("sparse", BasketArbConfig(method="sparse", **basket_only_kwargs), None, None),
    ]

    for label, config, strategy_fn, reset_fn in experiments:
        if config is not None:
            strategy = BasketArbStrategy(names, config)
            strategy_fn, reset_fn = make_strategy_callable(strategy)
        results_by_label[label] = evaluate_windows(prices, names, strategy_fn, reset_fn, windows)

    baseline_path = Path(baseline_strategy_path) if baseline_strategy_path else None
    if baseline_path is not None and baseline_path.exists():
        _, strategy_fn, reset_fn = load_strategy_namespace(baseline_path)
        results_by_label["core_baseline"] = evaluate_windows(prices, names, strategy_fn, reset_fn, windows)

    team_path = Path(team_strategy_path)
    _, final_strategy_fn, final_reset_fn = load_strategy_namespace(team_path)
    results_by_label["final_hybrid"] = evaluate_windows(prices, names, final_strategy_fn, final_reset_fn, windows)

    saved_paths: list[Path] = []
    comparison_path = output_path / "basket_method_comparison.png"
    plot_strategy_comparison(results_by_label, comparison_path)
    saved_paths.append(comparison_path)

    diagnostics_path = output_path / "final_strategy_diagnostics.png"
    plot_final_strategy_diagnostics(
        results_by_label["final_hybrid"]["official250"],
        diagnostics_path,
        title="Final hybrid strategy diagnostics on the official 250-day window",
    )
    saved_paths.append(diagnostics_path)

    live_models, team_namespace = extract_live_basket_models(prices, team_path, as_of_day=250)
    if live_models:
        overview_path = output_path / "live_basket_overview.png"
        plot_live_basket_overview(
            prices=prices,
            names=names,
            models=live_models,
            output_path=overview_path,
            title_suffix="at the official evaluation start",
        )
        saved_paths.append(overview_path)

        spread_path = output_path / "live_basket_spreads.png"
        plot_live_basket_spreads(
            prices=prices,
            names=names,
            models=live_models,
            zscore_window=int(team_namespace.get("BASKET_ZWIN", 18)),
            entry_z=float(team_namespace.get("BASKET_ENTRY", 1.7)),
            output_path=spread_path,
            as_of_day=250,
        )
        saved_paths.append(spread_path)

    return saved_paths


def run_research(price_path: str) -> None:
    prices, names = load_prices(price_path)
    _, n_days = prices.shape

    train_end = 250
    validation_end = 375
    full_end = n_days
    basket_only_kwargs = default_basket_only_kwargs()

    print("Repository-aware basket arbitrage research")
    print(f"Prices shape for strategy: {prices.shape}")
    print(f"Train split: days 0-{train_end - 1}")
    print(f"Validation split: days {train_end}-{validation_end - 1}")
    print(f"Holdout split: days {validation_end}-{full_end - 1}")
    print()

    experiments = [
        ("no_trade", None, no_trade_strategy, None),
        ("equal_weight", BasketArbConfig(method="equal", **basket_only_kwargs), None, None),
        ("ridge", BasketArbConfig(method="ridge", **basket_only_kwargs), None, None),
        ("sparse", BasketArbConfig(method="sparse", **basket_only_kwargs), None, None),
    ]

    for label, config, strategy_fn, reset_fn in experiments:
        if config is not None:
            strategy = BasketArbStrategy(names, config)
            strategy_fn, reset_fn = make_strategy_callable(strategy)
        print(f"Method: {label}")
        val_result = backtest_window(
            prices=prices,
            names=names,
            strategy_fn=strategy_fn,
            reset_fn=reset_fn,
            start_day=train_end,
            end_day=validation_end,
        )
        test_result = backtest_window(
            prices=prices,
            names=names,
            strategy_fn=strategy_fn,
            reset_fn=reset_fn,
            start_day=validation_end,
            end_day=full_end,
        )
        eval_result = backtest_window(
            prices=prices,
            names=names,
            strategy_fn=strategy_fn,
            reset_fn=reset_fn,
            start_day=n_days - 250,
            end_day=n_days,
        )
        print_window_summary("validation", val_result)
        print_window_summary("holdout", test_result)
        print_window_summary("official250", eval_result)
        print()

    sparse_strategy = BasketArbStrategy(names, BasketArbConfig(method="sparse"))
    sparse_strategy.reset_state()
    _ = sparse_strategy.getMyPosition(prices[:, : sparse_strategy.config.min_history])
    selected = sparse_strategy.cached_models[:15]
    if selected:
        print("Top discovered sparse basket relationships on first refit")
        for model in selected:
            hl = "nan" if not np.isfinite(model.spread_half_life) else f"{model.spread_half_life:5.1f}"
            print(
                f"target={model.target:2d} {names[model.target]:4s} "
                f"basket={model.hedge_names} q={model.quality_score:5.3f} "
                f"valR2={model.validation_r2:5.3f} valSR={model.validation_sharpe:5.2f} "
                f"corr={model.test_corr_proxy:5.2f} hl={hl} stab={model.stability:5.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research basket arbitrage strategy variants.")
    parser.add_argument("--prices", default="prices.txt", help="Path to whitespace separated price matrix.")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate matplotlib PNGs showing discovered baskets and strategy performance.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualisations/basket_arb_report",
        help="Directory where matplotlib report images should be saved.",
    )
    parser.add_argument(
        "--team-strategy",
        default="teamName.py",
        help="Path to the live team strategy file used for basket overlays.",
    )
    parser.add_argument(
        "--baseline-strategy",
        default="submissions/day3/teamName_pre_basket_overlay.py",
        help="Optional path to the pre-overlay baseline strategy for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_research(args.prices)
    if args.save_plots:
        saved_paths = generate_matplotlib_report(
            price_path=args.prices,
            output_dir=args.output_dir,
            team_strategy_path=args.team_strategy,
            baseline_strategy_path=args.baseline_strategy,
        )
        print()
        print("Saved matplotlib report files")
        for path in saved_paths:
            print(path)


if __name__ == "__main__":
    main()
