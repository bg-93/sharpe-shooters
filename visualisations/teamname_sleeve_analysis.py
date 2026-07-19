from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import strategies.teamName_triple_pairs as strategy

NUM_TEST_DAYS = 250
COMM_RATE = np.array([0.00002] + [0.0001] * 50, dtype=float)

BOOK_ORDER = [
    "Combined",
    "Main Book",
    "Core MR",
    "Pairs",
    "ALGO Fade",
    "Lead-Lag",
]

BOOK_COLORS = {
    "Combined": "#111827",
    "Main Book": "#6b7280",
    "Core MR": "#2563eb",
    "Pairs": "#059669",
    "ALGO Fade": "#d97706",
    "Lead-Lag": "#dc2626",
}


def load_prices(prices_file: str | Path | None = None) -> np.ndarray:
    """Load the released price matrix with instruments on rows."""
    path = Path(prices_file) if prices_file is not None else REPO_ROOT / "prices.txt"
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T.astype(float)


def _shares_from_dollars(target_dollars: np.ndarray, prices: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Mirror the evaluator: clip by dollar limits after converting to shares."""
    safe_prices = np.maximum(prices, 1.0)
    pos_limits = (limits / safe_prices).astype(int)
    target_shares = (target_dollars / safe_prices).astype(int)
    return np.clip(target_shares, -pos_limits, pos_limits).astype(int)


@dataclass
class BookState:
    n_inst: int
    cash: float = 0.0
    tot_dvolume: float = 0.0
    value: float = 0.0
    pending_commission: float = 0.0
    commission_total: float = 0.0
    cur_pos: np.ndarray = field(init=False)
    daily_pl: list[float] = field(default_factory=list)
    cumulative_value: list[float] = field(default_factory=list)
    traded_dollars: list[float] = field(default_factory=list)
    gross_exposure: list[float] = field(default_factory=list)
    net_exposure: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cur_pos = np.zeros(self.n_inst, dtype=int)

    def step(self, new_pos: np.ndarray, cur_prices: np.ndarray, record: bool) -> None:
        delta_pos = new_pos - self.cur_pos
        self.cash -= cur_prices.dot(delta_pos) + self.pending_commission

        dvolumes = cur_prices * np.abs(delta_pos)
        dvolume = float(dvolumes.sum())
        self.tot_dvolume += dvolume

        self.pending_commission = float((dvolumes * COMM_RATE).sum())
        self.commission_total += self.pending_commission

        self.cur_pos = new_pos.astype(int)
        pos_value = float(self.cur_pos.dot(cur_prices))
        today_pl = self.cash + pos_value - self.value
        self.value = self.cash + pos_value

        if record:
            self.daily_pl.append(today_pl)
            self.cumulative_value.append(self.value)
            self.traded_dollars.append(dvolume)
            self.gross_exposure.append(float(np.abs(self.cur_pos * cur_prices).sum()))
            self.net_exposure.append(float(self.cur_pos.dot(cur_prices)))
            self.positions.append(self.cur_pos.copy())


class SleeveDecomposer:
    """Research copy of the live strategy that exposes sleeve-level targets."""

    def __init__(self, strat_module):
        self.s = strat_module
        self.reset()

    def reset(self) -> None:
        self.prev_target_dollars = np.zeros(self.s.N_INST, dtype=float)
        self.prev_nt = -1
        self.pair_pos = [0] * len(self.s.PAIRS)
        self.ll_W = None
        self.ll_mu = None
        self.ll_sd = None
        self.ll_resid_sd = None
        self.ll_last_fit = -1
        self.ll_wf_preds = []
        self.ll_wf_reals = []
        self.ll_pending = None

    def _leadlag_target_dollars(self, prc_so_far: np.ndarray) -> np.ndarray:
        nt = prc_so_far.shape[1]
        if nt < self.s.LL_MIN_HIST:
            return np.zeros(self.s.N_INST)

        r = np.diff(np.log(np.maximum(prc_so_far, self.s.EPS)), axis=1)

        if self.ll_pending is not None:
            self.ll_wf_preds.append(self.ll_pending)
            self.ll_wf_reals.append(r[:, -1])
            self.ll_pending = None

        if self.ll_W is None or (nt - self.ll_last_fit) >= self.s.LL_RETRAIN:
            X = r[:, :-1].T
            Y = r[:, 1:].T
            self.ll_mu = X.mean(0)
            self.ll_sd = X.std(0)
            self.ll_sd = np.where(self.ll_sd > 1e-12, self.ll_sd, 1.0)
            Xs = (X - self.ll_mu) / self.ll_sd
            self.ll_W = np.linalg.solve(
                Xs.T @ Xs + self.s.LL_LAM * np.eye(self.s.N_INST), Xs.T @ Y
            )
            self.ll_resid_sd = np.maximum(Y.std(0), 1e-8)
            self.ll_last_fit = nt

        x = (r[:, -1] - self.ll_mu) / self.ll_sd
        pred = x @ self.ll_W
        self.ll_pending = pred.copy()

        mask = np.ones(self.s.N_INST)
        if len(self.ll_wf_preds) >= self.s.LL_IC_MIN_OBS:
            P = np.array(self.ll_wf_preds)
            R = np.array(self.ll_wf_reals)
            ics = np.zeros(self.s.N_INST)
            for j in range(self.s.N_INST):
                if P[:, j].std() > 1e-12 and R[:, j].std() > 1e-12:
                    ics[j] = np.corrcoef(P[:, j], R[:, j])[0, 1]
            mask = (ics > self.s.LL_SEL_IC).astype(float)

        tgt = self.s.LIMITS * np.sign(pred) * mask
        tgt[self.s.PAIR_OWNED] = 0.0
        return tgt

    def compute_sleeves(self, prc_so_far: np.ndarray) -> dict[str, np.ndarray]:
        n_inst, nt = prc_so_far.shape
        zeros = np.zeros(n_inst, dtype=float)

        if n_inst != self.s.N_INST:
            return {
                "core_mr": zeros,
                "pairs": zeros,
                "algo_fade": zeros,
                "main_book": zeros,
                "lead_lag": zeros,
                "combined": zeros,
            }

        if nt <= self.prev_nt:
            self.reset()
        self.prev_nt = nt

        if nt < self.s.SLOW_LB + 1:
            return {
                "core_mr": zeros,
                "pairs": zeros,
                "algo_fade": zeros,
                "main_book": zeros,
                "lead_lag": zeros,
                "combined": zeros,
            }

        cur = np.maximum(prc_so_far[:, -1], 1.0)

        def zscore_to_past(lb: int) -> np.ndarray:
            hist = prc_so_far[:, -lb - 1:-1]
            mu = hist.mean(axis=1)
            sig = hist.std(axis=1)
            sig = np.where(sig > 1e-8, sig, 1.0)
            return (mu - cur) / sig

        z_fast = zscore_to_past(self.s.FAST_LB)
        z_slow = zscore_to_past(self.s.SLOW_LB)
        signal = self.s.FAST_WEIGHT * z_fast + (1.0 - self.s.FAST_WEIGHT) * z_slow

        holding_side = np.sign(self.prev_target_dollars)
        keep = (
            (holding_side != 0)
            & (np.sign(signal) == holding_side)
            & (np.abs(signal) >= self.s.EXIT_ABS_SIGNAL)
        )
        active = (np.abs(signal) >= self.s.MIN_ABS_SIGNAL) | keep
        signal = np.where(active, signal, 0.0)

        core = self.s.LIMITS * np.tanh(self.s.SIGNAL_SCALE * signal) * self.s.POSITION_MULT

        mask = np.zeros(n_inst, dtype=float)
        mask[self.s.CORE_ASSETS] = 1.0
        mask[self.s.PAIR_OWNED] = 0.0
        core *= mask

        if nt > self.s.BASE_VOL_WIN + self.s.RECENT_VOL_WIN + 1:
            log_prices = np.log(np.maximum(prc_so_far, self.s.EPS))
            log_rets = np.diff(log_prices, axis=1)

            asset_scale = np.ones(n_inst, dtype=float)
            global_scale = 1.0

            recent_market_vol = log_rets[0, -self.s.RECENT_VOL_WIN:].std()
            base_market_vol = log_rets[
                0,
                -(self.s.BASE_VOL_WIN + self.s.RECENT_VOL_WIN):-self.s.RECENT_VOL_WIN,
            ].std()

            if base_market_vol > 1e-8:
                market_vol_ratio = recent_market_vol / base_market_vol
                if market_vol_ratio > self.s.VOL_DANGER:
                    global_scale *= self.s.VOL_CUT

            recent_asset_vol = log_rets[:, -self.s.RECENT_VOL_WIN:].std(axis=1)
            base_asset_vol = log_rets[
                :,
                -(self.s.BASE_VOL_WIN + self.s.RECENT_VOL_WIN):-self.s.RECENT_VOL_WIN,
            ].std(axis=1)
            vol_ratio = recent_asset_vol / np.where(base_asset_vol > 1e-8, base_asset_vol, 1.0)
            asset_scale *= np.where(vol_ratio > self.s.VOL_DANGER, self.s.VOL_CUT, 1.0)

            if nt > self.s.TREND_LOOKBACK + self.s.TREND_WIN:
                recent_trend = log_prices[:, -1] - log_prices[:, -self.s.TREND_WIN - 1]
                daily_vol = log_rets[:, -self.s.TREND_LOOKBACK:].std(axis=1)
                trend_z = recent_trend / np.where(
                    daily_vol > 1e-8,
                    daily_vol * np.sqrt(self.s.TREND_WIN),
                    1.0,
                )

                fighting_trend = signal * trend_z < 0
                asset_scale *= np.where(
                    fighting_trend & (np.abs(trend_z) > self.s.TREND_DANGER),
                    self.s.TREND_CUT,
                    1.0,
                )

            core *= asset_scale * global_scale

        log_all = np.log(np.maximum(prc_so_far, self.s.EPS))
        pairs = np.zeros(n_inst, dtype=float)
        if nt > self.s.PAIR_ROLL + 1:
            for k, (i, j, g) in enumerate(self.s.PAIRS):
                spread = log_all[i] - g * log_all[j]
                win = spread[-self.s.PAIR_ROLL - 1:-1]
                z = (spread[-1] - win.mean()) / (win.std() + self.s.EPS)
                pos = self.pair_pos[k]
                if pos == 0:
                    if z > self.s.PAIR_ENTRY:
                        pos = -1
                    elif z < -self.s.PAIR_ENTRY:
                        pos = 1
                elif pos == 1 and z > -self.s.PAIR_EXIT:
                    pos = 0
                elif pos == -1 and z < self.s.PAIR_EXIT:
                    pos = 0
                self.pair_pos[k] = pos
                if pos != 0:
                    pairs[i] += pos * self.s.PAIR_LEG
                    pairs[j] -= pos * g * self.s.PAIR_LEG

        algo_fade = np.zeros(n_inst, dtype=float)
        if nt > self.s.ALGO_FADE_LB + self.s.ALGO_FADE_VOL_WIN + 1:
            lp0 = np.log(np.maximum(prc_so_far[0], self.s.EPS))
            fade_ret = lp0[-1] - lp0[-1 - self.s.ALGO_FADE_LB]
            fade_vol = np.diff(lp0[-(self.s.ALGO_FADE_VOL_WIN + 1):]).std()
            fade_z = fade_ret / max(fade_vol * np.sqrt(self.s.ALGO_FADE_LB), 1e-9)
            algo_fade[0] = -np.clip(fade_z / self.s.ALGO_FADE_SCALE, -1.0, 1.0) * self.s.ALGO_FADE_CAP

        main_book = core + pairs + algo_fade

        small_change = (
            (np.abs(main_book - self.prev_target_dollars) < self.s.DEAD_BAND_FRAC * self.s.LIMITS)
            & (main_book != 0.0)
        )
        main_book = np.where(small_change, self.prev_target_dollars, main_book)
        self.prev_target_dollars = main_book.copy()

        lead_lag = self._leadlag_target_dollars(prc_so_far)
        combined = np.clip(main_book + lead_lag, -self.s.LIMITS, self.s.LIMITS)

        return {
            "core_mr": core,
            "pairs": pairs,
            "algo_fade": algo_fade,
            "main_book": main_book,
            "lead_lag": lead_lag,
            "combined": combined,
        }


def _max_drawdown(cumulative_value: np.ndarray) -> float:
    if cumulative_value.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(cumulative_value)
    drawdown = cumulative_value - running_peak
    return float(drawdown.min())


def run_teamname_sleeve_analysis(
    prices: np.ndarray | None = None,
    num_test_days: int = NUM_TEST_DAYS,
) -> dict[str, object]:
    """Backtest the live strategy and its sleeves on the official test split."""
    prices = load_prices() if prices is None else prices

    strat = importlib.reload(strategy)
    strat.reset_state()
    decomposer = SleeveDecomposer(strat)

    n_inst, nt = prices.shape
    start_day = nt - num_test_days

    books = {
        "Combined": BookState(n_inst),
        "Main Book": BookState(n_inst),
        "Core MR": BookState(n_inst),
        "Pairs": BookState(n_inst),
        "ALGO Fade": BookState(n_inst),
        "Lead-Lag": BookState(n_inst),
    }

    max_share_diff = 0
    day_index = []

    for t in range(start_day, nt + 1):
        hist = prices[:, :t]
        cur_prices = hist[:, -1]

        if t < nt:
            sleeves = decomposer.compute_sleeves(hist)
            official_combined = np.asarray(strat.getMyPosition(hist), dtype=int)
            diag_combined = _shares_from_dollars(sleeves["combined"], cur_prices, strat.LIMITS)
            max_share_diff = max(max_share_diff, int(np.max(np.abs(official_combined - diag_combined))))

            targets = {
                "Combined": official_combined,
                "Main Book": _shares_from_dollars(sleeves["main_book"], cur_prices, strat.LIMITS),
                "Core MR": _shares_from_dollars(sleeves["core_mr"], cur_prices, strat.LIMITS),
                "Pairs": _shares_from_dollars(sleeves["pairs"], cur_prices, strat.LIMITS),
                "ALGO Fade": _shares_from_dollars(sleeves["algo_fade"], cur_prices, strat.LIMITS),
                "Lead-Lag": _shares_from_dollars(sleeves["lead_lag"], cur_prices, strat.LIMITS),
            }
        else:
            targets = {name: state.cur_pos.copy() for name, state in books.items()}

        record = t > start_day
        for name, state in books.items():
            state.step(targets[name], cur_prices, record)

        if record:
            day_index.append(t)

    daily_pl = pd.DataFrame(
        {name: state.daily_pl for name, state in books.items()},
        index=day_index,
    )
    cumulative_value = pd.DataFrame(
        {name: state.cumulative_value for name, state in books.items()},
        index=day_index,
    )
    traded_dollars = pd.DataFrame(
        {name: state.traded_dollars for name, state in books.items()},
        index=day_index,
    )
    gross_exposure = pd.DataFrame(
        {name: state.gross_exposure for name, state in books.items()},
        index=day_index,
    )
    net_exposure = pd.DataFrame(
        {name: state.net_exposure for name, state in books.items()},
        index=day_index,
    )

    rows = []
    for name, state in books.items():
        pl = np.asarray(state.daily_pl, dtype=float)
        mean_pl = float(pl.mean()) if pl.size else 0.0
        std_pl = float(pl.std()) if pl.size else 0.0
        ann_sharpe = float(np.sqrt(250) * mean_pl / std_pl) if std_pl > 1e-12 else 0.0
        rows.append(
            {
                "book": name,
                "final_pnl": float(state.value),
                "mean_daily_pl": mean_pl,
                "std_daily_pl": std_pl,
                "ann_sharpe": ann_sharpe,
                "max_drawdown": _max_drawdown(np.asarray(state.cumulative_value, dtype=float)),
                "turnover": float(state.tot_dvolume),
                "commission": float(state.commission_total),
                "win_rate": float((pl > 0).mean()) if pl.size else 0.0,
                "active_days": float((np.asarray(state.gross_exposure) > 0).mean()) if state.gross_exposure else 0.0,
                "avg_gross_exposure": float(np.mean(state.gross_exposure)) if state.gross_exposure else 0.0,
                "trade_days": int(np.sum(np.asarray(state.traded_dollars) > 0)),
            }
        )

    summary = pd.DataFrame(rows).set_index("book").loc[BOOK_ORDER]

    return {
        "prices": prices,
        "summary": summary,
        "daily_pl": daily_pl[BOOK_ORDER],
        "cumulative_value": cumulative_value[BOOK_ORDER],
        "traded_dollars": traded_dollars[BOOK_ORDER],
        "gross_exposure": gross_exposure[BOOK_ORDER],
        "net_exposure": net_exposure[BOOK_ORDER],
        "validation": {
            "test_start_day": start_day,
            "test_end_day": nt - 1,
            "num_test_days": num_test_days,
            "combined_max_share_diff_vs_teamname": max_share_diff,
        },
    }


def plot_suite(results: dict[str, object], rolling_window: int = 20) -> tuple[plt.Figure, np.ndarray]:
    """Create a compact matplotlib dashboard for the sleeve backtest."""
    import matplotlib.pyplot as plt

    daily_pl = results["daily_pl"]
    cumulative_value = results["cumulative_value"]
    traded_dollars = results["traded_dollars"]
    gross_exposure = results["gross_exposure"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    x = cumulative_value.index.to_numpy()

    for name in BOOK_ORDER:
        lw = 2.8 if name == "Combined" else 1.8
        ls = "--" if name == "Main Book" else "-"
        axes[0, 0].plot(x, cumulative_value[name], label=name, color=BOOK_COLORS[name], linewidth=lw, linestyle=ls)
    axes[0, 0].set_title("Cumulative Test PnL")
    axes[0, 0].set_xlabel("Backtest day")
    axes[0, 0].set_ylabel("PnL")
    axes[0, 0].legend(ncol=2, fontsize=9)

    rolling_pl = daily_pl.rolling(rolling_window, min_periods=max(5, rolling_window // 2)).sum()
    for name in BOOK_ORDER:
        lw = 2.6 if name == "Combined" else 1.6
        axes[0, 1].plot(x, rolling_pl[name], color=BOOK_COLORS[name], linewidth=lw)
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[0, 1].set_title(f"{rolling_window}-Day Rolling PnL")
    axes[0, 1].set_xlabel("Backtest day")
    axes[0, 1].set_ylabel("Rolling PnL")

    traded_ma = traded_dollars.rolling(10, min_periods=1).mean() / 1_000.0
    for name in BOOK_ORDER:
        lw = 2.6 if name == "Combined" else 1.6
        axes[1, 0].plot(x, traded_ma[name], color=BOOK_COLORS[name], linewidth=lw)
    axes[1, 0].set_title("10-Day Average Traded Dollars")
    axes[1, 0].set_xlabel("Backtest day")
    axes[1, 0].set_ylabel("$ traded (thousands)")

    gross_ma = gross_exposure.rolling(10, min_periods=1).mean() / 1_000.0
    for name in BOOK_ORDER:
        lw = 2.6 if name == "Combined" else 1.6
        axes[1, 1].plot(x, gross_ma[name], color=BOOK_COLORS[name], linewidth=lw)
    axes[1, 1].set_title("10-Day Average Gross Exposure")
    axes[1, 1].set_xlabel("Backtest day")
    axes[1, 1].set_ylabel("Gross exposure ($k)")

    return fig, axes
