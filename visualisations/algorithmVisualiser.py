from pathlib import Path
import ast
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Algorithm Visualiser", page_icon="📉", layout="wide")


PRICE_PATH = Path(__file__).resolve().parent.parent / "prices.txt"
ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DOLLAR_LIMIT = 10_000.0
INST0_DOLLAR_LIMIT = 100_000.0
DEFAULT_NUM_TEST_DAYS = 250
SCORE_PARAM = 1.0


def score(mu: float, sigma: float, param: float = SCORE_PARAM) -> float:
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


@st.cache_data(show_spinner=False)
def load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = [col.strip() for col in df.columns]
    return df.astype(float)


@st.cache_data(show_spinner=False)
def discover_strategy_files(root: str) -> list[str]:
    root_path = Path(root)
    candidates: list[str] = []
    for pattern in ("*.py", "strategies/*.py", "submissions/**/*.py"):
        for path in root_path.glob(pattern):
            if path.name.startswith("."):
                continue
            if path.name == Path(__file__).name:
                continue
            if path.name == "eval.py":
                continue
            try:
                source = path.read_text()
                module = ast.parse(source)
                has_entrypoint = any(
                    isinstance(node, ast.FunctionDef) and node.name == "getMyPosition"
                    for node in module.body
                )
            except Exception:
                has_entrypoint = False
            if not has_entrypoint:
                continue
            candidates.append(str(path.relative_to(root_path)))
    return sorted(set(candidates))


def load_strategy_namespace(path: Path) -> dict:
    namespace = runpy.run_path(str(path))
    if "getMyPosition" not in namespace:
        raise ValueError(f"{path} does not define getMyPosition(prcSoFar)")
    return namespace


def extract_strategy_parameters(path: Path, namespace: dict) -> pd.DataFrame:
    rows = []
    source = path.read_text()
    module = ast.parse(source)
    literal_values = {}

    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        literal_values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass

    for key, value in namespace.items():
        if not key.isupper():
            continue
        display_value = value
        if isinstance(display_value, np.ndarray):
            preview = np.array2string(display_value, threshold=12, precision=3)
            rows.append(
                {
                    "name": key,
                    "type": "ndarray",
                    "value": preview,
                    "shape": str(display_value.shape),
                }
            )
            continue

        if isinstance(display_value, (int, float, bool, str, np.integer, np.floating)):
            rows.append(
                {
                    "name": key,
                    "type": type(display_value).__name__,
                    "value": literal_values.get(key, display_value),
                    "shape": "",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["name", "type", "value", "shape"])

    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def simulate_strategy(price_df: pd.DataFrame, strategy_path: Path, num_test_days: int) -> dict:
    namespace = load_strategy_namespace(strategy_path)
    get_position = namespace["getMyPosition"]
    if "reset_state" in namespace and callable(namespace["reset_state"]):
        namespace["reset_state"]()

    prices = price_df.to_numpy().T
    n_inst, n_days = prices.shape
    num_test_days = min(max(1, num_test_days), n_days - 1)
    start_day = n_days - num_test_days

    comm_rate = np.full(n_inst, DEFAULT_COMM_RATE, dtype=float)
    comm_rate[0] = INST0_COMM_RATE
    dollar_limit = np.full(n_inst, DEFAULT_DOLLAR_LIMIT, dtype=float)
    dollar_limit[0] = INST0_DOLLAR_LIMIT

    cash = 0.0
    value = 0.0
    total_volume = 0.0
    pending_commission = 0.0

    cur_pos = np.zeros(n_inst, dtype=int)
    prev_prices = None
    prev_pos = np.zeros(n_inst, dtype=int)
    prev_trade_dollars = np.zeros(n_inst, dtype=float)

    positions = []
    desired_positions = []
    trade_shares = []
    trade_dollars = []
    exposures = []
    prices_used = []
    pnl_attr = []
    day_rows = []

    for t in range(start_day, n_days + 1):
        history = prices[:, :t]
        cur_prices = history[:, -1]

        if t < n_days:
            desired_raw = np.asarray(get_position(history), dtype=float).reshape(-1)
            if desired_raw.shape[0] != n_inst:
                raise ValueError(
                    f"Strategy returned shape {desired_raw.shape}, expected ({n_inst},)"
                )
            pos_limits = (dollar_limit / np.maximum(cur_prices, 1e-12)).astype(int)
            new_pos = np.clip(desired_raw, -pos_limits, pos_limits).astype(int)
        else:
            desired_raw = cur_pos.astype(float).copy()
            new_pos = cur_pos.astype(int).copy()

        delta_pos = new_pos - cur_pos
        cash -= float(cur_prices.dot(delta_pos)) + pending_commission

        dvolumes = cur_prices * np.abs(delta_pos)
        dvolume = float(np.sum(dvolumes))
        total_volume += dvolume
        pending_commission = float(np.sum(dvolumes * comm_rate))

        cur_pos = new_pos.copy()
        pos_value = float(cur_pos.dot(cur_prices))
        today_value = cash + pos_value
        today_pl = today_value - value
        value = today_value

        gross_exposure = float(np.sum(np.abs(cur_pos * cur_prices)))
        long_exposure = float(np.sum(np.maximum(cur_pos * cur_prices, 0.0)))
        short_exposure = float(-np.sum(np.minimum(cur_pos * cur_prices, 0.0)))
        net_exposure = float(np.sum(cur_pos * cur_prices))
        active_positions = int(np.count_nonzero(cur_pos))
        traded_instruments = int(np.count_nonzero(delta_pos))
        clipped_instruments = int(np.count_nonzero(new_pos != desired_raw.astype(int)))
        turnover_ratio = dvolume / gross_exposure if gross_exposure > 1e-12 else 0.0
        capital_limit_total = float(np.sum(dollar_limit))
        capital_utilisation = gross_exposure / capital_limit_total if capital_limit_total > 0 else 0.0
        ret_on_volume = value / total_volume if total_volume > 0 else 0.0

        if prev_prices is None:
            per_inst_pnl = np.zeros(n_inst, dtype=float)
        else:
            per_inst_pnl = prev_pos * (cur_prices - prev_prices) - prev_trade_dollars * comm_rate

        day_rows.append(
            {
                "eval_day": t,
                "price_day_index": t - 1,
                "portfolio_value": value,
                "daily_pl": today_pl,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "dollar_traded": dvolume,
                "cum_dollar_traded": total_volume,
                "turnover_ratio": turnover_ratio,
                "capital_utilisation": capital_utilisation,
                "active_positions": active_positions,
                "traded_instruments": traded_instruments,
                "clipped_instruments": clipped_instruments,
                "pending_commission": pending_commission,
                "return_on_volume": ret_on_volume,
            }
        )

        positions.append(cur_pos.copy())
        desired_positions.append(desired_raw.copy())
        trade_shares.append(delta_pos.copy())
        trade_dollars.append(dvolumes.copy())
        exposures.append((cur_pos * cur_prices).copy())
        prices_used.append(cur_prices.copy())
        pnl_attr.append(per_inst_pnl.copy())

        prev_prices = cur_prices.copy()
        prev_pos = cur_pos.copy()
        prev_trade_dollars = dvolumes.copy()

    day_df = pd.DataFrame(day_rows)
    scored_df = day_df.iloc[1:].copy()
    scored_pl = scored_df["daily_pl"].to_numpy()
    mean_pl = float(np.mean(scored_pl)) if len(scored_pl) else 0.0
    std_pl = float(np.std(scored_pl)) if len(scored_pl) else 0.0
    ann_sharpe = float(np.sqrt(250) * mean_pl / std_pl) if std_pl > 1e-12 else 0.0
    score_val = float(score(mean_pl, std_pl, SCORE_PARAM))

    day_df["drawdown"] = day_df["portfolio_value"] - day_df["portfolio_value"].cummax()
    day_df["rolling_20_pl_vol"] = day_df["daily_pl"].rolling(20, min_periods=5).std()
    roll_mean = day_df["daily_pl"].rolling(20, min_periods=5).mean()
    roll_std = day_df["daily_pl"].rolling(20, min_periods=5).std()
    day_df["rolling_20_sharpe"] = np.where(
        roll_std > 1e-12, np.sqrt(250) * roll_mean / roll_std, np.nan
    )

    positions_arr = np.vstack(positions)
    desired_arr = np.vstack(desired_positions)
    trade_shares_arr = np.vstack(trade_shares)
    trade_dollars_arr = np.vstack(trade_dollars)
    exposures_arr = np.vstack(exposures)
    prices_arr = np.vstack(prices_used)
    pnl_attr_arr = np.vstack(pnl_attr)

    instrument_names = list(price_df.columns)
    limit_usage = np.abs(exposures_arr) / dollar_limit

    inst_rows = []
    for i, name in enumerate(instrument_names):
        inst_daily_pnl = pnl_attr_arr[1:, i]
        inst_mean = float(np.mean(inst_daily_pnl)) if len(inst_daily_pnl) else 0.0
        inst_std = float(np.std(inst_daily_pnl)) if len(inst_daily_pnl) else 0.0
        inst_sharpe = float(np.sqrt(250) * inst_mean / inst_std) if inst_std > 1e-12 else 0.0
        inst_rows.append(
            {
                "instrument": name,
                "total_pl": float(np.sum(inst_daily_pnl)),
                "mean_daily_pl": inst_mean,
                "ann_sharpe": inst_sharpe,
                "avg_abs_exposure": float(np.mean(np.abs(exposures_arr[:, i]))),
                "avg_limit_usage": float(np.mean(limit_usage[:, i])),
                "max_limit_usage": float(np.max(limit_usage[:, i])),
                "total_dollar_traded": float(np.sum(trade_dollars_arr[:, i])),
                "trade_days": int(np.count_nonzero(trade_shares_arr[:, i])),
                "final_position": int(positions_arr[-1, i]),
            }
        )
    inst_df = pd.DataFrame(inst_rows).sort_values("total_pl", ascending=False).reset_index(drop=True)

    summary = {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "ann_sharpe": ann_sharpe,
        "score": score_val,
        "final_value": float(day_df["portfolio_value"].iloc[-1]),
        "max_drawdown": float(day_df["drawdown"].min()),
        "win_rate": float((scored_df["daily_pl"] > 0).mean()) if len(scored_df) else 0.0,
        "total_dollar_traded": float(day_df["dollar_traded"].sum()),
        "avg_turnover": float(scored_df["turnover_ratio"].mean()) if len(scored_df) else 0.0,
        "avg_gross_exposure": float(scored_df["gross_exposure"].mean()) if len(scored_df) else 0.0,
        "avg_active_positions": float(scored_df["active_positions"].mean()) if len(scored_df) else 0.0,
        "num_test_days": num_test_days,
        "start_day": start_day,
        "end_day": n_days - 1,
    }

    return {
        "summary": summary,
        "day_df": day_df,
        "instrument_df": inst_df,
        "positions": positions_arr,
        "desired_positions": desired_arr,
        "trade_shares": trade_shares_arr,
        "trade_dollars": trade_dollars_arr,
        "exposures": exposures_arr,
        "prices": prices_arr,
        "pnl_attr": pnl_attr_arr,
        "limit_usage": limit_usage,
        "comm_rate": comm_rate,
        "dollar_limit": dollar_limit,
        "strategy_params": extract_strategy_parameters(strategy_path, namespace),
        "strategy_source": strategy_path.read_text(),
        "instrument_names": instrument_names,
    }


def main() -> None:
    st.title("Algorithm Visualiser")
    st.caption(
        "Replay a strategy through evaluator-style execution and inspect how it deploys capital, rebalances, and earns or loses PnL."
    )

    price_df = load_price_data(str(PRICE_PATH))
    available_strategies = discover_strategy_files(str(ROOT_DIR))
    if not available_strategies:
        st.error("No strategy files with getMyPosition(prcSoFar) were found.")
        return

    default_strategy = "teamName.py" if "teamName.py" in available_strategies else available_strategies[0]
    max_test_days = max(2, len(price_df) - 1)

    st.sidebar.header("Run Setup")
    strategy_rel = st.sidebar.selectbox("Strategy file", available_strategies, index=available_strategies.index(default_strategy))
    num_test_days = st.sidebar.slider(
        "Test window (days)",
        min_value=min(50, max_test_days),
        max_value=max_test_days,
        value=min(DEFAULT_NUM_TEST_DAYS, max_test_days),
    )
    selected_view_instrument = st.sidebar.selectbox("Instrument drilldown", list(price_df.columns), index=0)

    result = simulate_strategy(price_df, ROOT_DIR / strategy_rel, num_test_days)

    summary = result["summary"]
    day_df = result["day_df"]
    instrument_df = result["instrument_df"]
    instrument_names = result["instrument_names"]
    inst_idx = instrument_names.index(selected_view_instrument)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Score", f"{summary['score']:.2f}")
    top2.metric("Ann. Sharpe", f"{summary['ann_sharpe']:.2f}")
    top3.metric("Final value", f"${summary['final_value']:,.0f}")
    top4.metric("Max drawdown", f"${summary['max_drawdown']:,.0f}")

    top5, top6, top7, top8 = st.columns(4)
    top5.metric("Mean daily PL", f"${summary['mean_pl']:,.1f}")
    top6.metric("PL std dev", f"${summary['std_pl']:,.1f}")
    top7.metric("Total traded", f"${summary['total_dollar_traded']:,.0f}")
    top8.metric("Win rate", f"{summary['win_rate'] * 100:.1f}%")

    st.caption(
        f"Window: days {summary['start_day']} to {summary['end_day']} | "
        f"Average gross exposure ${summary['avg_gross_exposure']:,.0f} | "
        f"Average active positions {summary['avg_active_positions']:.1f} | "
        f"Average turnover {summary['avg_turnover'] * 100:.1f}%"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Portfolio overview", "Capital & trading", "Instrument drilldown", "Strategy introspection"]
    )

    with tab1:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(day_df["price_day_index"], day_df["portfolio_value"], color="#1f77b4", linewidth=1.4)
        axes[0].axhline(0, color="black", linestyle=":", linewidth=0.8)
        axes[0].set_title("Equity curve")
        axes[0].set_ylabel("Value ($)")
        axes[0].grid(alpha=0.3)

        colors = np.where(day_df["daily_pl"] >= 0, "#4CAF50", "#F44336")
        axes[1].bar(day_df["price_day_index"], day_df["daily_pl"], color=colors, width=0.9)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Daily PL")
        axes[1].set_ylabel("PL ($)")
        axes[1].grid(alpha=0.3)

        axes[2].fill_between(
            day_df["price_day_index"],
            day_df["drawdown"],
            0,
            color="#E57373",
            alpha=0.9,
        )
        axes[2].set_title("Drawdown from peak")
        axes[2].set_xlabel("Day")
        axes[2].set_ylabel("Drawdown ($)")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        hist_col, roll_col = st.columns(2)
        with hist_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            scored_pl = day_df["daily_pl"].iloc[1:]
            ax.hist(scored_pl, bins=30, color="#5DA5DA", edgecolor="white", alpha=0.9)
            if len(scored_pl):
                ax.axvline(scored_pl.mean(), color="#2CA02C", linestyle="-", linewidth=1.2, label=f"Mean: {scored_pl.mean():.1f}")
                ax.axvline(scored_pl.quantile(0.05), color="#FF9800", linestyle="--", linewidth=1.2, label=f"VaR 95%: {scored_pl.quantile(0.05):.1f}")
                ax.axvline(scored_pl.quantile(0.01), color="#D62728", linestyle="--", linewidth=1.2, label=f"CVaR-ish tail: {scored_pl[scored_pl <= scored_pl.quantile(0.05)].mean():.1f}")
            ax.set_title("Daily PL distribution")
            ax.set_xlabel("PL ($)")
            ax.set_ylabel("Frequency")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        with roll_col:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axes[0].plot(day_df["price_day_index"], day_df["rolling_20_pl_vol"], color="#9C27B0", linewidth=1.2)
            axes[0].set_title("Rolling 20-day PL volatility")
            axes[0].set_ylabel("Std dev ($)")
            axes[0].grid(alpha=0.3)

            axes[1].plot(day_df["price_day_index"], day_df["rolling_20_sharpe"], color="#673AB7", linewidth=1.1)
            axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
            axes[1].set_title("Rolling 20-day annualised Sharpe")
            axes[1].set_xlabel("Day")
            axes[1].set_ylabel("Sharpe")
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        pnl_col, sharpe_col = st.columns(2)
        with pnl_col:
            top_pnl = instrument_df.sort_values("total_pl")
            fig, ax = plt.subplots(figsize=(9, 10))
            colors = np.where(top_pnl["total_pl"] >= 0, "#4CAF50", "#E53935")
            ax.barh(top_pnl["instrument"], top_pnl["total_pl"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Per-instrument total PL attribution")
            ax.set_xlabel("Total PL ($)")
            ax.set_ylabel("Instrument")
            ax.grid(alpha=0.25, axis="x")
            st.pyplot(fig)

        with sharpe_col:
            top_sharpe = instrument_df.sort_values("ann_sharpe")
            fig, ax = plt.subplots(figsize=(9, 10))
            colors = np.where(top_sharpe["ann_sharpe"] >= 0, "#4CAF50", "#E53935")
            ax.barh(top_sharpe["instrument"], top_sharpe["ann_sharpe"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Per-instrument annualised Sharpe")
            ax.set_xlabel("Sharpe")
            ax.set_ylabel("Instrument")
            ax.grid(alpha=0.25, axis="x")
            st.pyplot(fig)

    with tab2:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(day_df["price_day_index"], day_df["gross_exposure"], label="Gross", color="#FF9800", linewidth=1.3)
        axes[0].plot(day_df["price_day_index"], day_df["long_exposure"], label="Long", color="#4CAF50", linewidth=1.1)
        axes[0].plot(day_df["price_day_index"], day_df["short_exposure"], label="Short", color="#E53935", linewidth=1.1)
        axes[0].set_title("Capital deployment")
        axes[0].set_ylabel("Exposure ($)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(day_df["price_day_index"], day_df["turnover_ratio"] * 100, color="#FB8C00", linewidth=1.2)
        axes[1].set_title("Daily turnover ratio")
        axes[1].set_ylabel("Turnover (%)")
        axes[1].grid(alpha=0.3)

        axes[2].plot(day_df["price_day_index"], day_df["active_positions"], color="#00ACC1", linewidth=1.2, label="Active positions")
        axes[2].plot(day_df["price_day_index"], day_df["traded_instruments"], color="#8E24AA", linewidth=1.2, label="Traded instruments")
        axes[2].plot(day_df["price_day_index"], day_df["clipped_instruments"], color="#6D4C41", linewidth=1.2, label="Clipped instruments")
        axes[2].set_title("How busy the strategy is")
        axes[2].set_xlabel("Day")
        axes[2].set_ylabel("Count")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        st.pyplot(fig)

        detail_day = st.slider(
            "Inspect a trading day",
            min_value=int(day_df["price_day_index"].min()),
            max_value=int(day_df["price_day_index"].max()),
            value=int(day_df["price_day_index"].iloc[min(len(day_df) - 1, 20)]),
        )
        detail_idx = int(day_df.index[day_df["price_day_index"] == detail_day][0])

        day_stats = day_df.iloc[detail_idx]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Portfolio value", f"${day_stats['portfolio_value']:,.0f}")
        d2.metric("Daily PL", f"${day_stats['daily_pl']:,.0f}")
        d3.metric("Gross exposure", f"${day_stats['gross_exposure']:,.0f}")
        d4.metric("Dollar traded", f"${day_stats['dollar_traded']:,.0f}")

        day_table = pd.DataFrame(
            {
                "instrument": instrument_names,
                "price": result["prices"][detail_idx],
                "desired_shares": np.round(result["desired_positions"][detail_idx]).astype(int),
                "actual_shares": result["positions"][detail_idx].astype(int),
                "trade_shares": result["trade_shares"][detail_idx].astype(int),
                "trade_dollars": result["trade_dollars"][detail_idx],
                "exposure_dollars": result["exposures"][detail_idx],
                "limit_usage": result["limit_usage"][detail_idx],
                "pnl_contribution": result["pnl_attr"][detail_idx],
            }
        )
        day_table["abs_trade_dollars"] = day_table["trade_dollars"].abs()
        day_table["abs_exposure"] = day_table["exposure_dollars"].abs()

        st.subheader(f"Top trades on day {detail_day}")
        st.dataframe(
            day_table.sort_values("abs_trade_dollars", ascending=False)
            .drop(columns=["abs_trade_dollars", "abs_exposure"])
            .reset_index(drop=True),
            use_container_width=True,
        )

        st.subheader("Most limit-using instruments")
        st.dataframe(
            day_table.sort_values("limit_usage", ascending=False)
            .drop(columns=["abs_trade_dollars", "abs_exposure"])
            .reset_index(drop=True)
            .head(15),
            use_container_width=True,
        )

    with tab3:
        inst_series = pd.DataFrame(
            {
                "day": day_df["price_day_index"],
                "price": result["prices"][:, inst_idx],
                "position_shares": result["positions"][:, inst_idx],
                "desired_shares": np.round(result["desired_positions"][:, inst_idx]).astype(int),
                "trade_shares": result["trade_shares"][:, inst_idx].astype(int),
                "trade_dollars": result["trade_dollars"][:, inst_idx],
                "exposure_dollars": result["exposures"][:, inst_idx],
                "pnl_contribution": result["pnl_attr"][:, inst_idx],
                "limit_usage": result["limit_usage"][:, inst_idx],
            }
        )

        inst_summary = instrument_df[instrument_df["instrument"] == selected_view_instrument].iloc[0]
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Instrument total PL", f"${inst_summary['total_pl']:,.0f}")
        i2.metric("Instrument ann. Sharpe", f"{inst_summary['ann_sharpe']:.2f}")
        i3.metric("Total traded", f"${inst_summary['total_dollar_traded']:,.0f}")
        i4.metric("Max limit usage", f"{inst_summary['max_limit_usage'] * 100:.1f}%")

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        axes[0].plot(inst_series["day"], inst_series["price"], color="#1f77b4", linewidth=1.3)
        axes[0].set_title(f"{selected_view_instrument} price")
        axes[0].set_ylabel("Price")
        axes[0].grid(alpha=0.3)

        axes[1].plot(inst_series["day"], inst_series["desired_shares"], color="#90CAF9", linewidth=1.0, label="Desired")
        axes[1].plot(inst_series["day"], inst_series["position_shares"], color="#1565C0", linewidth=1.3, label="Actual")
        axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
        axes[1].set_title("Desired vs actual position")
        axes[1].set_ylabel("Shares")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        trade_colors = np.where(inst_series["trade_shares"] >= 0, "#4CAF50", "#E53935")
        axes[2].bar(inst_series["day"], inst_series["trade_shares"], color=trade_colors, width=0.9)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_title("Trades")
        axes[2].set_ylabel("Delta shares")
        axes[2].grid(alpha=0.3)

        axes[3].plot(inst_series["day"], inst_series["exposure_dollars"], color="#FF9800", linewidth=1.2, label="Exposure")
        axes[3].plot(inst_series["day"], inst_series["pnl_contribution"].cumsum(), color="#8E24AA", linewidth=1.2, label="Cum PL")
        axes[3].axhline(0, color="black", linestyle=":", linewidth=0.8)
        axes[3].set_title("Exposure and cumulative PL attribution")
        axes[3].set_xlabel("Day")
        axes[3].set_ylabel("Dollars")
        axes[3].grid(alpha=0.3)
        axes[3].legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(inst_series, use_container_width=True)

    with tab4:
        st.subheader("Detected strategy parameters")
        st.dataframe(result["strategy_params"], use_container_width=True)

        st.subheader("Top instruments by contribution")
        st.dataframe(instrument_df, use_container_width=True)

        with st.expander("Strategy source code"):
            st.code(result["strategy_source"], language="python")


if __name__ == "__main__":
    main()
