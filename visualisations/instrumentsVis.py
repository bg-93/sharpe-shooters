from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.stattools import adfuller


st.set_page_config(page_title="Instrument Explorer", page_icon="📈", layout="wide")


EPS = 1e-12


@st.cache_data(show_spinner=False)
def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = [col.strip() for col in df.columns]
    return df.astype(float)


@st.cache_data(show_spinner=False)
def compute_log_diff(price_series: pd.Series) -> pd.Series:
    safe_prices = np.maximum(price_series.astype(float), EPS)
    return np.log(safe_prices).diff().dropna()


@st.cache_data(show_spinner=False)
def compute_stationarity_metrics(price_series: pd.Series) -> dict:
    log_diff = compute_log_diff(price_series)

    if len(log_diff) < 20:
        return {
            "adf_stat": np.nan,
            "adf_pvalue": np.nan,
            "lag1_autocorr_log_diff": np.nan,
            "mean_log_diff": np.nan,
            "vol_log_diff": np.nan,
            "stationary_flag": "insufficient data",
        }

    try:
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(log_diff.to_numpy(), autolag="AIC")
    except ValueError:
        adf_stat, adf_pvalue = np.nan, np.nan

    stationary_flag = "yes" if pd.notna(adf_pvalue) and adf_pvalue < 0.05 else "no"

    return {
        "adf_stat": float(adf_stat) if pd.notna(adf_stat) else np.nan,
        "adf_pvalue": float(adf_pvalue) if pd.notna(adf_pvalue) else np.nan,
        "lag1_autocorr_log_diff": float(log_diff.autocorr(lag=1)) if len(log_diff) > 1 else np.nan,
        "mean_log_diff": float(log_diff.mean()),
        "vol_log_diff": float(log_diff.std()),
        "stationary_flag": stationary_flag,
    }


@st.cache_data(show_spinner=False)
def compute_metrics(price_series: pd.Series) -> dict:
    returns = np.log(price_series / price_series.shift(1)).dropna()
    stationarity = compute_stationarity_metrics(price_series)

    latest_price = float(price_series.iloc[-1])
    start_price = float(price_series.iloc[0])
    total_return = (latest_price / start_price - 1.0) if start_price else np.nan

    mean_daily_return = float(returns.mean()) if len(returns) else np.nan
    vol_daily_return = float(returns.std()) if len(returns) else np.nan
    sharpe = float(np.sqrt(252) * mean_daily_return / vol_daily_return) if vol_daily_return else np.nan
    skew = float(returns.skew()) if len(returns) else np.nan
    kurt = float(returns.kurtosis()) if len(returns) else np.nan
    lag1_autocorr = float(returns.autocorr(lag=1)) if len(returns) > 1 else np.nan

    rolling_mean = price_series.rolling(20, min_periods=20).mean()
    rolling_std = price_series.rolling(20, min_periods=20).std().replace(0, np.nan)
    z_score = (price_series - rolling_mean) / rolling_std
    current_z_score = float(z_score.iloc[-1]) if len(z_score) else np.nan

    drawdown = 1 - price_series / price_series.cummax()
    max_drawdown = float(drawdown.max()) if len(drawdown) else np.nan

    momentum_5 = float(price_series.iloc[-1] / price_series.shift(5).iloc[-1] - 1.0) if len(price_series) > 5 else np.nan

    return {
        "latest_price": latest_price,
        "total_return": total_return,
        "mean_daily_return": mean_daily_return,
        "vol_daily_return": vol_daily_return,
        "sharpe": sharpe,
        "skew": skew,
        "kurt": kurt,
        "lag1_autocorr": lag1_autocorr,
        "current_z_score": current_z_score,
        "max_drawdown": max_drawdown,
        "momentum_5": momentum_5,
        **stationarity,
    }


@st.cache_data(show_spinner=False)
def compute_stationarity_table(price_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instrument in price_df.columns:
        metrics = compute_stationarity_metrics(price_df[instrument])
        rows.append(
            {
                "instrument": instrument,
                "adf_pvalue": metrics["adf_pvalue"],
                "adf_stat": metrics["adf_stat"],
                "lag1_autocorr_log_diff": metrics["lag1_autocorr_log_diff"],
                "mean_log_diff": metrics["mean_log_diff"],
                "vol_log_diff": metrics["vol_log_diff"],
                "stationary_flag": metrics["stationary_flag"],
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(["adf_pvalue", "adf_stat"], na_position="last").reset_index(drop=True)


PRICE_PATH = Path(__file__).resolve().parent.parent / "prices.txt"


def main() -> None:
    st.title("Instrument Explorer")
    st.caption("Inspect individual instruments, zoom into specific time windows, and compare trading-relevant statistics.")

    if not PRICE_PATH.exists():
        st.error(f"Could not find price data at {PRICE_PATH}")
        return

    price_df = load_price_data(PRICE_PATH)
    if price_df.empty:
        st.error("No price data was loaded.")
        return

    instrument_names = list(price_df.columns)
    stationarity_table = compute_stationarity_table(price_df)

    st.sidebar.header("Selection")
    selected_instrument = st.sidebar.selectbox("Choose an instrument", instrument_names, index=0)

    price_series = price_df[selected_instrument]
    metrics = compute_metrics(price_series)

    start_idx = st.sidebar.slider(
        "Start day",
        min_value=0,
        max_value=len(price_series) - 1,
        value=0,
    )
    end_idx = st.sidebar.slider(
        "End day",
        min_value=start_idx,
        max_value=len(price_series) - 1,
        value=len(price_series) - 1,
    )

    window = price_series.iloc[start_idx : end_idx + 1]
    window_returns = np.log(window / window.shift(1)).dropna()
    window_log = np.log(np.maximum(window, EPS))
    window_log_diff = window_log.diff().dropna()

    st.subheader(f"{selected_instrument}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest price", f"{metrics['latest_price']:.2f}")
    col2.metric("Total return", f"{metrics['total_return'] * 100:.2f}%")
    col3.metric("Daily vol", f"{metrics['vol_daily_return'] * 100:.2f}%")
    col4.metric("Sharpe", f"{metrics['sharpe']:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Mean daily return", f"{metrics['mean_daily_return'] * 100:.3f}%")
    col6.metric("Lag-1 autocorr", f"{metrics['lag1_autocorr']:.3f}")
    col7.metric("Current z-score", f"{metrics['current_z_score']:.2f}")
    col8.metric("Max drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")

    st.caption(
        f"Momentum (5-day): {metrics['momentum_5'] * 100:.2f}% | Skew: {metrics['skew']:.3f} | Kurtosis: {metrics['kurt']:.3f}"
    )

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Log-diff ADF p-value", f"{metrics['adf_pvalue']:.4f}" if pd.notna(metrics["adf_pvalue"]) else "n/a")
    col10.metric("Log-diff ADF stat", f"{metrics['adf_stat']:.3f}" if pd.notna(metrics["adf_stat"]) else "n/a")
    col11.metric("Log-diff lag-1 autocorr", f"{metrics['lag1_autocorr_log_diff']:.3f}" if pd.notna(metrics["lag1_autocorr_log_diff"]) else "n/a")
    col12.metric("Log-diff stationary?", metrics["stationary_flag"])

    tab1, tab2, tab3, tab4 = st.tabs(["Price view", "Return distribution", "Rolling signals", "Log-diff stationarity"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window.index, window.values, color="#1f77b4", linewidth=1.5)
        ax.plot(window.index, window.rolling(20, min_periods=20).mean().values, color="#ff7f0e", linewidth=1.2, linestyle="--")
        ax.set_title(f"{selected_instrument} price series")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(window_returns.values, bins=30, color="#4c78a8", edgecolor="black")
        ax.axvline(window_returns.mean(), color="#e45756", linestyle="--", label="Mean")
        ax.set_title(f"Daily return distribution for {selected_instrument}")
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with tab3:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        z_series = (window - window.rolling(20, min_periods=20).mean()) / window.rolling(20, min_periods=20).std().replace(0, np.nan)
        axes[0].plot(window.index, z_series.values, color="#2ca02c", linewidth=1.2)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Rolling z-score")
        axes[0].set_ylabel("Z-score")
        axes[0].grid(alpha=0.3)

        axes[1].plot(window.index, window.rolling(10, min_periods=10).std().values, color="#9467bd", linewidth=1.2)
        axes[1].set_title("Rolling 10-day volatility")
        axes[1].set_xlabel("Day")
        axes[1].set_ylabel("Volatility")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

        axes[0].plot(window.index, window_log.values, color="#1f77b4", linewidth=1.4)
        axes[0].plot(
            window.index,
            window_log.rolling(20, min_periods=20).mean().values,
            color="#ff7f0e",
            linewidth=1.1,
            linestyle="--",
        )
        axes[0].set_title("Log price")
        axes[0].set_ylabel("log(price)")
        axes[0].grid(alpha=0.3)

        axes[1].plot(window_log_diff.index, window_log_diff.values, color="#2ca02c", linewidth=1.1, label="log diff")
        axes[1].plot(
            window_log_diff.index,
            window_log_diff.rolling(20, min_periods=20).mean().values,
            color="#d62728",
            linewidth=1.1,
            linestyle="--",
            label="20-day mean",
        )
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle=":")
        axes[1].set_title("First difference of log price")
        axes[1].set_ylabel("Δ log(price)")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        axes[2].plot(
            window_log_diff.index,
            window_log_diff.rolling(20, min_periods=20).std().values,
            color="#9467bd",
            linewidth=1.1,
        )
        axes[2].set_title("Rolling 20-day volatility of log diff")
        axes[2].set_xlabel("Day")
        axes[2].set_ylabel("Std dev")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        if pd.notna(metrics["adf_pvalue"]):
            st.caption(
                f"ADF on full-sample log-diff: p-value={metrics['adf_pvalue']:.4f}. "
                "Lower values are more consistent with stationarity after the log-and-diff transform."
            )
        else:
            st.caption("ADF test could not be computed for this instrument.")

    with st.expander("Instrument summary table"):
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Latest price",
                    "Total return",
                    "Mean daily return",
                    "Daily volatility",
                    "Sharpe",
                    "Lag-1 autocorr",
                    "Current z-score",
                    "Max drawdown",
                    "5-day momentum",
                    "Skew",
                    "Kurtosis",
                    "Log-diff ADF p-value",
                    "Log-diff ADF stat",
                    "Log-diff lag-1 autocorr",
                    "Log-diff stationary?",
                ],
                "Value": [
                    f"{metrics['latest_price']:.2f}",
                    f"{metrics['total_return'] * 100:.2f}%",
                    f"{metrics['mean_daily_return'] * 100:.3f}%",
                    f"{metrics['vol_daily_return'] * 100:.3f}%",
                    f"{metrics['sharpe']:.3f}",
                    f"{metrics['lag1_autocorr']:.3f}",
                    f"{metrics['current_z_score']:.3f}",
                    f"{metrics['max_drawdown'] * 100:.3f}%",
                    f"{metrics['momentum_5'] * 100:.3f}%",
                    f"{metrics['skew']:.3f}",
                    f"{metrics['kurt']:.3f}",
                    f"{metrics['adf_pvalue']:.4f}" if pd.notna(metrics["adf_pvalue"]) else "n/a",
                    f"{metrics['adf_stat']:.3f}" if pd.notna(metrics["adf_stat"]) else "n/a",
                    f"{metrics['lag1_autocorr_log_diff']:.3f}" if pd.notna(metrics["lag1_autocorr_log_diff"]) else "n/a",
                    metrics["stationary_flag"],
                ],
            }
        )
        st.dataframe(summary, use_container_width=True)

    with st.expander("Stationarity ranking across all instruments"):
        display_table = stationarity_table.copy()
        display_table["adf_pvalue"] = display_table["adf_pvalue"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
        display_table["adf_stat"] = display_table["adf_stat"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
        display_table["lag1_autocorr_log_diff"] = display_table["lag1_autocorr_log_diff"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "n/a"
        )
        display_table["mean_log_diff"] = display_table["mean_log_diff"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "n/a")
        display_table["vol_log_diff"] = display_table["vol_log_diff"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "n/a")
        st.dataframe(display_table, use_container_width=True)


if __name__ == "__main__":
    main()
