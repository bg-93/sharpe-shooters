from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Instrument Explorer", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = [col.strip() for col in df.columns]
    return df.astype(float)


@st.cache_data(show_spinner=False)
def compute_metrics(price_series: pd.Series) -> dict:
    returns = np.log(price_series / price_series.shift(1)).dropna()

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
    }


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

    tab1, tab2, tab3 = st.tabs(["Price view", "Return distribution", "Rolling signals"])

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
                ],
            }
        )
        st.dataframe(summary, use_container_width=True)


if __name__ == "__main__":
    main()
