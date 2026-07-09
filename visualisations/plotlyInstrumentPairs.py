from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import coint


st.set_page_config(page_title="Plotly Instrument Explorer", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = [col.strip() for col in df.columns]
    return df.astype(float)


@st.cache_data(show_spinner=False)
def compute_pair_stats(price_df: pd.DataFrame) -> pd.DataFrame:
    instruments = list(price_df.columns)
    rows: List[dict] = []

    for i, left in enumerate(instruments):
        for right in instruments[i + 1 :]:
            x = np.log(price_df[left].astype(float).to_numpy())
            y = np.log(price_df[right].astype(float).to_numpy())
            if len(x) < 30:
                continue
            x_ret = np.diff(x)
            y_ret = np.diff(y)
            corr = float(np.corrcoef(x_ret, y_ret)[0, 1]) if len(x_ret) > 1 else np.nan
            if np.isnan(corr):
                continue

            slope, intercept = np.polyfit(x, y, 1)
            spread = y - (slope * x + intercept)
            spread_z = (spread - np.mean(spread)) / np.std(spread)
            current_z = float(spread_z[-1]) if len(spread_z) else np.nan
            rolling_corr = float(pd.Series(x_ret).rolling(20, min_periods=20).corr(pd.Series(y_ret)).iloc[-1]) if len(x_ret) >= 20 else np.nan

            try:
                _, pvalue, _ = coint(x, y)
            except Exception:
                pvalue = 1.0

            hedge_ratio = float(slope)
            hedge_r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 1 else np.nan
            hedge_score = float(0.5 * abs(corr) + 0.5 * max(0.0, 1.0 - min(pvalue, 1.0)))

            rows.append(
                {
                    "left": left,
                    "right": right,
                    "corr": corr,
                    "rolling_corr": rolling_corr,
                    "hedge_ratio": hedge_ratio,
                    "hedge_r2": hedge_r2,
                    "cointegration_pvalue": float(pvalue),
                    "current_spread_z": current_z,
                    "hedge_score": hedge_score,
                }
            )

    return pd.DataFrame(rows).sort_values("hedge_score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_instrument_stats(price_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for instrument in price_df.columns:
        series = price_df[instrument].astype(float)
        returns = np.log(series / series.shift(1)).dropna()
        rolling_mean = series.rolling(20, min_periods=20).mean()
        rolling_std = series.rolling(20, min_periods=20).std().replace(0, np.nan)
        z_score = (series - rolling_mean) / rolling_std
        rows.append(
            {
                "instrument": instrument,
                "latest_price": float(series.iloc[-1]),
                "total_return": float(series.iloc[-1] / series.iloc[0] - 1.0),
                "daily_vol": float(returns.std()),
                "sharpe": float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() else np.nan,
                "lag1_autocorr": float(returns.autocorr(lag=1)) if len(returns) > 1 else np.nan,
                "current_z_score": float(z_score.iloc[-1]) if len(z_score) else np.nan,
                "momentum_5d": float(series.iloc[-1] / series.shift(5).iloc[-1] - 1.0) if len(series) > 5 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("instrument").reset_index(drop=True)


PRICE_PATH = Path(__file__).resolve().parent.parent / "prices.txt"


def main() -> None:
    st.title("Plotly Instrument & Pair Explorer")
    st.caption("View several instruments together, inspect pair-tradability metrics, and find hedging candidates.")

    if not PRICE_PATH.exists():
        st.error(f"Could not locate price data at {PRICE_PATH}")
        return

    price_df = load_price_data(PRICE_PATH)
    if price_df.empty:
        st.error("No price data loaded.")
        return

    instruments = list(price_df.columns)

    with st.sidebar:
        st.header("Controls")
        selected_instruments = st.multiselect(
            "Choose instruments to overlay",
            instruments,
            default=instruments[:4],
            help="Select multiple instruments to compare them on one Plotly chart.",
        )
        if not selected_instruments:
            selected_instruments = [instruments[0]]

        start_idx = st.slider("Start day", 0, len(price_df) - 1, 0)
        end_idx = st.slider("End day", start_idx, len(price_df) - 1, len(price_df) - 1)
        norm_mode = st.radio("Normalize series", ["raw", "log", "z-score"], index=1)
        pair_focus = st.selectbox("Instrument for hedge lookup", instruments, index=0)

    window_df = price_df.iloc[start_idx : end_idx + 1].copy()

    if norm_mode == "raw":
        display_df = window_df[selected_instruments].copy()
    elif norm_mode == "log":
        display_df = np.log(window_df[selected_instruments]).copy()
    else:
        display_df = window_df[selected_instruments].copy()
        for col in selected_instruments:
            rolling_mean = display_df[col].rolling(20, min_periods=20).mean()
            rolling_std = display_df[col].rolling(20, min_periods=20).std().replace(0, np.nan)
            display_df[col] = (display_df[col] - rolling_mean) / rolling_std

    fig = px.line(
        display_df.reset_index(drop=True),
        x=display_df.index,
        y=selected_instruments,
        title=f"{', '.join(selected_instruments)} over time",
        labels={"value": "Series value", "index": "Day"},
    )
    fig.update_layout(template="plotly_white", legend_title_text="Instrument")
    st.plotly_chart(fig, use_container_width=True)

    stats_df = compute_instrument_stats(price_df)
    st.subheader("Instrument statistics")
    st.dataframe(stats_df, use_container_width=True)

    pair_df = compute_pair_stats(price_df)
    st.subheader("Pair tradability / hedge availability")

    pair_table = pair_df[
        [
            "left",
            "right",
            "corr",
            "rolling_corr",
            "hedge_ratio",
            "hedge_r2",
            "cointegration_pvalue",
            "current_spread_z",
            "hedge_score",
        ]
    ].copy()
    pair_table = pair_table.round(
        {
            "corr": 3,
            "rolling_corr": 3,
            "hedge_ratio": 3,
            "hedge_r2": 3,
            "cointegration_pvalue": 4,
            "current_spread_z": 3,
            "hedge_score": 3,
        }
    )
    st.dataframe(pair_table.head(80), use_container_width=True)

    candidate_pairs = pair_df[(pair_df["left"] == pair_focus) | (pair_df["right"] == pair_focus)].copy()
    candidate_pairs = candidate_pairs.sort_values("hedge_score", ascending=False)
    if not candidate_pairs.empty:
        st.subheader(f"Best hedge candidates for {pair_focus}")
        st.dataframe(
            candidate_pairs[["left", "right", "corr", "cointegration_pvalue", "current_spread_z", "hedge_score"]].round(3),
            use_container_width=True,
        )

    with st.expander("Spread view for a selected pair"):
        if len(selected_instruments) >= 2:
            left = selected_instruments[0]
            right = selected_instruments[1]
        else:
            left = pair_focus
            right = pair_df.loc[0, "right"] if not pair_df.empty else instruments[1]

        x = np.log(price_df[left].astype(float).to_numpy())
        y = np.log(price_df[right].astype(float).to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        spread = y - (slope * x + intercept)
        spread_series = pd.Series(spread, index=price_df.index)
        spread_z = (spread_series - spread_series.rolling(20, min_periods=20).mean()) / spread_series.rolling(20, min_periods=20).std().replace(0, np.nan)

        spread_fig = go.Figure()
        spread_fig.add_trace(go.Scatter(x=spread_series.index, y=spread_series.values, mode="lines", name="Spread"))
        spread_fig.add_trace(go.Scatter(x=spread_z.index, y=spread_z.values, mode="lines", yaxis="y2", name="Spread z-score"))
        spread_fig.update_layout(
            template="plotly_white",
            title=f"Spread between {left} and {right}",
            yaxis_title="Spread",
            yaxis2=dict(title="Z-score", overlaying="y", side="right"),
        )
        st.plotly_chart(spread_fig, use_container_width=True)


if __name__ == "__main__":
    main()
