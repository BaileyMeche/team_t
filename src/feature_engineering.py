from __future__ import annotations

import numpy as np
import pandas as pd

from .event_panels import (
    aggregate_event_time_intensity,
    build_beta_hedged_return_panel,
    build_event_time_abs_return_panel,
    build_event_time_metric_panel,
    build_global_trading_calendar,
    extract_fundamental_events,
)


def add_split_adjusted_intraday_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    out = prices_df.copy()
    for col in ["open", "close", "adj_close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["adj_factor"] = np.where(out["close"].ne(0), out["adj_close"] / out["close"], np.nan)
    out["adj_open"] = out["open"] * out["adj_factor"]
    out["adj_close_intraday"] = out["close"] * out["adj_factor"]
    return out


def compute_price_to_book(panel_df: pd.DataFrame) -> pd.DataFrame:
    out = panel_df.copy()
    out["price_to_book"] = np.where(
        out["book_val_per_share"].notna() & out["book_val_per_share"].ne(0),
        out["adj_close"] / out["book_val_per_share"],
        np.nan,
    )
    return out


def compute_rolling_beta_vs_spy(
    prices_df: pd.DataFrame,
    window: int = 252,
    min_obs: int = 126,
) -> pd.DataFrame:
    px = prices_df.copy()
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px.loc[px["adj_close"] <= 0, "adj_close"] = np.nan
    px["log_ret"] = px.groupby("ticker")["adj_close"].transform(lambda s: np.log(s).diff())

    spy = px[px["ticker"] == "SPY"][ ["date", "log_ret"] ].rename(columns={"log_ret": "spy_log_ret"})
    stock = px[px["ticker"] != "SPY"][ ["ticker", "date", "log_ret"] ].merge(spy, on="date", how="left")
    stock = stock.sort_values(["ticker", "date"]).reset_index(drop=True)

    if stock.empty:
        return pd.DataFrame(columns=["ticker", "date", "beta_252d", "beta_obs_count"])

    parts: list[pd.DataFrame] = []
    for ticker, group in stock.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        cov = group["log_ret"].rolling(window=window, min_periods=min_obs).cov(group["spy_log_ret"])
        var = group["spy_log_ret"].rolling(window=window, min_periods=min_obs).var()
        obs = group[["log_ret", "spy_log_ret"]].notna().all(axis=1).rolling(window=window, min_periods=min_obs).sum()

        group["beta_252d"] = cov / var
        group["beta_obs_count"] = obs
        parts.append(group[["ticker", "date", "beta_252d", "beta_obs_count"]])

    return pd.concat(parts, ignore_index=True)


def assign_time_split(
    panel_df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    dev_start: pd.Timestamp,
    dev_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    out = panel_df.copy()
    out["split"] = np.select(
        [
            (out["date"] >= train_start) & (out["date"] <= train_end),
            (out["date"] >= dev_start) & (out["date"] <= dev_end),
            (out["date"] >= test_start) & (out["date"] <= test_end),
        ],
        ["train", "dev", "test"],
        default="outside",
    )
    out.loc[out["split"] == "outside", "split"] = np.nan
    return out


def _build_sequences_for_split(
    split_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for _, group in split_df.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        feat = group[feature_cols].astype(float)
        target = pd.to_numeric(group[target_col], errors="coerce")

        valid = feat.notna().all(axis=1) & target.notna()
        group = group.loc[valid].copy()
        if len(group) <= lookback:
            continue

        feat_arr = group[feature_cols].to_numpy(dtype=float)
        target_arr = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float)

        for idx in range(lookback, len(group)):
            x_parts.append(feat_arr[idx - lookback : idx, :])
            y_parts.append(np.array(target_arr[idx], dtype=float))

    if not x_parts:
        return np.empty((0, lookback, len(feature_cols))), np.empty((0,))

    return np.stack(x_parts), np.asarray(y_parts, dtype=float)


def build_lstm_tensors(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    split_col: str = "split",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    required_cols = set(["ticker", "date", split_col, target_col, *feature_cols])
    missing = required_cols - set(panel_df.columns)
    if missing:
        raise KeyError(f"Missing columns for tensor build: {sorted(missing)}")

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ["train", "dev", "test"]:
        split_df = panel_df[panel_df[split_col] == split_name].copy()
        out[split_name] = _build_sequences_for_split(split_df, feature_cols, target_col, lookback)

    return out


def compute_event_intensity_diagnostics(
    mode: str,
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    window: int = 60,
    beta_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Compute plotting-ready diagnostics for event-time intensity figures.

    The notebook only passes mode, prebuilt price/fundamental panels, window,
    and optional beta table. Event alignment internals are handled here.
    """
    if mode not in {"raw", "beta_hedged"}:
        raise ValueError("mode must be one of {'raw', 'beta_hedged'}")

    global_dates = build_global_trading_calendar(prices_df, prefer_ticker="SPY")
    events_all = extract_fundamental_events(fundamentals_df, ticker_col="ticker_price", changed_only=True)

    price_tickers = list(dict.fromkeys(prices_df["ticker"].astype(str).tolist()))
    model_tickers = [ticker for ticker in price_tickers if ticker != "SPY"]

    events = events_all[events_all["ticker"].isin(model_tickers)].copy()

    if mode == "raw":
        event_panel_df = build_event_time_abs_return_panel(
            prices_df=prices_df[prices_df["ticker"].isin(model_tickers)].copy(),
            events_df=events,
            global_dates=global_dates,
            window=window,
        )
        value_col = "abs_log_ret"
        figure_title = "Price Response Around Fundamental Availability Dates"
        colorbar_label = "Median Absolute Daily Log Return"
        y_label = "|log return|"
    else:
        if beta_df is None:
            raise ValueError("beta_df is required when mode='beta_hedged'")

        # Rolling beta is estimated independently of event-time analysis and reflects
        # contemporaneous market exposure, not event-specific tuning.
        idio_daily = build_beta_hedged_return_panel(prices_df=prices_df, beta_df=beta_df, market_ticker="SPY")
        idio_daily = idio_daily[idio_daily["ticker"].isin(model_tickers)].copy()

        event_panel_df = build_event_time_metric_panel(
            metric_df=idio_daily,
            events_df=events,
            global_dates=global_dates,
            value_col="abs_idio_ret",
            window=window,
        )
        value_col = "abs_idio_ret"
        figure_title = "Idiosyncratic Price Response Around Fundamental Availability"
        colorbar_label = "Median |Beta-Hedged Log Return|"
        y_label = "|Beta-Hedged Log Return|"

    if event_panel_df.empty:
        raise ValueError("Event-time panel is empty for diagnostic computation")

    if not bool((event_panel_df["anchor_date"] >= event_panel_df["feature_available_date"]).all()):
        raise AssertionError("Found invalid event alignment: anchor_date < feature_available_date")

    heatmap_df, baseline_median = aggregate_event_time_intensity(
        event_panel_df=event_panel_df,
        ticker_order=model_tickers,
        window=window,
        agg="median",
        value_col=value_col,
    )
    _, baseline_mean = aggregate_event_time_intensity(
        event_panel_df=event_panel_df,
        ticker_order=model_tickers,
        window=window,
        agg="mean",
        value_col=value_col,
    )

    return {
        "heatmap_df": heatmap_df,
        "baseline_median": baseline_median,
        "baseline_mean": baseline_mean,
        "event_panel_df": event_panel_df,
        "figure_title": figure_title,
        "colorbar_label": colorbar_label,
        "y_label": y_label,
    }
