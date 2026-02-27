from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_EVENT_FEATURE_COLS = [
    "tot_debt_tot_equity",
    "ret_equity",
    "profit_margin",
    "book_val_per_share",
    "diluted_net_eps",
]


def build_global_trading_calendar(prices_df: pd.DataFrame, prefer_ticker: str = "SPY") -> pd.DatetimeIndex:
    px = prices_df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"]).copy()

    preferred = px[px["ticker"] == prefer_ticker]["date"].drop_duplicates().sort_values()
    if len(preferred) > 0:
        return pd.DatetimeIndex(preferred)

    return pd.DatetimeIndex(px["date"].drop_duplicates().sort_values())


def extract_fundamental_events(
    fundamentals_df: pd.DataFrame,
    ticker_col: str = "ticker_price",
    changed_only: bool = True,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    features = feature_cols or [col for col in DEFAULT_EVENT_FEATURE_COLS if col in fundamentals_df.columns]

    base_cols = [ticker_col, "feature_available_date", *features]
    events = fundamentals_df[base_cols].copy()
    events = events.rename(columns={ticker_col: "ticker"})
    events["feature_available_date"] = pd.to_datetime(events["feature_available_date"], errors="coerce")
    events = events.dropna(subset=["ticker", "feature_available_date"]).copy()
    events = events.sort_values(["ticker", "feature_available_date"]).reset_index(drop=True)
    events = events.drop_duplicates(subset=["ticker", "feature_available_date"], keep="last")

    if not changed_only or not features:
        return events[["ticker", "feature_available_date"]].reset_index(drop=True)

    e2 = events.copy()
    changed = pd.Series(False, index=e2.index)
    for col in features:
        prev = e2.groupby("ticker")[col].shift(1)
        changed = changed | (~e2[col].fillna(np.nan).eq(prev.fillna(np.nan)))

    first_mask = e2.groupby("ticker").cumcount().eq(0)
    keep = changed | first_mask
    filtered = e2[keep].copy()

    full_tickers = set(e2["ticker"].unique().tolist())
    kept_tickers = set(filtered["ticker"].unique().tolist())
    if full_tickers - kept_tickers:
        filtered = e2.copy()

    return filtered[["ticker", "feature_available_date"]].reset_index(drop=True)


def _build_event_time_panel_from_series(
    metric_series_by_ticker: dict[str, pd.Series],
    events_df: pd.DataFrame,
    global_dates: pd.DatetimeIndex,
    value_col: str,
    window: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    events = events_df.copy()
    events["feature_available_date"] = pd.to_datetime(events["feature_available_date"], errors="coerce")
    events = events.dropna(subset=["ticker", "feature_available_date"]).copy()

    for event in events.itertuples(index=False):
        ticker = event.ticker
        fa_date = event.feature_available_date
        if ticker not in metric_series_by_ticker:
            continue

        anchor_pos = global_dates.searchsorted(fa_date, side="left")
        if anchor_pos >= len(global_dates):
            continue

        anchor_date = global_dates[anchor_pos]
        series = metric_series_by_ticker[ticker]

        for event_day in range(-window, window + 1):
            idx = anchor_pos + event_day
            if idx < 0 or idx >= len(global_dates):
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "feature_available_date": fa_date,
                    "anchor_date": anchor_date,
                    "event_day": event_day,
                    "market_date": global_dates[idx],
                    value_col: series.iloc[idx],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["ticker", "feature_available_date", "anchor_date", "event_day", "market_date", value_col])

    return out.sort_values(["ticker", "feature_available_date", "event_day"]).reset_index(drop=True)


def build_event_time_abs_return_panel(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    global_dates: pd.DatetimeIndex,
    window: int = 60,
) -> pd.DataFrame:
    px = prices_df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["ticker", "date"]).copy()
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)

    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px.loc[px["adj_close"] <= 0, "adj_close"] = np.nan
    px["log_ret"] = px.groupby("ticker")["adj_close"].transform(lambda s: np.log(s).diff())
    px["abs_log_ret"] = px["log_ret"].abs()

    metric_series = {}
    for ticker, group in px.groupby("ticker", sort=False):
        series = group.set_index("date")["abs_log_ret"].sort_index()
        metric_series[ticker] = series.reindex(global_dates)

    return _build_event_time_panel_from_series(metric_series, events_df, global_dates, value_col="abs_log_ret", window=window)


def build_beta_hedged_return_panel(
    prices_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    px = prices_df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["ticker", "date"]).copy()
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)

    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px.loc[px["adj_close"] <= 0, "adj_close"] = np.nan
    px["r_i"] = px.groupby("ticker")["adj_close"].transform(lambda s: np.log(s).diff())

    market = (
        px[px["ticker"] == market_ticker][["date", "r_i"]]
        .rename(columns={"r_i": "r_m"})
        .drop_duplicates(subset=["date"]) 
        .sort_values("date")
    )

    beta = beta_df.copy()
    beta["date"] = pd.to_datetime(beta["date"], errors="coerce")
    keep = [col for col in ["ticker", "date", "beta_252d", "beta_obs_count"] if col in beta.columns]
    beta = beta[keep].drop_duplicates(subset=["ticker", "date"], keep="last")

    out = px[["ticker", "date", "r_i"]].merge(market, on="date", how="left")
    out = out.merge(beta, on=["ticker", "date"], how="left")
    out["idio_ret"] = out["r_i"] - out["beta_252d"] * out["r_m"]
    out["abs_idio_ret"] = out["idio_ret"].abs()

    return out[["ticker", "date", "r_i", "r_m", "beta_252d", "beta_obs_count", "idio_ret", "abs_idio_ret"]]


def build_event_time_metric_panel(
    metric_df: pd.DataFrame,
    events_df: pd.DataFrame,
    global_dates: pd.DatetimeIndex,
    value_col: str,
    window: int = 60,
) -> pd.DataFrame:
    if value_col not in metric_df.columns:
        raise KeyError(f"Missing value column in metric_df: {value_col}")

    m = metric_df.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["ticker", "date"]).copy()

    metric_series = {}
    for ticker, group in m.groupby("ticker", sort=False):
        series = group.set_index("date")[value_col].sort_index()
        metric_series[ticker] = series.reindex(global_dates)

    return _build_event_time_panel_from_series(metric_series, events_df, global_dates, value_col=value_col, window=window)


def aggregate_event_time_intensity(
    event_panel_df: pd.DataFrame,
    ticker_order: list[str],
    window: int = 60,
    agg: str = "median",
    value_col: str = "abs_log_ret",
) -> tuple[pd.DataFrame, pd.Series]:
    if value_col not in event_panel_df.columns:
        raise KeyError(f"Missing value column in event_panel_df: {value_col}")

    cols = list(range(-window, window + 1))
    if event_panel_df.empty:
        empty_heat = pd.DataFrame(index=ticker_order, columns=cols, dtype=float)
        empty_line = pd.Series(index=cols, dtype=float)
        return empty_heat, empty_line

    if agg == "mean":
        grouped = event_panel_df.groupby(["ticker", "event_day"], dropna=True)[value_col].mean()
        baseline = event_panel_df.groupby("event_day", dropna=True)[value_col].mean()
    else:
        grouped = event_panel_df.groupby(["ticker", "event_day"], dropna=True)[value_col].median()
        baseline = event_panel_df.groupby("event_day", dropna=True)[value_col].median()

    heat = grouped.unstack("event_day")
    heat = heat.reindex(index=ticker_order, columns=cols)
    baseline = baseline.reindex(cols)
    return heat, baseline
