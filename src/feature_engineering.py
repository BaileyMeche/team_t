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


def _safe_group_pct_change(group: pd.Series) -> pd.Series:
    prev = group.shift(1)
    out = np.where(prev.notna() & prev.ne(0), group / prev - 1.0, np.nan)
    return pd.Series(out, index=group.index, dtype=float)


def add_fundamental_change_features(
    panel_df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add change-based fundamental features to a PIT panel."""
    out = panel_df.copy()
    out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)

    level_to_change = {
        "tot_debt_tot_equity": "leverage_change",
        "ret_equity": "roe_change",
        "profit_margin": "margin_change",
    }
    for level_col, change_col in level_to_change.items():
        if level_col in out.columns:
            out[level_col] = pd.to_numeric(out[level_col], errors="coerce")
            out[change_col] = out.groupby(ticker_col)[level_col].diff()

    growth_map = {
        "diluted_net_eps": "eps_growth",
        "book_val_per_share": "book_value_growth",
    }
    for level_col, growth_col in growth_map.items():
        if level_col in out.columns:
            out[level_col] = pd.to_numeric(out[level_col], errors="coerce")
            out[growth_col] = out.groupby(ticker_col)[level_col].transform(_safe_group_pct_change)

    return out


def add_price_liquidity_features(
    panel_df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
    price_col: str = "adj_close",
    volume_col: str = "volume",
    volume_window: int = 20,
    min_volume_obs: int = 5,
) -> pd.DataFrame:
    """Add log return and relative volume features."""
    out = panel_df.copy()
    out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)

    if price_col in out.columns:
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
        out.loc[out[price_col] <= 0, price_col] = np.nan
        out["log_return"] = out.groupby(ticker_col)[price_col].transform(lambda s: np.log(s).diff())

    if volume_col in out.columns:
        out[volume_col] = pd.to_numeric(out[volume_col], errors="coerce")
        rolling_mean = out.groupby(ticker_col)[volume_col].transform(
            lambda s: s.rolling(window=volume_window, min_periods=min_volume_obs).mean()
        )
        out["volume_ratio"] = np.where(rolling_mean.notna() & rolling_mean.ne(0), out[volume_col] / rolling_mean, np.nan)

    return out


STAGE_1_BASELINE_FEATURE_COLS = [
    "roe_change",
    "margin_change",
    "leverage_change",
    "eps_growth",
    "book_value_growth",
    "log_return",
    "rolling_beta",
    "volume_ratio",
]
STAGE_2_DERIVED_FEATURE_COLS = [
    "roe_change_accel",
    "margin_change_accel",
]
STAGE_3_DERIVED_FEATURE_COLS = [
    "eps_growth_surprise",
]
STAGE_4_DERIVED_FEATURE_COLS = [
    "reaction_speed",
]
STAGE_5_DERIVED_FEATURE_COLS = [
    "vol_regime_ratio_20_60",
]
_STAGE_FEATURE_MAP: dict[int, list[str]] = {
    1: STAGE_1_BASELINE_FEATURE_COLS,
    2: STAGE_2_DERIVED_FEATURE_COLS,
    3: STAGE_3_DERIVED_FEATURE_COLS,
    4: STAGE_4_DERIVED_FEATURE_COLS,
    5: STAGE_5_DERIVED_FEATURE_COLS,
}


def get_stage_feature_columns(max_stage: int = 5) -> list[str]:
    """Return cumulative feature columns through max_stage."""
    if max_stage < 1 or max_stage > 5:
        raise ValueError(f"max_stage must be between 1 and 5 inclusive; got {max_stage}")

    cols: list[str] = []
    for stage in range(1, max_stage + 1):
        cols.extend(_STAGE_FEATURE_MAP[stage])
    return cols


def _build_report_level_feature_table(
    panel_df: pd.DataFrame,
    ticker_col: str,
    feature_available_col: str,
) -> pd.DataFrame:
    base_cols = [ticker_col, feature_available_col, "roe_change", "margin_change", "eps_growth"]
    missing = [col for col in base_cols if col not in panel_df.columns]
    if missing:
        raise KeyError(f"Missing columns required for report-level staged features: {missing}")

    report = (
        panel_df[base_cols]
        .copy()
        .dropna(subset=[ticker_col, feature_available_col])
        .sort_values([ticker_col, feature_available_col])
        # Keep first daily row in each availability regime so event-level
        # change/surprise features update only when a new report becomes tradable.
        .drop_duplicates([ticker_col, feature_available_col], keep="first")
        .reset_index(drop=True)
    )

    for col in ["roe_change", "margin_change", "eps_growth"]:
        report[col] = pd.to_numeric(report[col], errors="coerce")
    report[feature_available_col] = pd.to_datetime(report[feature_available_col], errors="coerce")
    return report


def _merge_report_features_asof(
    panel_df: pd.DataFrame,
    report_df: pd.DataFrame,
    feature_cols: list[str],
    ticker_col: str,
    date_col: str,
    feature_available_col: str,
) -> pd.DataFrame:
    helper_event_col = "__report_event_date"
    report_local = report_df.rename(columns={feature_available_col: helper_event_col}).copy()
    merged_parts: list[pd.DataFrame] = []

    for ticker, panel_ticker in panel_df.groupby(ticker_col, sort=False):
        panel_ticker = panel_ticker.sort_values(date_col).copy()
        report_ticker = report_local[report_local[ticker_col] == ticker].sort_values(helper_event_col).copy()

        if report_ticker.empty:
            for col in feature_cols:
                panel_ticker[col] = np.nan
            merged_parts.append(panel_ticker)
            continue

        merged = pd.merge_asof(
            panel_ticker,
            report_ticker[[helper_event_col, *feature_cols]],
            left_on=date_col,
            right_on=helper_event_col,
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=[helper_event_col])
        merged_parts.append(merged)

    out = pd.concat(merged_parts, ignore_index=True)
    out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)
    return out


def add_staged_features(
    panel_df: pd.DataFrame,
    max_stage: int = 5,
    ticker_col: str = "ticker",
    date_col: str = "date",
    feature_available_col: str = "feature_available_date",
    price_col: str = "adj_close",
    reaction_clip: tuple[float, float] = (-0.05, 0.05),
    surprise_window: int = 8,
    surprise_min_obs: int = 4,
    vol_short_window: int = 20,
    vol_short_min_obs: int = 20,
    vol_long_window: int = 60,
    vol_long_min_obs: int = 40,
    vol_epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Add cumulative staged features (Stages 2-5) to a PIT daily panel.

    Stage-1 baseline features are expected to be computed separately and are not
    modified here.
    """
    if max_stage < 1 or max_stage > 5:
        raise ValueError(f"max_stage must be between 1 and 5 inclusive; got {max_stage}")
    if ticker_col not in panel_df.columns or date_col not in panel_df.columns:
        raise KeyError(f"panel_df must contain '{ticker_col}' and '{date_col}' columns")

    out = panel_df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values([ticker_col, date_col]).reset_index(drop=True)

    if feature_available_col in out.columns:
        out[feature_available_col] = pd.to_datetime(out[feature_available_col], errors="coerce")

    if max_stage >= 2:
        if feature_available_col not in out.columns:
            raise KeyError(f"Missing required column for staged features: {feature_available_col}")

        report = _build_report_level_feature_table(
            panel_df=out,
            ticker_col=ticker_col,
            feature_available_col=feature_available_col,
        )

        report["roe_change_accel"] = report.groupby(ticker_col)["roe_change"].diff()
        report["margin_change_accel"] = report.groupby(ticker_col)["margin_change"].diff()

        if max_stage >= 3:
            grp = report.groupby(ticker_col)["eps_growth"]
            eps_mean = grp.transform(
                lambda s: s.shift(1).rolling(window=surprise_window, min_periods=surprise_min_obs).mean()
            )
            eps_std = grp.transform(
                lambda s: s.shift(1).rolling(window=surprise_window, min_periods=surprise_min_obs).std(ddof=0)
            )
            report["eps_growth_surprise"] = np.where(
                eps_std.notna() & eps_std.ne(0),
                (report["eps_growth"] - eps_mean) / eps_std,
                np.nan,
            )

        event_cols: list[str] = []
        if max_stage >= 2:
            event_cols.extend(STAGE_2_DERIVED_FEATURE_COLS)
        if max_stage >= 3:
            event_cols.extend(STAGE_3_DERIVED_FEATURE_COLS)

        out = _merge_report_features_asof(
            panel_df=out,
            report_df=report[[ticker_col, feature_available_col, *event_cols]].copy(),
            feature_cols=event_cols,
            ticker_col=ticker_col,
            date_col=date_col,
            feature_available_col=feature_available_col,
        )

    if max_stage >= 4:
        if feature_available_col not in out.columns:
            raise KeyError(f"Missing required column for reaction_speed: {feature_available_col}")
        if price_col not in out.columns:
            raise KeyError(f"Missing required column for reaction_speed: {price_col}")

        out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
        out.loc[out[price_col] <= 0, price_col] = np.nan

        fa_key = out[feature_available_col].fillna(pd.Timestamp("1900-01-01"))
        out["regime_id"] = (
            out.assign(_fa_key=fa_key)
            .groupby(ticker_col)["_fa_key"]
            .transform(lambda s: s.ne(s.shift()).cumsum())
            .astype(int)
        )
        out["anchor_price"] = out.groupby([ticker_col, "regime_id"])[price_col].transform("first")
        out["drift_since_report"] = np.where(
            out[price_col].notna() & out["anchor_price"].notna() & out["anchor_price"].ne(0),
            np.log(out[price_col]) - np.log(out["anchor_price"]),
            np.nan,
        )
        out["days_since_report"] = out.groupby([ticker_col, "regime_id"]).cumcount()
        days_den = np.maximum(out["days_since_report"].to_numpy(dtype=float), 1.0)
        out["reaction_speed"] = np.clip(
            np.asarray(out["drift_since_report"], dtype=float) / days_den,
            reaction_clip[0],
            reaction_clip[1],
        )

    if max_stage >= 5:
        if price_col not in out.columns:
            raise KeyError(f"Missing required column for vol_regime_ratio_20_60: {price_col}")

        out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
        out.loc[out[price_col] <= 0, price_col] = np.nan
        ret = out.groupby(ticker_col)[price_col].transform(lambda s: np.log(s).diff())

        annualize = np.sqrt(252.0)
        out["realized_vol_20d"] = ret.groupby(out[ticker_col]).transform(
            lambda s: s.rolling(window=vol_short_window, min_periods=vol_short_min_obs).std(ddof=0) * annualize
        )
        out["realized_vol_60d"] = ret.groupby(out[ticker_col]).transform(
            lambda s: s.rolling(window=vol_long_window, min_periods=vol_long_min_obs).std(ddof=0) * annualize
        )
        out["vol_regime_ratio_20_60"] = out["realized_vol_20d"] / (out["realized_vol_60d"] + vol_epsilon)

    return out


def winsorize_cross_sectional(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Winsorize features cross-sectionally by date."""
    out = panel_df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        lo = vals.groupby(out[date_col]).transform(lambda s: s.quantile(lower_q))
        hi = vals.groupby(out[date_col]).transform(lambda s: s.quantile(upper_q))
        out[col] = vals.clip(lower=lo, upper=hi)
    return out


def zscore_cross_sectional(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    suffix: str = "_z",
) -> pd.DataFrame:
    """Create cross-sectional z-score normalized feature columns by date."""
    out = panel_df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        grp = vals.groupby(out[date_col])
        mean = grp.transform("mean")
        std = grp.transform(lambda s: s.std(ddof=0))
        out[f"{col}{suffix}"] = np.where(std.notna() & std.ne(0), (vals - mean) / std, np.nan)
    return out


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
        if len(group) < lookback:
            continue

        feat_arr = group[feature_cols].to_numpy(dtype=float)
        target_arr = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float)

        for idx in range(lookback - 1, len(group)):
            # X through t predicts return from t to t+1 (stored in target_arr[idx]).
            x_parts.append(feat_arr[idx - lookback + 1 : idx + 1, :])
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
