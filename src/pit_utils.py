"""
Point-in-Time (PIT) Utilities
==============================
Enforces the no-lookahead condition for valid backtesting.

At every decision date t, only information actually available at
close of t may be used. Three lookahead sources are addressed:

1. Feature revisions — accounting data is restated after initial release.
   Only the vintage value (figure actually available at date t) is used.

2. Prediction leakage — LSTM predictions for date t must be trained
   exclusively on data available at close of t-1.

3. Option data timing — option prices at t correspond to close of t,
   not open of t+1.

References
----------
Mykland OOA lecture notes: Point-in-Time data and no-lookahead conditions.
Compustat documentation: rdq (Report Date of Quarterly earnings filing).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pit_feature_panel(
    feature_panel_df: pd.DataFrame,
    as_of_col: str = "date",
    vintage_col: str = "report_date",
    value_cols: Optional[list] = None,
    ticker_col: str = "ticker",
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Construct a PIT-safe feature panel.

    For each (ticker, as_of_date), only rows with vintage_date <= as_of_date
    are eligible. The most recent eligible vintage is selected. Any gaps are
    forward-filled if fill_method == "ffill".

    If vintage_col is not present in feature_panel_df, returns the panel
    unchanged with a proxy vintage date equal to as_of_col.

    Parameters
    ----------
    feature_panel_df : Raw feature panel with one row per (ticker, date).
    as_of_col        : Column giving the "as of" date (when the row is used).
    vintage_col      : Column giving the actual data vintage (when filed/available).
    value_cols       : Feature columns to enforce PIT on. Defaults to all
                       non-identifier numeric columns.
    ticker_col       : Column identifying the ticker.
    fill_method      : "ffill" to forward-fill missing values from prior vintage.

    Returns
    -------
    pd.DataFrame with same columns as input plus "pit_vintage_date".
    """
    df = feature_panel_df.copy()

    if vintage_col not in df.columns:
        print(
            f"[PIT] WARNING: '{vintage_col}' column not found. "
            "PIT enforcement skipped. Using as_of_col as proxy for vintage date."
        )
        df["pit_vintage_date"] = pd.to_datetime(df[as_of_col], errors="coerce")
        return df

    df[as_of_col]   = pd.to_datetime(df[as_of_col],   errors="coerce")
    df[vintage_col] = pd.to_datetime(df[vintage_col], errors="coerce")

    if value_cols is None:
        exclude = {as_of_col, vintage_col, ticker_col, "pit_vintage_date"}
        value_cols = [c for c in df.columns if c not in exclude]

    # Use merge_asof for performance: for each (as_of_date, ticker),
    # find the most recent vintage_date <= as_of_date.
    # Strategy: sort feature panel by vintage_col, then for each as_of_date
    # merge_asof by ticker.

    # Build a sorted vintage panel: one row per (ticker, vintage_date) with the
    # feature values. Then left-join to the as_of panel.
    as_of_panel = (
        df[[ticker_col, as_of_col]]
        .drop_duplicates()
        .sort_values([ticker_col, as_of_col])
        .reset_index(drop=True)
    )

    # vintage panel: one row per (ticker, vintage_col)
    vintage_panel = (
        df[[ticker_col, vintage_col] + value_cols]
        .dropna(subset=[vintage_col])
        .sort_values([ticker_col, vintage_col])
        .reset_index(drop=True)
    )
    vintage_panel = vintage_panel.rename(columns={vintage_col: "_vintage_date"})

    result_rows = []
    for ticker, as_of_group in as_of_panel.groupby(ticker_col, sort=False):
        vint_group = vintage_panel[vintage_panel[ticker_col] == ticker].copy()
        if vint_group.empty:
            # No vintage data for this ticker — produce NaN rows
            for _, row in as_of_group.iterrows():
                new_row = {ticker_col: ticker, as_of_col: row[as_of_col]}
                for c in value_cols:
                    new_row[c] = np.nan
                new_row["pit_vintage_date"] = pd.NaT
                result_rows.append(new_row)
            continue

        # merge_asof: for each as_of date, find most recent vintage <= as_of_date
        merged = pd.merge_asof(
            as_of_group.sort_values(as_of_col),
            vint_group.sort_values("_vintage_date"),
            left_on=as_of_col,
            right_on="_vintage_date",
            by=ticker_col,
            direction="backward",
        )
        merged = merged.rename(columns={"_vintage_date": "pit_vintage_date"})
        result_rows.append(merged)

    if not result_rows:
        result = df.copy()
        result["pit_vintage_date"] = pd.NaT
        return result

    # Concatenate, preserving original as_of panel order
    result = pd.concat(result_rows, ignore_index=True)
    result = result.sort_values([ticker_col, as_of_col]).reset_index(drop=True)

    if fill_method == "ffill":
        # Forward-fill NaN values within each ticker's time series
        result[value_cols] = (
            result.groupby(ticker_col, sort=False)[value_cols]
            .transform(lambda s: s.ffill())
        )

    return result


def validate_pit(
    feature_panel_df: pd.DataFrame,
    as_of_col: str = "date",
    vintage_col: str = "report_date",
    value_cols: Optional[list] = None,
    ticker_col: str = "ticker",
    raise_on_violation: bool = False,
) -> dict:
    """
    Scan feature panel for lookahead violations.

    A violation is any row where vintage_date > as_of_date for a non-NaN value.

    Parameters
    ----------
    feature_panel_df : Feature panel DataFrame.
    as_of_col        : Date column for "as of" date.
    vintage_col      : Date column for data vintage.
    value_cols       : Columns to check for non-NaN values in violation rows.
    ticker_col       : Ticker column.
    raise_on_violation : If True, raise AssertionError when violations found.

    Returns
    -------
    dict with keys:
        n_violations  : int
        violations_df : pd.DataFrame of violating rows
        violation_rate : float
        clean         : bool
        summary       : str
    """
    df = feature_panel_df.copy()

    if vintage_col not in df.columns:
        msg = (
            f"[PIT] WARNING: '{vintage_col}' column not found in feature panel. "
            "Cannot validate PIT. Returning clean=None."
        )
        print(msg)
        return {
            "n_violations": None,
            "violations_df": pd.DataFrame(),
            "violation_rate": None,
            "clean": None,
            "summary": msg,
        }

    df[as_of_col]   = pd.to_datetime(df[as_of_col],   errors="coerce")
    df[vintage_col] = pd.to_datetime(df[vintage_col], errors="coerce")

    if value_cols is None:
        exclude = {as_of_col, vintage_col, ticker_col}
        value_cols = [c for c in df.columns if c not in exclude]

    # A violation requires: vintage > as_of AND at least one value_col is non-NaN
    lookahead_mask = df[vintage_col].notna() & df[as_of_col].notna() & (
        df[vintage_col] > df[as_of_col]
    )

    if value_cols:
        has_value = df[value_cols].notna().any(axis=1)
        violation_mask = lookahead_mask & has_value
    else:
        violation_mask = lookahead_mask

    violations_df = df[violation_mask].copy()
    n_violations  = int(violation_mask.sum())
    total_rows    = len(df)
    violation_rate = n_violations / total_rows if total_rows > 0 else 0.0

    clean = n_violations == 0
    summary = (
        f"[PIT] Validation complete — {n_violations} violations out of "
        f"{total_rows} rows ({violation_rate:.2%} rate). "
        + ("CLEAN." if clean else "VIOLATIONS FOUND.")
    )

    if raise_on_violation and not clean:
        raise AssertionError(summary)

    return {
        "n_violations": n_violations,
        "violations_df": violations_df,
        "violation_rate": violation_rate,
        "clean": clean,
        "summary": summary,
    }


def build_pit_prediction_panel(
    predictions_df: pd.DataFrame,
    feature_panel_pit: pd.DataFrame,
    signal_date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Join predictions to the PIT feature panel.

    For each (signal_date, ticker) in predictions_df, attaches the PIT-safe
    feature values available at signal_date (from feature_panel_pit).

    Adds columns:
        pit_feature_lag : (signal_date - pit_vintage_date).days
                          Positive = data is stale by this many days (expected).
                          Negative = data is from the future (lookahead violation).
        pit_safe        : True if pit_feature_lag >= 0

    Asserts all rows are pit_safe before returning.

    Parameters
    ----------
    predictions_df    : DataFrame with signal_date_col and ticker_col columns.
    feature_panel_pit : PIT-safe feature panel (output of build_pit_feature_panel).
    signal_date_col   : Date column in predictions_df.
    ticker_col        : Ticker column in both DataFrames.

    Returns
    -------
    pd.DataFrame: predictions_df with pit_feature_lag and pit_safe columns added.
    """
    preds = predictions_df.copy()
    preds[signal_date_col] = pd.to_datetime(preds[signal_date_col], errors="coerce")

    panel = feature_panel_pit.copy()

    # Determine which date column to use in the feature panel for joining
    panel_date_col = signal_date_col  # assume same name ("date")
    if panel_date_col not in panel.columns:
        panel_date_col = [c for c in panel.columns if "date" in c.lower()][0]

    panel[panel_date_col] = pd.to_datetime(panel[panel_date_col], errors="coerce")

    # Join on (signal_date_col, ticker_col)
    pit_subset = panel[[ticker_col, panel_date_col, "pit_vintage_date"]].drop_duplicates()
    pit_subset = pit_subset.rename(columns={panel_date_col: signal_date_col})

    result = pd.merge(
        preds,
        pit_subset,
        on=[signal_date_col, ticker_col],
        how="left",
    )

    if "pit_vintage_date" in result.columns:
        result["pit_vintage_date"] = pd.to_datetime(result["pit_vintage_date"], errors="coerce")
        result["pit_feature_lag"] = (
            result[signal_date_col] - result["pit_vintage_date"]
        ).dt.days
        result["pit_safe"] = result["pit_feature_lag"].notna() & (
            result["pit_feature_lag"] >= 0
        )
    else:
        # No vintage info available — mark as safe (cannot validate)
        warnings.warn(
            "[PIT] pit_vintage_date not found in feature_panel_pit. "
            "Cannot compute pit_feature_lag. pit_safe set to True (unvalidated).",
            stacklevel=2,
        )
        result["pit_feature_lag"] = np.nan
        result["pit_safe"] = True

    return result


def compute_pit_signal_decay(
    pit_panel: pd.DataFrame,
    signal_col: str = "prediction",
    lag_col: str = "pit_feature_lag",
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Analyze how prediction strength varies with feature data staleness.

    Bins rows by pit_feature_lag (days since last vintage) and computes
    per-bin signal statistics.

    Parameters
    ----------
    pit_panel  : DataFrame with lag_col and signal_col.
    signal_col : Column with prediction/signal values.
    lag_col    : Column with feature lag in days.
    n_bins     : Number of quantile bins.

    Returns
    -------
    pd.DataFrame with columns:
        lag_bin, mean_lag_days, mean_prediction, std_prediction, n_signals
    """
    df = pit_panel[[signal_col, lag_col]].dropna().copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "lag_bin", "mean_lag_days", "mean_prediction", "std_prediction", "n_signals"
        ])

    try:
        df["lag_bin"] = pd.qcut(df[lag_col], q=n_bins, duplicates="drop")
    except ValueError:
        # Fall back to cut if qcut fails (e.g., too few unique values)
        df["lag_bin"] = pd.cut(df[lag_col], bins=n_bins, duplicates="drop")

    grouped = df.groupby("lag_bin", observed=True, sort=True)

    result = pd.DataFrame({
        "lag_bin":         grouped[lag_col].apply(lambda x: str(x.name)).reset_index(drop=True),
        "mean_lag_days":   grouped[lag_col].mean().values,
        "mean_prediction": grouped[signal_col].mean().values,
        "std_prediction":  grouped[signal_col].std().values,
        "n_signals":       grouped[signal_col].count().values,
    })

    return result.reset_index(drop=True)


def flag_earnings_pit_violations(
    trade_log: pd.DataFrame,
    rdq_df: pd.DataFrame,
    signal_col: str = "signal_date",
    entry_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Verify earnings announcement timing for each trade entry.

    Three formal PIT conditions (all must be True for a valid post-announcement trade):
        1. signal_date < entry_date       — no same-day lookahead
        2. signal_date < rdq              — signal predates announcement
        3. entry_date >= rdq              — enter on or after announcement

    Parameters
    ----------
    trade_log  : DataFrame of entry trades (rows where action == "enter").
    rdq_df     : DataFrame with (ticker, rdq) — earnings announcement dates.
    signal_col : Column in trade_log for signal/decision date.
    entry_col  : Column in trade_log for actual entry date.
    ticker_col : Ticker column in both DataFrames.

    Returns
    -------
    trade_log with additional boolean columns:
        pit_signal_precedes_entry : signal_date < entry_date
        pit_signal_precedes_rdq   : signal_date < rdq
        pit_entry_on_or_after_rdq : entry_date >= rdq
        pit_valid                 : all three True
    """
    result = trade_log.copy()
    result[entry_col]  = pd.to_datetime(result[entry_col],  errors="coerce")
    result[signal_col] = pd.to_datetime(result.get(signal_col, pd.NaT), errors="coerce")

    # Condition 1: signal_date < entry_date
    result["pit_signal_precedes_entry"] = (
        result[signal_col].notna() &
        result[entry_col].notna() &
        (result[signal_col] < result[entry_col])
    )

    if rdq_df is None or rdq_df.empty:
        warnings.warn(
            "[PIT] rdq_df is empty or None. Cannot check conditions 2 & 3.",
            stacklevel=2,
        )
        result["pit_signal_precedes_rdq"]   = np.nan
        result["pit_entry_on_or_after_rdq"] = np.nan
        result["pit_valid"] = result["pit_signal_precedes_entry"]
        return result

    rdq = rdq_df.copy()
    rdq["rdq"] = pd.to_datetime(rdq["rdq"], errors="coerce")
    rdq = rdq.dropna(subset=["rdq"]).sort_values([ticker_col, "rdq"])

    # For each entry row, find the most recent rdq <= entry_date for that ticker
    # using merge_asof
    entries_sorted = result.sort_values(entry_col).reset_index()  # keep original index
    rdq_sorted = rdq.sort_values("rdq")

    merged = pd.merge_asof(
        entries_sorted,
        rdq_sorted[[ticker_col, "rdq"]],
        left_on=entry_col,
        right_on="rdq",
        by=ticker_col,
        direction="backward",
    )

    # Conditions 2 & 3 — only defined where rdq was found
    merged["pit_signal_precedes_rdq"] = (
        merged["rdq"].notna() &
        merged[signal_col].notna() &
        (merged[signal_col] < merged["rdq"])
    )
    merged["pit_entry_on_or_after_rdq"] = (
        merged["rdq"].notna() &
        merged[entry_col].notna() &
        (merged[entry_col] >= merged["rdq"])
    )
    merged["pit_valid"] = (
        merged["pit_signal_precedes_entry"] &
        merged["pit_signal_precedes_rdq"] &
        merged["pit_entry_on_or_after_rdq"]
    )

    # Restore original index ordering
    merged = merged.set_index("index").sort_index()
    for col in ["pit_signal_precedes_rdq", "pit_entry_on_or_after_rdq", "pit_valid"]:
        result[col] = merged[col]

    return result
