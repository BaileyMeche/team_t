# Stage 9 — Option Selection
# For each top-K stock on signal date t, select the best ATM call to enter on t+1.
# Selection criteria (in order): closest to ATM, highest open interest, tightest spread.
# DTE target: 30–45 days (relaxes to full available range if no match found).
# Data source: data/options/optionmetrics_calls_atm_20_60d_full_history.parquet

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def select_option_for_entry(
    options_df: pd.DataFrame,
    ticker: str,
    entry_date: pd.Timestamp,
    dte_min: int = 30,
    dte_max: int = 45,
) -> pd.Series | None:
    """Select the best ATM call option for a given ticker on entry_date.

    Selection order:
    1. Prefer DTE in [dte_min, dte_max]; relax to full range if none found.
    2. Among candidates: minimize |moneyness - 1|, maximize open_interest, minimize spread.

    Parameters
    ----------
    options_df : Full options DataFrame (pre-loaded, filtered to ATM calls).
    ticker : Stock ticker (must match options_df 'ticker' column).
    entry_date : Date to enter the position (t+1 after signal).
    dte_min, dte_max : Preferred DTE window.

    Returns
    -------
    pd.Series row from options_df, or None if no option available.
    """
    entry_ts = pd.Timestamp(entry_date)

    candidates = options_df[
        (options_df["ticker"] == ticker) & (options_df["date"] == entry_ts)
    ].copy()

    if candidates.empty:
        return None

    # Prefer target DTE window
    preferred = candidates[
        (candidates["dte"] >= dte_min) & (candidates["dte"] <= dte_max)
    ].copy()

    if preferred.empty:
        preferred = candidates.copy()
        warnings.warn(
            f"[option_selection] {ticker} {entry_ts.date()}: no option in DTE [{dte_min},{dte_max}], "
            f"using full available DTE range.",
            stacklevel=2,
        )

    preferred["_atm_gap"] = (preferred["moneyness"] - 1.0).abs()
    preferred["_spread"] = pd.to_numeric(preferred["best_offer"], errors="coerce") - pd.to_numeric(
        preferred["best_bid"], errors="coerce"
    )
    preferred["open_interest"] = pd.to_numeric(preferred["open_interest"], errors="coerce").fillna(0)

    preferred = preferred.sort_values(
        ["_atm_gap", "open_interest", "_spread"],
        ascending=[True, False, True],
        na_position="last",
    )

    return preferred.iloc[0]


def build_entry_table(
    top_k_df: pd.DataFrame,
    options_df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    dte_min: int = 30,
    dte_max: int = 45,
) -> pd.DataFrame:
    """Build entry records for all (signal_date, ticker) pairs in top_k_df.

    For each row, the entry_date is the next available option trading date
    after signal_date (i.e., t+1 per the canonical timing convention).
    Options with no data on entry_date are silently dropped.

    Parameters
    ----------
    top_k_df : Output of ranking.select_top_k — one row per (date, ticker).
    options_df : Full options DataFrame.
    dte_min, dte_max : Preferred DTE window passed to select_option_for_entry.

    Returns
    -------
    DataFrame with one row per tradeable entry signal. Columns:
    signal_date, entry_date, ticker, rank, rank_from_bottom, signal_side, prediction,
    option_mid_entry, delta_entry, gamma_entry, vega_entry,
    strike, expiry, dte_entry, underlying_entry, implied_vol_entry,
    open_interest_entry.
    """
    options_df = options_df.copy()
    options_df[date_col] = pd.to_datetime(options_df[date_col], errors="coerce")

    all_option_dates = sorted(options_df[date_col].dropna().unique())
    date_to_next: dict[pd.Timestamp, pd.Timestamp | None] = {}
    for i, d in enumerate(all_option_dates):
        date_to_next[d] = all_option_dates[i + 1] if i + 1 < len(all_option_dates) else None

    def _f(val, default=np.nan):
        """Convert val to float, returning default for pd.NA / None / non-numeric."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    records: list[dict] = []
    skipped = 0

    for _, row in top_k_df.iterrows():
        signal_date = pd.Timestamp(row[date_col])
        ticker = str(row[ticker_col])

        entry_date = date_to_next.get(signal_date)
        if entry_date is None:
            skipped += 1
            continue

        opt = select_option_for_entry(options_df, ticker, entry_date, dte_min, dte_max)
        if opt is None:
            skipped += 1
            continue

        records.append(
            {
                "signal_date": signal_date,
                "entry_date": pd.Timestamp(entry_date),
                "ticker": ticker,
                "rank": int(row.get("rank", 0)),
                "rank_from_bottom": int(row.get("rank_from_bottom", 0))
                if pd.notna(row.get("rank_from_bottom", np.nan))
                else np.nan,
                "signal_side": int(np.sign(row.get("signal_side", 1)) or 1),
                "prediction": _f(row.get("prediction", np.nan)),
                "option_mid_entry": _f(opt["mid_price"]),
                "delta_entry": _f(opt["delta"]),
                "gamma_entry": _f(opt.get("gamma", np.nan)),
                "vega_entry": _f(opt.get("vega", np.nan)),
                "strike": _f(opt["strike_price"]),
                "expiry": pd.Timestamp(opt["exdate"]),
                "dte_entry": int(opt["dte"]),
                "underlying_entry": _f(opt["underlying_price"]),
                "implied_vol_entry": _f(opt.get("implied_vol", np.nan)),
                "open_interest_entry": _f(opt.get("open_interest", np.nan)),
            }
        )

    result = pd.DataFrame(records) if records else pd.DataFrame()
    print(
        f"[option_selection] entries built={len(records)} | skipped={skipped} "
        f"(no option data or end-of-series)"
    )
    return result
