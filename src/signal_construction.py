# Stage 7 — Trading Signal Construction
# Output directory: data/signals/
# Timing convention: signal formed after close t; execution at open t+1; PnL open(t+1)->close(t+1)
# Tie-breaking: on equal y_pred, higher volume on date t wins (more liquid signal preferred)

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd


def validate_no_lookahead(predictions_df: pd.DataFrame) -> None:
    """Assert no forward-look columns are present in predictions_df.

    Specifically checks for columns that would introduce information from t+1 or later:
    exact name matches against a forbidden set, and substring matches for ``_t1`` or ``_next``.

    Parameters
    ----------
    predictions_df : DataFrame to validate.

    Raises
    ------
    ValueError
        If any forbidden forward-look column is detected. Hard fail — never warns and continues.
    """
    FORBIDDEN_EXACT = {
        "adj_close_intraday",
        "adj_open",
        "open_t1",
        "close_t1",
        "next_open",
        "next_close",
    }
    FORBIDDEN_SUBSTRINGS = ["_t1", "_next"]

    offending: list[str] = []
    for col in predictions_df.columns:
        col_lower = col.lower()
        if col_lower in {c.lower() for c in FORBIDDEN_EXACT}:
            offending.append(col)
            continue
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in col_lower:
                offending.append(col)
                break

    if offending:
        raise ValueError(
            f"[validate_no_lookahead] FAILED — forward-look columns detected: {offending}. "
            "These columns contain t+1 or later information and must be removed before signal construction."
        )

    print("[validate_no_lookahead] PASSED — no forward-look columns detected.")


def rank_predictions_cross_sectionally(
    predictions_df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "y_pred",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Rank assets by predicted return within each date, with volume tie-breaking.

    For each date t, assets are ranked by ``y_pred`` descending (rank 1 = highest predicted return).
    Tie-breaking: equal ``y_pred`` on the same date → higher ``volume`` receives better (lower) rank.
    This reflects a preference for liquid signals.

    Parameters
    ----------
    predictions_df : DataFrame with at least ``date_col``, ``pred_col``, and ``ticker`` columns.
    date_col : Column name for date.
    pred_col : Column name for model prediction.
    volume_col : Column name for volume; used for tie-breaking within equal ``y_pred``.

    Returns
    -------
    DataFrame with added columns ``pred_rank`` (int) and ``n_assets`` (int),
    sorted by ``date_col`` asc, ``pred_rank`` asc.
    """
    df = predictions_df.copy()

    if volume_col not in df.columns:
        warnings.warn(
            f"[rank_predictions_cross_sectionally] Volume column '{volume_col}' not found. "
            "Falling back to method='first' on unsorted order — no volume tie-breaking applied.",
            stacklevel=2,
        )
        df_sorted = df.sort_values(by=[date_col, pred_col], ascending=[True, False])
        df_sorted["pred_rank"] = (
            df_sorted.groupby(date_col)[pred_col]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        # Sort by date asc, then y_pred desc, then volume desc for tie-breaking.
        # method="first" assigns rank by order of appearance, so higher-volume rows
        # (sorted earlier) receive better ranks within equal y_pred values.
        df_sorted = df.sort_values(
            by=[date_col, pred_col, volume_col],
            ascending=[True, False, False],
        )
        df_sorted["pred_rank"] = (
            df_sorted.groupby(date_col)[pred_col]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    df_sorted["n_assets"] = (
        df_sorted.groupby(date_col)[pred_col].transform("count").astype(int)
    )

    df_sorted = df_sorted.sort_values([date_col, "pred_rank"]).reset_index(drop=True)
    return df_sorted


def build_long_short_signal_book(
    ranked_df: pd.DataFrame,
    K: int = 3,
    date_col: str = "date",
    ticker_col: str = "ticker",
    rank_col: str = "pred_rank",
    n_assets_col: str = "n_assets",
    pred_col: str = "y_pred",
    true_col: str = "y_true",
) -> pd.DataFrame:
    """Build long-short signal book: long top K, short bottom K by cross-sectional rank.

    For each date t:
    - Long: assets with ``pred_rank <= K`` → ``signal = +1``
    - Short: assets with ``pred_rank > n_assets - K`` → ``signal = -1``

    Dates where ``n_assets < 2 * K`` are skipped entirely (warn and continue).
    A ``ValueError`` is raised if any ticker appears in both long and short on the same date.

    Parameters
    ----------
    ranked_df : Output from :func:`rank_predictions_cross_sectionally`.
    K : Number of long and short positions per date.
    date_col, ticker_col, rank_col, n_assets_col, pred_col, true_col : Column name overrides.

    Returns
    -------
    DataFrame with columns (in order):
    ``date``, ``ticker``, ``signal``, ``pred_rank``, ``y_pred``, ``y_true``, ``K``,
    ``execution_date_note``.
    Sorted by date asc, signal desc (+1 before -1), pred_rank asc.
    """
    df = ranked_df.copy()
    if true_col not in df.columns:
        df[true_col] = np.nan

    records: list[dict] = []

    for date_val, group in df.groupby(date_col, sort=True):
        n = int(group[n_assets_col].iloc[0])

        if n < 2 * K:
            warnings.warn(
                f"[Stage 7] Date {date_val}: only {n} assets available, need {2 * K} for K={K}. Skipping.",
                stacklevel=2,
            )
            continue

        long_mask = group[rank_col] <= K
        short_mask = group[rank_col] > n - K

        long_tickers = set(group.loc[long_mask, ticker_col])
        short_tickers = set(group.loc[short_mask, ticker_col])
        overlap = long_tickers & short_tickers
        if overlap:
            raise ValueError(
                f"[Stage 7] Date {date_val}: ticker(s) {overlap} appear in both long and short. "
                f"This should not happen with K={K} and n_assets={n}."
            )

        for _, row in group.loc[long_mask].iterrows():
            records.append({
                "date": row[date_col],
                "ticker": row[ticker_col],
                "signal": 1,
                "pred_rank": int(row[rank_col]),
                "y_pred": row[pred_col],
                "y_true": row[true_col] if true_col in row.index else np.nan,
                "K": K,
                "execution_date_note": "execute_at_open_t+1",
            })

        for _, row in group.loc[short_mask].iterrows():
            records.append({
                "date": row[date_col],
                "ticker": row[ticker_col],
                "signal": -1,
                "pred_rank": int(row[rank_col]),
                "y_pred": row[pred_col],
                "y_true": row[true_col] if true_col in row.index else np.nan,
                "K": K,
                "execution_date_note": "execute_at_open_t+1",
            })

    col_order = ["date", "ticker", "signal", "pred_rank", "y_pred", "y_true", "K", "execution_date_note"]

    if not records:
        return pd.DataFrame(columns=col_order)

    result = pd.DataFrame(records)[col_order]
    result = result.sort_values(
        by=["date", "signal", "pred_rank"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return result


def build_long_only_signal_book(
    ranked_df: pd.DataFrame,
    K: int = 3,
    date_col: str = "date",
    ticker_col: str = "ticker",
    rank_col: str = "pred_rank",
    n_assets_col: str = "n_assets",
    pred_col: str = "y_pred",
    true_col: str = "y_true",
) -> pd.DataFrame:
    """Build long-only signal book: long top K by cross-sectional rank.

    For each date t:
    - Long only: assets with ``pred_rank <= K`` → ``signal = +1``

    Dates where ``n_assets < K`` are skipped entirely (warn and continue).

    Parameters
    ----------
    ranked_df : Output from :func:`rank_predictions_cross_sectionally`.
    K : Number of long positions per date.
    date_col, ticker_col, rank_col, n_assets_col, pred_col, true_col : Column name overrides.

    Returns
    -------
    DataFrame with columns (in order):
    ``date``, ``ticker``, ``signal``, ``pred_rank``, ``y_pred``, ``y_true``, ``K``,
    ``execution_date_note``.
    Sorted by date asc, pred_rank asc. ``signal`` is always ``+1``.
    """
    df = ranked_df.copy()
    if true_col not in df.columns:
        df[true_col] = np.nan

    records: list[dict] = []

    for date_val, group in df.groupby(date_col, sort=True):
        n = int(group[n_assets_col].iloc[0])

        if n < K:
            warnings.warn(
                f"[Stage 7] Date {date_val}: only {n} assets available, need {K} for K={K}. Skipping.",
                stacklevel=2,
            )
            continue

        long_mask = group[rank_col] <= K

        for _, row in group.loc[long_mask].iterrows():
            records.append({
                "date": row[date_col],
                "ticker": row[ticker_col],
                "signal": 1,
                "pred_rank": int(row[rank_col]),
                "y_pred": row[pred_col],
                "y_true": row[true_col] if true_col in row.index else np.nan,
                "K": K,
                "execution_date_note": "execute_at_open_t+1",
            })

    col_order = ["date", "ticker", "signal", "pred_rank", "y_pred", "y_true", "K", "execution_date_note"]

    if not records:
        return pd.DataFrame(columns=col_order)

    result = pd.DataFrame(records)[col_order]
    result = result.sort_values(
        by=["date", "pred_rank"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return result


def generate_signal_books(
    predictions_path: str,
    output_dir: str = "data/signals",
    K: int = 3,
    volume_col: str = "volume",
) -> dict:
    """Orchestrate full signal construction pipeline from predictions CSV.

    Loads predictions, validates for no lookahead, ranks cross-sectionally,
    and writes long-short and long-only signal books to ``output_dir``.

    Parameters
    ----------
    predictions_path : Path to predictions CSV with columns ticker, date, y_pred, y_true, volume, split.
    output_dir : Directory to write signal book CSVs (created if it does not exist).
    K : Number of long (and short) positions per date.
    volume_col : Column name for volume in the predictions CSV.

    Returns
    -------
    dict with keys ``"long_short"`` and ``"long_only"`` mapping to the respective DataFrames.
    """
    # 1. Load predictions
    df = pd.read_csv(predictions_path, parse_dates=["date"])

    # 2. Validate no lookahead columns
    validate_no_lookahead(df)

    # 3. Rank cross-sectionally with volume tie-breaking
    ranked = rank_predictions_cross_sectionally(df, volume_col=volume_col)

    # 4. Build signal books
    long_short_df = build_long_short_signal_book(ranked, K=K)
    long_only_df = build_long_only_signal_book(ranked, K=K)

    # 5. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 6. Write CSVs
    ls_path = os.path.join(output_dir, "signal_book_long_short.csv")
    lo_path = os.path.join(output_dir, "signal_book_long_only.csv")
    long_short_df.to_csv(ls_path, index=False)
    long_only_df.to_csv(lo_path, index=False)

    # 7. Print summary
    ls_dates = long_short_df["date"].nunique() if not long_short_df.empty else 0
    lo_dates = long_only_df["date"].nunique() if not long_only_df.empty else 0

    all_date_set = set(ranked["date"].unique())
    ls_date_set = set(long_short_df["date"].unique()) if not long_short_df.empty else set()
    skipped = sorted(all_date_set - ls_date_set)
    if skipped:
        skipped_str = str([d.date() if hasattr(d, "date") else str(d) for d in skipped])
    else:
        skipped_str = "none"

    print(f"[Stage 7] Signal books written to: {output_dir}")
    print(f"Long-Short book: {ls_dates} dates, {len(long_short_df)} rows ({K} long + {K} short per date)")
    print(f"Long-Only book:  {lo_dates} dates, {len(long_only_df)} rows ({K} long per date)")
    print(f"Dates skipped (insufficient assets): {skipped_str}")
    print("Output files:")
    print(f"  - {ls_path}")
    print(f"  - {lo_path}")

    return {"long_short": long_short_df, "long_only": long_only_df}
