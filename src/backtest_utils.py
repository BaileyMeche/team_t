# Backtest Utilities — Stages 8–13 Orchestrator
# Delta-hedged long calls strategy driven by LSTM cross-sectional predictions.
#
# Full loop per trading day t:
#   1. Enter queued positions (signaled on t-1, entered at open t ≈ option close t)
#   2. Update open positions: look up today's option/stock prices, rebalance delta hedge,
#      compute daily mark-to-market P&L, check exit conditions
#   3. Exit triggered positions (DTE < threshold, signal gone, hold-period cap)
#   4. Queue new entries from today's top-K signals (for entry tomorrow t+1)
#   5. Record daily portfolio P&L and position snapshot
#
# Optional earnings-cycle mode:
#   - Enter one ranked cohort per signal cycle
#   - Hold for max_holding_days trading bars, then exit
#   - Stay flat until the next earnings signal cycle
#
# Timing convention (Option A):
#   Signal formed after close t → entry at open t+1 (approximated as option close t+1)
#   PnL measured close-to-close on the option + stock hedge
#
# Default exit rules (first triggered wins):
#   - DTE < exit_dte_threshold (default 10 days-to-expiry) — avoid pin risk
#   - Ticker drops out of top-K — signal exit
#   - days_held >= max_holding_days (default 20 trading bars)
# In earnings_cycle_mode:
#   - Exit is fixed at days_held >= max_holding_days

from __future__ import annotations

import contextlib
import io
import warnings
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .hedge import CONTRACT_SIZE, hedge_adjustment, initial_stock_position, rebalance_position
from .performance import compute_metrics, drawdown_series, equity_curve
from .pnl import daily_pnl, exit_pnl
from .ranking import build_signal_table

EXIT_REASON_STOP_LOSS = "stop_loss"
EXIT_REASON_DTE = "dte_threshold"
EXIT_REASON_SIGNAL = "signal_exit"
EXIT_REASON_HPR = "hpr_limit_exit"

# Module-level miss counter — incremented by _lookup_option on every cache miss.
# Reset to 0 at the start of each run_backtest call.
_LOOKUP_MISSES: int = 0


def _f(val, default: float = float("nan")) -> float:
    """float() that handles pd.NA / None without raising TypeError."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _build_underlying_vol_lookup(
    options_df: pd.DataFrame,
    lookback_days: int = 21,
) -> dict[tuple[pd.Timestamp, str], float]:
    """Build (date, ticker) -> rolling daily realized vol from underlying prices."""
    if lookback_days <= 1:
        return {}
    if not {"date", "ticker", "underlying_price"}.issubset(options_df.columns):
        return {}

    px = (
        options_df[["date", "ticker", "underlying_price"]]
        .dropna(subset=["date", "ticker", "underlying_price"])
        .groupby(["date", "ticker"], as_index=False)["underlying_price"]
        .first()
    )
    if px.empty:
        return {}

    px = px.sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker", sort=False)["underlying_price"].pct_change()
    window = int(lookback_days)
    min_periods = min(window, max(2, int(window // 2)))
    px["realized_vol"] = px.groupby("ticker", sort=False)["ret"].transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )

    out: dict[tuple[pd.Timestamp, str], float] = {}
    for row in px[["date", "ticker", "realized_vol"]].itertuples(index=False):
        vol = _f(row.realized_vol)
        if np.isfinite(vol) and vol > 0:
            out[(pd.Timestamp(row.date), str(row.ticker))] = float(vol)
    return out


def _compute_signal_weights(
    candidates_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    allow_shorts: bool = False,
    side_col: str = "signal_side",
    use_vol_scaling: bool = True,
    vol_lookup: dict[tuple[pd.Timestamp, str], float] | None = None,
    vol_floor: float = 0.01,
) -> dict[str, float]:
    """Cross-sectional weights from signal strength (optionally vol-scaled)."""
    if candidates_df.empty:
        return {}
    if "ticker" not in candidates_df.columns:
        return {}

    work = candidates_df.copy()
    if "prediction" in work.columns:
        work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce").fillna(0.0)
    else:
        work["prediction"] = 0.0

    if allow_shorts and side_col in work.columns:
        side = pd.to_numeric(work[side_col], errors="coerce").fillna(0.0)
        side = np.sign(side).replace(0.0, 1.0)
        work["score"] = side * work["prediction"].abs()
        if float(work["score"].abs().sum()) <= 0.0:
            work["score"] = side
    else:
        # Long-only book: keep positive conviction and fall back to equal-weight.
        work["score"] = work["prediction"].clip(lower=0.0)
        if float(work["score"].sum()) <= 0.0:
            work["score"] = 1.0

    if use_vol_scaling:
        lookup = vol_lookup or {}
        floor = max(float(vol_floor), 1e-6)
        sig_date = pd.Timestamp(signal_date)
        vols = [lookup.get((sig_date, str(t)), np.nan) for t in work["ticker"]]
        work["sigma"] = pd.to_numeric(pd.Series(vols, index=work.index), errors="coerce")
        work["sigma"] = work["sigma"].where(work["sigma"] > floor, floor).fillna(floor)
        work["score"] = work["score"] / work["sigma"]

    denom = float(work["score"].abs().sum())
    if not np.isfinite(denom) or denom <= 0.0:
        n = len(work)
        work["weight"] = 1.0 / n if n > 0 else 0.0
    else:
        work["weight"] = work["score"] / denom

    weights = work.groupby("ticker", sort=False)["weight"].sum()
    return {str(k): float(v) for k, v in weights.items()}


def _contracts_for_target_delta_exposure(
    target_delta_exposure: float,
    option_delta: float,
    underlying_price: float,
) -> tuple[int, float]:
    """Convert target delta-equivalent notional into whole option contracts."""
    per_contract_exposure = abs(float(option_delta)) * CONTRACT_SIZE * abs(float(underlying_price))
    if (
        not np.isfinite(per_contract_exposure)
        or per_contract_exposure <= 0.0
        or not np.isfinite(target_delta_exposure)
        or target_delta_exposure <= 0.0
    ):
        return 0, per_contract_exposure
    return int(float(target_delta_exposure) // per_contract_exposure), float(per_contract_exposure)


def _build_round_trip_trade_book(trade_df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-trade book by pairing entry/exit event rows.

    The event log (`trade_df`) intentionally has entry-only and exit-only columns.
    This helper merges both sides into a denormalized trade book so key fields are
    populated on a single row for each trade lifecycle.
    """
    if trade_df is None or trade_df.empty:
        return pd.DataFrame()
    if not {"action", "date", "ticker"}.issubset(trade_df.columns):
        return pd.DataFrame()

    work = trade_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    entries = (
        work[work["action"] == "enter"]
        .sort_values(["ticker", "date"], kind="stable")
        .copy()
    )
    exits = (
        work[work["action"] == "exit"]
        .sort_values(["ticker", "date"], kind="stable")
        .copy()
    )

    entries["trade_seq"] = entries.groupby("ticker", sort=False).cumcount()
    exits["trade_seq"] = exits.groupby("ticker", sort=False).cumcount()

    entry_rename = {
        "date": "entry_date",
        "option_price": "entry_option_price",
        "stock_price": "entry_stock_price",
        "delta": "entry_delta",
        "delta_raw": "entry_delta_raw",
        "stock_position": "entry_stock_position",
        "dte": "entry_dte",
    }
    exit_rename = {
        "date": "exit_date",
        "option_price": "exit_option_price",
        "stock_price": "exit_stock_price",
    }

    e = entries.rename(columns=entry_rename)
    x = exits.rename(columns=exit_rename)

    keep_e = [
        "ticker", "trade_seq", "signal_date", "entry_date", "entry_option_price",
        "entry_stock_price", "entry_delta", "entry_delta_raw", "signal_side",
        "option_qty", "entry_stock_position", "num_contracts", "strike", "expiry",
        "entry_dte", "entry_cost", "rank", "prediction", "target_weight",
        "target_delta_exposure", "per_contract_delta_exposure",
    ]
    keep_x = [
        "ticker", "trade_seq", "exit_date", "exit_option_price", "exit_stock_price",
        "exit_reason", "days_held", "bars_held", "realized_pnl", "exit_cost",
        "signal_side", "option_qty", "num_contracts",
    ]
    e = e[[c for c in keep_e if c in e.columns]]
    x = x[[c for c in keep_x if c in x.columns]]

    merged = e.merge(x, on=["ticker", "trade_seq"], how="outer", suffixes=("", "_exit"))

    # Fill side/size fields from whichever side is available.
    for base in ["signal_side", "option_qty", "num_contracts"]:
        alt = f"{base}_exit"
        if base in merged.columns and alt in merged.columns:
            merged[base] = merged[base].combine_first(merged[alt])
            merged = merged.drop(columns=[alt])

    def _col(df: pd.DataFrame, name: str, fill=np.nan) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(fill, index=df.index)

    merged["action"] = "round_trip"
    merged["date"] = _col(merged, "entry_date", pd.NaT).combine_first(_col(merged, "exit_date", pd.NaT))
    merged["option_price"] = _col(merged, "entry_option_price").combine_first(_col(merged, "exit_option_price"))
    merged["stock_price"] = _col(merged, "entry_stock_price").combine_first(_col(merged, "exit_stock_price"))
    merged["delta"] = _col(merged, "entry_delta")
    merged["delta_raw"] = _col(merged, "entry_delta_raw")
    merged["stock_position"] = _col(merged, "entry_stock_position")
    merged["dte"] = _col(merged, "entry_dte")

    # Keep target fields populated in fixed-size mode.
    entry_price = pd.to_numeric(_col(merged, "entry_stock_price"), errors="coerce")
    entry_delta_raw = pd.to_numeric(_col(merged, "entry_delta_raw"), errors="coerce")
    contracts = pd.to_numeric(_col(merged, "num_contracts"), errors="coerce").abs()
    per_contract = (entry_delta_raw.abs() * CONTRACT_SIZE * entry_price.abs())
    actual_delta_exposure = per_contract * contracts

    merged["per_contract_delta_exposure"] = pd.to_numeric(
        _col(merged, "per_contract_delta_exposure"), errors="coerce"
    ).combine_first(per_contract)
    merged["target_delta_exposure"] = pd.to_numeric(
        _col(merged, "target_delta_exposure"), errors="coerce"
    ).combine_first(actual_delta_exposure)
    merged["target_weight"] = pd.to_numeric(
        _col(merged, "target_weight"), errors="coerce"
    ).fillna(0.0)

    has_entry = _col(merged, "entry_date", pd.NaT).notna()
    has_exit = _col(merged, "exit_date", pd.NaT).notna()
    merged["trade_status"] = np.where(has_entry & has_exit, "closed", np.where(has_entry, "open", "orphan_exit"))

    core_cols = [
        "action", "date", "ticker", "signal_date", "option_price", "stock_price",
        "delta", "delta_raw", "signal_side", "option_qty", "stock_position",
        "num_contracts", "strike", "expiry", "dte", "entry_cost", "rank",
        "prediction", "target_weight", "target_delta_exposure",
        "per_contract_delta_exposure", "exit_reason", "days_held", "bars_held",
        "realized_pnl", "exit_cost",
    ]
    extra_cols = [
        "trade_status", "trade_seq", "entry_date", "exit_date",
        "entry_option_price", "exit_option_price", "entry_stock_price", "exit_stock_price",
    ]
    ordered = [c for c in core_cols + extra_cols if c in merged.columns]
    merged = merged[ordered].sort_values(["date", "ticker", "trade_seq"], kind="stable")
    merged = merged.reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Position schema (dict keys)
# ---------------------------------------------------------------------------
# ticker             : str
# entry_date         : pd.Timestamp
# signal_date        : pd.Timestamp
# entry_option_price : float  — mid_price at entry
# entry_stock_price  : float  — underlying_price at entry
# prev_option_price  : float  — updated each day
# prev_stock_price   : float  — updated each day
# current_delta      : float  — updated each day
# raw_delta          : float  — option delta from data (unsigned call delta)
# signal_side        : int    — +1 long option, -1 short option
# option_qty         : int    — signed contracts (+long / -short)
# stock_position     : float  — shares short (negative), updated on rebalance
# num_contracts      : int
# expiry             : pd.Timestamp
# strike             : float
# dte_at_entry       : int
# days_held          : int
# cumulative_pnl     : float
# last_hedge_adjustment : float


def _lookup_option(
    options_indexed: pd.DataFrame,
    ticker: str,
    date: "pd.Timestamp",
    expiry: "pd.Timestamp | None",
    dte_min: "int | None" = None,
    dte_max: "int | None" = None,
    moneyness_min: "float | None" = None,
    moneyness_max: "float | None" = None,
    min_open_interest: "int | None" = None,
    max_spread_frac: "float | None" = None,
    _opt_lookup: "dict | None" = None,  # NEW: pre-built (date, ticker) → list[dict]
) -> "pd.Series | None":
    """Fast O(1) option lookup via pre-built dict.

    When _opt_lookup is provided (the normal path inside run_backtest), the
    function does a single dict key access then iterates the pre-sorted
    candidate list.  The list is already ordered ATM-closest first, highest
    open-interest second (set up once in the pre-build block), so the first
    row that passes all filters is the correct result.

    Falls back to the original DataFrame-slice path when _opt_lookup is None.
    """
    global _LOOKUP_MISSES

    # ── Fast path ────────────────────────────────────────────────────────────
    if _opt_lookup is not None:
        date_ts = pd.Timestamp(date)
        candidates = _opt_lookup.get((date_ts, str(ticker)), [])
        if not candidates:
            _LOOKUP_MISSES += 1
            return None
        for row in candidates:  # pre-sorted: ATM-closest, highest OI first
            # Expiry match
            if expiry is not None:
                row_exp = row.get("exdate")
                if row_exp is not None and pd.Timestamp(row_exp) != pd.Timestamp(expiry):
                    continue
            # DTE filter
            dte_val = row.get("dte")
            if dte_val is None:
                continue
            dte_f = float(dte_val)
            if np.isnan(dte_f):
                continue
            if dte_min is not None and dte_f < dte_min:
                continue
            if dte_max is not None and dte_f > dte_max:
                continue
            # Moneyness filter
            mny = row.get("moneyness")
            if mny is not None:
                mny_f = float(mny)
                if not np.isnan(mny_f):
                    if moneyness_min is not None and mny_f < moneyness_min:
                        continue
                    if moneyness_max is not None and mny_f > moneyness_max:
                        continue
            # Open interest filter
            if min_open_interest is not None:
                oi = row.get("open_interest")
                if oi is None or float(oi) < min_open_interest:
                    continue
            # Spread filter
            if max_spread_frac is not None:
                bid = row.get("best_bid")
                ask = row.get("best_offer")
                mid = row.get("mid_price")
                if bid is not None and ask is not None and mid and float(mid) > 0:
                    if (float(ask) - float(bid)) / float(mid) > max_spread_frac:
                        continue
            # Passed all filters — return as pd.Series for drop-in compatibility
            return pd.Series(row)
        # All candidates were filtered out
        _LOOKUP_MISSES += 1
        return None

    # ── Legacy fallback (original DataFrame-slice path) ──────────────────────
    try:
        sub = options_indexed.loc[(pd.Timestamp(date), str(ticker))]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        if expiry is not None:
            match = sub[pd.to_datetime(sub["exdate"]) == expiry]
            if not match.empty:
                sub = match
        if dte_min is not None and "dte" in sub.columns:
            sub = sub[pd.to_numeric(sub["dte"], errors="coerce") >= int(dte_min)]
        if dte_max is not None and "dte" in sub.columns:
            sub = sub[pd.to_numeric(sub["dte"], errors="coerce") <= int(dte_max)]
        if moneyness_min is not None and "moneyness" in sub.columns:
            sub = sub[pd.to_numeric(sub["moneyness"], errors="coerce") >= float(moneyness_min)]
        if moneyness_max is not None and "moneyness" in sub.columns:
            sub = sub[pd.to_numeric(sub["moneyness"], errors="coerce") <= float(moneyness_max)]
        if min_open_interest is not None and "open_interest" in sub.columns:
            sub = sub[pd.to_numeric(sub["open_interest"], errors="coerce") >= int(min_open_interest)]
        if max_spread_frac is not None and {"best_bid", "best_offer", "mid_price"}.issubset(sub.columns):
            bid = pd.to_numeric(sub["best_bid"], errors="coerce")
            ask = pd.to_numeric(sub["best_offer"], errors="coerce")
            mid = pd.to_numeric(sub["mid_price"], errors="coerce").replace(0.0, np.nan)
            sub = sub[(ask - bid) / mid <= float(max_spread_frac)]
        if not sub.empty:
            out = sub.copy()
            if "moneyness" in out.columns:
                out["_atm_dist"] = (pd.to_numeric(out["moneyness"], errors="coerce") - 1.0).abs()
            else:
                out["_atm_dist"] = np.nan
            if "open_interest" in out.columns:
                out["_oi"] = pd.to_numeric(out["open_interest"], errors="coerce")
            else:
                out["_oi"] = np.nan
            out = out.sort_values(
                by=[c for c in ["_atm_dist", "_oi"] if c in out.columns],
                ascending=[True, False],
                kind="stable",
                na_position="last",
            )
            return out.iloc[0]
    except KeyError:
        pass
    _LOOKUP_MISSES += 1
    return None


def run_backtest(
    predictions_df: pd.DataFrame,
    options_df: pd.DataFrame,
    K: int = 3,
    initial_capital: float = 100_000.0,
    max_holding_days: int = 20,
    exit_dte_threshold: int = 10,
    num_contracts: int = 1,
    commission_per_contract: float = 1.0,
    half_spread_pct_stock: float = 0.0005,
    half_spread_pct_option: float = 0.02,
    dte_min: int = 30,
    dte_max: int = 45,
    entry_moneyness_min: float | None = None,
    entry_moneyness_max: float | None = None,
    entry_min_open_interest: int | None = None,
    entry_max_spread_frac: float | None = None,
    use_signal_exit: bool = True,
    top_k_df: pd.DataFrame | None = None,
    earnings_cycle_mode: bool | None = None,
    entry_prediction_threshold: float | None = None,
    stop_loss_frac_of_entry_cost: float | None = None,
    allow_short_signals: bool = False,
    signal_side_col: str = "signal_side",
    sizing_mode: str = "fixed",
    risk_target_gross_exposure_frac: float = 0.10,
    risk_target_daily_vol: float | None = None,
    risk_max_name_exposure_frac: float | None = 0.05,
    risk_use_vol_scaling: bool = True,
    risk_vol_lookback_days: int = 21,
    risk_vol_floor: float = 0.01,
    risk_max_option_oi_frac: float | None = 0.01,
    risk_max_contracts_per_trade: int | None = None,
) -> dict[str, Any]:
    """Run the full delta-hedged options backtest.

    Parameters
    ----------
    predictions_df : LSTM predictions CSV as DataFrame. Columns: date, ticker, prediction.
    options_df : OptionMetrics ATM calls DataFrame. Columns: date, ticker, mid_price,
                 delta, underlying_price, exdate, dte, open_interest, etc.
    K : Number of top-ranked stocks to hold at any time.
    initial_capital : Starting portfolio value in dollars.
    max_holding_days : Force exit after this many trading bars/sessions
                       (not calendar days).
    exit_dte_threshold : Exit when option DTE falls below this.
    num_contracts : Contracts per position (default 1).
    commission_per_contract : Flat commission per contract (entry + exit).
    half_spread_pct_stock : Half stock bid-ask spread for hedge rebalance cost.
    half_spread_pct_option : Half option spread for exit cost.
    dte_min, dte_max : DTE window for option selection at entry.
    entry_moneyness_min, entry_moneyness_max : Optional moneyness bounds at entry.
    entry_min_open_interest : Optional minimum open interest filter at entry.
    entry_max_spread_frac : Optional maximum relative spread
                            ``(ask - bid) / mid`` filter at entry.
    earnings_cycle_mode : If True, enforce earnings-cycle trading:
                         enter a non-overlapping cohort, hold max_holding_days,
                         exit, then wait for the next earnings signal cycle.
                         If None, auto-enables when top_k_df includes
                         ``entry_date_hint``.
    entry_prediction_threshold : If set, only enter signals where
                                 prediction >= threshold (long-only mode) or
                                 abs(prediction) >= threshold (when shorts enabled).
    stop_loss_frac_of_entry_cost : If set (e.g., 0.20), exit a position early
                                   when cumulative mark-to-market P&L falls below
                                   -threshold * entry_cost.
    allow_short_signals : If True, support signed long/short signals in
                          ``top_k_df`` via ``signal_side_col`` (+1 long, -1 short).
                          Shorts are implemented as short calls plus delta hedge.
    signal_side_col : Column name in signal table holding signal side.
    sizing_mode : ``"fixed"`` uses scalar ``num_contracts`` for every trade.
                  ``"risk"`` sizes each trade from cross-sectional signal weights,
                  optional inverse-vol scaling, and option delta-equivalent exposure.
    risk_target_gross_exposure_frac : Target gross delta-equivalent exposure as a
                                      fraction of current equity when
                                      ``sizing_mode="risk"``.
    risk_target_daily_vol : Optional target daily portfolio volatility
                            (e.g., 0.01 for 1%). Uses a diagonal approximation
                            from ticker-level realized vols. If provided, the
                            gross target is derived from this risk budget.
    risk_max_name_exposure_frac : Optional per-ticker cap (fraction of equity) in
                                  risk sizing mode.
    risk_use_vol_scaling : If True, scale signal scores by inverse realized vol.
    risk_vol_lookback_days : Lookback window for daily realized vol estimate.
    risk_vol_floor : Minimum daily vol used in inverse-vol scaling to avoid
                     unstable leverage.
    risk_max_option_oi_frac : Optional liquidity cap as fraction of option open
                              interest allowed per trade.
    risk_max_contracts_per_trade : Optional hard cap on contracts per trade.

    Returns
    -------
    dict with keys:
        equity_curve   : pd.Series — daily cumulative equity
        daily_pnl_df   : pd.DataFrame — daily P&L log
        trade_log      : pd.DataFrame — entry/exit records
        trade_book     : pd.DataFrame — one-row-per-trade round-trip book
        position_log   : pd.DataFrame — daily position snapshots
        metrics        : dict — performance metrics
        drawdown       : pd.Series
    """
    # --- Prep ---
    predictions_df = predictions_df.copy()
    predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors="coerce")

    options_df = options_df.copy()
    options_df["date"] = pd.to_datetime(options_df["date"], errors="coerce")
    options_df["exdate"] = pd.to_datetime(options_df["exdate"], errors="coerce")

    # Build ranked signal table — use pre-built top_k_df if provided (e.g. earnings mode)
    if top_k_df is not None:
        ranked = top_k_df.copy()
        ranked["date"] = pd.to_datetime(ranked["date"], errors="coerce")
    else:
        if allow_short_signals:
            ranked_all = predictions_df.copy()
            ranked_all["date"] = pd.to_datetime(ranked_all["date"], errors="coerce")
            ranked_all = ranked_all.dropna(subset=["date", "ticker", "prediction"]).copy()
            ranked_all["prediction"] = pd.to_numeric(ranked_all["prediction"], errors="coerce")
            ranked_all = ranked_all.dropna(subset=["prediction"]).copy()
            ranked_all["rank"] = (
                ranked_all.groupby("date")["prediction"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            ranked_all["n_assets"] = ranked_all.groupby("date")["ticker"].transform("count").astype(int)
            ranked_all["rank_from_bottom"] = (
                ranked_all.groupby("date")["prediction"]
                .rank(method="first", ascending=True)
                .astype(int)
            )

            long_df = ranked_all[ranked_all["rank"] <= int(K)].copy()
            long_df[signal_side_col] = 1
            short_df = ranked_all[ranked_all["rank_from_bottom"] <= int(K)].copy()
            short_df[signal_side_col] = -1
            ranked = pd.concat([long_df, short_df], ignore_index=True)
            ranked = ranked.drop_duplicates(subset=["date", "ticker"], keep="first")
            ranked["K"] = int(K)
            n_dates = ranked["date"].nunique() if not ranked.empty else 0
            print(
                "[ranking] long/short signal table: "
                f"K_long={K}, K_short={K} | signal dates={n_dates} | "
                f"rows={len(ranked)} | longs={(ranked[signal_side_col] > 0).sum()} | "
                f"shorts={(ranked[signal_side_col] < 0).sum()}"
            )
        else:
            ranked = build_signal_table(predictions_df, K=K)
    if ranked.empty:
        raise ValueError("[backtest] No ranked signals produced — check predictions input.")

    if signal_side_col in ranked.columns:
        ranked[signal_side_col] = pd.to_numeric(ranked[signal_side_col], errors="coerce").fillna(0.0)
        ranked[signal_side_col] = np.sign(ranked[signal_side_col]).replace(0.0, 1.0).astype(int)
    else:
        ranked[signal_side_col] = 1
    if not allow_short_signals:
        ranked[signal_side_col] = 1

    if earnings_cycle_mode is None:
        earnings_cycle_mode = bool(top_k_df is not None and "entry_date_hint" in ranked.columns)

    sizing_mode = str(sizing_mode).strip().lower()
    if sizing_mode not in {"fixed", "risk"}:
        raise ValueError(f"[backtest] sizing_mode must be 'fixed' or 'risk' (got: {sizing_mode!r})")
    num_contracts = max(1, int(num_contracts))

    risk_target_gross_exposure_frac = max(0.0, float(risk_target_gross_exposure_frac))
    if risk_target_daily_vol is not None:
        risk_target_daily_vol = max(0.0, float(risk_target_daily_vol))
    if risk_max_name_exposure_frac is not None:
        risk_max_name_exposure_frac = max(0.0, float(risk_max_name_exposure_frac))
    if risk_max_option_oi_frac is not None:
        risk_max_option_oi_frac = max(0.0, float(risk_max_option_oi_frac))
    if risk_max_contracts_per_trade is not None:
        risk_max_contracts_per_trade = max(1, int(risk_max_contracts_per_trade))
    risk_vol_floor = max(float(risk_vol_floor), 1e-6)
    risk_vol_lookback_days = max(2, int(risk_vol_lookback_days))

    vol_lookup: dict[tuple[pd.Timestamp, str], float] = {}
    if sizing_mode == "risk":
        vol_lookup = _build_underlying_vol_lookup(
            options_df=options_df,
            lookback_days=risk_vol_lookback_days,
        )
        print(
            f"[backtest] risk sizing enabled: gross_target={risk_target_gross_exposure_frac:.1%}, "
            f"daily_vol_target={risk_target_daily_vol if risk_target_daily_vol is not None else 'None'}, "
            f"max_name={risk_max_name_exposure_frac if risk_max_name_exposure_frac is not None else 'None'}, "
            f"vol_scaling={risk_use_vol_scaling}, shorts={allow_short_signals}, vol_points={len(vol_lookup):,}",
            flush=True,
        )

    ticker_signal_dates: dict[str, list[pd.Timestamp]] = {}
    next_allowed_signal_date: dict[str, pd.Timestamp | None] = {}
    if earnings_cycle_mode:
        for ticker, g in ranked.groupby("ticker", sort=False):
            signal_dates = sorted(pd.to_datetime(g["date"], errors="coerce").dropna().unique())
            ticker_signal_dates[str(ticker)] = [pd.Timestamp(d) for d in signal_dates]

    def _next_signal_date_for_ticker(ticker: str, after_date: pd.Timestamp) -> pd.Timestamp | None:
        for d in ticker_signal_dates.get(ticker, []):
            if d > after_date:
                return d
        return None

    # ── Pre-build fast option lookup dict ──────────────────────────────────
    # Key: (date, ticker) → list of option rows (as dicts), pre-sorted by
    # ATM proximity asc, open_interest desc.  _lookup_option then does an
    # O(1) dict access instead of a DataFrame .loc + filter chain per call.
    global _LOOKUP_MISSES
    _LOOKUP_MISSES = 0  # reset counter for this run

    print("[backtest] Building option lookup dict...", flush=True)
    _t0_lookup = __import__("time").time()

    # Parse numeric columns once (avoids repeated pd.to_numeric in hot loop)
    for _col in ["mid_price", "delta", "underlying_price", "dte", "strike_price",
                 "open_interest", "best_bid", "best_offer", "moneyness"]:
        if _col in options_df.columns:
            options_df[_col] = pd.to_numeric(options_df[_col], errors="coerce")

    # Compute ATM distance once
    if "moneyness" in options_df.columns:
        options_df["_atm_dist"] = (options_df["moneyness"] - 1.0).abs()
    else:
        options_df["_atm_dist"] = np.nan

    # Sort globally so each group list is already in lookup-priority order
    _sort_cols: list[str] = []
    _sort_asc: list[bool] = []
    if "_atm_dist" in options_df.columns:
        _sort_cols.append("_atm_dist");  _sort_asc.append(True)
    if "open_interest" in options_df.columns:
        _sort_cols.append("open_interest"); _sort_asc.append(False)
    if _sort_cols:
        options_df = options_df.sort_values(_sort_cols, ascending=_sort_asc, kind="stable")

    # Build dict: (date, ticker) → list[dict]
    _opt_lookup: dict[tuple, list[dict]] = {}
    for (_date, _ticker), _grp in options_df.groupby(["date", "ticker"], sort=False):
        _opt_lookup[(_date, _ticker)] = _grp.to_dict("records")

    # Keep options_indexed for any code paths that call _lookup_option without
    # _opt_lookup (e.g. external callers); not used in the hot loop below.
    options_indexed = options_df.set_index(["date", "ticker"]).sort_index()

    _elapsed_lookup = __import__("time").time() - _t0_lookup
    print(
        f"[backtest] Option lookup dict built: {len(_opt_lookup):,} keys "
        f"in {_elapsed_lookup:.1f}s",
        flush=True,
    )

    # All prediction dates in order
    all_pred_dates = sorted(ranked["date"].unique())
    all_option_dates = sorted(options_df["date"].dropna().unique())

    # Map each prediction date → next option trading date (for entry queue)
    pred_to_next_opt: dict[pd.Timestamp, pd.Timestamp | None] = {}
    for pd_date in all_pred_dates:
        future = [d for d in all_option_dates if d > pd_date]
        pred_to_next_opt[pd_date] = future[0] if future else None

    # --- State ---
    open_positions: dict[str, dict] = {}   # ticker -> position dict
    entry_queue: list[dict] = []            # pending entries keyed by entry_date

    daily_rows: list[dict] = []
    trade_log: list[dict] = []
    position_snapshots: list[dict] = []
    running_equity = float(initial_capital)

    # Iterate over all dates in the union of prediction + option dates
    all_dates = sorted(set(all_pred_dates) | set(all_option_dates))

    # Yearly progress tracker (state stored on function object to avoid a
    # closure variable that would survive across calls in the same process)
    _last_year_printed: list[int | None] = [None]  # mutable container

    for current_date in all_dates:
        current_ts = pd.Timestamp(current_date)
        daily_portfolio_pnl = 0.0
        exited_any_today = False

        # Progress: print once per calendar year (real-time via flush=True)
        _cur_year = current_ts.year
        if _cur_year != _last_year_printed[0]:
            _last_year_printed[0] = _cur_year
            print(
                f"[backtest] {current_ts.date()}  "
                f"open={len(open_positions)}  queued={len(entry_queue)}  "
                f"trades_logged={len(trade_log)}",
                flush=True,
            )

        # ================================================================
        # Step 1: Enter queued positions whose entry_date == today
        # ================================================================
        remaining_queue: list[dict] = []
        for queued in entry_queue:
            if queued["entry_date"] != current_ts:
                remaining_queue.append(queued)
                continue

            ticker = queued["ticker"]
            if ticker in open_positions:
                continue  # already holding — no pyramid

            opt_row = _lookup_option(
                options_indexed,
                ticker,
                current_ts,
                queued.get("expiry"),
                dte_min=dte_min,
                dte_max=dte_max,
                moneyness_min=entry_moneyness_min,
                moneyness_max=entry_moneyness_max,
                min_open_interest=entry_min_open_interest,
                max_spread_frac=entry_max_spread_frac,
                _opt_lookup=_opt_lookup,
            )
            if opt_row is None:
                warnings.warn(
                    f"[backtest] No option data for {ticker} on entry {current_ts.date()} — skipped.",
                    stacklevel=2,
                )
                continue

            entry_opt_price = _f(opt_row["mid_price"])
            entry_stock_price = _f(opt_row["underlying_price"])
            entry_delta_raw = _f(opt_row["delta"])
            expiry = pd.Timestamp(opt_row["exdate"])
            dte_entry = int(opt_row["dte"])
            strike = _f(opt_row["strike_price"])
            position_num_contracts = max(1, int(queued.get("num_contracts", num_contracts)))
            option_side = int(np.sign(queued.get("signal_side", 1)) or 1)
            option_qty = int(option_side * position_num_contracts)
            effective_delta = float(entry_delta_raw) * float(option_side)
            stock_pos = initial_stock_position(effective_delta, position_num_contracts)

            open_positions[ticker] = {
                "ticker": ticker,
                "entry_date": current_ts,
                "signal_date": queued["signal_date"],
                "entry_option_price": entry_opt_price,
                "entry_stock_price": entry_stock_price,
                "prev_option_price": entry_opt_price,
                "prev_stock_price": entry_stock_price,
                "current_delta": effective_delta,
                "raw_delta": entry_delta_raw,
                "signal_side": option_side,
                "option_qty": option_qty,
                "stock_position": stock_pos,
                "num_contracts": position_num_contracts,
                "expiry": expiry,
                "strike": strike,
                "dte_at_entry": dte_entry,
                "days_held": 0,
                "cumulative_pnl": 0.0,
                "last_hedge_adjustment": 0.0,
            }

            entry_cost = (
                entry_opt_price * CONTRACT_SIZE * abs(option_qty)
                + commission_per_contract * abs(option_qty)
            )
            stop_loss_pnl_threshold = (
                -abs(float(stop_loss_frac_of_entry_cost)) * entry_cost
                if stop_loss_frac_of_entry_cost is not None
                else None
            )
            open_positions[ticker]["stop_loss_pnl_threshold"] = stop_loss_pnl_threshold

            trade_log.append({
                "action": "enter",
                "date": current_ts,
                "ticker": ticker,
                "signal_date": queued["signal_date"],
                "option_price": entry_opt_price,
                "stock_price": entry_stock_price,
                "delta": effective_delta,
                "delta_raw": entry_delta_raw,
                "signal_side": option_side,
                "option_qty": option_qty,
                "stock_position": stock_pos,
                "num_contracts": position_num_contracts,
                "strike": strike,
                "expiry": expiry,
                "dte": dte_entry,
                "entry_cost": entry_cost,
                "rank": queued.get("rank"),
                "prediction": queued.get("prediction"),
                "target_weight": queued.get("target_weight"),
                "target_delta_exposure": queued.get("target_delta_exposure"),
                "per_contract_delta_exposure": queued.get("per_contract_delta_exposure"),
            })

        entry_queue = remaining_queue

        # ================================================================
        # Step 2: Today's top-K for signal-exit check and new entry queue
        # ================================================================
        today_signals = ranked[ranked["date"] == current_ts]
        today_top_k_tickers = set(today_signals["ticker"].tolist()) if not today_signals.empty else set()
        today_signal_sides: dict[str, set[int]] = {}
        if not today_signals.empty:
            for row in today_signals.itertuples(index=False):
                _t = str(getattr(row, "ticker"))
                _s = int(np.sign(getattr(row, signal_side_col, 1)) or 1)
                today_signal_sides.setdefault(_t, set()).add(_s)

        # ================================================================
        # Step 3: Update open positions — rebalance, P&L, exit checks
        # ================================================================
        positions_to_exit: list[tuple[str, str, float, float]] = []

        for ticker, pos in list(open_positions.items()):
            opt_row = _lookup_option(
                options_indexed, ticker, current_ts, pos["expiry"],
                _opt_lookup=_opt_lookup,
            )
            if opt_row is None:
                open_positions[ticker]["days_held"] += 1
                # Still enforce exit rules even without fresh option data (use last known prices)
                sl_thr = open_positions[ticker].get("stop_loss_pnl_threshold")
                if sl_thr is not None and open_positions[ticker]["cumulative_pnl"] <= sl_thr:
                    positions_to_exit.append(
                        (ticker, EXIT_REASON_STOP_LOSS, pos["prev_option_price"], pos["prev_stock_price"])
                    )
                elif open_positions[ticker]["days_held"] >= max_holding_days:
                    positions_to_exit.append(
                        (ticker, EXIT_REASON_HPR, pos["prev_option_price"], pos["prev_stock_price"])
                    )
                elif (
                    (not earnings_cycle_mode)
                    and use_signal_exit
                    and not today_signals.empty
                    and (
                        ticker not in today_top_k_tickers
                        or int(np.sign(pos.get("signal_side", 1)) or 1)
                        not in today_signal_sides.get(ticker, set())
                    )
                ):
                    positions_to_exit.append(
                        (ticker, EXIT_REASON_SIGNAL, pos["prev_option_price"], pos["prev_stock_price"])
                    )
                continue

            curr_opt_price = _f(opt_row["mid_price"])
            curr_stock_price = _f(opt_row["underlying_price"])
            curr_delta_raw = _f(opt_row["delta"])
            curr_delta = float(curr_delta_raw) * float(pos.get("signal_side", 1))
            curr_dte = int(opt_row.get("dte", 999))

            adj_shares = hedge_adjustment(
                pos["current_delta"], curr_delta, pos["num_contracts"]
            )

            pnl = daily_pnl(
                option_price_prev=pos["prev_option_price"],
                option_price_curr=curr_opt_price,
                stock_price_prev=pos["prev_stock_price"],
                stock_price_curr=curr_stock_price,
                stock_position=pos["stock_position"],
                num_contracts=pos.get("option_qty", pos["num_contracts"]),
                hedge_adjustment_shares=adj_shares,
                half_spread_pct_stock=half_spread_pct_stock,
            )

            daily_portfolio_pnl += pnl["total_pnl"]
            open_positions[ticker]["cumulative_pnl"] += pnl["total_pnl"]
            open_positions[ticker] = rebalance_position(
                open_positions[ticker], curr_delta, curr_stock_price
            )
            open_positions[ticker]["prev_option_price"] = curr_opt_price
            open_positions[ticker]["prev_stock_price"] = curr_stock_price
            open_positions[ticker]["raw_delta"] = curr_delta_raw
            open_positions[ticker]["days_held"] += 1

            position_snapshots.append({
                "date": current_ts,
                "ticker": ticker,
                "option_price": curr_opt_price,
                "stock_price": curr_stock_price,
                "delta": curr_delta,
                "delta_raw": curr_delta_raw,
                "signal_side": open_positions[ticker].get("signal_side", 1),
                "option_qty": open_positions[ticker].get("option_qty", open_positions[ticker]["num_contracts"]),
                "num_contracts": open_positions[ticker]["num_contracts"],
                "stock_position": open_positions[ticker]["stock_position"],
                "dte": curr_dte,
                "days_held": open_positions[ticker]["days_held"],
                "bars_held": open_positions[ticker]["days_held"],
                "daily_pnl": pnl["total_pnl"],
                "cumulative_pnl": open_positions[ticker]["cumulative_pnl"],
                "option_pnl": pnl["option_pnl"],
                "stock_pnl": pnl["stock_pnl"],
            })

            # Exit condition checks
            exit_reason: str | None = None
            sl_thr = open_positions[ticker].get("stop_loss_pnl_threshold")
            if sl_thr is not None and open_positions[ticker]["cumulative_pnl"] <= sl_thr:
                exit_reason = EXIT_REASON_STOP_LOSS
            elif earnings_cycle_mode:
                if open_positions[ticker]["days_held"] >= max_holding_days:
                    exit_reason = EXIT_REASON_HPR
            else:
                if curr_dte < exit_dte_threshold:
                    exit_reason = EXIT_REASON_DTE
                elif (
                    use_signal_exit
                    and not today_signals.empty
                    and (
                        ticker not in today_top_k_tickers
                        or int(np.sign(pos.get("signal_side", 1)) or 1)
                        not in today_signal_sides.get(ticker, set())
                    )
                ):
                    exit_reason = EXIT_REASON_SIGNAL
                elif open_positions[ticker]["days_held"] >= max_holding_days:
                    exit_reason = EXIT_REASON_HPR

            if exit_reason:
                positions_to_exit.append(
                    (ticker, exit_reason, curr_opt_price, curr_stock_price)
                )

        exited_any_today = bool(positions_to_exit)
        for ticker, reason, exit_opt_price, exit_stock_price in positions_to_exit:
            pos = open_positions.pop(ticker)
            if earnings_cycle_mode:
                next_allowed_signal_date[ticker] = _next_signal_date_for_ticker(
                    ticker=ticker,
                    after_date=pd.Timestamp(pos["signal_date"]),
                )
            ep = exit_pnl(
                entry_option_price=pos["entry_option_price"],
                exit_option_price=exit_opt_price,
                entry_stock_price=pos["entry_stock_price"],
                exit_stock_price=exit_stock_price,
                stock_position=pos["stock_position"],
                num_contracts=pos.get("option_qty", pos["num_contracts"]),
                commission_per_contract=commission_per_contract,
                half_spread_pct_option=half_spread_pct_option,
                half_spread_pct_stock=half_spread_pct_stock,
            )
            trade_log.append({
                "action": "exit",
                "date": current_ts,
                "ticker": ticker,
                "exit_reason": reason,
                "option_price": exit_opt_price,
                "stock_price": exit_stock_price,
                "signal_side": pos.get("signal_side", 1),
                "option_qty": pos.get("option_qty", pos["num_contracts"]),
                "num_contracts": pos["num_contracts"],
                "days_held": pos["days_held"],
                "bars_held": pos["days_held"],
                "realized_pnl": ep["total_pnl"],
                "exit_cost": ep["exit_cost"],
            })

        # ================================================================
        # Step 4: Queue new entries from today's top-K for tomorrow
        # ================================================================
        can_queue_today = not today_signals.empty
        if earnings_cycle_mode and (len(open_positions) > 0 or len(entry_queue) > 0 or exited_any_today):
            can_queue_today = False

        if can_queue_today:
            next_opt_date = pred_to_next_opt.get(current_ts)
            if next_opt_date is not None:
                candidate_entries: list[dict[str, Any]] = []
                for _, sig_row in today_signals.iterrows():
                    ticker = str(sig_row["ticker"])
                    if ticker in open_positions:
                        continue

                    pred_val = float(sig_row.get("prediction", np.nan))
                    signal_side = int(np.sign(sig_row.get(signal_side_col, 1)) or 1)
                    if not allow_short_signals:
                        signal_side = 1
                    if entry_prediction_threshold is not None:
                        thr_probe = abs(pred_val) if allow_short_signals else pred_val
                        if pd.isna(thr_probe) or thr_probe < float(entry_prediction_threshold):
                            continue

                    if earnings_cycle_mode:
                        allowed_signal_date = next_allowed_signal_date.get(ticker)
                        if allowed_signal_date is not None and current_ts < allowed_signal_date:
                            continue

                    opt_row = _lookup_option(
                        options_indexed,
                        ticker,
                        next_opt_date,
                        expiry=None,
                        dte_min=dte_min,
                        dte_max=dte_max,
                        moneyness_min=entry_moneyness_min,
                        moneyness_max=entry_moneyness_max,
                        min_open_interest=entry_min_open_interest,
                        max_spread_frac=entry_max_spread_frac,
                        _opt_lookup=_opt_lookup,
                    )
                    if opt_row is None:
                        continue

                    candidate_entries.append({
                        "sig_row": sig_row,
                        "ticker": ticker,
                        "prediction": pred_val,
                        "signal_side": signal_side,
                        "opt_row": opt_row,
                    })

                if candidate_entries:
                    equity_for_sizing = max(running_equity + daily_portfolio_pnl, 0.0)
                    signal_weights: dict[str, float] = {}
                    gross_target_dollars = float("nan")
                    if sizing_mode == "risk":
                        candidate_df = pd.DataFrame(
                            [
                                {
                                    "ticker": c["ticker"],
                                    "prediction": c["prediction"],
                                    signal_side_col: c["signal_side"],
                                }
                                for c in candidate_entries
                            ]
                        )
                        signal_weights = _compute_signal_weights(
                            candidate_df,
                            signal_date=current_ts,
                            allow_shorts=allow_short_signals,
                            side_col=signal_side_col,
                            use_vol_scaling=risk_use_vol_scaling,
                            vol_lookup=vol_lookup,
                            vol_floor=risk_vol_floor,
                        )
                        gross_target_dollars = equity_for_sizing * risk_target_gross_exposure_frac
                        if risk_target_daily_vol is not None and risk_target_daily_vol > 0.0:
                            weighted_var = 0.0
                            for t, w in signal_weights.items():
                                sigma = vol_lookup.get((pd.Timestamp(current_ts), str(t)), np.nan)
                                if not np.isfinite(sigma) or sigma <= 0.0:
                                    sigma = risk_vol_floor
                                weighted_var += (abs(float(w)) * float(sigma)) ** 2
                            unit_portfolio_vol = float(np.sqrt(weighted_var))
                            if np.isfinite(unit_portfolio_vol) and unit_portfolio_vol > 0.0:
                                gross_from_risk = (
                                    equity_for_sizing * float(risk_target_daily_vol) / unit_portfolio_vol
                                )
                                if risk_target_gross_exposure_frac > 0.0:
                                    gross_target_dollars = min(gross_target_dollars, gross_from_risk)
                                else:
                                    gross_target_dollars = gross_from_risk

                    for cand in candidate_entries:
                        sig_row = cand["sig_row"]
                        ticker = cand["ticker"]
                        pred_val = cand["prediction"]
                        signal_side = int(np.sign(cand.get("signal_side", 1)) or 1)
                        opt_row = cand["opt_row"]

                        entry_contracts = int(num_contracts)
                        target_weight = np.nan
                        target_delta_exposure = np.nan
                        per_contract_delta_exposure = np.nan

                        if sizing_mode == "risk":
                            target_weight = float(signal_weights.get(ticker, 0.0))
                            target_delta_exposure = abs(target_weight) * gross_target_dollars
                            if risk_max_name_exposure_frac is not None:
                                target_delta_exposure = min(
                                    target_delta_exposure,
                                    float(risk_max_name_exposure_frac) * equity_for_sizing,
                                )

                            entry_contracts, per_contract_delta_exposure = _contracts_for_target_delta_exposure(
                                target_delta_exposure=target_delta_exposure,
                                option_delta=_f(opt_row.get("delta")),
                                underlying_price=_f(opt_row.get("underlying_price")),
                            )

                            if risk_max_option_oi_frac is not None:
                                oi = _f(opt_row.get("open_interest"))
                                if np.isfinite(oi) and oi > 0:
                                    liq_cap = int(np.floor(float(oi) * float(risk_max_option_oi_frac)))
                                    entry_contracts = min(entry_contracts, max(liq_cap, 0))

                            if risk_max_contracts_per_trade is not None:
                                entry_contracts = min(entry_contracts, int(risk_max_contracts_per_trade))

                            if entry_contracts < 1:
                                continue

                        entry_queue.append({
                            "signal_date": current_ts,
                            "entry_date": pd.Timestamp(next_opt_date),
                            "ticker": ticker,
                            "rank": int(sig_row.get("rank", 0)),
                            "prediction": pred_val,
                            "signal_side": signal_side,
                            "expiry": pd.Timestamp(opt_row["exdate"]),
                            "num_contracts": int(entry_contracts),
                            "target_weight": target_weight,
                            "target_delta_exposure": target_delta_exposure,
                            "per_contract_delta_exposure": per_contract_delta_exposure,
                        })

        # ================================================================
        # Step 5: Record daily portfolio state
        # ================================================================
        daily_rows.append({
            "date": current_ts,
            "daily_pnl": daily_portfolio_pnl,
            "equity": running_equity + daily_portfolio_pnl,
            "n_open_positions": len(open_positions),
            "open_tickers": sorted(open_positions.keys()),
        })
        running_equity = max(0.0, running_equity + daily_portfolio_pnl)

    print(
        f"[backtest] Loop complete. Dates processed: {len(all_dates):,}  "
        f"Trade log rows: {len(trade_log)}  "
        f"Option lookup misses (no data found): {_LOOKUP_MISSES:,}",
        flush=True,
    )

    # --- Post-processing ---
    daily_df = pd.DataFrame(daily_rows).set_index("date")
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    trade_book_df = _build_round_trip_trade_book(trade_df)
    snapshot_df = pd.DataFrame(position_snapshots) if position_snapshots else pd.DataFrame()

    eq = equity_curve(daily_df["daily_pnl"], initial_capital)
    metrics = compute_metrics(eq)
    dd = drawdown_series(eq)

    n_entered = int((trade_df["action"] == "enter").sum()) if not trade_df.empty else 0
    n_exited = int((trade_df["action"] == "exit").sum()) if not trade_df.empty else 0

    exits_df = trade_df[trade_df["action"] == "exit"] if not trade_df.empty else pd.DataFrame()
    avg_bars = float(exits_df["days_held"].mean()) if not exits_df.empty and "days_held" in exits_df.columns else float("nan")
    max_bars = int(exits_df["days_held"].max()) if not exits_df.empty and "days_held" in exits_df.columns else 0
    exit_reason_counts = exits_df["exit_reason"].value_counts().to_dict() if not exits_df.empty else {}

    print("\n[backtest] === Results ===")
    print(f"  Period:         {daily_df.index.min().date()} → {daily_df.index.max().date()}")
    print(f"  Total return:   {metrics['total_return']:.2%}")
    print(f"  Sharpe ratio:   {metrics['sharpe']:.3f}")
    print(f"  Max drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"  Hit rate:       {metrics['hit_rate']:.2%}")
    print(f"  Trades entered: {n_entered}")
    print(f"  Trades exited:  {n_exited}")
    print(f"  Avg bars held:  {avg_bars:.1f}")
    print(f"  Max bars held:  {max_bars}")
    print("  Note: 'bars held' are trading sessions in this backtest timeline, not calendar days.")
    print(f"  Exit reasons:   {exit_reason_counts}")

    return {
        "equity_curve": eq,
        "daily_pnl_df": daily_df,
        "trade_log": trade_df,
        "trade_book": trade_book_df,
        "position_log": snapshot_df,
        "metrics": metrics,
        "drawdown": dd,
    }


def evaluate_performance(
    results: dict[str, Any],
    prices_df: pd.DataFrame | None = None,
    universe_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Compute strategy performance vs benchmarks and return a comparison table.

    Parameters
    ----------
    results : Output dict from run_backtest.
    prices_df : Optional price panel for benchmark curves (date, ticker, adj_close).
    universe_tickers : Tickers for equal-weight universe benchmark.

    Returns
    -------
    dict with keys: metrics, benchmark_equities, performance_table.
    """
    from .performance import benchmark_equity_curve, build_performance_table

    eq = results["equity_curve"]
    start = eq.index.min()
    end = eq.index.max()

    benchmarks: dict[str, pd.Series] = {}

    if prices_df is not None:
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")
        ticker_col = "ticker" if "ticker" in prices_df.columns else None

        if ticker_col and "SPY" in prices_df[ticker_col].unique():
            benchmarks["SPY Buy & Hold"] = benchmark_equity_curve(
                prices_df, ["SPY"], start, end, initial_capital=float(eq.iloc[0])
            )

        if universe_tickers and ticker_col:
            avail = [t for t in universe_tickers if t in prices_df[ticker_col].unique()]
            if avail:
                benchmarks["Equal-Weight Universe"] = benchmark_equity_curve(
                    prices_df, avail, start, end, initial_capital=float(eq.iloc[0])
                )

    perf_table = build_performance_table(eq, benchmarks)

    print("\n[evaluate_performance] === Performance Table ===")
    print(perf_table.to_string())

    return {
        "metrics": results["metrics"],
        "benchmark_equities": benchmarks,
        "performance_table": perf_table,
    }


def optimize_backtest_grid(
    predictions_df: pd.DataFrame,
    options_df: pd.DataFrame,
    *,
    K: int = 3,
    top_k_df: pd.DataFrame | None = None,
    initial_capital: float = 100_000.0,
    hold_days_grid: list[int] | range = range(1, 21),
    stop_loss_grid: list[float | None] = (None, 0.10, 0.20, 0.30, 0.40),
    entry_threshold_grid: list[float | None] = (None, 0.0, 0.0005, 0.0010),
    dte_min: int = 30,
    dte_max: int = 45,
    exit_dte_threshold: int = 10,
    entry_moneyness_min: float | None = None,
    entry_moneyness_max: float | None = None,
    entry_min_open_interest: int | None = None,
    entry_max_spread_frac: float | None = None,
    dte_min_grid: list[int] | None = None,
    dte_max_grid: list[int] | None = None,
    exit_dte_threshold_grid: list[int] | None = None,
    entry_moneyness_min_grid: list[float | None] | None = None,
    entry_moneyness_max_grid: list[float | None] | None = None,
    entry_min_open_interest_grid: list[int | None] | None = None,
    entry_max_spread_frac_grid: list[float | None] | None = None,
    num_contracts: int = 1,
    commission_per_contract: float = 1.0,
    half_spread_pct_stock: float = 0.0005,
    half_spread_pct_option: float = 0.02,
    use_signal_exit: bool = False,
    earnings_cycle_mode: bool | None = True,
    allow_short_signals: bool = False,
    signal_side_col: str = "signal_side",
    sizing_mode: str = "fixed",
    risk_target_gross_exposure_frac: float = 0.10,
    risk_target_daily_vol: float | None = None,
    risk_max_name_exposure_frac: float | None = 0.05,
    risk_use_vol_scaling: bool = True,
    risk_vol_lookback_days: int = 21,
    risk_vol_floor: float = 0.01,
    risk_max_option_oi_frac: float | None = 0.01,
    risk_max_contracts_per_trade: int | None = None,
    drawdown_penalty: float = 0.25,
    suppress_run_output: bool = True,
) -> pd.DataFrame:
    """Run a parameter grid search over holding period, stop-loss and threshold.

    Returns a DataFrame with one row per parameter combination and summary metrics.
    The ``score`` column is a simple risk-adjusted objective:

    ``score = total_return_pct - drawdown_penalty * abs(max_drawdown_pct)``.
    """
    rows: list[dict[str, Any]] = []

    if dte_min_grid is None:
        dte_min_grid = [dte_min]
    if dte_max_grid is None:
        dte_max_grid = [dte_max]
    if exit_dte_threshold_grid is None:
        exit_dte_threshold_grid = [exit_dte_threshold]
    if entry_moneyness_min_grid is None:
        entry_moneyness_min_grid = [entry_moneyness_min]
    if entry_moneyness_max_grid is None:
        entry_moneyness_max_grid = [entry_moneyness_max]
    if entry_min_open_interest_grid is None:
        entry_min_open_interest_grid = [entry_min_open_interest]
    if entry_max_spread_frac_grid is None:
        entry_max_spread_frac_grid = [entry_max_spread_frac]

    for (
        hold_days,
        stop_loss_frac,
        entry_threshold,
        cfg_dte_min,
        cfg_dte_max,
        cfg_exit_dte,
        cfg_mny_min,
        cfg_mny_max,
        cfg_min_oi,
        cfg_max_spr,
    ) in product(
        list(hold_days_grid),
        list(stop_loss_grid),
        list(entry_threshold_grid),
        list(dte_min_grid),
        list(dte_max_grid),
        list(exit_dte_threshold_grid),
        list(entry_moneyness_min_grid),
        list(entry_moneyness_max_grid),
        list(entry_min_open_interest_grid),
        list(entry_max_spread_frac_grid),
    ):
        if cfg_dte_min is not None and cfg_dte_max is not None and int(cfg_dte_min) > int(cfg_dte_max):
            continue
        if suppress_run_output:
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_backtest(
                    predictions_df=predictions_df,
                    options_df=options_df,
                    K=K,
                    initial_capital=initial_capital,
                    max_holding_days=int(hold_days),
                    exit_dte_threshold=int(cfg_exit_dte),
                    num_contracts=num_contracts,
                    commission_per_contract=commission_per_contract,
                    half_spread_pct_stock=half_spread_pct_stock,
                    half_spread_pct_option=half_spread_pct_option,
                    dte_min=int(cfg_dte_min),
                    dte_max=int(cfg_dte_max),
                    entry_moneyness_min=cfg_mny_min,
                    entry_moneyness_max=cfg_mny_max,
                    entry_min_open_interest=cfg_min_oi,
                    entry_max_spread_frac=cfg_max_spr,
                    use_signal_exit=use_signal_exit,
                    top_k_df=top_k_df,
                    earnings_cycle_mode=earnings_cycle_mode,
                    entry_prediction_threshold=entry_threshold,
                    stop_loss_frac_of_entry_cost=stop_loss_frac,
                    allow_short_signals=allow_short_signals,
                    signal_side_col=signal_side_col,
                    sizing_mode=sizing_mode,
                    risk_target_gross_exposure_frac=risk_target_gross_exposure_frac,
                    risk_target_daily_vol=risk_target_daily_vol,
                    risk_max_name_exposure_frac=risk_max_name_exposure_frac,
                    risk_use_vol_scaling=risk_use_vol_scaling,
                    risk_vol_lookback_days=risk_vol_lookback_days,
                    risk_vol_floor=risk_vol_floor,
                    risk_max_option_oi_frac=risk_max_option_oi_frac,
                    risk_max_contracts_per_trade=risk_max_contracts_per_trade,
                )
        else:
            res = run_backtest(
                predictions_df=predictions_df,
                options_df=options_df,
                K=K,
                initial_capital=initial_capital,
                max_holding_days=int(hold_days),
                exit_dte_threshold=int(cfg_exit_dte),
                num_contracts=num_contracts,
                commission_per_contract=commission_per_contract,
                half_spread_pct_stock=half_spread_pct_stock,
                half_spread_pct_option=half_spread_pct_option,
                dte_min=int(cfg_dte_min),
                dte_max=int(cfg_dte_max),
                entry_moneyness_min=cfg_mny_min,
                entry_moneyness_max=cfg_mny_max,
                entry_min_open_interest=cfg_min_oi,
                entry_max_spread_frac=cfg_max_spr,
                use_signal_exit=use_signal_exit,
                top_k_df=top_k_df,
                earnings_cycle_mode=earnings_cycle_mode,
                entry_prediction_threshold=entry_threshold,
                stop_loss_frac_of_entry_cost=stop_loss_frac,
                allow_short_signals=allow_short_signals,
                signal_side_col=signal_side_col,
                sizing_mode=sizing_mode,
                risk_target_gross_exposure_frac=risk_target_gross_exposure_frac,
                risk_target_daily_vol=risk_target_daily_vol,
                risk_max_name_exposure_frac=risk_max_name_exposure_frac,
                risk_use_vol_scaling=risk_use_vol_scaling,
                risk_vol_lookback_days=risk_vol_lookback_days,
                risk_vol_floor=risk_vol_floor,
                risk_max_option_oi_frac=risk_max_option_oi_frac,
                risk_max_contracts_per_trade=risk_max_contracts_per_trade,
            )

        metrics = res["metrics"]
        trade_df = res["trade_log"]
        exits_df = trade_df[trade_df["action"] == "exit"] if not trade_df.empty else pd.DataFrame()

        total_return_pct = float(metrics.get("total_return", np.nan)) * 100.0
        max_drawdown_pct = float(metrics.get("max_drawdown", np.nan)) * 100.0
        sharpe = float(metrics.get("sharpe", np.nan))
        score = total_return_pct - drawdown_penalty * abs(max_drawdown_pct)

        rows.append({
            "hold_days": int(hold_days),
            "stop_loss_frac": stop_loss_frac,
            "entry_threshold": entry_threshold,
            "dte_min": int(cfg_dte_min),
            "dte_max": int(cfg_dte_max),
            "exit_dte_threshold": int(cfg_exit_dte),
            "entry_moneyness_min": cfg_mny_min,
            "entry_moneyness_max": cfg_mny_max,
            "entry_min_open_interest": cfg_min_oi,
            "entry_max_spread_frac": cfg_max_spr,
            "total_return_pct": total_return_pct,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "hit_rate_pct": float(metrics.get("hit_rate", np.nan)) * 100.0,
            "entries": int((trade_df["action"] == "enter").sum()) if not trade_df.empty else 0,
            "exits": int((trade_df["action"] == "exit").sum()) if not trade_df.empty else 0,
            "avg_days_held": float(exits_df["days_held"].mean()) if not exits_df.empty else np.nan,
            "avg_bars_held": float(exits_df["days_held"].mean()) if not exits_df.empty else np.nan,
            "max_days_held": int(exits_df["days_held"].max()) if not exits_df.empty else 0,
            "max_bars_held": int(exits_df["days_held"].max()) if not exits_df.empty else 0,
            "stop_loss_exit_count": int((exits_df["exit_reason"] == EXIT_REASON_STOP_LOSS).sum()) if not exits_df.empty else 0,
            "score": score,
        })

    return pd.DataFrame(rows)


def select_best_backtest_config(
    grid_df: pd.DataFrame,
    *,
    strict_only: bool = False,
    max_stop_loss_frac: float | None = None,
    min_entry_threshold: float | None = None,
    min_hold_days: int = 1,
    max_hold_days: int | None = None,
    min_exits: int = 1,
    min_avg_days_held: float | None = None,
    min_avg_bars_held: float | None = None,
    prefer_fast_exit: bool = True,
    prefer_tighter_stop: bool = True,
    prefer_higher_threshold: bool = True,
) -> dict[str, Any] | None:
    """Select a best config row from a grid search result DataFrame.

    This applies optional constraints and then uses a deterministic sort order
    anchored on ``score`` and ``total_return_pct``.
    """
    if grid_df.empty:
        return None

    cand = grid_df.copy()
    cand = cand[cand["hold_days"] >= int(min_hold_days)]
    if max_hold_days is not None:
        cand = cand[cand["hold_days"] <= int(max_hold_days)]

    if strict_only:
        cand = cand[cand["stop_loss_frac"].notna() & cand["entry_threshold"].notna()]

    if max_stop_loss_frac is not None:
        cand = cand[cand["stop_loss_frac"].notna() & (cand["stop_loss_frac"] <= float(max_stop_loss_frac))]

    if min_entry_threshold is not None:
        cand = cand[cand["entry_threshold"].notna() & (cand["entry_threshold"] >= float(min_entry_threshold))]

    cand = cand[cand["exits"] >= int(min_exits)]
    avg_bar_constraint = min_avg_bars_held if min_avg_bars_held is not None else min_avg_days_held
    if avg_bar_constraint is not None:
        col = "avg_bars_held" if "avg_bars_held" in cand.columns else "avg_days_held"
        cand = cand[cand[col].fillna(0.0) >= float(avg_bar_constraint)]

    if cand.empty:
        return None

    cand = cand.copy()
    cand["sort_hold"] = cand["hold_days"]
    cand["sort_stop"] = cand["stop_loss_frac"].fillna(np.inf if prefer_tighter_stop else -np.inf)
    cand["sort_threshold"] = cand["entry_threshold"].fillna(-np.inf if prefer_higher_threshold else np.inf)
    cand["sort_sharpe"] = cand["sharpe"].fillna(-np.inf)

    sort_cols = [
        "score",
        "total_return_pct",
        "max_drawdown_pct",
        "sort_sharpe",
        "sort_hold",
        "sort_stop",
        "sort_threshold",
    ]
    sort_asc = [
        False,  # score higher is better
        False,  # return higher is better
        False,  # drawdown closer to 0 is better (less negative)
        False,  # sharpe higher is better
        prefer_fast_exit,            # lower hold if prefer fast exit
        prefer_tighter_stop,         # lower stop-loss fraction if tighter preferred
        not prefer_higher_threshold, # higher threshold if stricter entry preferred
    ]

    best = cand.sort_values(sort_cols, ascending=sort_asc).iloc[0].to_dict()
    return {k: v for k, v in best.items() if not str(k).startswith("sort_")}
