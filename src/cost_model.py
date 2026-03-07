"""
Cost Model — OOA Framework (Mykland §2–3)
==========================================
Transaction cost attribution for delta-hedged options strategy.

Per Mykland OOA lecture notes:
  c_i = spread_cost + commission            (§2.2.5, §3.1)
  PnL_h(t_i) = q_i * (p_h_i - p_i) * multiplier - c_entry - c_exit    (§3.2)
  Slip_h(t_i) = sign(q_i) * (execution_price - decision_price)          (§3.3)

Aggressive orders (market orders) pay half the bid-ask spread.
Passive orders (limit orders) earn a rebate equal to half the spread.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def compute_trade_costs(
    trade_type: str,
    quantity: float,
    price: float,
    bid_ask_spread_frac: float,
    order_side: str = "aggressive",
    num_contracts: int = 1,
    commission_per_contract: float = 1.00,
    commission_per_share: float = 0.005,
    min_stock_commission: float = 1.00,
    contract_multiplier: int = 100,
) -> dict:
    """
    Compute all-in transaction cost for one fill (OOA §2.2.5, §3.1).

    Parameters
    ----------
    trade_type : "option_entry" | "option_exit" | "stock_entry" | "stock_exit"
    quantity   : Number of contracts (options) or shares (stock). Positive.
    price      : Execution mid price.
    bid_ask_spread_frac : (ask - bid) / mid — full relative spread.
                          Function halves it internally to get the half-spread cost.
    order_side : "aggressive" (pays half-spread) | "passive" (earns half-spread rebate).
    num_contracts : Number of option contracts (used for option legs).
    commission_per_contract : $ flat per options contract (one-way).
    commission_per_share : $ per share for stock leg.
    min_stock_commission : $ minimum per stock order.
    contract_multiplier : 100 for standard equity options, 1 for stock.

    Returns
    -------
    dict with keys:
        spread_cost : half-spread cost (negative if passive rebate)
        commission  : fixed per-contract or per-share commission
        total_cost  : max(0, spread_cost + commission)
        cost_bps    : total_cost / (price * quantity * multiplier) * 10000
        order_side  : echoed back
    """
    half_spread = bid_ask_spread_frac / 2.0  # half-spread from full spread

    if trade_type in ("option_entry", "option_exit"):
        # Spread cost: half-spread * mid * multiplier * contracts
        spread_cost = half_spread * price * contract_multiplier * num_contracts
        if order_side == "passive":
            spread_cost = -spread_cost  # rebate; will be capped at 0 in total_cost
        commission = commission_per_contract * num_contracts
        notional = price * num_contracts * contract_multiplier

    elif trade_type in ("stock_entry", "stock_exit"):
        spread_cost = half_spread * price * abs(quantity)
        if order_side == "passive":
            spread_cost = -spread_cost
        commission = max(min_stock_commission, commission_per_share * abs(quantity))
        notional = price * abs(quantity)

    else:
        raise ValueError(
            f"trade_type must be one of 'option_entry', 'option_exit', "
            f"'stock_entry', 'stock_exit'. Got: {trade_type!r}"
        )

    # Costs are non-negative after netting passive rebate against commission
    total_cost = max(0.0, spread_cost + commission)

    # Cost in basis points of notional
    if notional > 0:
        cost_bps = (total_cost / notional) * 10_000.0
    else:
        cost_bps = float("nan")

    return {
        "spread_cost": float(spread_cost),
        "commission": float(commission),
        "total_cost": float(total_cost),
        "cost_bps": float(cost_bps),
        "order_side": order_side,
    }


def compute_markout_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    multiplier: int,
    entry_cost: float,
    exit_cost: float,
) -> float:
    """
    OOA mark-out P&L (§3.2).

    PnL_h(t_i) = q_i * (p_h_i - p_i) * multiplier - entry_cost - exit_cost

    Signed quantity: positive for long, negative for short.
    """
    return float(
        quantity * (exit_price - entry_price) * multiplier - entry_cost - exit_cost
    )


def compute_slippage(
    execution_price: float,
    decision_price: float,
    signed_quantity: float,
) -> float:
    """
    OOA slippage (§3.3).

    Slip(t_i) = sign(q_i) * (execution_price - decision_price)

    Positive = paid more than decision price (adverse for buyer).
    """
    sign_q = 1.0 if signed_quantity > 0 else (-1.0 if signed_quantity < 0 else 0.0)
    return float(sign_q * (execution_price - decision_price))


def mark_trades(
    trade_log: pd.DataFrame,
    half_spread_pct_option: float = 0.02,
    half_spread_pct_stock: float = 0.0005,
    stock_order_side: str = "aggressive",
    commission_per_contract: float = 1.00,
    commission_per_share: float = 0.005,
    min_stock_commission: float = 1.00,
    num_contracts: int = 1,
    contract_multiplier: int = 100,
) -> pd.DataFrame:
    """
    Decorate trade_log with OOA cost marks (§2.2).

    For each matched entry/exit pair, adds:
        m_type           : "AGGRESSIVE" or "PASSIVE" for option leg
        m_spread_option  : relative half-spread for option leg (as fraction)
        m_spread_stock   : relative half-spread for stock leg (as fraction)
        cost_option      : all-in option fill cost ($) — entry + exit combined
        cost_stock       : all-in stock fill cost ($) — entry + exit combined
        cost_total       : cost_option + cost_stock
        cost_bps_option  : cost_option in bps of option notional
        slippage         : NaN (decision-price proxy unavailable; see docstring)
        net_pnl          : realized_pnl - cost_total  (on exit rows)
        profit_factor_input : net_pnl (signed; for profit factor aggregation)

    Note: slippage requires decision-price (mid at signal time). Since the
    trade_log does not carry the prior-day close of the option, slippage is
    set to NaN. A richer data source (e.g., NBBO quote at signal time) would
    be needed for accurate slippage attribution.

    Unmatched exit rows (no corresponding entry found) receive NaN costs
    and a logged warning rather than crashing.
    """
    if trade_log.empty:
        return trade_log.copy()

    result = trade_log.copy()

    # Initialise output columns on ALL rows (entry rows get NaN for cost/pnl fields)
    for col in [
        "m_type", "m_spread_option", "m_spread_stock",
        "cost_option", "cost_stock", "cost_total", "cost_bps_option",
        "slippage", "net_pnl", "profit_factor_input",
    ]:
        if col not in result.columns:
            result[col] = np.nan

    # Build entry lookup: ticker -> FIFO queue of entry row indices
    entry_rows = result[result["action"] == "enter"].copy()
    exit_rows  = result[result["action"] == "exit"].copy()

    # FIFO queue per ticker
    from collections import defaultdict, deque
    entry_queue: dict[str, deque] = defaultdict(deque)
    for idx, row in entry_rows.iterrows():
        entry_queue[str(row["ticker"])].append(idx)

    for ex_idx, ex_row in exit_rows.iterrows():
        ticker = str(ex_row["ticker"])
        queue = entry_queue.get(ticker)

        if not queue:
            warnings.warn(
                f"[mark_trades] No matching entry found for exit row idx={ex_idx} "
                f"ticker={ticker} date={ex_row.get('date', '?')}. "
                f"Costs set to NaN for this row.",
                stacklevel=2,
            )
            continue

        en_idx = queue.popleft()
        en_row = result.loc[en_idx]

        # --- Prices ---
        opt_price_entry = float(en_row.get("option_price", np.nan))
        opt_price_exit  = float(ex_row.get("option_price", np.nan))
        stk_price_entry = float(en_row.get("stock_price",  np.nan))
        stk_price_exit  = float(ex_row.get("stock_price",  np.nan))
        stock_pos       = float(en_row.get("stock_position", np.nan))

        # --- Option costs (entry + exit, both aggressive / market-taker) ---
        bid_ask_opt = half_spread_pct_option * 2  # full spread = 2 × half-spread

        opt_entry_result = compute_trade_costs(
            trade_type="option_entry",
            quantity=num_contracts,
            price=opt_price_entry if not np.isnan(opt_price_entry) else 0.0,
            bid_ask_spread_frac=bid_ask_opt,
            order_side="aggressive",
            num_contracts=num_contracts,
            commission_per_contract=commission_per_contract,
            contract_multiplier=contract_multiplier,
        )
        opt_exit_result = compute_trade_costs(
            trade_type="option_exit",
            quantity=num_contracts,
            price=opt_price_exit if not np.isnan(opt_price_exit) else 0.0,
            bid_ask_spread_frac=bid_ask_opt,
            order_side="aggressive",
            num_contracts=num_contracts,
            commission_per_contract=commission_per_contract,
            contract_multiplier=contract_multiplier,
        )
        cost_option = opt_entry_result["total_cost"] + opt_exit_result["total_cost"]

        # --- Stock costs (entry + exit) ---
        bid_ask_stk = half_spread_pct_stock * 2
        abs_shares = abs(stock_pos) if not np.isnan(stock_pos) else 0.0

        stk_entry_result = compute_trade_costs(
            trade_type="stock_entry",
            quantity=abs_shares,
            price=stk_price_entry if not np.isnan(stk_price_entry) else 0.0,
            bid_ask_spread_frac=bid_ask_stk,
            order_side=stock_order_side,
            commission_per_share=commission_per_share,
            min_stock_commission=min_stock_commission,
            contract_multiplier=1,
        )
        stk_exit_result = compute_trade_costs(
            trade_type="stock_exit",
            quantity=abs_shares,
            price=stk_price_exit if not np.isnan(stk_price_exit) else 0.0,
            bid_ask_spread_frac=bid_ask_stk,
            order_side=stock_order_side,
            commission_per_share=commission_per_share,
            min_stock_commission=min_stock_commission,
            contract_multiplier=1,
        )
        cost_stock = stk_entry_result["total_cost"] + stk_exit_result["total_cost"]

        cost_total = cost_option + cost_stock

        # cost_bps for option leg (entry notional)
        entry_notional = opt_price_entry * num_contracts * contract_multiplier
        cost_bps_opt = (cost_option / entry_notional * 10_000.0
                        if entry_notional > 0 else np.nan)

        # net_pnl
        realized_pnl = float(ex_row.get("realized_pnl", np.nan))
        net_pnl = realized_pnl - cost_total if not np.isnan(realized_pnl) else np.nan

        # Slippage: requires decision-price proxy (prior-day option close).
        # Not available in trade_log — documented as NaN.
        slippage = np.nan

        # Write to result DataFrame on the EXIT row
        result.loc[ex_idx, "m_type"]            = "AGGRESSIVE"
        result.loc[ex_idx, "m_spread_option"]   = half_spread_pct_option
        result.loc[ex_idx, "m_spread_stock"]    = half_spread_pct_stock
        result.loc[ex_idx, "cost_option"]       = cost_option
        result.loc[ex_idx, "cost_stock"]        = cost_stock
        result.loc[ex_idx, "cost_total"]        = cost_total
        result.loc[ex_idx, "cost_bps_option"]   = cost_bps_opt
        result.loc[ex_idx, "slippage"]          = slippage
        result.loc[ex_idx, "net_pnl"]           = net_pnl
        result.loc[ex_idx, "profit_factor_input"] = net_pnl

    # --- Assertions ---
    exit_mask = result["action"] == "exit"
    _costs = result.loc[exit_mask, "cost_total"].dropna()
    if not (_costs >= 0).all():
        neg_count = (_costs < 0).sum()
        raise AssertionError(
            f"[mark_trades] {neg_count} exit rows have cost_total < 0. "
            "Check spread/commission parameters."
        )

    # Profitable gross trades should have net_pnl < realized_pnl (costs always drag)
    _ex = result[exit_mask].copy()
    _profitable = _ex["realized_pnl"].gt(0) & _ex["net_pnl"].notna() & _ex["realized_pnl"].notna()
    if _profitable.any():
        _bad = _ex[_profitable & (_ex["net_pnl"] > _ex["realized_pnl"] + 1e-8)]
        if not _bad.empty:
            raise AssertionError(
                f"[mark_trades] {len(_bad)} rows where net_pnl > realized_pnl "
                "for profitable trades. Costs must always reduce PnL."
            )

    return result
