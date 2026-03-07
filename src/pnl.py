# Stage 12 — P&L Computation
# Daily mark-to-market P&L for delta-hedged option positions.
#
# Two P&L sources per position each day:
#   1. Option price change:  (mid_t - mid_{t-1}) * CONTRACT_SIZE * signed_contracts
#   2. Stock hedge P&L:      stock_position * (S_t - S_{t-1})
#      (stock_position is negative for short, so stock gains hurt)
#
# Transaction costs applied on hedge rebalance (bid-ask half-spread on shares traded).
# Exit costs include: option spread cost + stock cover cost + commission.
#
# The strategy profits from gamma (large moves in either direction exceed
# the theta decay cost) and from implied vol expansion.

from __future__ import annotations

import numpy as np

from .hedge import CONTRACT_SIZE


def daily_pnl(
    option_price_prev: float,
    option_price_curr: float,
    stock_price_prev: float,
    stock_price_curr: float,
    stock_position: float,
    num_contracts: int = 1,
    hedge_adjustment_shares: float = 0.0,
    half_spread_pct_stock: float = 0.0005,
) -> dict[str, float]:
    """Compute mark-to-market P&L for one day on an open delta-hedged call position.

    Parameters
    ----------
    option_price_prev : Option mid_price at previous day's close.
    option_price_curr : Option mid_price at today's close.
    stock_price_prev : Underlying price at previous day's close.
    stock_price_curr : Underlying price at today's close.
    stock_position : Current short stock position (negative shares) BEFORE today's rebalance.
    num_contracts : Signed contracts held (+long / -short).
    hedge_adjustment_shares : Shares traded today for delta rebalance (from hedge.py).
    half_spread_pct_stock : Half bid-ask spread as fraction of stock price for rebalance cost.

    Returns
    -------
    dict with keys: option_pnl, stock_pnl, rebalance_cost, total_pnl.
    """
    option_pnl = (option_price_curr - option_price_prev) * CONTRACT_SIZE * num_contracts
    stock_pnl = stock_position * (stock_price_curr - stock_price_prev)
    rebalance_cost = abs(hedge_adjustment_shares) * stock_price_curr * half_spread_pct_stock

    return {
        "option_pnl": float(option_pnl),
        "stock_pnl": float(stock_pnl),
        "rebalance_cost": float(rebalance_cost),
        "total_pnl": float(option_pnl + stock_pnl - rebalance_cost),
    }


def exit_pnl(
    entry_option_price: float,
    exit_option_price: float,
    entry_stock_price: float,
    exit_stock_price: float,
    stock_position: float,
    num_contracts: int = 1,
    commission_per_contract: float = 1.0,
    half_spread_pct_option: float = 0.02,
    half_spread_pct_stock: float = 0.0005,
) -> dict[str, float]:
    """Compute realized P&L when closing a position.

    Parameters
    ----------
    entry_option_price : Option mid_price at entry.
    exit_option_price : Option mid_price at exit.
    entry_stock_price : Underlying at entry.
    exit_stock_price : Underlying at exit.
    stock_position : Final short stock position to cover (negative shares).
    num_contracts : Signed contracts held (+long / -short).
    commission_per_contract : Flat per-contract commission (both legs).
    half_spread_pct_option : Half option spread as fraction of option price.
    half_spread_pct_stock : Half stock spread as fraction of stock price.

    Returns
    -------
    dict with keys: option_pnl, stock_pnl, exit_cost, total_pnl.
    """
    option_pnl = (exit_option_price - entry_option_price) * CONTRACT_SIZE * num_contracts
    stock_pnl = stock_position * (exit_stock_price - entry_stock_price)
    contracts_abs = abs(float(num_contracts))

    # Closing costs: sell option (spread + commission) + cover short (spread)
    option_exit_cost = (
        exit_option_price * half_spread_pct_option * CONTRACT_SIZE * contracts_abs
        + commission_per_contract * contracts_abs
    )
    stock_exit_cost = abs(stock_position) * exit_stock_price * half_spread_pct_stock
    total_exit_cost = option_exit_cost + stock_exit_cost

    return {
        "option_pnl": float(option_pnl),
        "stock_pnl": float(stock_pnl),
        "exit_cost": float(total_exit_cost),
        "total_pnl": float(option_pnl + stock_pnl - total_exit_cost),
    }
