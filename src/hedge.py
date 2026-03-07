# Stage 10 — Delta Hedge
# Each option position is delta-hedged by trading delta * CONTRACT_SIZE shares.
# The hedge is rebalanced daily as delta drifts with the underlying price.
# Net delta ≈ 0 isolates gamma/vega exposure rather than directional equity risk.
#
# Hedge mechanics:
#   stock_position = -effective_delta * CONTRACT_SIZE * num_contracts
# where effective_delta is signed (+ for long call, - for short call).
#   Daily adjustment = -(new_delta - old_delta) * CONTRACT_SIZE * num_contracts
#
# Convention: stock_position < 0 means short; adjustment < 0 means sell more shares.

from __future__ import annotations

CONTRACT_SIZE: int = 100
"""Shares per standard option contract."""


def initial_stock_position(delta: float, num_contracts: int = 1) -> float:
    """Compute initial stock position to delta-neutralize an option position.

    Parameters
    ----------
    delta : Signed effective delta at entry (e.g. +0.52 long call, -0.52 short call).
    num_contracts : Number of call contracts held long.

    Returns
    -------
    Shares to short (negative float). Example: delta=0.52 → -52.0 shares.
    """
    return -float(delta) * CONTRACT_SIZE * num_contracts


def hedge_adjustment(
    old_delta: float,
    new_delta: float,
    num_contracts: int = 1,
) -> float:
    """Additional shares to trade to restore delta neutrality after delta drift.

    Parameters
    ----------
    old_delta : Delta from previous day.
    new_delta : Delta today (updated from options data).
    num_contracts : Contracts held.

    Returns
    -------
    Shares to short (negative) or cover (positive).
    Example: old_delta=0.52, new_delta=0.57 → -5.0 (short 5 more shares).
    """
    return -(float(new_delta) - float(old_delta)) * CONTRACT_SIZE * num_contracts


def rebalance_position(position: dict, new_delta: float, new_stock_price: float) -> dict:
    """Apply daily delta rebalance to a position state dictionary in-place copy.

    Updates: stock_position, current_delta, last_hedge_adjustment.
    Does NOT update option price or P&L — those are handled in pnl.py.

    Parameters
    ----------
    position : Position state dict (see backtest_utils.py for schema).
    new_delta : Today's option delta from options data.
    new_stock_price : Today's underlying price (for logging).

    Returns
    -------
    Updated copy of position dict.
    """
    adj = hedge_adjustment(position["current_delta"], new_delta, position["num_contracts"])
    pos = position.copy()
    pos["stock_position"] = position["stock_position"] + adj
    pos["current_delta"] = float(new_delta)
    pos["last_hedge_adjustment"] = float(adj)
    return pos
