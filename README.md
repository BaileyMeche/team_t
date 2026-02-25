# Team T: Treasury vs Synthetic Nominal Yield Carry Trade (Levered Relative-Value Arbitrage)

## 1. Executive Summary
This project proposes a **levered, relative-value carry strategy** that exploits a persistent wedge between:

- **Nominal U.S. Treasury yields** at key tenors \(\tau \in \{2,5,10,20\}\), and
- A **synthetic nominal yield** constructed from a **TIPS real yield** plus the corresponding **inflation swap rate**.

The core claim is that a **persistent nominal–synthetic basis** exists and is tradeable via a hedged long/short structure. Profitability arises from (i) **basis convergence/mean reversion**, (ii) **net carry / roll-down** while convergence occurs, and (iii) the ability to apply **bounded leverage** under transparent risk controls and realistic funding and transaction-cost assumptions.

This approach naturally supports holding **multiple distinct instruments simultaneously** (across tenors and legs) and is designed to produce sufficient trade frequency through a disciplined entry/exit “recipe.”

---

## 2. Trade Definition and Intuition

### 2.1 Synthetic nominal yield
For each tenor \(\tau\), define:
\[
y^{S}_{\tau,t} \equiv y^{TIPS}_{\tau,t} + \pi^{swap}_{\tau,t}
\]
where:
- \(y^{TIPS}_{\tau,t}\) is the real yield implied by TIPS,
- \(\pi^{swap}_{\tau,t}\) is the inflation swap rate (market-implied inflation compensation).

### 2.2 Wedge / basis signal
Define the basis (wedge):
\[
w_{\tau,t} \equiv y^{N}_{\tau,t} - y^{S}_{\tau,t}
\]
where \(y^{N}_{\tau,t}\) is the nominal Treasury yield.

Interpretation:
- \(w_{\tau,t} > 0\): nominal yield is high relative to synthetic → nominal may be “cheap” vs synthetic, or synthetic “rich”.
- \(w_{\tau,t} < 0\): nominal yield is low relative to synthetic → nominal “rich” vs synthetic, or synthetic “cheap”.

The strategy is a **relative-value trade** that seeks to profit from **reversion of \(w_{\tau,t}\)** toward a stable long-run level (possibly 0, or a nonzero equilibrium reflecting frictions).

---

## 3. Position Construction (Per Tenor)

### 3.1 Core position: DV01-hedged long/short
For each tenor \(\tau\), the strategy takes a paired position designed to be approximately **duration-neutral**:

- One leg references nominal Treasuries (cash or futures proxy).
- The synthetic leg is implemented using:
  - a TIPS instrument (cash or proxy), and
  - an inflation swap (or a proxy series if swaps are not directly tradable in the project setting).

Hedge ratios are chosen to minimize residual rate-level exposure using DV01 (and optionally convexity adjustments).

### 3.2 Multi-asset portfolio
The portfolio runs across multiple tenors (2/5/10/20), so the strategy can hold a collection of instruments simultaneously:
- Nominal leg: up to 4 tenors
- TIPS leg: up to 4 tenors
- Inflation swap exposure: up to 4 tenors (or equivalent proxy)

This yields a naturally diversified book of relative-value positions across the curve.

---

## 4. Return Decomposition (Why it can make money)

Daily P&L for a DV01-hedged position decomposes into:

### 4.1 Convergence (basis) P&L
\[
\text{P\&L}^{conv}_{t} \approx \text{DV01}_{\tau}\cdot \Delta w_{\tau,t}
\]
(where the sign depends on whether the portfolio is positioned for \(w\) to narrow or widen).

### 4.2 Carry / roll-down
Even if the wedge does not move, each leg may generate carry and roll-down. Net carry is the difference in the “nothing changes except time passes” return between the long and short legs.

### 4.3 Funding and financing costs
Because the strategy is levered and often involves financing via repo / collateralized funding, the net return is reduced by:
- funding rate (e.g., SOFR) plus a borrowing spread,
- haircuts/margin that constrain maximum leverage.

### 4.4 Transaction costs and slippage
The strategy pays bid/ask and trading costs at entry, exit, and periodic re-hedging. These costs increase with:
- turnover frequency,
- number of legs,
- and leverage-scaled notional.

---

## 5. Signal and Trading Rules (Making it implementable)

### 5.1 Entry/exit via statistical bands
A simple, implementable rule:
- Compute a rolling mean and volatility of \(w_{\tau,t}\) (or z-score).
- Enter positions only when the wedge is sufficiently extreme:
  - enter if \(|z_{\tau,t}| \ge z_{enter}\)
- Exit when the wedge mean reverts:
  - exit if \(|z_{\tau,t}| \le z_{exit}\) (with \(z_{exit} < z_{enter}\))

This prevents constant exposure and creates discrete, countable trades.

### 5.2 Position aging
To avoid indefinite holding and to produce a steady trading cadence:
- impose a max holding period \(H_{max}\)
- exit/resize on schedule even if convergence is incomplete

---

## 6. Leverage and Risk Controls

### 6.1 Volatility targeting with leverage caps
Define gross leverage \(L_t\) as:
\[
L_t = \min\left(L_{max}, \frac{\sigma^{target}}{\hat{\sigma}_{port,t}}\right)
\]
where \(\hat{\sigma}_{port,t}\) is an EWMA volatility estimate of unlevered returns.

### 6.2 Hard drawdown stops / de-leveraging
To mitigate “carry-like” crash behavior:
- reduce leverage when drawdown exceeds a threshold,
- optionally flatten exposures during high-volatility stress regimes.

### 6.3 Concentration limits and turnover penalties
Impose:
- per-tenor gross exposure caps,
- turnover caps per rebalance period,
- and/or transaction-cost-aware rebalancing.

---

## 7. Why Leverage Helps (and when it stops helping)

### 7.1 Low leverage regime
At modest leverage, the strategy’s net return can scale roughly linearly with leverage if:
- basis convergence dominates costs,
- funding spreads are small,
- and turnover is controlled.

### 7.2 High leverage regime
At higher leverage, net returns can deteriorate because:
- funding costs scale with gross notional,
- transaction costs scale with turnover and size,
- tail events produce amplified drawdowns and potential margin breaches.

Therefore, an explicit goal is to identify a **net-performance-maximizing leverage region** rather than assuming “more leverage = better.”

---

## 8. Backtest Outputs (What this project will report)

For each leverage cap \(L_{max}\) (e.g., 1x, 2x, 4x, 6x, 8x, 10x):
- Gross vs net performance (with and without costs/funding)
- CAGR, Sharpe, Sortino
- Maximum drawdown
- Turnover and cost drag
- Stress-period performance and tail risk metrics (e.g., worst month, CVaR)
- Trade counts and clustering diagnostics
- Per-tenor attribution: which maturities contribute most to P&L

A key figure will be a **profitability vs leverage frontier**, showing how performance changes as leverage increases under realistic cost assumptions.

---

## 9. Practical Notes / Implementation Choices
- Instrument mapping must be defined explicitly (cash vs futures proxies for nominal and TIPS legs; inflation swap or proxy series).
- Hedge ratios should be DV01-based and re-estimated periodically.
- Costs should be modeled conservatively: bid/ask crossing, slippage assumptions, and funding spreads.
- Robustness checks will include alternative entry/exit thresholds, alternative lookbacks, and stress regime filters.

---

## 10. Working Hypothesis
A persistent wedge \(w_{\tau,t}\) exists due to market frictions (liquidity, balance-sheet constraints, segmented investor demand). This wedge is sufficiently mean reverting that a disciplined, hedged, levered strategy can earn positive net returns at moderate leverage—subject to strict risk controls that limit exposure to tail events.

---
