---
title: Home
---
Project architecture for the LSTM draft workflow: [docs/ARCHITECTURE.md](/Users/assortedsphinx/Desktop/team_t/docs/ARCHITECTURE.md)

# Treasury vs Synthetic Nominal Yield Carry Trade  
## Levered Relative-Value Arbitrage Strategy

---

## 1. Executive Summary

This project proposes a **levered relative-value carry strategy** that exploits a persistent wedge between:

- Nominal U.S. Treasury yields at tenors \( \tau \in \{2,5,10,20\} \), and  
- A synthetic nominal yield constructed as **TIPS real yield + inflation swap rate**.

The strategy trades deviations between these two yields using a DV01-hedged long/short structure. Profitability arises from:

1. **Basis convergence (mean reversion)**
2. **Net carry / roll-down**
3. **Disciplined leverage with bounded risk controls**

The strategy holds multiple instruments simultaneously across tenors and legs, naturally satisfying multi-asset exposure requirements.

---

## 2. Trade Definition

### 2.1 Synthetic Nominal Yield

For each tenor \( \tau \), define the synthetic nominal yield as:

$$
y^{S}_{\tau,t} = y^{TIPS}_{\tau,t} + \pi^{swap}_{\tau,t}
$$

where:

- \( y^{TIPS}_{\tau,t} \) = real yield implied by TIPS  
- \( \pi^{swap}_{\tau,t} \) = inflation swap rate  

---

### 2.2 Wedge (Basis) Signal

Define the wedge:

$$
w_{\tau,t} = y^{N}_{\tau,t} - y^{S}_{\tau,t}
$$

where \( y^{N}_{\tau,t} \) is the nominal Treasury yield.

Interpretation:

- \( w_{\tau,t} > 0 \): nominal yield is high relative to synthetic  
- \( w_{\tau,t} < 0 \): nominal yield is low relative to synthetic  

The strategy trades expected **mean reversion of \( w_{\tau,t} \)**.

---

## 3. Position Construction

### 3.1 DV01-Hedged Structure

For each tenor \( \tau \), construct a long/short position designed to be approximately duration-neutral:

- Nominal Treasury leg  
- TIPS leg  
- Inflation swap leg  

Hedge ratios are determined using DV01 matching:

$$
\text{DV01}_{nominal} \approx \text{DV01}_{synthetic}
$$

This minimizes exposure to parallel shifts in the yield curve.

---

## 4. Return Decomposition

Daily P&L decomposes into four components:

---

### 4.1 Convergence (Basis) P&L

$$
\text{P\&L}^{conv}_{t} \approx \text{DV01}_{\tau} \cdot \Delta w_{\tau,t}
$$

This is the primary driver of returns.

---

### 4.2 Carry / Roll-Down

If the wedge does not move, the position still earns:

- Coupon carry  
- Roll-down along the curve  

Net carry equals the difference between long and short leg carry.

---

### 4.3 Funding Cost

With leverage, funding reduces returns:

$$
\text{P\&L}^{fund}_{t} = L_t \cdot r^{fund}_t \cdot E_t
$$

where:

- \( L_t \) = leverage  
- \( r^{fund}_t \) = funding rate (e.g., SOFR + spread)  
- \( E_t \) = equity  

---

### 4.4 Transaction Costs

Costs include:

- Bid/ask spread crossing  
- Slippage  
- Rebalancing costs  

Transaction costs increase with turnover and leverage.

---

## 5. Trading Rules

### 5.1 Entry / Exit via Statistical Bands

Compute rolling mean and volatility of \( w_{\tau,t} \):

$$
z_{\tau,t} = \frac{w_{\tau,t} - \mu_{\tau,t}}{\sigma_{\tau,t}}
$$

Trading rule:

- Enter when \( |z_{\tau,t}| \ge z_{enter} \)  
- Exit when \( |z_{\tau,t}| \le z_{exit} \)  

with \( z_{exit} < z_{enter} \).

---

### 5.2 Maximum Holding Period

Impose a maximum holding period \( H_{max} \) to ensure turnover and prevent capital lock-up.

---

## 6. Leverage and Risk Controls

### 6.1 Volatility-Targeted Leverage

Define leverage:

$$
L_t = \min\left(L_{max}, \frac{\sigma^{target}}{\hat{\sigma}_{port,t}}\right)
$$

where:

- \( \hat{\sigma}_{port,t} \) = EWMA portfolio volatility  
- \( \sigma^{target} \) = target volatility  

---

### 6.2 Drawdown-Based Deleveraging

If drawdown exceeds threshold \( D^* \), reduce or eliminate leverage.

---

### 6.3 Concentration Limits

- Cap per-tenor exposure  
- Cap gross notional  
- Penalize turnover  

---

## 7. Profitability vs Leverage Framework

Evaluate performance across leverage levels:

$$
L_{max} \in \{1,2,4,6,8,10\}
$$

For each level compute:

- CAGR  
- Sharpe ratio  
- Sortino ratio  
- Maximum drawdown  
- CVaR  
- Turnover  
- Cost drag  

Expect:

- Returns scale approximately linearly at low leverage  
- Net returns peak at moderate leverage  
- Tail risk dominates at high leverage  

---

## 8. Backtest Deliverables

For each tenor and the aggregate portfolio:

- Trade counts  
- Holding periods  
- Stress-period performance  
- Gross vs net returns  
- Profitability vs leverage frontier  

---

## 9. Working Hypothesis

A persistent wedge exists due to liquidity segmentation and balance-sheet constraints. The wedge is sufficiently mean-reverting that a disciplined, hedged, moderately levered strategy can generate positive net returns under realistic funding and cost assumptions.

The objective is to determine the **optimal leverage region** that maximizes net risk-adjusted returns without inducing unacceptable tail risk.
