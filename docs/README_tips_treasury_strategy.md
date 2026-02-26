# Dual-Basis Segmented-Arbitrage Carry Portfolio (DB-SACP)
**Core idea:** run a *levered, risk-controlled* fixed-income carry book that harvests two persistent “segmented arbitrage” wedges:
1) **Treasury–OIS (Treasury-swap) basis**: the difference between the fixed rate on OIS and the matched-maturity Treasury yield.  
2) **TIPS–Treasury basis**: the yield difference between a *synthetic nominal Treasury* (constructed from TIPS + inflation swaps) and the true nominal Treasury.

These two spreads are empirically large and persistent, but they are not “free money”: they reflect funding + balance-sheet frictions and can widen sharply in stress. The design goal is **Sharpe maximization** via (i) cross-tenor diversification (2/5/10/20), (ii) cross-strategy diversification (swap-basis + TIPS-basis), and (iii) strict volatility/VaR + stop-loss controls.

- This strategy systematically monetizes persistent wedges created by segmented funding and balance-sheet constraints, using the fact that Treasury-swap spreads and TIPS-Treasury basis spreads persist and are not explained by a single common factor.
- The strategy is a **carry book**: it tends to earn when markets are quiet and can lose when bases gap wider, so risk controls are central—not optional.


---

## Instruments and trade construction

### Treasury–OIS (Treasury-swap) basis package
**Spread definition (by tenor m ∈ {2,5,10,20}):**
\[
S^{swap}_m(t) = r^{OIS}_m(t) - y^{UST}_m(t)
\]
Negative \(S^{swap}_m\) implies Treasury yields exceed OIS fixed rates.

**Trade direction (carry-harvesting):**
- If \(S^{swap}_m(t) < 0\): **Long Treasury / Pay OIS fixed** (receive overnight compounded float).
- If \(S^{swap}_m(t) > 0\): either (a) **no trade**, or (b) reverse *only if* your implementation truly locks positive carry after funding + margin (conservative baseline is “no trade”).

**Implementation note:** treat this as an “asset-swap-like” package:
- Long on-the-run (or liquid benchmark) Treasury with maturity ~m, financed in repo.
- OIS swap: pay fixed \(r^{OIS}_m\), receive compounded overnight (SOFR/FF).

**Carry (first-order approximation):**
\[
\text{Carry}^{swap}_m \approx \big(y^{UST}_m - r^{OIS}_m\big) - c^{fund}_m - c^{swap}_m - c^{repo}_m
= -S^{swap}_m - c^{all}_m
\]
where \(c^{all}_m\) includes funding spreads, bid/ask + execution, and margin drag.

---

### TIPS–Treasury basis package
**Synthetic nominal construction (conceptual):**
- Hold a TIPS of maturity ~m with real yield \(y^{TIPS,real}_m\).
- Overlay a zero-coupon inflation swap (or equivalent CPI swap strip) that converts the uncertain realized inflation uplift into a fixed inflation rate \( \pi^{swap}_m\).
- The resulting cashflows replicate (approximately) a *nominal* bond; its yield is (heuristically) \(y^{TIPS,real}_m + \pi^{swap}_m\).

**Basis definition:**
\[
S^{tips}_m(t) = y^{synth\_nom}_m(t) - y^{UST}_m(t)
\quad \text{with} \quad
y^{synth\_nom}_m(t) \approx y^{TIPS,real}_m(t) + \pi^{swap}_m(t)
\]

**Trade direction (carry-harvesting):**
- If \(S^{tips}_m(t) > 0\): synthetic nominal yields more → synthetic is cheap → **Long synthetic / Short nominal Treasury**.
- If \(S^{tips}_m(t) < 0\): **Short synthetic / Long nominal Treasury**.

**Carry (first-order approximation):**
\[
\text{Carry}^{tips}_m \approx \text{sign}(S^{tips}_m)\cdot |S^{tips}_m| - c^{fund}_m - c^{inflswap}_m - c^{repo}_m
\]

---

## Data inputs (minimal viable set)

### Daily inputs (tenor-by-tenor)
For each \(m \in \{2,5,10,20\}\):

**Treasury leg**
- Nominal Treasury par yields or zero yields \(y^{UST}_m(t)\)
- On-the-run price/yield + DV01 (preferred for execution realism)

**OIS leg**
- OIS fixed swap rates \(r^{OIS}_m(t)\) (SOFR-OIS or FF-OIS, consistent curve)

**TIPS + inflation swap leg**
- TIPS real yields \(y^{TIPS,real}_m(t)\) matched by maturity
- Zero-coupon inflation swap fixed rates \(\pi^{swap}_m(t)\) matched by maturity
- (Optional) TIPS indexation lag model / seasonality adjustment (implementation detail)

### Risk and state variables (used only for sizing/filters)
- Spread history \(S^{swap}_m(t), S^{tips}_m(t)\)
- Realized spread volatility estimates (EWMA + cones)
- Funding stress proxy set (choose one consistent set; examples: TED-like proxy, SOFR dislocations, dealer balance-sheet proxies)
- Yield-curve factors (level/slope/curvature PCs)
- (Optional) term-premium / return-forecast factor (Cochrane–Piazzesi style)

---

## Profitability model (how this earns money)

###  P&L decomposition (per tenor, per package)
Each package is a classic “carry + mark-to-market” trade:

\[
\Delta \Pi \approx \underbrace{\text{Carry}\cdot \Delta t}_{\text{earns when nothing moves}}
\;-\;\underbrace{DV01^{pkg}\cdot \Delta S}_{\text{basis risk}}
\;-\;\underbrace{\text{costs}}_{\text{bid/ask + funding + margin drag}}
\]

 **The risk is not Treasury duration if you hedge DV01; it is spread/basis widening**.


### Feasibility
Cross-tenor and cross-strategy diversification matters because arbitrage spreads are **not** all one factor. Empirically, arbitrage-spread correlations are low and require many principal components to explain most variation. This supports building a *risk-budgeted portfolio* rather than a single concentrated basis bet.

---


##  Portfolio construction, sizing, entry/exit (bounded risk + leverage)

### Portfolio weights (Sharpe-oriented, constrained)
Let the trade universe be all tenor×strategy pairs, indexed by i.
- Expected return vector: \(\hat{\mu}_i\) (from §4.1)
- Covariance of spread changes: \(\hat{\Sigma}\) (EWMA or rolling window; shrink if needed)

**Unconstrained Sharpe-optimal direction:**
\[
w^* \propto \hat{\Sigma}^{-1}\hat{\mu}
\]

**Practical constraints:**
- **Long-only in “carry” sense**: take direction that earns \(|S|\) (no discretionary directional macro bets).
- **Factor exposure bounds**: keep net exposure to yield-curve PCs (level/slope/curvature) near zero.
- **Concentration caps**: max weight per tenor, max weight per strategy.
- **Turnover penalty**: rebalance only when score changes materially.

### Convert weights → notionals using a volatility target
Choose target annualized portfolio volatility \(\sigma_{target}\) (e.g., 6–10%).
Let \(\hat{\sigma}_{p,unlev}\) be predicted vol of the portfolio (in $ terms per $1 notional unit).
Set gross scaling:
\[
\text{Scale}(t)=\min\Big(\frac{\sigma_{target}}{\hat{\sigma}_{p,unlev}(t)},\;\text{LeverageCap}\Big)
\]

### Leverage and capital assumptions (documented)
We assume a prime-broker + repo + cleared swap implementation:
- Treasuries financed in repo with haircut \(h_{UST}\) (assume 1–3%).
- TIPS financed in repo with haircut \(h_{TIPS}\) (assume 2–5%).
- Cleared OIS and inflation swaps require initial margin (assume 1–3% notional) + daily VM.

**Constraint:**
\[
\text{Total initial margin + haircuts} \le 60\%-70\% \text{ of equity capital}
\]
Keep remaining capital as liquidity buffer for VM in stress.

### Entry rules (executed daily; scaled via VWAP if needed)
For each candidate trade i:
1) Check \(\widehat{\mu}_i>0\), \(|z_i|\ge z_{entry}\), and vol-cone not extreme.
2) Add risk in *steps* (scale-in):
   - 25% of target notional on day 0,
   - 25% on day 1 if conditions persist,
   - remainder over days 2–4 (or VWAP over liquidity window).

### Exit rules
A trade is closed if any trigger hits:

**(A) Mean-reversion / normalization**
- Exit when \(|z_i(t)| \le z_{exit}\) (e.g., 0.25–0.5).

**(B) Stop-loss (PnL-based)**
- Define per-trade stop \(SL_i\) as a fixed % of equity or as k×daily-$vol:
  \[
  SL_i = \min(0.75\%\text{ equity},\; 6\cdot \hat{\sigma}_{\$,i})
  \]
- If cumulative PnL on trade i ≤ −SL_i: exit fully and impose a cooldown (e.g., 5–10 trading days).

**(C) Position age rule**
- Close any position older than \(T_{max}\) (e.g., 60–90 trading days) unless it remains in the top attractiveness bucket.

**(D) Stress-off switch**
- If funding-stress proxy spikes or spread-vol cones hit extreme bands: cut gross exposure by a fixed fraction (e.g., halve) immediately.

---

## Risk management (what can go wrong, and how we bound it)

### The dominant tail risk: basis widening under leverage
Even DV01-neutral packages can suffer large mark-to-market losses if the basis widens.
Therefore, risk must be set in **DV01 × spread-move** space:

**Stress test template (per trade):**
- Assume adverse basis move of 25–50 bps (scenario set by historical episodes).
- Loss estimate:
  \[
  Loss_i \approx DV01^{pkg}_i \cdot \Delta S^{stress}
  \]
Require: worst-case scenario loss ≤ 10–15% of equity after accounting for diversification.

### Portfolio limits (hard)
- **1-day 99% VaR** ≤ 2% equity
- **10-day 99% VaR** ≤ 6% equity
- **Max drawdown stop**: at −12% equity, cut gross risk by 50%; at −18%, cut to minimal “monitor only.”

### Factor neutrality (soft but monitored)
Because yield movements are dominated by a small set of factors (level/slope/curvature), keep exposure bounded:
- Compute portfolio loading on PCs daily.
- Enforce: |level loading| ≤ threshold, |slope loading| ≤ threshold, |curve loading| ≤ threshold.
Residual exposure is acceptable but must be intentional.

### Carry-trade diagnostics (monitoring)
Carry strategies typically show many small gains and occasional sharp losses; therefore, monitor:
- downside beta to broad risk indicators,
- left-tail / drawdown clustering,
- funding-stress sensitivity.

---


## Profitability (worked example)

Assume equity capital \(E = \$10\)mm.  
Assume the strategy runs two “core” tenors (10y, 20y) in both packages, diversified:
- Treasury–OIS carry: ~25 bps annualized net of costs at 10y and ~30 bps at 20y
- TIPS basis carry: ~25–30 bps annualized net of costs

Suppose the portfolio targets gross exposure of \$250mm notional equivalent (≈25× equity *not allowed* unless haircuts+IM permit; treat this as “upper bound scenario”), but risk targeting and stress tests typically force less.

**Expected carry (rough):**
- If net carry ~50 bps/year on \$250mm → \$1.25mm/year → 12.5% ROE *before* drawdowns.
- Realized Sharpe depends on basis volatility; thus vol targeting is the binding constraint.

**Stress reality check:**
- A 30 bps adverse basis widening on a large DV01 package can produce multi-million losses quickly.
- Therefore, the *actual* traded size should be whatever keeps 1–10 day VaR and stress losses within limits.

