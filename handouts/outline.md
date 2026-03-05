# Unified Quant Research Execution Plan (Stages 0–16)

## Summary
This plan integrates hypothesis design, feature economics, modeling, signal diagnostics, portfolio construction, backtesting, risk/exposure, benchmarks, inference, and manuscript structure into one continuous research workflow.  
All status flags below are based on direct inspection of: [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb), [data_utils.py](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py), [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py), [event_panels.py](/Users/assortedsphinx/Desktop/team_t/src/event_panels.py), [backtest_utils.py](/Users/assortedsphinx/Desktop/team_t/src/backtest_utils.py), [ARCHITECTURE.md](/Users/assortedsphinx/Desktop/team_t/docs/ARCHITECTURE.md), and data artifacts in [data/lstm_draft/processed](/Users/assortedsphinx/Desktop/team_t/data/lstm_draft/processed).

---

## Stage 0 — Repository audit and architecture overview

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: define the formal research problem, audit architecture, and lock end-to-end scope.

### Existing Implementation
- Existing implementation (`COMPLETE`): modular pipeline split between notebook orchestration and reusable `src` modules is documented in [ARCHITECTURE.md](/Users/assortedsphinx/Desktop/team_t/docs/ARCHITECTURE.md).

### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): hypothesis narrative exists in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:23), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:31), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:33), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:708), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:710), but not formalized as testable hypotheses.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): formalize research problem definition with:
- Economic hypothesis: delayed/heterogeneous incorporation of fundamentals into prices creates short-horizon return predictability.
- Empirical predictions: fundamentals+price context should improve out-of-sample cross-sectional ranking quality and tradable spread performance.
- Model role: LSTM tests nonlinear/state-dependent lag structure against simpler baselines.

### Files Involved
- Files involved: [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb), [ARCHITECTURE.md](/Users/assortedsphinx/Desktop/team_t/docs/ARCHITECTURE.md), [README.md](/Users/assortedsphinx/Desktop/team_t/README.md).

### Outputs
- Outputs: `docs/RESEARCH_PROBLEM_DEFINITION.md`, updated notebook Introduction.

### Recommended Approach
- Recommended approach: keep existing architecture unchanged; tighten problem statement and empirical test criteria only.

## Stage 1 — Data ingestion pipeline

### Stage Summary
- Status: `COMPLETE`
- Purpose: deterministic and reproducible acquisition of raw inputs.

### Existing Implementation
- Existing implementation (`COMPLETE`): API config/loading/filtering and required `PRICES.csv` ingestion via [data_utils.py](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py) (`configure_api_from_env`, `fetch_zacks_table`, `load_prices_csv_required`, `build_static_top10_universe`), orchestrated in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:214) and [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:234).

### Tasks to Implement
- Tasks to implement: none for core ingestion logic.

### Files Involved
- Files involved: [data_utils.py](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb), [PRICES.csv](/Users/assortedsphinx/Desktop/team_t/data/PRICES.csv).

### Outputs
- Outputs: already produced universe and panel draft files in [data/lstm_draft/processed](/Users/assortedsphinx/Desktop/team_t/data/lstm_draft/processed).

### Recommended Approach
- Recommended approach: preserve functions as-is; only add reproducibility logging in Stage 16.

## Stage 2 — Feature engineering and preprocessing

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: build leak-free, model-ready features and justify them economically.

#### Feature Construction
- Feature construction

##### Stage Summary
- Status: `COMPLETE`

##### Existing Implementation
- Existing implementation: PIT lag merge and validations in [data_utils.py](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py:257) and [data_utils.py](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py:331); feature creation in [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:27) and [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:37).

#### Feature Economic Rationale
- Feature economic rationale

##### Stage Summary
- Status: `PARTIALLY COMPLETE`

##### Existing Implementation
- Existing implementation: feature list exists in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:328) and [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:736), but no explicit per-feature economic table.

##### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): add notebook table mapping each feature (`debt/equity`, `ROE`, `P/B`, `profit margin`, `EPS`, `beta`, `adj_close/log_ret`, `volume`) to return-predictive intuition.

#### Feature Stability Diagnostics
- Feature stability diagnostics

##### Stage Summary
- Status: `PARTIALLY COMPLETE`

##### Existing Implementation
- Existing implementation: missingness heatmap exists ([lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:428)).

##### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): add split-by-split feature distribution drift checks, train-only scaling/imputation, and outlier handling policy.

##### Files Involved
- Files involved: [data_utils.py](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py), [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

##### Outputs
- Outputs: `data/lstm_final/processed/feature_panel_final.parquet`, `data/lstm_final/processed/feature_metadata.json`.

##### Recommended Approach
- Recommended approach: keep current feature set and add documentation/diagnostics around it.

## Stage 3 — Dataset construction and lookback window generation

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: convert panel into supervised sequence datasets with sample traceability.

### Existing Implementation
- Existing implementation (`COMPLETE`): lookback tensor builder exists in [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:124) and is used in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:741).

### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): tensors are generated and printed, but sample index metadata is not persisted.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): export `ticker/date/sample_id` mapping for each tensor row and save train/dev/test tensor artifacts.

### Files Involved
- Files involved: [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

### Outputs
- Outputs: `data/lstm_final/model_inputs/{train,dev,test}.npz`, `data/lstm_final/model_inputs/sample_index.csv`.

### Recommended Approach
- Recommended approach: extend current tensor function instead of replacing it.

## Stage 4 — Model architecture implementation

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: implement executable predictive models tied to hypotheses.

### Existing Implementation
- Existing implementation: notebook model rationale exists, but no trainable model code in repo.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): implement pooled LSTM regressor and a simple baseline model for comparison.

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/model_utils.py`, [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

### Outputs
- Outputs: `data/lstm_final/models/lstm_config.json`, `lstm_best.pt`, baseline model artifact.

### Recommended Approach
- Recommended approach: pooled cross-sectional training as primary; per-stock models optional robustness branch.


### Training Structure
- `Pooled cross-sectional LSTM (primary)` — `PARTIALLY COMPLETE`
- Evidence: sequence construction already stacks samples from all tickers via [build_lstm_tensors](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:124) and notebook usage in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:741).
- Remaining: actual trainable model/training code is still missing in `src` (`NOT COMPLETE`).
- Note: pooled training stacks samples across tickers; no ticker-ID embedding by default (optional extension only if needed).
- `Per-stock model variant (robustness only)` — `NOT COMPLETE`
- Implementation: add optional per-ticker training path for robustness comparison only.
- `Baseline 1: Linear model (lagged returns + fundamentals)` — `NOT COMPLETE`
- Implementation: add linear baseline (start with Ridge or OLS; keep it simple) with identical train/dev/test split and same timing convention as primary strategy.
- `Baseline 2: Price-only LSTM` — `NOT COMPLETE`
- Implementation: same LSTM architecture, but restrict inputs to price/return features to test incremental value of fundamentals.

## Stage 5 — Model training pipeline

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: train robustly with strict anti-overfitting controls.

### Existing Implementation
- Existing implementation (`COMPLETE`): deterministic calendar splits are already defined in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:186) and assigned via [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:69); PIT safeguards already exist.

### Existing Implementation
- Existing implementation (`NOT COMPLETE`): no actual training loop, hyperparameter search, or walk-forward workflow.

### Tasks to Implement
- Tasks to implement:
- `NOT COMPLETE`: dev-based hyperparameter selection, early stopping, checkpointing, run logging.
- `NOT COMPLETE`: rolling walk-forward robustness (frozen hyperparameters) to validate temporal stability.

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/model_utils.py`, [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

### Outputs
- Outputs: `training_log.csv`, `hyperparam_results.csv`, `walk_forward_results.csv`.

### Recommended Approach
- Recommended approach: preserve the current train/dev/test structure as primary evaluation and treat walk-forward as robustness evidence.

## Stage 6 — Prediction generation

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: generate out-of-sample predictions and evaluate raw signal quality before portfolio construction.

#### Prediction Generation
- Prediction generation

##### Stage Summary
- Status: `NOT COMPLETE`

##### Existing Implementation
- Existing implementation: none in `src`; notebook explicitly stops before training/evaluation.
- Tasks: produce `y_pred` for dev/test with aligned `ticker/date/y_true`.

#### Signal Evaluation Diagnostics
- Signal evaluation diagnostics

##### Stage Summary
- Status: `NOT COMPLETE` for model-based diagnostics; `PARTIALLY COMPLETE` for pre-model data/event diagnostics.

##### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): event-time diagnostics around fundamentals exist in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:559) and [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:643), but these are not prediction diagnostics.

##### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`):
- information coefficient (IC) by day
- rank-IC by day
- prediction decile analysis
- long-short spread analysis
- calibration diagnostics
- permutation feature importance
- SHAP feature importance

##### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/prediction_utils.py`, new `/Users/assortedsphinx/Desktop/team_t/src/signal_diagnostics.py`, [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

##### Outputs
- Outputs: `predictions_test.csv`, `signal_diagnostics.csv`, feature-importance figures.

##### Recommended Approach
- Recommended approach: run diagnostics on dev first for specification checks, then lock and report on test.


##### Feature Interpretability Scope
- `Permutation feature importance` — `NOT COMPLETE` (required)
- Implementation: compute on dev first, then lock and report on test.
- Scoring metric: use dev set ranking quality (rank-IC preferred; fallback to MSE if needed).
- Permutation unit: permute one feature channel at a time across samples (preserve time order within each sequence) to avoid breaking sequence structure.
- `SHAP feature importance` — `NOT COMPLETE` (optional, if time permits)
- Implementation: run only after core diagnostics are complete; treat as exploratory/appendix.
- Primary method note: permutation importance is the default because it is computationally lighter and typically more stable for sequence-model diagnostics in this workflow.

## Stage 7 — Trading signal construction

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: map predictions into explicit trade signals and stance definition.

### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): notebook states next-step return target and ranking intent in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:31), [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:710), and target column in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:290).

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`):
- formalize target as next-day log return for signal generation
- formalize pooled training structure as primary model design
- implement cross-sectional rank-to-signal function
- implement both long-short and long-only signal books
- define entry/exit and rebalance cadence in code

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/signal_utils.py`, [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

### Outputs
- Outputs: `signal_book_long_short.csv`, `signal_book_long_only.csv`.

### Recommended Approach
- Recommended approach: keep ranking mechanism simple and reproducible first, then run sensitivity in Stage 12.


### Timing Convention and Information Availability

- `Price-field audit` — `COMPLETE`
- Evidence from data inspection:
- Raw [PRICES.csv](/Users/assortedsphinx/Desktop/team_t/data/PRICES.csv): for top-10+SPY (2006–2013), `open`, `close`, `adj_open`, `adj_close`, `volume` each have full non-null coverage.
- Processed panel [lstm_feature_panel_2006_2013.csv](/Users/assortedsphinx/Desktop/team_t/data/lstm_draft/processed/lstm_feature_panel_2006_2013.csv): `open`, `close`, `adj_open`, `adj_close_intraday`, `adj_factor` each have full non-null coverage.
- Split-adjusted intraday construction exists in [add_split_adjusted_intraday_prices](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:16).

- `Canonical execution convention` — `COMPLETE` (selected)
- Use **Option A** project-wide:
- Features use info available through close of day `t` (PIT fundamentals + price/volume through `t` close).
- Prediction targets next-day return at `t+1`.
- Signal is formed after close `t`.
- Execution occurs at open `t+1`.
- PnL realized from `open(t+1) -> close(t+1)` using split-adjusted fields (`adj_open`, `adj_close_intraday`).

### Target vs Execution Horizon

Status: `PARTIALLY COMPLETE`

Current predictive target:

`target_next_log_ret = log(adj_close_(t+1) / adj_close_t)`

Evidence: implemented in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:289) and [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:290).

Clarification:

The predictive target remains **close-to-close return**, which is standard in cross-sectional return forecasting.

Execution occurs at `open(t+1)`, and realized PnL is measured over `open(t+1) -> close(t+1)`.

This difference is intentional.

The model forecasts cross-sectional return differences using close-to-close returns, while trades are executed at `open(t+1)` to avoid lookahead bias.

Strategy performance is therefore evaluated primarily through **cross-sectional ranking quality**, not exact return magnitude.

Remaining work:

Confirm this convention is propagated consistently through signal generation and backtest engine.

### Cross-Sectional Ranking Rule — `NOT COMPLETE`

Daily signal generation procedure:
1. For each trading day `t`, rank all assets by predicted return.
2. Construct signals based on ranks.

Baseline rule:

Long-short portfolio

• Long top `K` assets
• Short bottom `K` assets

Long-only portfolio

• Long top `K` assets only

Baseline parameter:

`K = 3`

Sensitivity analysis will test `K ∈ {2,3,4}` in Stage 12.

- `No Lookahead assertion (signal layer)` — `NOT COMPLETE`
- Implementation rule: at signal date `t`, allowed columns must be derived only from dates `<= t`; forbid any use of `t+1` or later values in feature engineering.
- Reason: signals are frozen after close `t`, and `t+1` prices are used only for execution/PnL, never for feature construction.

### Stage 7 timing table (insert exactly)

| Timeline row | Definition |
|---|---|
| `(t) data used` | PIT fundamentals with `feature_available_date <= t`, plus prices/volume through close `t` |
| `(t) signal computed` | model prediction and cross-sectional rank formed after close `t` |
| `(t+1) execution + return realized` | execute at open `t+1`, realize PnL over `open(t+1) -> close(t+1)` |

## Stage 8 — Portfolio construction

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: convert signals to sized positions under constraints.

### Existing Implementation
- Existing implementation: none.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): sizing rules (equal-weight baseline + vol-scaled variant), concentration caps, gross/net bounds, leverage assumptions, minimum simultaneous holdings checks.

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/portfolio_utils.py`.

### Outputs
- Outputs: `positions_daily.csv`.

### Recommended Approach
- Recommended approach: generate one canonical positions table consumed by Stages 9 and 10.

## Stage 9 — Backtesting engine

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: produce executable PnL and trade logs.

### Existing Implementation
- Existing implementation (`NOT COMPLETE`): [backtest_utils.py](/Users/assortedsphinx/Desktop/team_t/src/backtest_utils.py) is placeholder-only.

### Tasks to Implement
- Tasks to implement: replace placeholders with position-to-return engine, trade event generation, and daily ledger.

### Files Involved
- Files involved: [backtest_utils.py](/Users/assortedsphinx/Desktop/team_t/src/backtest_utils.py), new `/Users/assortedsphinx/Desktop/team_t/src/trade_utils.py`.

### Outputs
- Outputs: `backtest_gross.csv`, `trade_log.csv`.

### Recommended Approach
- Recommended approach: vectorized daily simulation with explicit trade entry/exit records.


### Timing Convention and Information Availability
- `Backtest timing convention alignment with Stage 7` — `NOT COMPLETE`
- Implementation: enforce same canonical Option A timeline in backtest ledger.
- `Existing PIT safeguard` — `COMPLETE`
- Evidence: [validate_point_in_time_panel](/Users/assortedsphinx/Desktop/team_t/src/data_utils.py:331).
- `Backtest no-lookahead assertions` — `NOT COMPLETE`
- Required assertions:
- Signal date `t` joins only features built from data dated `<= t`.
- Execution date must be next trading day `t+1`.
- PnL for decision `t` uses only `t+1` execution-day prices and never feeds back into features.
- Any merge exposing `t+1` close/open to feature rows dated `t` must fail hard.

## Stage 10 — Risk management framework

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: enforce portfolio risk controls and exposure monitoring using Stage 8 positions.

### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): asset-level rolling beta feature exists in [feature_engineering.py](/Users/assortedsphinx/Desktop/team_t/src/feature_engineering.py:37); no portfolio risk framework exists.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`), explicitly using `positions_daily.csv` from Stage 8:
- portfolio beta exposure to SPY
- sector concentration metrics
- gross vs net exposure monitoring
- drawdown-triggered de-risking rules

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/risk_utils.py`, new `/Users/assortedsphinx/Desktop/team_t/src/exposure_utils.py`, Stage 8 output.

### Outputs
- Outputs: `risk_exposure_daily.csv`, `risk_events.csv`.

### Recommended Approach
- Recommended approach: compute exposures from realized positions, not from raw signals.

## Stage 11 — Transaction cost modeling

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: transition from gross to realistic net performance.

### Existing Implementation
- Existing implementation: none.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): commissions, slippage, borrow, and funding spread modeling with component attribution.

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/cost_utils.py`, [backtest_utils.py](/Users/assortedsphinx/Desktop/team_t/src/backtest_utils.py).

### Outputs
- Outputs: `backtest_net.csv`, `cost_breakdown_daily.csv`.

### Recommended Approach
- Recommended approach: run paired gross/net simulations with identical signals/positions.

## Stage 12 — Strategy evaluation metrics

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: evaluate strategy quality, compare benchmarks, and test statistical significance.

#### Performance Metrics
- Performance metrics

##### Stage Summary
- Status: `NOT COMPLETE`
- Tasks: CAGR, annualized vol, Sharpe, Sortino, max drawdown, hit rate, holding period, turnover, net-vs-gross deltas.

#### Benchmark Comparisons
- Benchmark comparisons

##### Stage Summary
- Status: `NOT COMPLETE`
- Tasks:
- SPY buy-and-hold
- equal-weight top-10 universe buy-and-hold
- long-only signal portfolio
- long-short signal portfolio

#### Statistical Inference Tests
- Statistical inference tests

##### Stage Summary
- Status: `NOT COMPLETE`
- Tasks:
- t-test of mean daily returns
- bootstrap confidence intervals for Sharpe
- turnover diagnostics (`turnover vs net returns`, turnover distribution)
- parameter sensitivity analysis (`top-k` sensitivity 2–4, lookback sensitivity)

##### Existing Implementation
- Existing implementation note: notebook states benchmarks/training diagnostics are not yet run in [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb:712).

##### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/src/performance_utils.py`, [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

##### Outputs
- Outputs: `performance_table.csv`, `benchmark_table.csv`, `inference_results.json`, `sensitivity_results.csv`.

##### Recommended Approach
- Recommended approach: evaluate all benchmark tracks on identical date windows and cost assumptions.


##### Benchmark Execution Alignment
- `Execution-consistent benchmarks` — `NOT COMPLETE`
- All benchmarks must use the same canonical Option A timing and return definition as the strategy.
- All benchmark comparisons must use identical evaluation windows.
- All applicable benchmark portfolios must use the same cost/funding assumptions as strategy, except explicitly documented buy-and-hold minimal/zero turnover cases.
- Benchmarks to align:
1. `SPY buy-and-hold` — `NOT COMPLETE`
1. Evaluate with same return timing convention; use minimal/explicitly zero trading costs consistent with buy-and-hold assumption.
2. `Equal-weight top-10 universe` — `NOT COMPLETE`
2. Same dates and same Option A return timing.
3. `Long-only signal portfolio` — `NOT COMPLETE`
3. Same execution timing and cost model as main framework.
4. `Long-short signal portfolio` — `NOT COMPLETE`
4. Same execution timing and funding assumptions as main framework.

Benchmark returns must use the SAME timestamp convention as the strategy to avoid apples-to-oranges comparisons.
Example: if strategy realized returns use `open(t+1) -> close(t+1)`, benchmark returns must be computed using that same realized-return definition.

## Stage 13 — Visualization and performance reporting

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: visualize empirical evidence from data diagnostics through strategy behavior.

### Existing Implementation
- Existing implementation (`COMPLETE` for data diagnostics): five diagnostic figures are already generated in [docs/figures](/Users/assortedsphinx/Desktop/team_t/docs/figures).

### Existing Implementation
- Existing implementation (`NOT COMPLETE` for strategy reporting): no prediction/backtest/risk figures yet.

### Tasks to Implement
- Tasks to implement:
- equity and drawdown plots by strategy/benchmark
- IC and rank-IC plots
- stability analysis by subperiod
- stability analysis by volatility regime
- turnover and cost-drag visuals

### Files Involved
- Files involved: [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb), [docs/figures](/Users/assortedsphinx/Desktop/team_t/docs/figures).

### Outputs
- Outputs: `docs/figures/lstm_results_*.png`.

### Recommended Approach
- Recommended approach: keep existing figures as “Data Diagnostics” section and add “Strategy Results” section.

## Stage 14 — Technical notebook documentation

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: finalize notebook as a coherent research manuscript with stage-to-section mapping.

### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): Introduction/Data/Cleaning/Diagnostics/Preliminary Model sections exist.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): complete manuscript structure and map sections to stages:
1. Introduction → Stage 0
2. Data → Stage 1
3. Feature Construction → Stage 2
4. Model → Stages 4–5
5. Signal Analysis → Stages 6–7
6. Portfolio Construction → Stage 8
7. Backtest → Stages 9–11
8. Evaluation → Stages 12–13
9. Limitations → Stage 14
10. Conclusion → Stage 14
- Constraint handling (`PARTIALLY COMPLETE`): notebook currently does not reference the inspiration paper; keep that invariant in final narrative.

### Files Involved
- Files involved: [lstm.ipynb](/Users/assortedsphinx/Desktop/team_t/notebooks/lstm.ipynb).

### Outputs
- Outputs: final technical notebook ready for submission.

### Recommended Approach
- Recommended approach: preserve current text and insert missing sections after existing diagnostics/model setup.

## Stage 15 — Pitchbook creation

### Stage Summary
- Status: `NOT COMPLETE`
- Purpose: produce non-technical slide narrative consistent with technical notebook findings.

### Existing Implementation
- Existing implementation: none.

### Tasks to Implement
- Tasks to implement: create investor-facing summary with hypothesis, strategy, net/gross performance, benchmarks, risk, and limitations.

### Files Involved
- Files involved: new `/Users/assortedsphinx/Desktop/team_t/deliverables/pitchbook.pdf`.

### Outputs
- Outputs: final pitchbook PDF.

### Recommended Approach
- Recommended approach: align all figures/tables with Stage 12–13 outputs to avoid narrative drift.

## Stage 16 — Repository cleanup and submission packaging

### Stage Summary
- Status: `PARTIALLY COMPLETE`
- Purpose: package reproducible, grader-ready submission.

### Existing Implementation
- Existing implementation (`PARTIALLY COMPLETE`): modular source layout and draft exports are already present.

### Tasks to Implement
- Tasks to implement (`NOT COMPLETE`): dependency lock, run-order docs, output manifest, and assignment-compliant zip packaging.

### Files Involved
- Files involved: [README.md](/Users/assortedsphinx/Desktop/team_t/README.md), `/Users/assortedsphinx/Desktop/team_t/deliverables/`.

### Outputs
- Outputs: final zip package and reproducibility guide.

### Recommended Approach
- Recommended approach: package only required artifacts and references to large raw data sources.

---

## Important API / Interface Additions
- `src/model_utils.py`: model build/train/predict and walk-forward routines.
- `src/prediction_utils.py`: standardized prediction table creation.
- `src/signal_diagnostics.py`: IC/rank-IC/decile/spread/calibration/permutation/SHAP outputs.
- `src/signal_utils.py`: rank-to-signal mapping for long-only and long-short.
- `src/portfolio_utils.py`: signals-to-positions conversion.
- `src/backtest_utils.py`: replace placeholders with executable backtest/evaluation.
- `src/cost_utils.py`: explicit cost/funding layer.
- `src/risk_utils.py` and `src/exposure_utils.py`: position-based risk and exposure analytics.
- `src/performance_utils.py`: benchmark/inference/sensitivity/turnover analytics.

## Test Cases and Acceptance Scenarios
1. PIT integrity and no-lookahead checks pass on full panel.
2. Train/dev/test boundaries are deterministic and non-overlapping.
3. Tensor row count aligns with exported sample index.
4. Prediction table has one record per valid ticker-date sample.
5. IC/rank-IC and decile spread are computed for all test dates.
6. Long-short and long-only signal books are reproducible from predictions.
7. Stage 8 constraints (simultaneous assets, gross/net caps) are enforced.
8. Trade count and turnover outputs match portfolio transitions.
9. Gross and net backtests reconcile to cost breakdown components.
10. Benchmark comparison, t-tests, and bootstrap intervals are reproducible from fixed seeds.
11. Sensitivity analyses (`top-k`, lookback) run on consistent evaluation windows.
12. Subperiod and volatility-regime stability reports are generated.

13. Timing-leakage unit tests fail if any feature row at date `t` uses data from `t+1+`.
14. Execution-lag test verifies all trades from signal date `t` execute on `t+1` only.
15. Benchmark timing parity test verifies identical return timestamps and evaluation windows.

## Assumptions and Defaults
1. Primary predictive target remains `target_next_log_ret` (next-day log return).
2. Primary model setup is pooled cross-sectional LSTM; per-stock variant is optional robustness.
3. Strategy includes both long-short and long-only tracks.
4. Stage 8 positions are the single source for Stage 9–13 analytics.
5. Net performance is reported alongside gross for all benchmarked strategies.
6. Final notebook must remain independent in narrative and must not cite the inspiration paper.


- `Canonical timing convention` — `COMPLETE`
- The project uses one timing convention end-to-end: close-`t` information, signal after close `t`, execution at open `t+1`, PnL open-to-close `t+1`.
- `Pipeline consistency implementation` — `NOT COMPLETE`
- Remaining: propagate this convention consistently across Stage 4 target definition, Stage 7 signal logic, Stage 9 backtest engine, and Stage 12 benchmarks.
