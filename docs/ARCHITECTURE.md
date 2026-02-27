# Project Architecture and Draft Notebook Design

## 1. Purpose of This Document
This document explains the architecture of the LSTM draft workflow in this repository:

- why the notebook is intentionally thin
- why implementation logic lives in `src/`
- what belongs in each module
- what is included in draft scope versus deferred to final scope
- how teammates should safely extend the codebase

The goal is a clean draft submission now, with minimal rework for final-stage training and backtesting.

## 2. High-Level Philosophy
The project follows a narrative-versus-engine separation model.

| Layer | Role |
| --- | --- |
| `notebooks/lstm.ipynb` | Reader-facing technical narrative and orchestration |
| `src/data_utils.py` | Data loading, cleaning, and point-in-time (PIT) correctness |
| `src/feature_engineering.py` | Feature construction, splits, tensor building, and diagnostic wrappers |
| `src/event_panels.py` | Event-time diagnostics internals (calendar, anchoring, panel construction) |
| `src/backtest_utils.py` | Deferred final-stage simulation placeholders |

The notebook is the research draft artifact. `src/` is the reusable implementation layer.

## 3. Why the Notebook Is Thin
Draft requirements ask for:

- substantial project exposition
- initial code for obtaining, cleaning, formatting, and arranging data
- at least three graphs
- Jupyter notebook format

They do not require that low-level infrastructure code be embedded inline. The project handout also encourages keeping source code volume small in the technical paper and importing from source modules.

Therefore, `lstm.ipynb` demonstrates and verifies the pipeline, while reusable internals are implemented in `src/`.

## 4. Data Flow Diagram
```text
Nasdaq Data Link + data/PRICES.csv
                |
                v
         src/data_utils.py
                |
                |-- API credential validation
                |-- table fetch and filtering
                |-- required prices load
                |-- universe selection
                |-- PIT merge + PIT validation
                v
     src/feature_engineering.py
                |
                |-- feature creation (P/B, beta, splits)
                |-- tensor construction
                |-- event diagnostics wrapper
                v
        notebooks/lstm.ipynb
                |
                |-- pipeline summary checkpoints
                |-- Figures 1-5 (static diagnostics)
                |-- LSTM framework section
                |-- tensor outputs (no training)
```

## 5. File Responsibilities in Detail
### `src/data_utils.py`
Primary objective: data integrity and PIT correctness.

Key responsibilities:

- `configure_api_from_env(...)` only loads and validates credentials (no fetch side effects)
- `fetch_zacks_table(...)` for controlled API access
- `load_prices_csv_required(...)` for deterministic `data/PRICES.csv` loading
- `build_static_top10_universe(...)` for fixed universe construction
- `prepare_fundamentals_with_availability(...)` for lagged availability setup
- `asof_join_point_in_time(...)` for PIT-safe merge
- `validate_point_in_time_panel(...)` for no-lookahead checks

### `src/feature_engineering.py`
Primary objective: model-ready transformations and lightweight orchestration helpers.

Key responsibilities:

- split-consistent intraday adjustment
- feature creation (`price_to_book`, rolling beta)
- split assignment
- LSTM tensor construction
- `compute_event_intensity_diagnostics(...)` wrapper used by notebook for Figures 4-5

### `src/event_panels.py`
Primary objective: heavy event-time internals.

Key responsibilities:

- market calendar construction
- fundamental event extraction and change filtering
- event-window panel construction for raw and beta-hedged metrics
- event-day aggregation logic

Notebook users should not need to reason about event-alignment internals.

### `src/backtest_utils.py`
Primary objective: reserve final-stage execution layer.

Current state:

- placeholder functions only
- intentionally unused in draft notebook

This prevents accidental performance analysis leakage into draft scope.

## 6. Draft Notebook Scope
`notebooks/lstm.ipynb` includes:

- title/compliance header
- motivation and strategy narrative
- data source and construction documentation
- data cleaning and formatting flow
- pipeline summary checkpoint cell
- five static diagnostic figures
- LSTM modeling framework explanation
- tensor construction only

It intentionally excludes training diagnostics and performance reporting.

## 7. Why Event Diagnostics Are Allowed
Figures 4 and 5 are diagnostic feature-behavior visualizations:

- they are not PnL curves
- they are not strategy evaluation metrics
- they do not include execution assumptions

They are used to justify modeling choices and temporal feature structure.

## 8. What Is Explicitly Not Included in Draft
The following are intentionally out of scope:

- hyperparameter tuning
- model training diagnostics
- backtesting and execution simulation
- transaction cost modeling
- leverage simulation
- risk-adjusted performance metrics (Sharpe, Sortino, etc.)

These items are deferred to final-stage deliverables.

## 9. How to Extend for Final Submission
To move from draft to final:

1. add model training/evaluation cells
2. implement simulation logic in `src/backtest_utils.py`
3. integrate transaction costs and funding assumptions
4. add portfolio construction and risk-control logic
5. produce performance diagnostics and attribution

Current architecture is designed to support this progression without restructuring notebook narrative cells.

## 10. Design Decisions Summary
| Decision | Reason |
| --- | --- |
| Require `data/PRICES.csv` | Deterministic, grader-safe input contract |
| No fallback filename | Avoid silent data mismatches |
| Thin notebook | Professional narrative, reduced clutter |
| Logic in `src/` | Maintainability, testability, reuse |
| Event wrapper in `feature_engineering` | Clean notebook API with hidden alignment internals |
| Deferred backtest module | Preserves strict draft scope |

## 11. Mental Model for Teammates
Use this model when contributing:

- notebook = technical draft paper
- `src/` = reusable quant library
- figures = feature and data-behavior diagnostics
- LSTM section = modeling blueprint
- final stage = add execution/performance layer, not rewrite data layer

## 12. Why This Structure Is Professional
This layout mirrors common quant-research standards:

- clean separation of narrative and implementation
- explicit PIT and leakage controls
- reproducible deterministic data inputs
- modular extensibility for later simulation work

It reduces notebook sprawl, hidden state, duplicated logic, and grading ambiguity.

## Wrapper Contract Reference
`compute_event_intensity_diagnostics(...)` in `src/feature_engineering.py` intentionally keeps notebook inputs minimal:

```python
compute_event_intensity_diagnostics(
    mode: str,
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    window: int = 60,
    beta_df: pd.DataFrame | None = None,
) -> dict[str, object]
```

Notebook passes only:

- `mode`
- prebuilt `prices_df`
- prebuilt `fundamentals_df`
- `window`
- optional `beta_df`

The wrapper handles event alignment internals and returns only plotting-ready outputs:

- `heatmap_df`
- `baseline_median`
- `baseline_mean`
- `event_panel_df`
- `figure_title`
- `colorbar_label`
- `y_label`

## Contributor Pre-PR Checklist
- Notebook does not directly import `src.event_panels` or `src.backtest_utils`.
- No hardcoded absolute data paths are introduced.
- `data/PRICES.csv` requirement remains deterministic.
- Notebook narrative remains concise and non-replicative.
- Draft notebook remains free of performance analysis content.
