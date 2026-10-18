# Glovo Courier RTO · Orphan Order Validation

A Streamlit prototype for reviewing courier recommendations for orphan orders. It brings together product estimates, a deterministic vehicle-capacity decision engine, human review, action history, and order-level analytics in one agent-facing workspace.

> **Prototype, not a dispatch system.** Recommendations and capacity values are planning aids based on configurable assumptions. Agent actions and “logged in internal app” statuses are records of what an agent reported; they do not dispatch couriers, update Glovo systems, or verify delivery outcomes.

## What it does

- Reads original orders and product-occurrence estimates from local data files.
- Uses available cached product enrichment to calculate order-level estimates and deterministic delivery recommendations.
- Separates complete product occurrences, valid physical estimates, recommendation availability, automatic eligibility, and human-review requirements.
- Routes orders needing attention to an agent review flow, with confirmation, override, escalation, and resolution history.
- Tracks reported internal-app logging separately from review decisions.
- Summarizes recommendation groups, overlapping review reasons, and workflow activity in Analytics.

The dashboard does **not** call Gemini when it starts or when agents navigate the app. Optional enrichment is a separate, explicitly run command described below.

## Quick start

Requirements: Python 3.10 or later and `pip`.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# Install dependencies and start the dashboard
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

Streamlit prints a local URL—usually <http://localhost:8501>—to open in your browser.

To run the test suite:

```bash
python -m pytest -q
```

## Dashboard guide

Use the navigation at the top of the page to switch between the four sections:

### Order Queue

Browse orders and filter by processing state, recommendation, review requirement, or agent status. The **Internal app status** column appears beside the order ID so it remains visible on narrower screens. Select an order row to open it in **Order Review**.

### Order Review

Inspect the order, product-level estimates, recommendation, and review reasons. Record a confirmation or override, or escalate the order. An escalation can have a human-verified resolution recorded separately.

The dashboard offers a distinct action to mark the latest eligible decision as logged in Glovo’s internal application. For an escalation, record its resolution first, then mark that resolution as logged. These controls only record the agent’s report and timestamp; reviewer identity is not captured.

### Analytics

Analytics use unique original orders as the denominator and show:

1. Dataset completeness and coverage at distinct stages.
2. Mutually exclusive recommendation categories, including split delivery and no valid recommendation.
3. Human-review categories and explicit product evidence for special handling.
4. Agent workflow activity, reported internal-app logging, and confirmation/override outcomes.

Review-reason groups may overlap, so their counts should not be summed to estimate the total number of orders requiring review. Recommendation categories are mutually exclusive and reconcile to the number of unique orders. **FTE and workload-impact calculations are not included in the dashboard.**

### Action History

Review saved actions and resolutions, including historical actions for the same order. Each action shows whether that specific decision was logged; a logged escalation resolution is identified separately. History is append-only: later actions update the latest status shown elsewhere but do not erase earlier records.

## Data and persistence

By default, the app reads:

| File | Purpose |
| --- | --- |
| `data/orphan_orders_case_dataset.csv` | Original order records and the source population for queue and analytics counts. |
| `data/processed_products.csv` | Product occurrences used to associate estimates with each order. |
| `data/ai_enrichment_cache.json` | Successful cached product estimates used by the dashboard. |
| `data/ai_enriched_sample.csv` | Legacy enrichment fallback, used when the cache does not produce enriched rows. |
| `data/agent_actions.db` | SQLite history for agent actions, resolutions, and agent-reported internal-app logging. |

You can change the data paths in the **Data Sources / Developer Settings** expander in the sidebar. The app can load a queue without enriched rows, but missing or incomplete product information will limit evaluation and may leave orders pending or requiring review.

The SQLite database is initialized automatically. Existing action records are retained. In Action History, each action shows whether that specific decision was logged in the internal app; a logged resolution is identified separately.

Preprocessing also writes `unique_products.csv`, `unmatched_products.csv`, `malformed_orders.csv`, and `summary_report.txt` as inspection outputs. They are not inputs to dashboard evaluation and can be regenerated by running preprocessing again.

## Optional product enrichment

The enrichment utility can request structured estimates from Gemini for a bounded sample of processed products. It is separate from the dashboard and requires a Gemini API key supplied through the `GEMINI_API_KEY` environment variable (a local `.env` file is also supported).

First preview the work without making API requests:

```bash
python -m src.ai_enrichment --dry-run
```

To run enrichment explicitly, for example for the default 20-product sample:

```bash
python -m src.ai_enrichment --limit 20
```

The command writes its output to `data/ai_enriched_sample.csv` and stores reusable successful estimates in `data/ai_enrichment_cache.json` by default. Review the command options with:

```bash
python -m src.ai_enrichment --help
```

Enrichment makes external API requests and may incur usage or quota costs. Do not run it unless you intend to use the provider; the dashboard itself only reads available files and cache entries.

## How recommendations should be interpreted

The decision engine compares estimated order weight, packaged volume, and item dimensions with configured vehicle capacities. It can recommend a single courier or a split delivery; incomplete, invalid, materially inconsistent, low-confidence, special-handling, or otherwise uncertain cases can be sent for human review. Liquid-content plausibility checks tolerate discrepancies up to 5% of estimated external packaged volume; smaller differences do not block a recommendation.

Vehicle capacities, safety margins, packing efficiency, confidence adjustments, and relative delivery costs are prototype assumptions—not official Glovo specifications. Special-handling labels are based on available product evidence and do not replace operational handling guidance.

Analytics are descriptive of the loaded dataset and saved agent activity. They do not establish ground-truth recommendation accuracy, real-world delivery success, representative staffing impact, or a time period for the dataset. Confirmation and override rates are not model-accuracy metrics; prototype and test actions may be present in the saved history.

## Code map

| Path | Responsibility |
| --- | --- |
| `dashboard.py` | Streamlit application, queue, review workflow, analytics, and action history. |
| `src/decision_engine.py` | Deterministic vehicle-capacity evaluation and recommendation logic. |
| `src/analytics.py` | Unique-order coverage, recommendation distributions, review categories, and workflow aggregates. |
| `src/dashboard_store.py` | SQLite persistence for agent actions, resolutions, and reported internal-app logging. |
| `src/ai_enrichment.py` | Optional structured Gemini enrichment command and cache handling. |
| `src/preprocessing.py` | Product and order preprocessing utilities. |
| `src/product_estimator.py` | Product measurement extraction and estimation helpers. |
| `tests/` | Unit and dashboard regression tests. |

## Current scope

This is an interview/demo prototype intended to make the decision and review workflow inspectable. Before operational use, validate the input data, vehicle constraints, special-handling policy, estimation quality, and workflow controls with the appropriate Glovo owners. The dashboard does not connect to Glovo’s dispatch or internal application systems.
