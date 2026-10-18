## Streamlit dashboard

Launch the agent-facing dashboard with:

```bash
streamlit run dashboard.py
```

The dashboard reads original orders from `data/orphan_orders_case_dataset.csv`, processed occurrences from `data/processed_products.csv`, and cached Gemini results from `data/ai_enrichment_cache.json`. It never calls Gemini automatically. The enrichment CSV is retained only as a fallback. Orders without complete cached enrichment remain pending or require review and are not treated as validated.

Agent confirmations, overrides, and escalations are stored in `data/agent_actions.db` and persist across restarts. Vehicle capacities, safety margins, relative costs, and FTE calculations are explicitly prototype assumptions, not official Glovo data or verified operational measurements.
