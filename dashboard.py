"""Streamlit operations dashboard for Glovo orphan-order validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_store import load_agent_actions, record_agent_action
from src.decision_engine import (
	DEFAULT_VEHICLES,
	DecisionConfig,
	DecisionRecord,
	calculate_order_weight_breakdown,
	evaluate_order,
	liquid_volume_plausibility_warning,
)
from src.ai_enrichment import context_from_row, is_cacheable_result, product_identifier, GeminiEnrichment


ROOT = Path(__file__).parent
DEFAULT_ORIGINAL_PATH = ROOT / "data/orphan_orders_case_dataset.csv"
DEFAULT_PROCESSED_PATH = ROOT / "data/processed_products.csv"
DEFAULT_ENRICHED_PATH = ROOT / "data/ai_enriched_sample.csv"
DEFAULT_CACHE_PATH = ROOT / "data/ai_enrichment_cache.json"
DEFAULT_DATABASE_PATH = ROOT / "data/agent_actions.db"


def _source_token(*paths: Path) -> tuple[tuple[str, int, int], ...]:
	return tuple(
		(str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (str(path), 0, 0)
		for path in paths
	)


@st.cache_data(show_spinner=False)
def load_csv(path: str, source_token: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
	"""Cache a CSV read and invalidate it when the source file changes."""
	file_path = Path(path)
	if not file_path.exists():
		return pd.DataFrame()
	return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_json_cache(path: str, source_token: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
	"""Load successful enrichment cache data without making provider requests."""
	file_path = Path(path)
	if not file_path.exists():
		return {}
	try:
		value = json.loads(file_path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}
	return value if isinstance(value, dict) else {}


@st.cache_data(show_spinner=False)
def build_cached_enrichment(processed: pd.DataFrame, cache: dict[str, Any]) -> pd.DataFrame:
	"""Expand valid cached unique estimates back onto every processed occurrence."""
	if processed.empty or not cache:
		return pd.DataFrame()
	rows: list[dict[str, Any]] = []
	for _, row in processed.iterrows():
		try:
			context = context_from_row(row)
			key = product_identifier(context)
			cached = cache.get(key)
			result = GeminiEnrichment.model_validate(cached) if isinstance(cached, dict) else None
			if result is None or not is_cacheable_result(result):
				continue
		except Exception:
			continue
		output = row.to_dict()
		output.update({
			"ai_status": "completed",
			"ai_cache_hit": True,
			"ai_error": None,
			"ai_error_kind": None,
		})
		output.update({f"ai_{field}": value for field, value in result.model_dump().items()})
		rows.append(output)
	return pd.DataFrame(rows)


def _parse_product_json(value: Any) -> list[dict[str, Any]]:
	if not isinstance(value, str):
		return []
	try:
		parsed = json.loads(value)
	except json.JSONDecodeError:
		return []
	return parsed if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed) else []


def _decision_status(decision: DecisionRecord, enriched_rows: pd.DataFrame) -> str:
	if enriched_rows.empty:
		return "pending enrichment"
	if not decision.order_complete:
		return "incomplete enrichment"
	if (enriched_rows.get("ai_status", pd.Series(dtype=str)) != "completed").any():
		return "enrichment review"
	if decision.requires_human_review:
		return "decision review"
	return "decision available"


@st.cache_data(show_spinner=False)
def build_order_queue(original: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
	"""Build one queue row per original order without making API calls."""
	if original.empty:
		return pd.DataFrame()
	enriched = enriched.copy()
	if not enriched.empty:
		enriched["order_reference"] = enriched["order_reference"].astype(str)
	rows: list[dict[str, Any]] = []
	for _, order in original.iterrows():
		reference = str(order["order_id"])
		order_enriched = enriched[enriched["order_reference"] == reference] if not enriched.empty else enriched
		product_rows = order_enriched.to_dict("records")
		decision = evaluate_order(reference, product_rows, original_products_json=order.get("products_json"))
		rows.append({
			"order_reference": reference,
			"product_count": int(order.get("product_lines", len(_parse_product_json(order.get("products_json")))) or 0),
			"enrichment_status": _decision_status(decision, order_enriched),
			"recommended_vehicle": decision.recommended_vehicle or "Pending",
			"recommended_courier_count": str(decision.recommended_courier_count or "Pending"),
			"decision_confidence": decision.decision_confidence,
			"requires_human_review": decision.requires_human_review,
			"order_complete": decision.order_complete,
			"review_reasons": " | ".join(decision.review_reasons),
			"decision": decision,
			"original_products_json": order.get("products_json"),
		})
	return pd.DataFrame(rows)


def _display_queue(queue: pd.DataFrame) -> str | None:
	st.subheader("Order Queue")
	st.caption("All original orders remain visible. Pending and incomplete orders are never treated as validated.")
	if queue.empty:
		st.info("No original orders are available.")
		return None
	search = st.text_input("Search order ID", placeholder="e.g. 862376000")
	statuses = st.multiselect("Processing status", sorted(queue["enrichment_status"].unique()))
	vehicles = st.multiselect("Recommended vehicle", sorted(queue["recommended_vehicle"].unique()))
	review_filter = st.selectbox("Human review", ["All", "Review required", "No review required"])
	filtered = queue.copy()
	if search:
		filtered = filtered[filtered["order_reference"].str.contains(search, case=False, na=False)]
	if statuses:
		filtered = filtered[filtered["enrichment_status"].isin(statuses)]
	if vehicles:
		filtered = filtered[filtered["recommended_vehicle"].isin(vehicles)]
	if review_filter == "Review required":
		filtered = filtered[filtered["requires_human_review"]]
	elif review_filter == "No review required":
		filtered = filtered[~filtered["requires_human_review"]]
	st.dataframe(
		filtered.drop(columns=["decision", "original_products_json"], errors="ignore"),
		width="stretch",
		hide_index=True,
	)
	if filtered.empty:
		st.info("No orders match the current filters.")
		return None
	return st.selectbox("Select an order to review", filtered["order_reference"].tolist())


def _save_action(database_path: Path, order_reference: str, action: str, vehicle: str | None, couriers: int | None, reason: str) -> None:
	record_agent_action(database_path, order_reference, action, vehicle, couriers, reason)
	st.success(f"Saved {action.lower()} for order {order_reference}.")


def _aggregate_ai_confidence(enriched_rows: pd.DataFrame) -> str:
	"""Aggregate product-level AI confidence conservatively for display only."""
	if enriched_rows.empty or "ai_confidence" not in enriched_rows:
		return "Pending"
	confidences = {str(value).lower() for value in enriched_rows["ai_confidence"].dropna()}
	if not confidences:
		return "Pending"
	if "low" in confidences:
		return "Low"
	if "medium" in confidences:
		return "Medium"
	return "High" if confidences == {"high"} else "Unknown"


def _display_review(selected_reference: str, queue: pd.DataFrame, enriched: pd.DataFrame, database_path: Path) -> None:
	st.subheader("Order Review")
	selected = queue[queue["order_reference"] == selected_reference].iloc[0]
	decision: DecisionRecord = selected["decision"]
	original_products = _parse_product_json(selected["original_products_json"])
	order_enriched = enriched[enriched["order_reference"].astype(str) == selected_reference] if not enriched.empty else pd.DataFrame()
	weight_rows = order_enriched.to_dict("records")
	base_weight, safety_margin_weight, conservative_weight = calculate_order_weight_breakdown(weight_rows, DecisionConfig())
	left, right = st.columns([1.35, 1])
	with left:
		st.markdown(f"### Order `{selected_reference}`")
		st.caption("This dashboard is an internal validation tool. It does not dispatch couriers or modify Glovo systems.")
		st.markdown("**Original products**")
		st.dataframe(pd.DataFrame(original_products), width="stretch", hide_index=True)
		if order_enriched.empty:
			st.warning("No cached Gemini enrichment exists for this order.")
		else:
			columns = [
				"product_name", "quantity", "ai_estimated_transport_weight_kg",
				"ai_estimated_packaged_length_cm", "ai_estimated_packaged_width_cm",
				"ai_estimated_packaged_height_cm", "ai_confidence", "ai_estimate_explanation",
			]
			order_enriched = order_enriched.copy()
			order_enriched["liquid_volume_warning"] = order_enriched.apply(
				lambda row: liquid_volume_plausibility_warning(row.to_dict()) or "",
				axis=1,
			)
			columns.append("liquid_volume_warning")
			st.markdown("**Enriched product estimates**")
			st.dataframe(order_enriched[[column for column in columns if column in order_enriched.columns]], width="stretch", hide_index=True)
	with right:
		st.metric("AI transport weight", f"{base_weight:.2f} kg" if base_weight is not None else "Pending")
		st.metric("Weight safety margin", f"{safety_margin_weight:.2f} kg" if safety_margin_weight is not None else "Pending")
		st.metric("Conservative dispatch weight", f"{conservative_weight:.2f} kg" if conservative_weight is not None else "Pending")
		st.metric("Estimated packaged volume", f"{decision.estimated_total_packaged_volume_litres:.2f} L" if decision.estimated_total_packaged_volume_litres is not None else "Pending")
		st.metric("Recommended vehicle", decision.recommended_vehicle or "Pending")
		st.metric("Couriers", decision.recommended_courier_count or "Pending")
		st.write("**AI estimation confidence:**", _aggregate_ai_confidence(order_enriched))
		st.write("**Operational decision confidence:**", decision.decision_confidence)
		st.write("**Complete order:**", "Yes" if decision.order_complete else "No")
		st.write("**Review reasons:**", " | ".join(decision.review_reasons) or "None")
		warnings = [liquid_volume_plausibility_warning(row) for row in weight_rows]
		for warning in [warning for warning in warnings if warning]:
			st.warning(warning)
		st.write("**Decision explanation:**", decision.decision_explanation)
		st.write("**Capacity assumptions:**", " ".join(decision.vehicle_capacity_assumptions))
		if decision.alternative_options:
			st.markdown("**Alternative options**")
			st.dataframe(pd.DataFrame([option.to_dict() for option in decision.alternative_options]), width="stretch", hide_index=True)

	if decision.recommended_vehicle is None:
		st.warning("This order cannot be automatically validated until enrichment or completeness issues are resolved.")
	st.markdown("**Agent action**")
	actions = ["Confirm recommendation", "Override recommendation", "Escalate for review"] if decision.recommended_vehicle else ["Escalate for review"]
	action = st.radio("Select an action", actions, horizontal=True, key=f"action_{selected_reference}")
	vehicle_names = [vehicle.name for vehicle in DEFAULT_VEHICLES]
	vehicle = st.selectbox("Vehicle", vehicle_names, index=vehicle_names.index(decision.recommended_vehicle) if decision.recommended_vehicle in vehicle_names else 0, disabled=action == "Confirm recommendation", key=f"vehicle_{selected_reference}")
	couriers = st.number_input("Courier count", min_value=1, max_value=10, value=decision.recommended_courier_count or 1, disabled=action == "Confirm recommendation", key=f"couriers_{selected_reference}")
	reason = st.text_area("Action reason", placeholder="Required for overrides and escalations", key=f"reason_{selected_reference}")
	if st.button("Save action", type="primary", key=f"save_{selected_reference}"):
		if action != "Confirm recommendation" and not reason.strip():
			st.error("Add a reason for an override or escalation.")
		else:
			_save_action(database_path, selected_reference, action, vehicle if action != "Confirm recommendation" else decision.recommended_vehicle, int(couriers) if action != "Confirm recommendation" else decision.recommended_courier_count, reason.strip() or "Agent confirmed the deterministic recommendation.")


def _display_analytics(queue: pd.DataFrame, actions: pd.DataFrame) -> None:
	st.subheader("Analytics")
	if queue.empty:
		st.info("Analytics are unavailable until original orders are loaded.")
		return
	complete = int(queue["order_complete"].sum())
	review_pct = float(queue["requires_human_review"].mean() * 100)
	metrics = st.columns(4)
	metrics[0].metric("Total orders", len(queue))
	metrics[1].metric("Complete orders", complete)
	metrics[2].metric("Incomplete orders", len(queue) - complete)
	metrics[3].metric("Human review", f"{review_pct:.1f}%")
	vehicle_counts = queue.loc[queue["recommended_vehicle"] != "Pending", "recommended_vehicle"].value_counts().rename_axis("vehicle").reset_index(name="orders")
	if vehicle_counts.empty:
		st.info("No reliable vehicle recommendations are available yet.")
	else:
		st.plotly_chart(px.bar(vehicle_counts, x="vehicle", y="orders", title="Recommended vehicles"), width="stretch")
	reasons = Counter(reason for values in queue["review_reasons"] for reason in values.split(" | ") if reason)
	if reasons:
		st.markdown("**Common review reasons**")
		st.dataframe(pd.DataFrame(reasons.most_common(), columns=["reason", "orders"]), width="stretch", hide_index=True)
	st.markdown("**Agent actions**")
	if actions.empty:
		st.info("No agent actions have been recorded yet.")
	else:
		confirmations = int((actions["action"] == "Confirm recommendation").sum())
		overrides = int((actions["action"] == "Override recommendation").sum())
		action_count = len(actions)
		st.write(f"Confirmation rate: {confirmations / action_count:.1%} | Override rate: {overrides / action_count:.1%}")
		escalation_reasons = actions.loc[actions["action"] == "Escalate for review", "reason"].dropna()
		if not escalation_reasons.empty:
			st.markdown("**Recorded escalation reasons**")
			st.dataframe(escalation_reasons.value_counts().rename_axis("reason").reset_index(name="escalations"), width="stretch", hide_index=True)
		st.dataframe(actions, width="stretch", hide_index=True)
	st.markdown("**FTE-equivalent workload estimate**")
	st.caption("Hypothetical planning estimate, not measured operational savings or a Glovo staffing commitment.")
	manual_minutes = st.number_input("Assumed manual review minutes per order", min_value=1.0, value=8.0, step=1.0)
	automated_minutes = st.number_input("Assumed automated handling minutes per eligible order", min_value=0.0, value=1.0, step=0.5)
	weekly_hours = st.number_input("Assumed productive hours per FTE per week", min_value=1.0, value=35.0, step=1.0)
	eligible = int(((queue["order_complete"]) & (~queue["requires_human_review"]) & (queue["recommended_vehicle"] != "Pending")).sum())
	saved_hours = eligible * max(manual_minutes - automated_minutes, 0) / 60
	st.write(f"Eligible reliable orders: {eligible}. Estimated workload reduction: {saved_hours:.1f} hours, or {saved_hours / weekly_hours:.2f} FTE-equivalent per comparable week.")


def main() -> None:
	st.set_page_config(page_title="Glovo RTO Validation", page_icon="▣", layout="wide")
	st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:3rem}.small-note{color:#64748b}</style>", unsafe_allow_html=True)
	st.title("Courier RTO · Orphan Validation")
	st.caption("Agent workspace for deterministic review of enriched grocery orders")
	with st.sidebar:
		st.markdown("### Data sources")
		original_path = Path(st.text_input("Original orders CSV", str(DEFAULT_ORIGINAL_PATH)))
		processed_path = Path(st.text_input("Processed product occurrences CSV", str(DEFAULT_PROCESSED_PATH)))
		cache_path = Path(st.text_input("Gemini enrichment cache JSON", str(DEFAULT_CACHE_PATH)))
		enriched_path = Path(st.text_input("Legacy enrichment CSV fallback", str(DEFAULT_ENRICHED_PATH)))
		database_path = Path(st.text_input("Agent action database", str(DEFAULT_DATABASE_PATH)))
		st.info("No API requests are made by this dashboard. Gemini results are read from the local cache JSON.")
	token = _source_token(original_path, processed_path, cache_path, enriched_path)
	original = load_csv(str(original_path), token)
	processed = load_csv(str(processed_path), token)
	cache = load_json_cache(str(cache_path), token)
	enriched = build_cached_enrichment(processed, cache)
	if enriched.empty:
		enriched = load_csv(str(enriched_path), token)
	if original.empty:
		st.error("Original order data could not be loaded. Check the configured CSV path.")
		return
	queue = build_order_queue(original, enriched)
	actions = load_agent_actions(database_path)
	tab_queue, tab_review, tab_analytics = st.tabs(["Order Queue", "Order Review", "Analytics"])
	with tab_queue:
		selected_reference = _display_queue(queue)
	with tab_review:
		if "selected_order" not in st.session_state:
			st.session_state.selected_order = selected_reference if selected_reference else queue.iloc[0]["order_reference"]
		if selected_reference:
			st.session_state.selected_order = selected_reference
		if queue.empty:
			st.info("Select an order after loading the queue.")
		else:
			_review_reference = st.selectbox("Order to inspect", queue["order_reference"].tolist(), index=queue["order_reference"].tolist().index(st.session_state.selected_order) if st.session_state.selected_order in queue["order_reference"].tolist() else 0)
			_display_review(_review_reference, queue, enriched, database_path)
	with tab_analytics:
		_display_analytics(queue, actions)


if __name__ == "__main__":
	main()