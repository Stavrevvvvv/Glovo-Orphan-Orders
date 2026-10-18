"""Streamlit operations dashboard for Glovo orphan-order validation."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
	agent_workflow_analysis,
	dataset_coverage_metrics,
	recommendation_distribution,
	review_reason_analysis,
)
from src.dashboard_store import (
	latest_agent_actions,
	load_agent_actions,
	load_internal_app_logs,
	load_internal_resolution_logs,
	load_manual_resolutions,
	record_agent_action,
	record_internal_app_log,
	record_internal_resolution_log,
	record_manual_resolution,
)
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


def _processing_indicator(status: str) -> str:
	"""Map internal processing states to concise queue labels."""
	return {
		"pending enrichment": "🟡 Pending",
		"incomplete enrichment": "🟠 Incomplete",
		"enrichment review": "🟠 Enrichment review",
		"decision review": "🔴 Decision review",
		"decision available": "🟢 Ready",
	}.get(status, "⚪ Unknown")


def _review_reason_group(reason: str) -> str:
	"""Group decision messages by the operational issue they describe."""
	value = reason.lower()
	if "liquid volume plausibility" in value or "physical" in value or "below regex" in value:
		return "Physical inconsistency"
	if "missing" in value or "incomplete" in value or "does not have a completed" in value:
		return "Missing estimates or incomplete order"
	if "capacity" in value or "splitting" in value or "allocation" in value or "no safe" in value:
		return "Capacity or allocation"
	if "fragile" in value or "special handling" in value:
		return "Special handling"
	if "confidence" in value or "manual review" in value:
		return "Confidence or manual review"
	return "Other review reason"


def _unique_review_reasons(reasons: list[str]) -> list[str]:
	"""Remove exact repeats and equivalent duplicate fragile-handling wording."""
	unique: list[str] = []
	seen: set[str] = set()
	for reason in reasons:
		key = reason.strip().lower()
		if key in {"fragile or special handling", "one or more products require special handling"}:
			key = "special handling"
		if key not in seen:
			seen.add(key)
			unique.append(reason)
	return unique


def _group_review_reasons(decision: DecisionRecord, product_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
	"""Combine engine and product plausibility warnings without equivalent duplicates."""
	reasons = list(decision.review_reasons)
	reasons.extend(filter(None, (liquid_volume_plausibility_warning(row) for row in product_rows)))
	grouped: dict[str, list[str]] = {}
	for reason in _unique_review_reasons(reasons):
		grouped.setdefault(_review_reason_group(reason), []).append(reason)
	return grouped


def add_operational_review_status(queue: pd.DataFrame, latest_actions: pd.DataFrame, resolutions: pd.DataFrame | None = None) -> pd.DataFrame:
	"""Attach latest agent action status independently of engine review requirements."""
	queue = queue.copy()
	by_reference = {
		str(row["order_reference"]): row
		for _, row in latest_actions.iterrows()
	} if not latest_actions.empty else {}
	latest_resolutions: dict[str, Any] = {}
	if resolutions is not None and not resolutions.empty:
		latest_resolutions = {
			str(row["order_reference"]): row
			for _, row in resolutions.sort_values("id", ascending=False).drop_duplicates("order_reference").iterrows()
		}
	statuses = []
	for reference in queue["order_reference"].astype(str):
		action = by_reference.get(reference)
		if action is None:
			statuses.append({"operational_review_status": "⚪ Not reviewed", "latest_action": None, "latest_resolution": None})
		elif action["action"] == "Escalate for review":
			resolution = latest_resolutions.get(reference)
			if resolution is not None and int(resolution["escalation_action_id"]) == int(action["id"]):
				statuses.append({"operational_review_status": "🟣 Resolved", "latest_action": action, "latest_resolution": resolution})
			else:
				statuses.append({"operational_review_status": "🟠 Escalated", "latest_action": action, "latest_resolution": resolution})
		else:
			statuses.append({"operational_review_status": "🟢 Reviewed", "latest_action": action, "latest_resolution": latest_resolutions.get(reference)})
	queue["operational_review_status"] = [item["operational_review_status"] for item in statuses]
	queue["latest_action"] = [item["latest_action"] for item in statuses]
	queue["latest_resolution"] = [item["latest_resolution"] for item in statuses]
	return queue


def add_internal_app_log_status(queue: pd.DataFrame, internal_logs: pd.DataFrame, resolution_logs: pd.DataFrame | None = None) -> pd.DataFrame:
	"""Mark each order from the internal-app log for its latest action or resolution."""
	queue = queue.copy()
	logged_action_times = {int(row["agent_action_id"]): row["created_at"] for _, row in internal_logs.iterrows()} if not internal_logs.empty else {}
	logged_resolution_times = {int(row["resolution_id"]): row["created_at"] for _, row in resolution_logs.iterrows()} if resolution_logs is not None and not resolution_logs.empty else {}
	queue["decision_action_id"] = queue["latest_action"].map(
		lambda action: int(action["id"]) if action is not None and pd.notna(action.get("id")) else None
	)
	statuses, logged, timestamps = [], [], []
	for _, row in queue.iterrows():
		resolution = row.get("latest_resolution")
		resolution_id = int(resolution["id"]) if resolution is not None and pd.notna(resolution.get("id")) else None
		if row.get("operational_review_status") == "🟣 Resolved" and resolution_id is not None:
			is_logged = resolution_id in logged_resolution_times
			statuses.append("Resolution logged" if is_logged else "Not logged")
			logged.append(is_logged)
			timestamps.append(logged_resolution_times.get(resolution_id))
		else:
			action_id = row.get("decision_action_id")
			action_id = int(action_id) if action_id is not None and pd.notna(action_id) else None
			is_logged = action_id in logged_action_times if action_id is not None else False
			statuses.append("Logged" if is_logged else "Not logged")
			logged.append(is_logged)
			timestamps.append(logged_action_times.get(action_id) if action_id is not None else None)
	queue["internal_app_status"] = statuses
	queue["internal_app_logged"] = logged
	queue["internal_app_logged_at"] = timestamps
	return queue


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
	st.caption("Pending means no reliable recommendation; agent review and internal-app logging are separate states.")
	if queue.empty:
		st.info("No original orders are available.")
		return None
	logged = queue["internal_app_logged"].astype(bool)
	latest = queue["latest_action"]
	confirmed_or_overridden = latest.map(lambda action: action is not None and action["action"] in {"Confirm recommendation", "Override recommendation"})
	unresolved_escalation = latest.map(lambda action: action is not None and action["action"] == "Escalate for review")
	active_count = int((~logged).sum())
	reviewed_not_logged = int((confirmed_or_overridden & ~logged).sum())
	logged_count = int(logged.sum())
	escalated_count = int(unresolved_escalation.sum())
	kpis = st.columns(5)
	for column, label, value in zip(kpis, ("Dataset orders", "Active orders", "Reviewed, not logged", "Logged internally", "Unresolved escalations"), (len(queue), active_count, reviewed_not_logged, logged_count, escalated_count)):
		column.metric(label, value)

	scope_column, search_column, status_column, vehicle_column, engine_review_column, agent_status_column = st.columns([0.8, 1.2, 1.2, 0.9, 0.9, 1.0])
	scope = scope_column.selectbox("Orders", ["Active", "Logged", "All"], index=0)
	search = search_column.text_input("Order ID", placeholder="Search")
	status_options = ["All"] + sorted(queue["enrichment_status"].unique())
	status = status_column.selectbox("Processing", status_options)
	vehicle_options = ["All"] + sorted(queue["recommended_vehicle"].unique())
	vehicle = vehicle_column.selectbox("Vehicle", vehicle_options)
	review_filter = engine_review_column.selectbox("Human review", ["All", "Required", "Not required"])
	agent_status_options = ["All"] + sorted(queue["operational_review_status"].unique())
	agent_status = agent_status_column.selectbox("Agent status", agent_status_options)
	if scope == "Active":
		filtered = queue.loc[~logged].copy()
	elif scope == "Logged":
		filtered = queue.loc[logged].copy()
	else:
		filtered = queue.copy()
	if search:
		filtered = filtered[filtered["order_reference"].str.contains(search, case=False, na=False)]
	if status != "All":
		filtered = filtered[filtered["enrichment_status"] == status]
	if vehicle != "All":
		filtered = filtered[filtered["recommended_vehicle"] == vehicle]
	if review_filter == "Required":
		filtered = filtered[filtered["requires_human_review"]]
	elif review_filter == "Not required":
		filtered = filtered[~filtered["requires_human_review"]]
	if agent_status != "All":
		filtered = filtered[filtered["operational_review_status"] == agent_status]
	filtered = filtered.assign(
		_attention=filtered["requires_human_review"] & filtered["operational_review_status"].eq("⚪ Not reviewed"),
	).sort_values(["_attention", "order_reference"], ascending=[False, True])
	display = filtered.rename(columns={
		"order_reference": "Order ID",
		"product_count": "Items",
		"recommended_vehicle": "Recommended Vehicle",
		"recommended_courier_count": "Couriers",
		"decision_confidence": "Confidence",
		"requires_human_review": "Human Review",
		"order_complete": "Complete",
		"operational_review_status": "Agent Status",
	}).copy()
	display["Processing Status"] = display["enrichment_status"].map(_processing_indicator)
	display["Human Review"] = display["Human Review"].map({True: "🔴 Required", False: "—"})
	display["Complete"] = display["Complete"].map({True: "Yes", False: "No"})
	display["Internal app status"] = filtered["internal_app_status"].to_numpy()
	display = display[["Order ID", "Internal app status", "Items", "Processing Status", "Recommended Vehicle", "Couriers", "Confidence", "Human Review", "Complete", "Agent Status"]]
	st.caption("Select an order row to open its review.")
	selection = st.dataframe(
		display.reset_index(drop=True),
		width="stretch",
		hide_index=True,
		on_select="rerun",
		selection_mode="single-row",
		key="order_queue_table",
	)
	if filtered.empty:
		st.info("No orders match the current filters.")
		return None
	selected_rows = selection.selection.rows
	if selected_rows:
		return str(filtered.iloc[selected_rows[0]]["order_reference"])
	return None


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


def _display_numeric(value: Any, suffix: str = "") -> str:
	try:
		if pd.isna(value):
			return "—"
		return f"{float(value):.2f}{suffix}"
	except (TypeError, ValueError):
		return "—"


def _product_estimate_status(row: Any) -> str:
	fields = (
		"ai_estimated_transport_weight_kg", "ai_estimated_packaged_length_cm",
		"ai_estimated_packaged_width_cm", "ai_estimated_packaged_height_cm",
	)
	for field in fields:
		try:
			value = float(row.get(field))
			if not math.isfinite(value) or value <= 0:
				return "⚠ Missing / invalid estimate"
		except (TypeError, ValueError):
			return "⚠ Missing / invalid estimate"
	if liquid_volume_plausibility_warning(dict(row)):
		return "⚠ Physical inconsistency"
	return "✓ Estimate available"


def _action_form_defaults(decision: DecisionRecord) -> tuple[list[str], str | None, int | None]:
	"""Return safe action choices and defaults without inventing a recommendation."""
	if not decision.recommended_vehicle or not decision.recommended_courier_count:
		return ["Escalate for review"], None, None
	return ["Confirm recommendation", "Override recommendation", "Escalate for review"], decision.recommended_vehicle, decision.recommended_courier_count


def _action_summary(order_reference: str, action: str, vehicle: str | None, couriers: int | None, reason: str | None) -> str:
	"""Format an agent decision for manual transfer into Glovo internal tools."""
	return "\n".join((
		f"Order ID: {order_reference}",
		f"Action: {action}",
		f"Selected vehicle: {vehicle or 'Not specified'}",
		f"Courier count: {couriers if couriers is not None else 'Not specified'}",
		f"Reason: {reason or 'Not provided'}",
	))


def _local_action_timestamp(value: Any) -> tuple[str, date | None]:
	"""Format a stored UTC timestamp in local time and return its local date."""
	try:
		timestamp = pd.Timestamp(value)
		if timestamp.tzinfo is None:
			timestamp = timestamp.tz_localize("UTC")
		local_timestamp = timestamp.to_pydatetime().astimezone()
		return local_timestamp.strftime("%b %d, %Y · %I:%M %p %Z"), local_timestamp.date()
	except (TypeError, ValueError):
		return str(value), None


def build_action_history_view(
	actions: pd.DataFrame,
	queue: pd.DataFrame,
	latest_only: bool = True,
	internal_logs: pd.DataFrame | None = None,
	resolutions: pd.DataFrame | None = None,
	resolution_logs: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""Build a readable latest-only or full action history with current order status."""
	columns = [
		"Order ID", "Action", "Selected vehicle", "Courier count", "Reason",
		"Timestamp", "Current review status", "Internal app status",
	]
	if actions.empty:
		return pd.DataFrame(columns=columns + ["_local_date"])
	latest_rows = latest_agent_actions_from_frame(actions)
	rows = latest_rows if latest_only else actions.copy()
	queue_status = {
		str(row["order_reference"]): (row["recommended_vehicle"], row["operational_review_status"])
		for _, row in queue.iterrows()
	} if not queue.empty else {}
	current_status_by_order = {
		str(row["order_reference"]): row["operational_review_status"]
		for _, row in queue.iterrows()
	} if not queue.empty else {}
	action_labels = {
		"Confirm recommendation": "Confirmed",
		"Override recommendation": "Overridden",
		"Escalate for review": "Escalated",
	}
	logged_action_ids = {
		int(value) for value in internal_logs.get("agent_action_id", pd.Series(dtype=int)).dropna()
	} if internal_logs is not None else set()
	logged_resolution_ids = {
		int(value) for value in resolution_logs.get("resolution_id", pd.Series(dtype=int)).dropna()
	} if resolution_logs is not None else set()
	logged_escalation_ids = set()
	if resolutions is not None and not resolutions.empty and logged_resolution_ids:
		logged_escalation_ids = {
			int(row["escalation_action_id"])
			for _, row in resolutions.iterrows()
			if int(row["id"]) in logged_resolution_ids
		}
	records = []
	for _, row in rows.iterrows():
		reference = str(row["order_reference"])
		recommended_vehicle, _ = queue_status.get(reference, ("Pending", "⚪ Not reviewed"))
		operational_status = current_status_by_order.get(reference, "⚪ Not reviewed")
		if operational_status == "🟠 Escalated":
			current_status = "Escalated"
		elif operational_status == "🟣 Resolved":
			current_status = "Resolved"
		elif operational_status == "🟢 Reviewed":
			current_status = "Reviewed"
		else:
			current_status = "Pending" if recommended_vehicle == "Pending" else "Not Reviewed"
		action_id = int(row["id"]) if pd.notna(row.get("id")) else None
		if action_id in logged_action_ids:
			internal_app_status = "Logged"
		elif action_id in logged_escalation_ids:
			internal_app_status = "Resolution logged"
		else:
			internal_app_status = "Not logged"
		formatted_timestamp, local_date = _local_action_timestamp(row["created_at"])
		records.append({
			"Order ID": reference,
			"Action": action_labels.get(row["action"], row["action"]),
			"Selected vehicle": row["selected_vehicle"],
			"Courier count": row["courier_count"],
			"Reason": row["reason"],
			"Timestamp": formatted_timestamp,
			"Current review status": current_status,
			"Internal app status": internal_app_status,
			"_local_date": local_date,
			"_id": row.get("id"),
		})
	return pd.DataFrame(records).sort_values(["_id", "Order ID"], ascending=[False, True], na_position="last").reset_index(drop=True)


def latest_agent_actions_from_frame(actions: pd.DataFrame) -> pd.DataFrame:
	"""Select latest rows from already-loaded history without another DB read."""
	order_column = "id" if "id" in actions else "created_at"
	return actions.sort_values(order_column, ascending=False).drop_duplicates("order_reference", keep="first")


def filter_action_history(
	history: pd.DataFrame,
	order_search: str = "",
	action_types: list[str] | None = None,
	vehicles: list[str] | None = None,
	date_range: tuple[date, date] | None = None,
) -> pd.DataFrame:
	"""Apply the history page's order, action, vehicle and local-date filters."""
	filtered = history.copy()
	if order_search:
		filtered = filtered[filtered["Order ID"].astype(str).str.contains(order_search, case=False, na=False)]
	if action_types:
		filtered = filtered[filtered["Action"].isin(action_types)]
	if vehicles:
		filtered = filtered[filtered["Selected vehicle"].fillna("Not specified").isin(vehicles)]
	if date_range and len(date_range) == 2:
		start_date, end_date = date_range
		filtered = filtered[filtered["_local_date"].map(lambda value: value is not None and start_date <= value <= end_date)]
	return filtered.reset_index(drop=True)


def _format_action_timestamp(value: Any) -> str:
	try:
		timestamp = pd.Timestamp(value)
		if timestamp.tzinfo is None:
			timestamp = timestamp.tz_localize("UTC")
		else:
			timestamp = timestamp.tz_convert("UTC")
		return timestamp.strftime("%b %d, %Y · %H:%M UTC")
	except (TypeError, ValueError):
		return str(value)


def _metric_card(column: Any, label: str, value: Any) -> None:
	"""Render one recommendation metric with a native Streamlit border."""
	with column.container(border=True):
		st.metric(label, value)


def _display_review(
	selected_reference: str,
	queue: pd.DataFrame,
	enriched: pd.DataFrame,
	database_path: Path,
	actions: pd.DataFrame,
	internal_logs: pd.DataFrame,
	resolutions: pd.DataFrame,
	internal_resolution_logs: pd.DataFrame,
) -> None:
	st.subheader("Order Review")
	selected = queue[queue["order_reference"] == selected_reference].iloc[0]
	decision: DecisionRecord = selected["decision"]
	latest_action = selected.get("latest_action")
	original_products = _parse_product_json(selected["original_products_json"])
	order_enriched = enriched[enriched["order_reference"].astype(str) == selected_reference] if not enriched.empty else pd.DataFrame()
	weight_rows = order_enriched.to_dict("records")
	base_weight, safety_margin_weight, conservative_weight = calculate_order_weight_breakdown(weight_rows, DecisionConfig())
	saved_message = st.session_state.pop("action_saved_message", None)
	if saved_message:
		st.success(saved_message)
	st.markdown(f"### Order `{selected_reference}`")
	latest_resolution = selected.get("latest_resolution")
	is_resolved = selected["operational_review_status"] == "🟣 Resolved" and latest_resolution is not None
	valid_internal_log_decision = (
		latest_action is not None
		and latest_action["action"] in {"Confirm recommendation", "Override recommendation"}
		and latest_action["selected_vehicle"] in {vehicle.name for vehicle in DEFAULT_VEHICLES}
		and pd.notna(latest_action["courier_count"])
		and int(latest_action["courier_count"]) > 0
		and decision.order_complete
	)
	with st.container(border=True):
		status_columns = st.columns(5)
		status_columns[0].caption(f"Processing: {_processing_indicator(selected['enrichment_status'])}")
		status_columns[1].caption(f"Order complete: {'Yes' if decision.order_complete else 'No'}")
		status_columns[2].caption(f"Agent status: {selected['operational_review_status']}")
		status_columns[3].caption(f"Internal app: {'🔵 Logged' if selected.get('internal_app_logged', False) else '⚪ Not logged'}")
		status_columns[4].caption(f"Action required: {'Yes' if decision.requires_human_review else 'No'}")
	if latest_action is None:
		st.info("Not reviewed yet. Record an agent action below to mark this order as reviewed.")
	else:
		st.caption(f"Latest action: {latest_action['action']} · {_format_action_timestamp(latest_action['created_at'])}. Reviewer identity is not recorded.")
	if selected.get("internal_app_logged", False):
		st.success(f"Agent reports the latest decision was logged in the internal app · {_format_action_timestamp(selected.get('internal_app_logged_at'))}. Dispatch is not verified.")
	elif valid_internal_log_decision:
		st.caption("After completing the corresponding step in Glovo’s internal app, mark it here. This records your report; it does not verify dispatch.")
		if st.button("Mark latest decision as logged in internal app", key=f"mark_internal_log_{selected_reference}"):
			try:
				record_internal_app_log(database_path, selected_reference, int(latest_action["id"]))
				st.rerun()
			except ValueError as error:
				st.error(str(error))
	elif is_resolved:
		st.caption("After recording the verified resolution in Glovo’s internal app, mark it here. This records your report; it does not verify dispatch.")
		if st.button("Mark resolution as logged in internal app", key=f"mark_resolution_logged_{selected_reference}"):
			try:
				record_internal_resolution_log(database_path, selected_reference, int(latest_resolution["id"]))
				st.rerun()
			except ValueError as error:
				st.error(str(error))
	elif latest_action is not None and latest_action["action"] == "Escalate for review":
		st.info("Record a human-verified resolution before marking an escalation as logged.")
	st.markdown("#### Recommendation summary")
	recommendation_columns = st.columns(4)
	_metric_card(recommendation_columns[0], "Vehicle", decision.recommended_vehicle or "Awaiting estimation")
	_metric_card(recommendation_columns[1], "Couriers", decision.recommended_courier_count or "Awaiting estimation")
	_metric_card(recommendation_columns[2], "Raw weight", f"{base_weight:.2f} kg" if base_weight is not None else "Not available")
	_metric_card(recommendation_columns[3], "Conservative weight", f"{conservative_weight:.2f} kg" if conservative_weight is not None else "Not available")
	confidence_columns = st.columns(3)
	_metric_card(confidence_columns[0], "Packaged volume", f"{decision.estimated_total_packaged_volume_litres:.2f} L" if decision.estimated_total_packaged_volume_litres is not None else "Not available")
	_metric_card(confidence_columns[1], "AI confidence", _aggregate_ai_confidence(order_enriched))
	_metric_card(confidence_columns[2], "Decision confidence", decision.decision_confidence)
	st.caption(f"Weight safety margin: +{safety_margin_weight:.2f} kg" if safety_margin_weight is not None else "Weight safety margin: Not available")
	left, right = st.columns([1.15, 1])
	with left:
		st.markdown("#### Decision explanation and capacity assumptions")
		st.write(decision.decision_explanation)
		for assumption in decision.vehicle_capacity_assumptions:
			st.caption(assumption)
		st.caption("Original product names, quantities and source values are preserved.")
		original_frame = pd.DataFrame(original_products).rename(columns={
			"name": "Original product",
			"quantity": "Qty",
			"unit_price": "Unit price",
			"declared_weight_kg": "Declared weight (kg)",
		})
		if original_frame.empty:
			original_frame = pd.DataFrame(columns=["Original product", "Qty", "Unit price", "Declared weight (kg)"])
		original_tab, estimates_tab, details_tab = st.tabs(["Original Products", "AI Estimates", "Detailed Calculations"])
		with original_tab:
			def original_estimate_status(name: str) -> str:
				matches = order_enriched[order_enriched["product_name"] == name] if not order_enriched.empty else pd.DataFrame()
				if matches.empty:
					return "⚪ Awaiting enrichment"
				return _product_estimate_status(matches.iloc[0])
			original_frame["Estimate status"] = original_frame["Original product"].map(original_estimate_status)
			st.dataframe(original_frame, width="stretch", hide_index=True)
		with estimates_tab:
			if order_enriched.empty:
				st.info("No cached Gemini estimates are available for this order.")
			else:
				estimates = order_enriched.copy()
				estimates["Estimate status"] = estimates.apply(_product_estimate_status, axis=1)
				estimates = estimates.rename(columns={
					"product_name": "Product",
					"quantity": "Qty",
					"ai_estimated_transport_weight_kg": "Transport wt (kg)",
					"ai_estimated_packaged_length_cm": "Length (cm)",
					"ai_estimated_packaged_width_cm": "Width (cm)",
					"ai_estimated_packaged_height_cm": "Height (cm)",
					"ai_confidence": "Confidence",
				})
				for field in ("Transport wt (kg)", "Length (cm)", "Width (cm)", "Height (cm)"):
					if field in estimates:
						estimates[field] = estimates[field].map(lambda value: _display_numeric(value))
				st.dataframe(estimates[["Product", "Qty", "Transport wt (kg)", "Length (cm)", "Width (cm)", "Height (cm)", "Confidence", "Estimate status"]], width="stretch", hide_index=True)
		with details_tab:
			if order_enriched.empty:
				st.info("Detailed estimates will appear after products are enriched.")
			else:
				details_columns = [field for field in (
					"product_name", "quantity", "ai_estimate_explanation", "matched_text",
					"extraction_warning", "extracted_volume_litres",
				) if field in order_enriched]
				details = order_enriched[details_columns].copy()
				if {"quantity", "ai_estimated_transport_weight_kg"}.issubset(order_enriched.columns):
					details["estimated_line_weight_kg"] = pd.to_numeric(order_enriched["quantity"], errors="coerce") * pd.to_numeric(order_enriched["ai_estimated_transport_weight_kg"], errors="coerce")
				if {"ai_estimated_packaged_length_cm", "ai_estimated_packaged_width_cm", "ai_estimated_packaged_height_cm"}.issubset(order_enriched.columns):
					details["external_volume_per_unit_l"] = (
						pd.to_numeric(order_enriched["ai_estimated_packaged_length_cm"], errors="coerce")
						* pd.to_numeric(order_enriched["ai_estimated_packaged_width_cm"], errors="coerce")
						* pd.to_numeric(order_enriched["ai_estimated_packaged_height_cm"], errors="coerce") / 1000
					)
					if "quantity" in order_enriched:
						details["external_volume_per_line_l"] = details["external_volume_per_unit_l"] * pd.to_numeric(order_enriched["quantity"], errors="coerce")
				details = details.rename(columns={
					"product_name": "Product", "quantity": "Qty", "ai_estimate_explanation": "AI estimate explanation",
					"matched_text": "Regex evidence", "extraction_warning": "Regex warning",
					"extracted_volume_litres": "Extracted liquid content (L)",
					"estimated_line_weight_kg": "Line weight (kg)",
					"external_volume_per_unit_l": "External volume / unit (L)",
					"external_volume_per_line_l": "External volume / line (L)",
				})
				st.dataframe(details, width="stretch", hide_index=True)
	with right:
		st.markdown("#### Review and action")
		selected_vehicle = next((vehicle for vehicle in DEFAULT_VEHICLES if vehicle.name == decision.recommended_vehicle), None)
		if selected_vehicle and decision.recommended_courier_count:
			courier_capacity = decision.recommended_courier_count
			weight_use = decision.estimated_total_weight_kg / (selected_vehicle.max_weight_kg * courier_capacity) if decision.estimated_total_weight_kg is not None else None
			volume_use = decision.estimated_total_packaged_volume_litres / (selected_vehicle.max_volume_litres * courier_capacity) if decision.estimated_total_packaged_volume_litres is not None else None
			st.caption(f"Selected vehicle utilization · Weight {weight_use:.0%} / Volume {volume_use:.0%}" if weight_use is not None and volume_use is not None else "Capacity utilization unavailable")
		else:
			st.caption("Capacity utilization unavailable until a vehicle is recommended.")
		for category, messages in _group_review_reasons(decision, weight_rows).items():
			text = "\n".join(f"- {message}" for message in messages)
			if category in {"Physical inconsistency", "Missing estimates or incomplete order"}:
				st.error(f"**{category}**\n\n{text}")
			elif category in {"Capacity or allocation", "Confidence or manual review", "Special handling"}:
				st.warning(f"**{category}**\n\n{text}")
			else:
				st.info(f"**{category}**\n\n{text}")
		action_required = decision.requires_human_review
		st.write("**Human review:**", "🔴 Required" if action_required else "No")
		if latest_action is not None:
			st.caption(f"Latest action: {latest_action['action']} · {_format_action_timestamp(latest_action['created_at'])}")
			if latest_action["reason"]:
				st.caption(f"Action reason: {latest_action['reason']}")
		if selected["operational_review_status"] == "🟠 Escalated" and latest_action is not None:
			st.warning("This order remains escalated. The AI recommendation is still Pending; resolve only with externally verified information.")
			st.markdown("#### Resolve Escalation")
			outcome_label = st.radio(
				"Human-verified outcome",
				["Verified delivery decision", "Customer intervention required"],
				horizontal=True,
				key=f"resolution_outcome_{selected_reference}",
			)
			resolution_vehicle = None
			resolution_couriers = None
			if outcome_label == "Verified delivery decision":
				resolution_vehicle = st.selectbox("Externally verified vehicle", [vehicle.name for vehicle in DEFAULT_VEHICLES], index=None, placeholder="Select verified vehicle", key=f"resolution_vehicle_{selected_reference}")
				courier_text = st.text_input("Externally verified courier count", key=f"resolution_couriers_{selected_reference}")
				try:
					resolution_couriers = int(courier_text) if courier_text.strip() else None
				except ValueError:
					resolution_couriers = None
			verification_explanation = st.text_area("How was this decision verified? (required)", key=f"resolution_explanation_{selected_reference}")
			verification_source = st.text_input("Verification source (optional)", placeholder="Partner catalogue, supervisor confirmation, etc.", key=f"resolution_source_{selected_reference}")
			if st.button("Save human-verified resolution", type="primary", key=f"resolve_{selected_reference}"):
				if not verification_explanation.strip():
					st.error("Enter an explanation of how the resolution was verified.")
				elif outcome_label == "Verified delivery decision" and (resolution_vehicle is None or resolution_couriers is None or resolution_couriers < 1):
					st.error("Select a verified vehicle and enter a positive courier count.")
				else:
					outcome = "delivery" if outcome_label == "Verified delivery decision" else "customer_intervention"
					try:
						record_manual_resolution(
							database_path, selected_reference, int(latest_action["id"]), outcome,
							verification_explanation, verification_source or None,
							resolution_vehicle, resolution_couriers,
						)
						st.session_state["action_saved_message"] = f"Saved human-verified resolution for order {selected_reference}. AI recommendation remains Pending; internal-app logging is still separate."
						st.rerun()
					except ValueError as error:
						st.error(str(error))
		elif selected["operational_review_status"] == "🟣 Resolved" and latest_resolution is not None:
			st.success("Human-verified resolution · AI recommendation remains Pending")
			outcome_text = "Verified delivery decision" if latest_resolution["outcome"] == "delivery" else "Customer intervention required"
			st.write("**Resolution:**", outcome_text)
			if latest_resolution["outcome"] == "delivery":
				st.write("**Human-selected vehicle / couriers:**", f"{latest_resolution['selected_vehicle']} / {latest_resolution['courier_count']}")
			st.write("**Verification explanation:**", latest_resolution["explanation"])
			st.write("**Verification source:**", latest_resolution["verification_source"] or "Not specified")
			st.caption(f"Resolved {_format_action_timestamp(latest_resolution['created_at'])}. This does not verify dispatch or delivery.")
		elif selected["operational_review_status"] != "🟣 Resolved":
			st.markdown("#### Agent Action")
			st.caption("Recording an action does not dispatch a courier or modify Glovo systems. Complete the operational step in Glovo's internal tools; dispatch status is not verified here.")
			allow_rereview = True
			if latest_action is not None:
				allow_rereview = st.checkbox("Re-review this order", key=f"rereview_{selected_reference}")
			action_options, default_vehicle, default_couriers = _action_form_defaults(decision)
			action = st.radio("Action", action_options, index=0, horizontal=True, key=f"action_{selected_reference}")
			vehicle: str | None = None
			couriers: int | None = None
			if decision.recommended_vehicle and action != "Escalate for review":
				vehicle_names = [item.name for item in DEFAULT_VEHICLES]
				vehicle = st.selectbox("Vehicle", vehicle_names, index=vehicle_names.index(default_vehicle), disabled=action == "Confirm recommendation", key=f"vehicle_{selected_reference}")
				couriers = st.number_input("Couriers", min_value=1, max_value=10, value=default_couriers, disabled=action == "Confirm recommendation", key=f"couriers_{selected_reference}")
			if action == "Confirm recommendation":
				st.caption(f"Confirming recommendation: {vehicle} · {couriers} courier(s).")
			reason = st.text_input("Reason" if action != "Confirm recommendation" else "Note (optional)", key=f"reason_{selected_reference}")
			if st.button("Save action", type="primary", key=f"save_{selected_reference}", disabled=not allow_rereview):
				if action != "Confirm recommendation" and not reason.strip():
					st.error("A reason is required for overrides and escalations.")
				else:
					selected_vehicle = vehicle if action == "Override recommendation" else decision.recommended_vehicle if action == "Confirm recommendation" else None
					selected_couriers = int(couriers) if action == "Override recommendation" and couriers is not None else decision.recommended_courier_count if action == "Confirm recommendation" else None
					saved_reason = reason.strip() or None
					action_reason = saved_reason or ("Agent confirmed the deterministic recommendation." if action == "Confirm recommendation" else "")
					record_agent_action(
						database_path, selected_reference, action, selected_vehicle, selected_couriers,
						action_reason,
					)
					st.session_state["action_saved_message"] = (
						f"Recorded {action.lower()} for order {selected_reference} · {selected_vehicle or 'No vehicle'} · "
						f"{selected_couriers if selected_couriers is not None else 'No courier count'} courier(s). "
						"This prototype only records the decision. Complete the operational step in Glovo's internal tools; dispatch is not verified."
					)
					st.session_state.pop(f"rereview_{selected_reference}", None)
					st.rerun()
		with st.expander("Previous actions"):
			previous = actions[actions["order_reference"].astype(str) == selected_reference] if not actions.empty else pd.DataFrame()
			if previous.empty:
				st.caption("No previous actions recorded for this order.")
			else:
				history = build_action_history_view(
					previous, queue, latest_only=False, internal_logs=internal_logs,
					resolutions=resolutions, resolution_logs=internal_resolution_logs,
				)
				st.dataframe(history.drop(columns=["_local_date", "_id"], errors="ignore"), width="stretch", hide_index=True)
			previous_resolutions = resolutions[resolutions["order_reference"].astype(str) == selected_reference] if not resolutions.empty else pd.DataFrame()
			if not previous_resolutions.empty:
				st.markdown("**Human-verified resolutions**")
				resolution_view = previous_resolutions.copy()
				resolution_view["Outcome"] = resolution_view["outcome"].map({"delivery": "Human-verified delivery decision", "customer_intervention": "Customer intervention required"})
				resolution_view["Resolved at"] = resolution_view["created_at"].map(lambda value: _local_action_timestamp(value)[0])
				st.dataframe(resolution_view[["Outcome", "selected_vehicle", "courier_count", "explanation", "verification_source", "Resolved at"]].rename(columns={"selected_vehicle": "Vehicle", "courier_count": "Couriers", "explanation": "Verification explanation", "verification_source": "Verification source"}), width="stretch", hide_index=True)
		if latest_action is not None and latest_action["action"] in {"Confirm recommendation", "Override recommendation"}:
			st.markdown("**Action summary**")
			st.caption("Use the code-block copy control to transfer this to Glovo's internal tools.")
			st.code(_action_summary(selected_reference, latest_action["action"], latest_action["selected_vehicle"], latest_action["courier_count"], latest_action["reason"]), language=None)
		with st.expander("Previous human-verified resolutions"):
			previous_resolutions = resolutions[resolutions["order_reference"].astype(str) == selected_reference] if not resolutions.empty else pd.DataFrame()
			if previous_resolutions.empty:
				st.caption("No manual escalation resolutions recorded for this order.")
			else:
				resolution_history = previous_resolutions.copy()
				resolution_history["Outcome"] = resolution_history["outcome"].map({"delivery": "Human-verified delivery decision", "customer_intervention": "Customer intervention required"})
				resolution_history["Resolved at"] = resolution_history["created_at"].map(lambda value: _format_action_timestamp(value))
				st.dataframe(resolution_history[["Outcome", "selected_vehicle", "courier_count", "explanation", "verification_source", "Resolved at"]].rename(columns={"selected_vehicle": "Vehicle", "courier_count": "Couriers", "explanation": "Verification explanation", "verification_source": "Source"}), width="stretch", hide_index=True)
		if decision.alternative_options:
			with st.expander("Alternative delivery options"):
				st.dataframe(pd.DataFrame([option.to_dict() for option in decision.alternative_options]), width="stretch", hide_index=True)


def _open_order_review(order_reference: str) -> None:
	"""Select an order and navigate directly to its existing review page."""
	st.session_state["selected_order"] = str(order_reference)
	st.session_state["dashboard_section"] = "Order Review"


def _display_action_history(
	actions: pd.DataFrame,
	queue: pd.DataFrame,
	resolutions: pd.DataFrame | None = None,
	internal_logs: pd.DataFrame | None = None,
	resolution_logs: pd.DataFrame | None = None,
) -> None:
	st.subheader("Action History")
	st.caption("Append-only record of agent decisions. This is not courier dispatch history.")
	search = ""
	date_range = None
	if actions.empty:
		st.info("No agent actions have been recorded yet.")
	else:
		latest_only = not st.toggle("Show full history", value=False, help="By default, show the latest action for each order.")
		history = build_action_history_view(
			actions, queue, latest_only, internal_logs, resolutions, resolution_logs,
		)
		search_column, action_column, vehicle_column = st.columns([1.4, 1, 1])
		search = search_column.text_input("Search Order ID", key="action_history_search")
		action_types = action_column.multiselect("Action type", sorted(history["Action"].dropna().unique()), key="action_history_types")
		vehicles = sorted(history["Selected vehicle"].dropna().astype(str).unique())
		if history["Selected vehicle"].isna().any():
			vehicles.append("Not specified")
		selected_vehicles = vehicle_column.multiselect("Vehicle", vehicles, key="action_history_vehicles")
		dates = [value for value in history["_local_date"] if isinstance(value, date)]
		if dates:
			date_range = st.date_input(
				"Date range (local time)", value=(min(dates), max(dates)),
				min_value=min(dates), max_value=max(dates), key="action_history_dates",
			)
			if isinstance(date_range, date):
				date_range = (date_range, date_range)
			elif len(date_range) != 2:
				date_range = None
		filtered = filter_action_history(history, search, action_types, selected_vehicles, date_range)
		if filtered.empty:
			st.info("No actions match these filters.")
		else:
			display = filtered.drop(columns=["_local_date", "_id"], errors="ignore")
			st.dataframe(display, width="stretch", hide_index=True)
			st.download_button(
				"Export filtered history as CSV",
				display.to_csv(index=False).encode("utf-8"),
				file_name="glovo_action_history.csv",
				mime="text/csv",
			)
			order_ids = filtered["Order ID"].astype(str).drop_duplicates().tolist()
			selected_order = st.selectbox("Select a historical order", order_ids, key="historical_order_to_review")
			st.button("Open in Order Review", type="primary", on_click=_open_order_review, args=(selected_order,))
	if resolutions is not None and not resolutions.empty:
		st.markdown("#### Human-verified escalation resolutions")
		resolution_view = resolutions.copy()
		if search:
			resolution_view = resolution_view[resolution_view["order_reference"].astype(str).str.contains(search, case=False, na=False)]
		if date_range is not None:
			resolution_dates = resolution_view["created_at"].map(lambda value: _local_action_timestamp(value)[1])
			resolution_view = resolution_view.loc[resolution_dates.map(lambda value: value is not None and date_range[0] <= value <= date_range[1])]
		status_by_order = dict(zip(queue["order_reference"].astype(str), queue["operational_review_status"]))
		resolution_view["Outcome"] = resolution_view["outcome"].map({"delivery": "Human-verified delivery decision", "customer_intervention": "Customer intervention required"})
		resolution_view["Resolved at"] = resolution_view["created_at"].map(lambda value: _local_action_timestamp(value)[0])
		resolution_view["Current review status"] = resolution_view["order_reference"].astype(str).map(status_by_order).fillna("Resolved")
		resolution_display = resolution_view.rename(columns={
			"order_reference": "Order ID", "selected_vehicle": "Vehicle", "courier_count": "Couriers",
			"explanation": "Verification explanation", "verification_source": "Verification source",
		})[["Order ID", "Outcome", "Vehicle", "Couriers", "Verification explanation", "Verification source", "Resolved at", "Current review status"]]
		if resolution_display.empty:
			st.info("No human-verified resolutions match these filters.")
		else:
			st.dataframe(resolution_display, width="stretch", hide_index=True)
			st.download_button("Export filtered resolution history as CSV", resolution_display.to_csv(index=False).encode("utf-8"), file_name="glovo_resolution_history.csv", mime="text/csv")


def _display_analytics(queue: pd.DataFrame, actions: pd.DataFrame, enriched: pd.DataFrame) -> None:
	st.subheader("Analytics")
	if queue.empty:
		st.info("Analytics are unavailable until original orders are loaded.")
		return

	coverage = dataset_coverage_metrics(queue)
	total_orders = int(coverage["total_orders"])

	def show_order_metric(column: Any, label: str, key: str, help_text: str) -> None:
		count = int(coverage[key])
		percentage = 100.0 if key == "total_orders" else float(coverage[f"{key}_pct"])
		column.metric(label, f"{count:,}")
		column.caption(f"{count:,} / {total_orders:,} orders · {percentage:.1f}%")
		column.caption(help_text)

	st.markdown("### 1. Dataset Coverage")
	st.caption("Each KPI counts unique original orders. Completeness, evaluability, recommendation availability and automation eligibility are separate stages.")
	coverage_labels = [
		("Total dataset orders", "total_orders", "Denominator for all coverage percentages."),
		("Original occurrences accounted for", "occurrence_complete_orders", "Engine verified all original product occurrences."),
		("Physically evaluable", "physically_evaluable_orders", "Required transport weights and packaged dimensions are valid."),
		("Valid delivery recommendation", "recommended_orders", "A supported vehicle and positive courier count are available."),
		("Eligible for automatic validation", "eligible_orders", "Complete recommendation with no human-review requirement."),
		("Require human review", "human_review_orders", "Decision engine marks the order for review."),
		("No valid recommendation", "no_recommendation_orders", "No supported vehicle and courier-count pair was produced."),
	]
	first_row = st.columns(4)
	for column, metric in zip(first_row, coverage_labels[:4]):
		show_order_metric(column, *metric)
	second_row = st.columns(3)
	for column, metric in zip(second_row, coverage_labels[4:]):
		show_order_metric(column, *metric)

	st.markdown("### 2. Delivery Recommendations")
	distribution = recommendation_distribution(queue)
	distribution["Bar label"] = distribution.apply(
		lambda row: f"{int(row['Orders']):,} · {row['Percent of dataset']:.1f}%",
		axis=1,
	)
	recommendation_chart = px.bar(
		distribution,
		x="Orders",
		y="Category",
		orientation="h",
		text="Bar label",
		custom_data=["Percent of dataset"],
		labels={"Orders": "Orders", "Category": ""},
	)
	recommendation_chart.update_traces(
		textposition="outside",
		hovertemplate="%{y}<br>%{x:,} orders<br>%{customdata[0]:.1f}% of dataset<extra></extra>",
	)
	recommendation_chart.update_layout(
		height=330,
		yaxis={"categoryorder": "array", "categoryarray": distribution["Category"].tolist()[::-1]},
		margin={"l": 12, "r": 24, "t": 10, "b": 10},
		showlegend=False,
	)
	st.plotly_chart(recommendation_chart, width="stretch")
	multi_courier_orders = int(distribution.attrs["multi_courier_orders"])
	multi_columns = st.columns([1, 2])
	multi_columns[0].metric("Orders requiring multiple couriers", f"{multi_courier_orders:,}")
	multi_columns[0].caption(f"{multi_courier_orders:,} / {total_orders:,} original orders · {_percentage_display(multi_courier_orders, total_orders)}")
	multi_columns[1].caption("Split delivery is a mutually exclusive chart category. The vehicle breakdown below is a subset of those split orders, not an additional recommendation total.")
	split_detail = distribution.attrs["split_by_vehicle"]
	with st.expander("Split-delivery breakdown by recommended vehicle"):
		if split_detail.empty:
			st.caption("No multi-courier recommendations are present.")
		else:
			st.dataframe(split_detail, width="stretch", hide_index=True)

	st.markdown("### 3. Human Review and Bottlenecks")
	review_categories, special_subtypes, cold_chain_info, detailed_reasons = review_reason_analysis(queue, enriched)
	review_chart_data = review_categories.copy()
	review_chart_data["Bar label"] = review_chart_data.apply(
		lambda row: f"{int(row['Orders affected']):,} · {row['Percent of dataset']:.1f}%",
		axis=1,
	)
	review_chart = px.bar(
		review_chart_data.sort_values("Orders affected"),
		x="Orders affected",
		y="Review category",
		orientation="h",
		text="Bar label",
		custom_data=["Percent of dataset"],
		labels={"Orders affected": "Orders affected", "Review category": ""},
	)
	review_chart.update_traces(
		textposition="outside",
		hovertemplate="%{y}<br>%{x:,} orders<br>%{customdata[0]:.1f}% of dataset<extra></extra>",
	)
	review_chart.update_layout(height=350, margin={"l": 12, "r": 24, "t": 10, "b": 10}, showlegend=False)
	st.plotly_chart(review_chart, width="stretch")
	st.caption("Review categories may overlap; each percentage uses all unique dataset orders as its denominator. Do not sum these category counts into an overall review rate.")
	bottlenecks = review_categories[review_categories["Review category"] != "Other"].nlargest(3, "Orders affected")
	if not bottlenecks.empty:
		top_labels = ", ".join(
			f"{row['Review category']} ({int(row['Orders affected']):,})"
			for _, row in bottlenecks.iterrows()
		)
		st.info(f"Most frequently tagged review bottlenecks: {top_labels}. These categories overlap and are not estimates of orders that would become automatable if a single issue were addressed.")
	with st.expander("Special-handling evidence from existing product text"):
		st.caption("Subtypes are shown only where existing product names, descriptions or categories contain direct evidence. Refrigerated/perishable wording is informational and is not itself treated as a review trigger.")
		if not special_subtypes.empty:
			st.dataframe(special_subtypes, width="stretch", hide_index=True)
		if not cold_chain_info.empty:
			st.dataframe(cold_chain_info, width="stretch", hide_index=True)
		if special_subtypes.empty and int(cold_chain_info.loc[0, "Orders affected"]) == 0:
			st.caption("No directly identifiable special-handling or cold-chain wording was found.")
	with st.expander("Detailed original review reasons"):
		if detailed_reasons.empty:
			st.caption("No orders currently require human review.")
		else:
			st.dataframe(detailed_reasons, width="stretch", hide_index=True)

	st.markdown("### 4. Agent Workflow Activity")
	st.caption("Workflow status is separate from model coverage. Counts include all saved records, including prototype/test activity; they are not verified production outcomes or dispatch evidence.")
	status_counts, action_summary = agent_workflow_analysis(queue, actions)
	workflow_metrics = [
		("Not reviewed", "not_reviewed"),
		("Reviewed", "reviewed"),
		("Escalated", "escalated"),
		("Resolved", "resolved"),
		("Logged in internal app · agent-reported", "logged"),
		("Reviewed but not yet logged", "reviewed_not_logged"),
	]
	for row_start in (0, 3):
		columns = st.columns(3)
		for column, (label, key) in zip(columns, workflow_metrics[row_start:row_start + 3]):
			count = status_counts[key]
			column.metric(label, f"{count:,}")
			column.caption(f"{count:,} / {total_orders:,} unique orders · {_percentage_display(count, total_orders)}")
	st.markdown("#### Latest saved action per order")
	st.caption("Action counts use only the latest action for each order. Confirmation and override rates use only latest confirmation/override actions; historical records remain available in Action History.")
	action_metrics = [
		("Confirmed", "confirmations"),
		("Overridden", "overrides"),
		("Escalated", "escalations"),
		("Confirm/override actions", "confirm_override_total"),
	]
	columns = st.columns(4)
	for column, (label, key) in zip(columns, action_metrics):
		column.metric(label, f"{action_summary[key]:,}")
	columns = st.columns(2)
	columns[0].metric("Confirmation rate", f"{action_summary['confirmation_rate']:.1f}%")
	columns[1].metric("Override rate", f"{action_summary['override_rate']:.1f}%")
	st.caption(f"Rates use {action_summary['confirm_override_total']:,} latest confirmation/override actions as the denominator; they are not model-accuracy measures.")
	with st.expander("Append-only agent action records"):
		if actions.empty:
			st.caption("No agent actions have been recorded.")
		else:
			history = build_action_history_view(
				actions, queue, latest_only=False, internal_logs=internal_logs,
				resolutions=resolutions, resolution_logs=internal_resolution_logs,
			)
			st.dataframe(history.drop(columns=["_local_date", "_id"], errors="ignore"), width="stretch", hide_index=True)


def _percentage_display(count: int, denominator: int) -> str:
	return f"{count / denominator:.1%}" if denominator else "0.0%"


def main() -> None:
	st.set_page_config(page_title="Glovo RTO Validation", page_icon="▣", layout="wide")
	navigation_order = st.session_state.pop("open_order_review", None)
	if navigation_order is not None:
		st.session_state["selected_order"] = str(navigation_order)
		st.session_state["order_review_selector"] = str(navigation_order)
		st.session_state["dashboard_section"] = "Order Review"
	st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:3rem}.small-note{color:#64748b}</style>", unsafe_allow_html=True)
	st.title("Courier RTO · Orphan Validation")
	st.caption("Agent workspace · Validation support only, not courier dispatch")
	with st.sidebar:
		st.markdown("### Operations")
		st.caption("Review enriched orders, record decisions, and escalate exceptions.")
		st.info("Agent actions do not dispatch couriers or update Glovo systems.")
		with st.expander("Data Sources / Developer Settings"):
			original_path = Path(st.text_input("Original orders CSV", str(DEFAULT_ORIGINAL_PATH)))
			processed_path = Path(st.text_input("Processed occurrences CSV", str(DEFAULT_PROCESSED_PATH)))
			cache_path = Path(st.text_input("Enrichment cache JSON", str(DEFAULT_CACHE_PATH)))
			enriched_path = Path(st.text_input("Legacy enrichment CSV fallback", str(DEFAULT_ENRICHED_PATH)))
			database_path = Path(st.text_input("Agent action database", str(DEFAULT_DATABASE_PATH)))
			st.caption("Read-only data load; no Gemini API calls are made by the dashboard.")
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
	actions = load_agent_actions(database_path)
	internal_logs = load_internal_app_logs(database_path)
	resolutions = load_manual_resolutions(database_path)
	internal_resolution_logs = load_internal_resolution_logs(database_path)
	queue = add_operational_review_status(build_order_queue(original, enriched), latest_agent_actions(database_path), resolutions)
	queue = add_internal_app_log_status(queue, internal_logs, internal_resolution_logs)
	sections = ["Order Queue", "Order Review", "Analytics", "Action History"]
	section = st.radio("Dashboard section", sections, horizontal=True, label_visibility="collapsed", key="dashboard_section")
	if section == "Order Queue":
		selected_reference = _display_queue(queue)
		if selected_reference:
			st.session_state["open_order_review"] = selected_reference
			st.rerun()
	elif section == "Order Review":
		if queue.empty:
			st.info("No orders are available to review.")
		else:
			order_ids = queue["order_reference"].tolist()
			default_index = order_ids.index(st.session_state["selected_order"]) if st.session_state.get("selected_order") in order_ids else 0
			selected_reference = st.selectbox("Order ID", order_ids, index=default_index, key="order_review_selector")
			st.session_state["selected_order"] = selected_reference
			_display_review(selected_reference, queue, enriched, database_path, actions, internal_logs, resolutions, internal_resolution_logs)
	elif section == "Analytics":
		_display_analytics(queue, actions, enriched)
	else:
		_display_action_history(actions, queue, resolutions, internal_logs, internal_resolution_logs)


if __name__ == "__main__":
	main()