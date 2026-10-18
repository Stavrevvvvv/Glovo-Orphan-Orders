"""Per-order analytics helpers for the orphan-validation dashboard."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd


REVIEW_CATEGORIES = (
	"Missing physical information",
	"Physical inconsistencies",
	"Special handling",
	"Low confidence",
	"Capacity constraints",
	"Allocation uncertainty",
	"Other",
)
RECOMMENDATION_CATEGORIES = (
	"Bicycle",
	"Motorbike",
	"Car",
	"Split delivery",
	"No valid recommendation",
)
_ACTION_DECISIONS = {"Confirm recommendation", "Override recommendation"}


def _unique_orders(queue: pd.DataFrame) -> pd.DataFrame:
	if queue.empty or "order_reference" not in queue:
		return queue.iloc[0:0].copy()
	return queue.drop_duplicates("order_reference", keep="last").copy()


def _decision(row: pd.Series) -> Any:
	return row.get("decision")


def _has_recommendation(decision: Any) -> bool:
	return bool(
		decision is not None
		and decision.recommended_vehicle
		and decision.recommended_courier_count is not None
		and decision.recommended_courier_count > 0
	)


def _percentage(count: int, denominator: int) -> float:
	return count / denominator * 100 if denominator else 0.0


def dataset_coverage_metrics(queue: pd.DataFrame) -> dict[str, int | float]:
	"""Count order-level completeness, evaluation, recommendation and review states."""
	orders = _unique_orders(queue)
	total = len(orders)
	decisions = orders.apply(_decision, axis=1)
	occurrence_complete = decisions.map(lambda item: bool(item and item.order_complete))
	physically_evaluable = decisions.map(
		lambda item: bool(
			item
			and item.estimated_total_weight_kg is not None
			and item.estimated_total_packaged_volume_litres is not None
		)
	)
	has_recommendation = decisions.map(_has_recommendation)
	automatically_eligible = decisions.map(
		lambda item: bool(
			item and item.order_complete and not item.requires_human_review
			and item.estimated_total_weight_kg is not None
			and item.estimated_total_packaged_volume_litres is not None
			and _has_recommendation(item)
		)
	)
	requires_review = decisions.map(lambda item: bool(item and item.requires_human_review))
	counts = {
		"total_orders": total,
		"occurrence_complete_orders": int(occurrence_complete.sum()),
		"physically_evaluable_orders": int(physically_evaluable.sum()),
		"recommended_orders": int(has_recommendation.sum()),
		"eligible_orders": int(automatically_eligible.sum()),
		"human_review_orders": int(requires_review.sum()),
		"no_recommendation_orders": int((~has_recommendation).sum()),
	}
	counts.update({
		f"{key}_pct": _percentage(value, total)
		for key, value in counts.items()
		if key != "total_orders"
	})
	return counts


def recommendation_distribution(queue: pd.DataFrame) -> pd.DataFrame:
	"""Return mutually exclusive delivery-mode counts that reconcile to unique orders."""
	orders = _unique_orders(queue)
	counts: Counter[str] = Counter()
	split_by_vehicle: Counter[str] = Counter()
	multi_courier_order_ids: set[str] = set()
	for _, row in orders.iterrows():
		decision = _decision(row)
		if not _has_recommendation(decision):
			counts["No valid recommendation"] += 1
		elif decision.recommended_courier_count > 1:
			counts["Split delivery"] += 1
			split_by_vehicle[str(decision.recommended_vehicle)] += 1
			multi_courier_order_ids.add(str(row["order_reference"]))
		else:
			counts[str(decision.recommended_vehicle)] += 1
	for category in RECOMMENDATION_CATEGORIES:
		counts.setdefault(category, 0)
	for category in counts:
		if category not in RECOMMENDATION_CATEGORIES:
			counts[category] += 0
	data = pd.DataFrame([
		{"Category": category, "Orders": count, "Percent of dataset": _percentage(count, len(orders))}
		for category, count in counts.items()
	])
	data = data.sort_values("Orders", ascending=False).reset_index(drop=True)
	split_total = sum(split_by_vehicle.values())
	split_detail = pd.DataFrame([
		{
			"Recommended vehicle": vehicle,
			"Split-delivery orders": count,
			"Percent of split deliveries": _percentage(count, split_total),
		}
		for vehicle, count in sorted(split_by_vehicle.items())
	])
	data.attrs["multi_courier_orders"] = len(multi_courier_order_ids)
	data.attrs["split_by_vehicle"] = split_detail
	return data


def _reason_category(reason: str) -> str:
	value = reason.casefold()
	if "low-confidence" in value or "low confidence" in value:
		return "Low confidence"
	if any(term in value for term in (
		"missing", "incomplete", "does not have a completed", "not have a completed",
		"original products_json", "product quantity", "packaged dimensions",
		"transport weight", "ai confidence",
	)):
		return "Missing physical information"
	if any(term in value for term in (
		"liquid volume", "below regex", "regex extraction warning", "physical inconsistency",
	)):
		return "Physical inconsistencies"
	if any(term in value for term in ("fragile", "hazard", "frozen", "special handling")):
		return "Special handling"
	if any(term in value for term in ("capacity", "no safe allocation", "splitting is required", "item limits")):
		return "Capacity constraints"
	if any(term in value for term in ("allocation", "heuristic", "optimality", "could change the recommendation")):
		return "Allocation uncertainty"
	return "Other"


_SPECIAL_EVIDENCE_PATTERNS = {
	"Fragile / breakable": re.compile(
		r"\b(fragile|breakable|glass|eggs?|bruise\w*|leak\w*|spill\w*|crush\w*|delicate|ceramic)\b",
		re.IGNORECASE,
	),
	"Hazardous": re.compile(
		r"\b(hazard\w*|flammab\w*|explos\w*|pressur\w*|aerosol|lithium battery)\b",
		re.IGNORECASE,
	),
	"Frozen": re.compile(
		r"\b(frozen|freezer|dry ice|deep[- ]frozen|requires? freezing)\b",
		re.IGNORECASE,
	),
	"Unusually difficult": re.compile(
		r"\b(awkward|oversized|difficult to handle|unusually difficult)\b",
		re.IGNORECASE,
	),
}
_ROUTINE_COLD_CHAIN = re.compile(
	r"\b(refrigerat\w*|cold[- ]chain|perishable|chilled|keep cold|temperature[- ]controlled transport)\b",
	re.IGNORECASE,
)


def _product_evidence(row: dict[str, Any]) -> str:
	return " ".join(
		str(row.get(field) or "")
		for field in (
			"product_name", "ai_normalized_product_description",
			"ai_product_category", "ai_estimate_explanation",
		)
	)


def review_reason_analysis(
	queue: pd.DataFrame,
	enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Group review reasons per unique order and surface only explicit product evidence."""
	orders = _unique_orders(queue)
	total = len(orders)
	enriched_by_order: dict[str, list[dict[str, Any]]] = {}
	if not enriched.empty and "order_reference" in enriched:
		for reference, rows in enriched.groupby(enriched["order_reference"].astype(str), sort=False):
			enriched_by_order[str(reference)] = rows.to_dict("records")

	category_counts: Counter[str] = Counter()
	detailed_rows: list[dict[str, Any]] = []
	special_subtypes: dict[str, set[str]] = {}
	routine_cold_orders: set[str] = set()
	for _, row in orders.iterrows():
		decision = _decision(row)
		if decision is None:
			continue
		reasons = list(dict.fromkeys(str(reason) for reason in decision.review_reasons if str(reason).strip()))
		order_categories = {_reason_category(reason) for reason in reasons}
		if decision.requires_human_review and not order_categories:
			order_categories.add("Other")
		reference = str(row["order_reference"])
		for category in order_categories:
			category_counts[category] += 1
		if reasons or decision.requires_human_review:
			detailed_rows.append({
				"Order ID": reference,
				"Review categories": ", ".join(category for category in REVIEW_CATEGORIES if category in order_categories),
				"Original review reasons": " | ".join(reasons) if reasons else "No structured reason provided",
			})
		if "Special handling" in order_categories:
			subtypes: set[str] = set()
			for product in enriched_by_order.get(reference, []):
				evidence = _product_evidence(product)
				for label, pattern in _SPECIAL_EVIDENCE_PATTERNS.items():
					if pattern.search(evidence):
						subtypes.add(label)
				if _ROUTINE_COLD_CHAIN.search(evidence):
					routine_cold_orders.add(reference)
			special_subtypes[reference] = subtypes or {"Subtype not identifiable from available product text"}
		else:
			for product in enriched_by_order.get(reference, []):
				if _ROUTINE_COLD_CHAIN.search(_product_evidence(product)):
					routine_cold_orders.add(reference)

	category_data = pd.DataFrame([
		{
			"Review category": category,
			"Orders affected": category_counts[category],
			"Percent of dataset": _percentage(category_counts[category], total),
		}
		for category in REVIEW_CATEGORIES
	])
	subtype_counts: Counter[str] = Counter()
	for subtypes in special_subtypes.values():
		subtype_counts.update(subtypes)
	subtype_data = pd.DataFrame([
		{
			"Explicit product evidence": label,
			"Orders affected": count,
			"Percent of special-handling review orders": _percentage(count, len(special_subtypes)),
		}
		for label, count in sorted(subtype_counts.items(), key=lambda item: (-item[1], item[0]))
	])
	cold_chain_data = pd.DataFrame([{
		"Informational product evidence": "Refrigerated / perishable wording (not a review trigger by itself)",
		"Orders affected": len(routine_cold_orders),
		"Percent of dataset": _percentage(len(routine_cold_orders), total),
	}])
	detailed = pd.DataFrame(detailed_rows, columns=["Order ID", "Review categories", "Original review reasons"])
	return category_data, subtype_data, cold_chain_data, detailed


def agent_workflow_analysis(queue: pd.DataFrame, actions: pd.DataFrame) -> tuple[dict[str, int], dict[str, int | float]]:
	"""Summarize current unique-order workflow states and latest action outcomes."""
	orders = _unique_orders(queue)
	status_counts = {
		"not_reviewed": 0,
		"reviewed": 0,
		"escalated": 0,
		"resolved": 0,
		"logged": 0,
		"reviewed_not_logged": 0,
	}
	if not orders.empty:
		status_map = {
			"⚪ Not reviewed": "not_reviewed",
			"🟢 Reviewed": "reviewed",
			"🟠 Escalated": "escalated",
			"🟣 Resolved": "resolved",
		}
		for _, row in orders.iterrows():
			status_counts[status_map.get(row.get("operational_review_status"), "not_reviewed")] += 1
			logged = bool(row.get("internal_app_logged", False))
			status_counts["logged"] += int(logged)
			action = row.get("latest_action")
			action_type = action.get("action") if isinstance(action, (dict, pd.Series)) else None
			if action_type in _ACTION_DECISIONS and not logged:
				status_counts["reviewed_not_logged"] += 1

	latest = actions.copy()
	if not latest.empty and "order_reference" in latest:
		order_column = "id" if "id" in latest else "created_at"
		latest = latest.sort_values(order_column, ascending=False).drop_duplicates("order_reference", keep="first")
	action_types = latest.get("action", pd.Series(dtype=str))
	confirmations = int((action_types == "Confirm recommendation").sum())
	overrides = int((action_types == "Override recommendation").sum())
	escalations = int((action_types == "Escalate for review").sum())
	decision_count = confirmations + overrides
	action_summary: dict[str, int | float] = {
		"latest_orders_with_actions": len(latest),
		"confirmations": confirmations,
		"overrides": overrides,
		"escalations": escalations,
		"confirm_override_total": decision_count,
		"confirmation_rate": _percentage(confirmations, decision_count),
		"override_rate": _percentage(overrides, decision_count),
	}
	return status_counts, action_summary


def fte_impact(
	eligible_orders: int,
	total_orders: int,
	manual_minutes: float,
	automated_minutes: float,
	productive_hours_per_fte: float,
	current_fte: float,
) -> dict[str, float]:
	"""Return dataset hours and clearly separate staffing-scenario capacity equivalents."""
	if total_orders <= 0:
		raise ValueError("total orders must be positive")
	if manual_minutes <= 0:
		raise ValueError("manual handling time must be positive")
	if productive_hours_per_fte <= 0:
		raise ValueError("productive hours per FTE must be positive")
	eligibility_rate = eligible_orders / total_orders
	minutes_saved_per_eligible = manual_minutes - automated_minutes
	dataset_hours_saved = eligible_orders * minutes_saved_per_eligible / 60
	return {
		"eligibility_rate": eligibility_rate,
		"minutes_saved_per_eligible": minutes_saved_per_eligible,
		"dataset_hours_saved": dataset_hours_saved,
		"dataset_fte_work_units": dataset_hours_saved / productive_hours_per_fte,
		"scenario_fte_capacity": current_fte * eligibility_rate * minutes_saved_per_eligible / manual_minutes,
	}


def fte_sensitivity(
	total_orders: int,
	eligibility_rate: float,
	manual_minutes: float,
	automated_minutes: float,
	current_fte: float,
) -> pd.DataFrame:
	"""Vary one FTE assumption at a time around the current dataset-derived baseline."""
	if total_orders <= 0 or manual_minutes <= 0:
		raise ValueError("positive order count and manual handling time are required")
	rates = sorted({0.0, max(0.0, eligibility_rate / 2), eligibility_rate, min(1.0, eligibility_rate * 1.5), 1.0})
	manual_scenarios = sorted({max(1.0, manual_minutes / 2), manual_minutes, manual_minutes * 1.5})
	automated_scenarios = sorted({0.0, automated_minutes, automated_minutes + 1.0})
	rows: list[dict[str, float | str]] = []
	scenarios = (
		("Automation eligibility", [(f"{rate:.0%}", rate, manual_minutes, automated_minutes) for rate in rates]),
		("Manual handling time", [(f"{minutes:g} min", eligibility_rate, minutes, automated_minutes) for minutes in manual_scenarios]),
		("Automated handling time", [(f"{minutes:g} min", eligibility_rate, manual_minutes, minutes) for minutes in automated_scenarios]),
	)
	for parameter, values in scenarios:
		for scenario, rate, manual, automated in values:
			dataset_hours = total_orders * rate * (manual - automated) / 60
			scenario_fte = current_fte * rate * (manual - automated) / manual
			rows.append({
				"Sensitivity": parameter,
				"Scenario": scenario,
				"Eligibility rate": rate * 100,
				"Manual minutes": manual,
				"Automated minutes": automated,
				"Dataset hours saved (+) / added (-)": dataset_hours,
				"Scenario FTE capacity equivalent": scenario_fte,
			})
	return pd.DataFrame(rows)
