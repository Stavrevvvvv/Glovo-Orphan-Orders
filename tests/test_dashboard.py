import json

import pandas as pd
import pytest

from dashboard import (
	_processing_indicator,
	_product_estimate_status,
	_action_form_defaults,
	_action_summary,
	add_operational_review_status,
	add_internal_app_log_status,
	add_internal_app_log_status,
	build_action_history_view,
	build_cached_enrichment,
	build_order_queue,
	filter_action_history,
)
from datetime import date
from src.ai_enrichment import ProductContext, product_identifier
from src.decision_engine import DecisionRecord
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


def enriched_row(order_reference, name, quantity=1):
	return {
		"order_reference": order_reference,
		"product_name": name,
		"quantity": quantity,
		"ai_status": "completed",
		"ai_estimated_transport_weight_kg": 1.0,
		"ai_estimated_packaged_length_cm": 10.0,
		"ai_estimated_packaged_width_cm": 10.0,
		"ai_estimated_packaged_height_cm": 10.0,
		"ai_confidence": "high",
		"ai_fragile_or_special_handling": False,
		"ai_manual_review_recommended": False,
	}


def test_queue_keeps_original_orders_and_marks_missing_enrichment_pending():
	original = pd.DataFrame([
		{"order_id": "A", "product_lines": 1, "products_json": json.dumps([{"name": "Rice", "quantity": 1}])},
		{"order_id": "B", "product_lines": 1, "products_json": json.dumps([{"name": "Milk", "quantity": 1}])},
	])
	enriched = pd.DataFrame([enriched_row("A", "Rice")])
	queue = build_order_queue(original, enriched)
	assert set(queue["order_reference"]) == {"A", "B"}
	assert queue.loc[queue["order_reference"] == "B", "enrichment_status"].iloc[0] == "pending enrichment"
	assert queue.loc[queue["order_reference"] == "B", "requires_human_review"].iloc[0]


def test_queue_marks_partial_duplicate_occurrence_order_incomplete():
	original = pd.DataFrame([{
		"order_id": "A",
		"product_lines": 2,
		"products_json": json.dumps([
			{"name": "Same", "quantity": 1},
			{"name": "Same", "quantity": 1},
		]),
	}])
	enriched = pd.DataFrame([enriched_row("A", "Same")])
	queue = build_order_queue(original, enriched)
	assert queue.loc[0, "enrichment_status"] == "incomplete enrichment"
	assert not queue.loc[0, "order_complete"]


def test_agent_actions_persist_in_sqlite(tmp_path):
	database_path = tmp_path / "nested" / "actions.db"
	record_agent_action(database_path, "A", "Escalate for review", None, None, "Missing estimate")
	actions = load_agent_actions(database_path)
	assert len(actions) == 1
	assert actions.loc[0, "order_reference"] == "A"
	assert actions.loc[0, "action"] == "Escalate for review"


def test_cache_join_expands_one_unique_estimate_to_all_occurrences():
	processed = pd.DataFrame([
		{"order_reference": "A", "product_name": "Same", "quantity": 1},
		{"order_reference": "B", "product_name": "Same", "quantity": 3},
	])
	key = product_identifier(ProductContext(product_name="Same"))
	cache = {
		key: {
			"original_product_name": "Same",
			"product_id": key,
			"normalized_product_description": "A product",
			"confidence": "high",
			"estimate_explanation": "Known item",
			"manual_review_recommended": False,
			"estimated_transport_weight_kg": 1.0,
			"estimated_packaged_length_cm": 10.0,
			"estimated_packaged_width_cm": 10.0,
			"estimated_packaged_height_cm": 10.0,
		},
	}
	joined = build_cached_enrichment(processed, cache)
	assert len(joined) == 2
	assert set(joined["order_reference"]) == {"A", "B"}
	assert joined["ai_status"].tolist() == ["completed", "completed"]


def test_operational_review_statuses_are_separate_from_engine_review():
	queue = pd.DataFrame({
		"order_reference": ["N", "C", "O", "E"],
		"requires_human_review": [True, True, False, True],
	})
	actions = pd.DataFrame([
		{"order_reference": "C", "action": "Confirm recommendation", "selected_vehicle": "Bicycle", "courier_count": 1, "reason": "ok", "created_at": "2026-10-18T10:00:00+00:00"},
		{"order_reference": "O", "action": "Override recommendation", "selected_vehicle": "Car", "courier_count": 1, "reason": "large item", "created_at": "2026-10-18T10:01:00+00:00"},
		{"order_reference": "E", "action": "Escalate for review", "selected_vehicle": None, "courier_count": None, "reason": "missing info", "created_at": "2026-10-18T10:02:00+00:00"},
	])
	result = add_operational_review_status(queue, actions)
	assert result["operational_review_status"].tolist() == ["⚪ Not reviewed", "🟢 Reviewed", "🟢 Reviewed", "🟠 Escalated"]
	assert bool(result.loc[result.order_reference == "C", "requires_human_review"].iloc[0])


def test_latest_action_and_rereview_preserve_history(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Confirm recommendation", "Bicycle", 1, "initial")
	record_agent_action(database_path, "A", "Override recommendation", "Car", 1, "re-review")
	all_actions = load_agent_actions(database_path)
	latest = latest_agent_actions(database_path)
	assert len(all_actions) == 2
	assert len(latest) == 1
	assert latest.iloc[0]["action"] == "Override recommendation"
	assert latest.iloc[0]["reason"] == "re-review"


def test_internal_app_log_is_bound_to_action_and_cannot_repeat(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Confirm recommendation", "Bicycle", 1, "confirmed")
	action_id = int(load_agent_actions(database_path).iloc[0]["id"])
	record_internal_app_log(database_path, "A", action_id)
	logs = load_internal_app_logs(database_path)
	assert len(logs) == 1
	assert logs.loc[0, "agent_action_id"] == action_id
	with pytest.raises(ValueError, match="already been logged"):
		record_internal_app_log(database_path, "A", action_id)


def test_internal_app_log_rejects_escalation_and_stale_decision(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Confirm recommendation", "Bicycle", 1, "confirmed")
	old_id = int(load_agent_actions(database_path).iloc[0]["id"])
	record_agent_action(database_path, "A", "Escalate for review", None, None, "need review")
	with pytest.raises(ValueError, match="latest order decision"):
		record_internal_app_log(database_path, "A", old_id)
	latest_id = int(load_agent_actions(database_path).iloc[0]["id"])
	with pytest.raises(ValueError, match="valid confirmed or overridden"):
		record_internal_app_log(database_path, "A", latest_id)


def test_internal_logged_status_is_decision_id_specific():
	queue = pd.DataFrame([
		{"order_reference": "A", "latest_action": pd.Series({"id": 5})},
		{"order_reference": "B", "latest_action": pd.Series({"id": 6})},
	])
	logs = pd.DataFrame([{"order_reference": "A", "agent_action_id": 5, "created_at": "now"}])
	result = add_internal_app_log_status(queue, logs)
	assert result["internal_app_logged"].tolist() == [True, False]
	assert result["internal_app_status"].tolist() == ["Logged", "Not logged"]


def test_escalation_can_be_human_resolved_without_changing_ai_recommendation(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Escalate for review", None, None, "Missing estimate")
	escalation = load_agent_actions(database_path).iloc[0]
	resolution_id = record_manual_resolution(
		database_path, "A", int(escalation["id"]), "delivery",
		"Checked partner catalogue dimensions and confirmed vehicle fit.", "Partner catalogue", "Car", 1,
	)
	resolution = load_manual_resolutions(database_path)
	queue = pd.DataFrame([{
		"order_reference": "A", "decision": None, "recommended_vehicle": "Pending",
		"operational_review_status": "⚪ Not reviewed", "latest_action": escalation.to_dict(),
	}])
	status_queue = add_operational_review_status(queue, latest_agent_actions(database_path), resolution)
	assert status_queue.loc[0, "operational_review_status"] == "🟣 Resolved"
	assert status_queue.loc[0, "recommended_vehicle"] == "Pending"
	assert int(status_queue.loc[0, "latest_resolution"]["id"]) == resolution_id
	assert len(load_agent_actions(database_path)) == 1


def test_resolution_requires_verification_and_customer_intervention_is_supported(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Escalate for review", None, None, "No safe option")
	escalation_id = int(load_agent_actions(database_path).iloc[0]["id"])
	with pytest.raises(ValueError, match="explanation is required"):
		record_manual_resolution(database_path, "A", escalation_id, "customer_intervention", "  ")
	resolution_id = record_manual_resolution(
		database_path, "A", escalation_id, "customer_intervention",
		"No configured vehicle can safely transport the item; contacted customer for next steps.",
		"Supervisor confirmation",
	)
	resolution = load_manual_resolutions(database_path).iloc[0]
	assert int(resolution["id"]) == resolution_id
	assert resolution["selected_vehicle"] is None
	assert resolution["courier_count"] is None
	assert resolution["outcome"] == "customer_intervention"


def test_resolution_logging_is_separate_and_one_time(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Escalate for review", None, None, "Need human verification")
	escalation_id = int(load_agent_actions(database_path).iloc[0]["id"])
	resolution_id = record_manual_resolution(database_path, "A", escalation_id, "delivery", "Verified with partner catalogue.", None, "Car", 1)
	record_internal_resolution_log(database_path, "A", resolution_id)
	logs = load_internal_resolution_logs(database_path)
	assert len(logs) == 1
	assert int(logs.loc[0, "resolution_id"]) == resolution_id
	with pytest.raises(ValueError, match="already been logged"):
		record_internal_resolution_log(database_path, "A", resolution_id)
	assert len(load_agent_actions(database_path)) == 1


def test_only_latest_resolution_can_be_logged(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Escalate for review", None, None, "Review")
	escalation_id = int(load_agent_actions(database_path).iloc[0]["id"])
	old_id = record_manual_resolution(database_path, "A", escalation_id, "customer_intervention", "Asked customer to contact support.")
	new_id = record_manual_resolution(database_path, "A", escalation_id, "customer_intervention", "Supervisor verified customer intervention request.")
	with pytest.raises(ValueError, match="latest resolution"):
		record_internal_resolution_log(database_path, "A", old_id)
	record_internal_resolution_log(database_path, "A", new_id)


def test_resolution_status_uses_latest_escalation_and_log_state(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Escalate for review", None, None, "Review")
	escalation_id = int(load_agent_actions(database_path).iloc[0]["id"])
	resolution_id = record_manual_resolution(database_path, "A", escalation_id, "delivery", "Verified dimensions.", None, "Car", 1)
	queue = pd.DataFrame([{
		"order_reference": "A", "recommended_vehicle": "Pending",
		"operational_review_status": "🟠 Escalated",
		"latest_action": load_agent_actions(database_path).iloc[0].to_dict(),
	}])
	queue = add_operational_review_status(queue, latest_agent_actions(database_path), load_manual_resolutions(database_path))
	assert queue.loc[0, "operational_review_status"] == "🟣 Resolved"
	queue = add_internal_app_log_status(queue, load_internal_app_logs(database_path), load_internal_resolution_logs(database_path))
	assert not queue.loc[0, "internal_app_logged"]
	assert queue.loc[0, "internal_app_status"] == "Not logged"
	record_internal_resolution_log(database_path, "A", resolution_id)
	queue = add_internal_app_log_status(queue, load_internal_app_logs(database_path), load_internal_resolution_logs(database_path))
	assert queue.loc[0, "internal_app_logged"]
	assert queue.loc[0, "internal_app_status"] == "Resolution logged"


def test_concise_processing_and_product_estimate_indicators():
	assert _processing_indicator("decision available") == "🟢 Ready"
	assert _processing_indicator("pending enrichment") == "🟡 Pending"
	assert _product_estimate_status(enriched_row("A", "Milk")) == "✓ Estimate available"
	assert _product_estimate_status(enriched_row("A", "Milk") | {"ai_estimated_packaged_height_cm": None}) == "⚠ Missing / invalid estimate"


def test_order_without_recommendation_only_allows_escalation():
	decision = DecisionRecord("A", None, None, None, None, "low", True, (), "No estimates", (), order_complete=False)
	options, vehicle, couriers = _action_form_defaults(decision)
	assert options == ["Escalate for review"]
	assert vehicle is None
	assert couriers is None


def test_recommended_order_defaults_to_confirmation_values_and_summary():
	decision = DecisionRecord("A", 2.0, 1.0, "Motorbike", 1, "high", False, (), "Fits", (), order_complete=True)
	options, vehicle, couriers = _action_form_defaults(decision)
	assert options[0] == "Confirm recommendation"
	assert vehicle == "Motorbike"
	assert couriers == 1
	assert "Order ID: A" in _action_summary("A", "Override recommendation", "Car", 1, "large item")
	assert "Reason: large item" in _action_summary("A", "Override recommendation", "Car", 1, "large item")


def test_action_history_latest_and_full_modes_use_repeated_records():
	actions = pd.DataFrame([
		{"id": 1, "order_reference": "A", "action": "Confirm recommendation", "selected_vehicle": "Bicycle", "courier_count": 1, "reason": "initial", "created_at": "2026-10-17T10:00:00+00:00"},
		{"id": 2, "order_reference": "A", "action": "Override recommendation", "selected_vehicle": "Car", "courier_count": 1, "reason": "re-review", "created_at": "2026-10-18T12:00:00+00:00"},
	])
	queue = pd.DataFrame([{"order_reference": "A", "recommended_vehicle": "Car", "operational_review_status": "🟢 Reviewed"}])
	internal_logs = pd.DataFrame([{"agent_action_id": 2}])
	latest = build_action_history_view(actions, queue, latest_only=True, internal_logs=internal_logs)
	full = build_action_history_view(actions, queue, latest_only=False, internal_logs=internal_logs)
	assert len(latest) == 1
	assert latest.iloc[0]["Action"] == "Overridden"
	assert latest.iloc[0]["Current review status"] == "Reviewed"
	assert latest.iloc[0]["Internal app status"] == "Logged"
	assert len(full) == 2
	assert set(full["Current review status"]) == {"Reviewed"}
	assert full.set_index("Action")["Internal app status"].to_dict() == {
		"Confirmed": "Not logged",
		"Overridden": "Logged",
	}


def test_action_history_filters_order_action_vehicle_and_local_date():
	actions = pd.DataFrame([
		{"id": 1, "order_reference": "A", "action": "Confirm recommendation", "selected_vehicle": "Bicycle", "courier_count": 1, "reason": "ok", "created_at": "2026-10-17T10:00:00+00:00"},
		{"id": 2, "order_reference": "B", "action": "Escalate for review", "selected_vehicle": None, "courier_count": None, "reason": "missing", "created_at": "2026-10-18T12:00:00+00:00"},
	])
	queue = pd.DataFrame([
		{"order_reference": "A", "recommended_vehicle": "Bicycle", "operational_review_status": "🟢 Reviewed"},
		{"order_reference": "B", "recommended_vehicle": "Pending", "operational_review_status": "🟠 Escalated"},
	])
	history = build_action_history_view(actions, queue, latest_only=False)
	selected_date = history.loc[history["Order ID"] == "A", "_local_date"].iloc[0]
	filtered = filter_action_history(history, "A", ["Confirmed"], ["Bicycle"], (selected_date, selected_date))
	assert len(filtered) == 1
	assert filtered.iloc[0]["Order ID"] == "A"


def test_action_history_marks_logged_escalation_resolutions():
	actions = pd.DataFrame([{
		"id": 4,
		"order_reference": "A",
		"action": "Escalate for review",
		"selected_vehicle": None,
		"courier_count": None,
		"reason": "Needs human verification",
		"created_at": "2026-10-18T12:00:00+00:00",
	}])
	resolutions = pd.DataFrame([{"id": 17, "order_reference": "A", "escalation_action_id": 4}])
	resolution_logs = pd.DataFrame([{"resolution_id": 17}])
	history = build_action_history_view(
		actions, pd.DataFrame(), resolutions=resolutions, resolution_logs=resolution_logs,
	)
	assert history.iloc[0]["Internal app status"] == "Resolution logged"