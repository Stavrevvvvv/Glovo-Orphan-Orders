import json

import pandas as pd

from dashboard import build_cached_enrichment, build_order_queue
from src.ai_enrichment import ProductContext, product_identifier
from src.dashboard_store import load_agent_actions, record_agent_action


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