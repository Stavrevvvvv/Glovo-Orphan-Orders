import sqlite3

import pandas as pd
import pytest

from src.analytics import (
	agent_workflow_analysis,
	dataset_coverage_metrics,
	fte_impact,
	fte_sensitivity,
	recommendation_distribution,
	review_reason_analysis,
)
from src.dashboard_store import (
	load_agent_actions,
	load_internal_app_logs,
	record_agent_action,
	record_internal_app_log,
)
from dashboard import add_internal_app_log_status
from src.decision_engine import DecisionRecord


def decision(
	reference: str,
	*,
	complete: bool = True,
	weight: float | None = 1.0,
	volume: float | None = 1.0,
	vehicle: str | None = "Bicycle",
	couriers: int | None = 1,
	review: bool = False,
	reasons: tuple[str, ...] = (),
) -> DecisionRecord:
	return DecisionRecord(
		order_reference=reference,
		estimated_total_weight_kg=weight,
		estimated_total_packaged_volume_litres=volume,
		recommended_vehicle=vehicle,
		recommended_courier_count=couriers,
		decision_confidence="high",
		requires_human_review=review,
		review_reasons=reasons,
		decision_explanation="test fixture",
		vehicle_capacity_assumptions=(),
		order_complete=complete,
	)


def test_dataset_metrics_and_recommendation_distribution_count_each_of_930_once():
	queue = pd.DataFrame([
		{"order_reference": str(index), "decision": decision(
			str(index),
			vehicle=(None if index >= 548 else ("Motorbike" if index < 300 else "Car")),
			couriers=(None if index >= 548 else (2 if 500 <= index < 530 else 1)),
			review=index >= 209,
		)}
		for index in range(930)
	])
	metrics = dataset_coverage_metrics(queue)
	distribution = recommendation_distribution(queue)
	assert metrics["total_orders"] == 930
	assert metrics["eligible_orders"] == 209
	assert metrics["human_review_orders"] == 721
	assert sum(distribution["Orders"]) == 930
	assert distribution["Percent of dataset"].sum() == pytest.approx(100)
	assert int(distribution.attrs["multi_courier_orders"]) == 30
	assert distribution.attrs["split_by_vehicle"]["Split-delivery orders"].sum() == 30


def test_incomplete_physical_estimates_are_not_automatically_eligible():
	queue = pd.DataFrame([{
		"order_reference": "incomplete",
		"decision": decision("incomplete", weight=None, volume=None, vehicle="Bicycle", couriers=1, review=False),
	}])
	metrics = dataset_coverage_metrics(queue)
	assert metrics["occurrence_complete_orders"] == 1
	assert metrics["physically_evaluable_orders"] == 0
	assert metrics["eligible_orders"] == 0
	assert metrics["recommended_orders"] == 1


def test_review_categories_can_overlap_without_inflating_unique_review_total():
	queue = pd.DataFrame([
		{"order_reference": "A", "decision": decision(
			"A", review=True,
			reasons=("missing or invalid packaged dimensions", "low-confidence physical estimate"),
		)},
		{"order_reference": "B", "decision": decision(
			"B", review=True, reasons=("fragile, hazardous, frozen, or unusually difficult item requires review",),
		)},
	])
	categories, _, _, _ = review_reason_analysis(queue, pd.DataFrame())
	metrics = dataset_coverage_metrics(queue)
	assert categories["Orders affected"].sum() == 3
	assert metrics["human_review_orders"] == 2
	assert categories["Percent of dataset"].sum() == pytest.approx(150)


def test_dataset_and_50_fte_scenario_use_their_separate_formulas():
	result = fte_impact(209, 930, 8, 1, 35, 50)
	assert result["dataset_hours_saved"] == pytest.approx(209 * (8 - 1) / 60)
	assert result["dataset_fte_work_units"] == pytest.approx((209 * 7 / 60) / 35)
	assert result["scenario_fte_capacity"] == pytest.approx(50 * (209 / 930) * (8 - 1) / 8)
	sensitivity = fte_sensitivity(930, 209 / 930, 8, 1, 50)
	assert {"Automation eligibility", "Manual handling time", "Automated handling time"} == set(sensitivity["Sensitivity"])
	manual_case = sensitivity.loc[
		(sensitivity["Sensitivity"] == "Manual handling time")
		& (sensitivity["Manual minutes"] == 4)
	].iloc[0]
	assert manual_case["Dataset hours saved (+) / added (-)"] == pytest.approx(209 * (4 - 1) / 60)


def test_repeated_actions_do_not_inflate_latest_order_or_confirmation_rates():
	actions = pd.DataFrame([
		{"id": 1, "order_reference": "A", "action": "Confirm recommendation"},
		{"id": 2, "order_reference": "A", "action": "Override recommendation"},
		{"id": 3, "order_reference": "B", "action": "Confirm recommendation"},
	])
	_, summary = agent_workflow_analysis(pd.DataFrame(), actions)
	assert summary["latest_orders_with_actions"] == 2
	assert summary["confirmations"] == 1
	assert summary["overrides"] == 1
	assert summary["override_rate"] == 50
	assert summary["confirmation_rate"] == 50


def test_duplicate_queue_rows_for_one_order_count_once():
	order_decision = decision("A")
	queue = pd.DataFrame([
		{"order_reference": "A", "decision": order_decision},
		{"order_reference": "A", "decision": order_decision},
	])
	metrics = dataset_coverage_metrics(queue)
	distribution = recommendation_distribution(queue)
	assert metrics["total_orders"] == 1
	assert metrics["eligible_orders"] == 1
	assert sum(distribution["Orders"]) == 1


def test_internal_app_logging_changes_workflow_not_lifetime_review_totals(tmp_path):
	database_path = tmp_path / "actions.db"
	record_agent_action(database_path, "A", "Confirm recommendation", "Bicycle", 1, "reviewed")
	action_id = int(load_agent_actions(database_path).iloc[0]["id"])
	queue = pd.DataFrame([{
		"order_reference": "A",
		"operational_review_status": "🟢 Reviewed",
		"latest_action": {"action": "Confirm recommendation", "id": action_id},
		"internal_app_logged": False,
	}])
	before, _ = agent_workflow_analysis(queue, load_agent_actions(database_path))
	record_internal_app_log(database_path, "A", action_id)
	queue = add_internal_app_log_status(queue, load_internal_app_logs(database_path))
	after, _ = agent_workflow_analysis(queue, load_agent_actions(database_path))
	assert before["reviewed"] == after["reviewed"] == 1
	assert before["logged"] == 0
	assert after["logged"] == 1
	assert after["reviewed_not_logged"] == 0
	assert len(load_agent_actions(database_path)) == 1


def test_existing_actions_are_preserved_when_classification_column_is_removed(tmp_path):
	database_path = tmp_path / "legacy.db"
	with sqlite3.connect(database_path) as connection:
		connection.execute(
			"""CREATE TABLE agent_actions (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				order_reference TEXT NOT NULL,
				action TEXT NOT NULL,
				selected_vehicle TEXT,
				courier_count INTEGER,
				reason TEXT,
				created_at TEXT NOT NULL,
				activity_type TEXT NOT NULL DEFAULT 'unclassified'
			)"""
		)
		connection.execute(
			"INSERT INTO agent_actions (order_reference, action, reason, created_at, activity_type) VALUES (?, ?, ?, ?, ?)",
			("A", "Confirm recommendation", "For testing", "2026-10-18T10:00:00+00:00", "demo"),
		)
	migrated = load_agent_actions(database_path)
	assert len(migrated) == 1
	assert migrated.loc[0, "reason"] == "For testing"
	with sqlite3.connect(database_path) as connection:
		columns = {row[1] for row in connection.execute("PRAGMA table_info(agent_actions)")}
	assert "activity_type" not in columns


def test_fte_calculation_rejects_undefined_denominators():
	with pytest.raises(ValueError, match="total orders"):
		fte_impact(1, 0, 8, 1, 35, 50)
	with pytest.raises(ValueError, match="manual handling"):
		fte_impact(1, 1, 0, 1, 35, 50)
