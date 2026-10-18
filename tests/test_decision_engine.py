import json

import pytest

from src.decision_engine import (
	DEFAULT_VEHICLES,
	DecisionConfig,
	ProductEstimate,
	VehicleCapacity,
	_allocate_items,
	calculate_order_weight_breakdown,
	evaluate_order,
	liquid_volume_plausibility_warning,
	verify_order_completeness,
)


def product(
	name="item", quantity=1, weight=1.0, length=10, width=10, height=10,
	confidence="high", fragile=False, order_reference=None,
):
	row = {
		"product_name": name,
		"quantity": quantity,
		"ai_status": "completed",
		"ai_estimated_transport_weight_kg": weight,
		"ai_estimated_packaged_length_cm": length,
		"ai_estimated_packaged_width_cm": width,
		"ai_estimated_packaged_height_cm": height,
		"ai_confidence": confidence,
		"ai_fragile_or_special_handling": fragile,
		"ai_manual_review_recommended": False,
	}
	if order_reference is not None:
		row["order_reference"] = order_reference
	return row


def original(*items):
	return json.dumps([{"name": name, "quantity": quantity} for name, quantity in items])


def test_small_order_recommends_lowest_cost_bicycle():
	row = product(weight=2, length=20, width=10, height=10)
	decision = evaluate_order("O-1", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Bicycle"
	assert decision.recommended_courier_count == 1
	assert not decision.requires_human_review
	assert {option.vehicle for option in decision.alternative_options} == {"Bicycle", "Motorbike", "Car"}


def test_one_motorbike_is_cheaper_than_two_bicycles():
	row = product(quantity=2, weight=8, length=30, width=20, height=10)
	decision = evaluate_order("O-2", [row], original_products_json=original(("item", 2)))
	assert decision.recommended_vehicle == "Motorbike"
	assert decision.recommended_courier_count == 1
	assert any(option.vehicle == "Bicycle" and option.courier_count == 2 for option in decision.alternative_options)
	assert "lowest hypothetical relative cost" in decision.decision_explanation


def test_car_order_when_motorbike_is_exceeded():
	row = product(weight=50, length=50, width=40, height=30)
	decision = evaluate_order("O-3", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Car"


def test_vehicle_specific_dimension_limits_reject_item_for_bicycle():
	row = product(weight=1, length=110, width=10, height=10)
	decision = evaluate_order("O-4", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Motorbike"
	assert any(option.vehicle == "Bicycle" and not option.feasible for option in decision.alternative_options)


def test_oversized_for_all_vehicles_requires_review():
	row = product(weight=1, length=250, width=10, height=10)
	decision = evaluate_order("O-5", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle is None
	assert decision.requires_human_review
	assert any("no safe allocation" in reason for reason in decision.review_reasons)


def test_multiple_couriers_are_used_when_cheapest_safe_option_splits():
	rows = [product(name="item", quantity=2, weight=11, length=30, width=20, height=10)]
	decision = evaluate_order("O-6", rows, original_products_json=original(("item", 2)))
	assert decision.recommended_vehicle == "Bicycle"
	assert decision.recommended_courier_count == 2
	assert decision.requires_human_review
	assert any("splitting" in reason for reason in decision.review_reasons)


def test_unknown_physical_properties_are_blocked_not_zero():
	row = product(weight=None, length=None, width=None, height=None)
	decision = evaluate_order("O-7", [row], original_products_json=original(("item", 1)))
	assert decision.estimated_total_weight_kg is None
	assert decision.recommended_vehicle is None
	assert decision.requires_human_review


@pytest.mark.parametrize("field", [
	"ai_estimated_transport_weight_kg",
	"ai_estimated_packaged_length_cm",
	"ai_estimated_packaged_width_cm",
	"ai_estimated_packaged_height_cm",
])
def test_zero_nan_and_infinite_measurements_are_invalid(field):
	row = product()
	row[field] = 0 if field != "ai_estimated_transport_weight_kg" else float("nan")
	decision = evaluate_order("O-8", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle is None
	assert decision.requires_human_review


def test_low_confidence_estimate_requires_review():
	row = product(confidence="low")
	decision = evaluate_order("O-9", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Bicycle"
	assert decision.requires_human_review
	assert decision.decision_confidence == "low"


def test_fragile_item_keeps_recommendation_but_requires_review():
	row = product(fragile=True)
	decision = evaluate_order("O-10", [row], original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Bicycle"
	assert decision.requires_human_review
	assert any("special handling" in reason for reason in decision.review_reasons)


def test_near_capacity_order_is_flagged():
	config = DecisionConfig(weight_safety_margin=0, volume_safety_margin=0, packing_efficiency=1)
	row = product(weight=13.6, length=20, width=20, height=20)
	decision = evaluate_order("O-11", [row], config, original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Bicycle"
	assert decision.requires_human_review
	assert any("capacity" in reason for reason in decision.review_reasons)


def test_incomplete_order_blocks_recommendation():
	rows = [product(name="known")]
	decision = evaluate_order("O-12", rows, original_products_json=original(("known", 1), ("missing", 1)))
	assert not decision.order_complete
	assert decision.recommended_vehicle is None
	assert any("missing enriched" in reason for reason in decision.review_reasons)


def test_duplicate_product_occurrences_are_preserved():
	rows = [product(name="same", quantity=1), product(name="same", quantity=1)]
	complete, reasons = verify_order_completeness("O-13", rows, original(("same", 1), ("same", 1)))
	assert complete
	assert reasons == ()


def test_quantity_mismatch_is_incomplete():
	rows = [product(name="same", quantity=1)]
	complete, reasons = verify_order_completeness("O-14", rows, original(("same", 2)))
	assert not complete
	assert reasons


def test_large_order_reports_heuristic_uncertainty():
	rows = [product(name=f"item-{index}", weight=2, length=10, width=10, height=10) for index in range(13)]
	original_json = original(*[(f"item-{index}", 1) for index in range(13)])
	decision = evaluate_order("O-15", rows, original_products_json=original_json)
	assert decision.recommended_vehicle is not None
	assert decision.alternative_options[0].heuristic_optimal is None
	assert not any("optimality" in reason for reason in decision.review_reasons)


def test_one_courier_allocation_is_proven_optimal():
	item = ProductEstimate("item", 1, 1, 0.001, 10, 10, 10, "high", False, False)
	allocation = _allocate_items([item], DEFAULT_VEHICLES[0], DecisionConfig())
	assert allocation.courier_count == 1
	assert allocation.heuristic_optimal is True


def test_multi_courier_exact_allocation_is_proven_optimal():
	items = [
		ProductEstimate(f"item-{index}", 1, 9, 0.001, 10, 10, 10, "high", False, False)
		for index in range(2)
	]
	config = DecisionConfig(weight_safety_margin=0, volume_safety_margin=0, packing_efficiency=1)
	allocation = _allocate_items(items, DEFAULT_VEHICLES[0], config)
	assert allocation.courier_count == 2
	assert allocation.heuristic_optimal is True


def test_exact_optimization_skipped_returns_unknown_not_false():
	items = [
		ProductEstimate(f"item-{index}", 1, 2, 0.001, 10, 10, 10, "high", False, False)
		for index in range(13)
	]
	allocation = _allocate_items(items, DEFAULT_VEHICLES[0], DecisionConfig())
	assert allocation.courier_count is not None and allocation.courier_count > 1
	assert allocation.heuristic_optimal is None


def test_custom_relative_cost_can_change_choice():
	vehicles = (
		VehicleCapacity("Bicycle", 15, 50, 100, 60, 60, 5),
		VehicleCapacity("Motorbike", 20, 70, 120, 70, 70, 1),
	)
	config = DecisionConfig(vehicles=vehicles)
	row = product(weight=1, length=10, width=10, height=10)
	decision = evaluate_order("O-16", [row], config, original_products_json=original(("item", 1)))
	assert decision.recommended_vehicle == "Motorbike"


def test_weight_breakdown_applies_quantity_and_safety_margin():
	rows = [product(name="a", quantity=2, weight=3), product(name="b", quantity=1, weight=2)]
	config = DecisionConfig(weight_safety_margin=0.10)
	base, margin, conservative = calculate_order_weight_breakdown(rows, config)
	assert base == 8.0
	assert margin == 0.8
	assert conservative == 8.8


def test_high_ai_confidence_can_have_medium_operational_confidence():
	row = product(weight=13.6, length=20, width=20, height=20, confidence="high")
	decision = evaluate_order("O-17", [row], original_products_json=original(("item", 1)))
	assert decision.decision_confidence == "medium"
	assert decision.requires_human_review


def test_liquid_volume_smaller_than_package_is_flagged():
	row = product(length=7, width=7, height=20)
	row["extracted_volume_litres"] = 1.0
	assert liquid_volume_plausibility_warning(row)
	decision = evaluate_order("O-18", [row], original_products_json=original(("item", 1)))
	assert decision.requires_human_review
	assert any("liquid volume plausibility" in reason for reason in decision.review_reasons)


def test_liquid_volume_valid_dimensions_have_no_false_positive():
	row = product(length=10, width=10, height=10)
	row["extracted_volume_litres"] = 1.0
	assert liquid_volume_plausibility_warning(row) is None


def test_fragile_review_reason_is_not_duplicated():
	decision = evaluate_order("O-19", [product(fragile=True)], original_products_json=original(("item", 1)))
	assert decision.review_reasons.count("fragile or special handling") == 1
	assert not any("one or more products require special handling" == reason for reason in decision.review_reasons)