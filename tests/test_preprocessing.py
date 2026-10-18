import json

import pandas as pd
import pytest

from src.preprocessing import (
    extract_net_content,
    extract_products,
    build_summary,
    parse_products_json,
    process_orders,
)


@pytest.mark.parametrize(
    ("name", "unit", "weight", "volume"),
    [("Rice 1kg", "kg", 1.0, None), ("Flour 500g", "g", 0.5, None),
     ("Juice 1L", "l", None, 1.0), ("Milk 500ml", "ml", None, 0.5),
     ("Yogurt 250gr", "g", 0.25, None)],
)
def test_extract_regular_content(name, unit, weight, volume):
    result = extract_net_content(name)
    assert result["multipack_quantity"] is None
    assert result["net_content_unit"] == unit
    assert result["extracted_weight_kg"] == weight
    assert result["extracted_volume_litres"] == volume


@pytest.mark.parametrize(
    ("name", "quantity", "expected_pack", "expected_weight"),
    [("Yogurt 4x125g", 2, 4, 0.5), ("Water 4 x 90gr", 1, 4, 0.36)],
)
def test_multipack_is_separate_from_product_quantity(name, quantity, expected_pack, expected_weight):
    orders = pd.DataFrame([{
        "order_id": "O-1",
        "products_json": json.dumps([{
            "name": name, "quantity": quantity, "unit_price": 2.5,
            "declared_weight_kg": 0.7,
        }]),
    }])
    result = extract_products(orders).iloc[0]
    assert result["quantity"] == quantity
    assert result["multipack_quantity"] == expected_pack
    assert result["extracted_weight_kg"] == expected_weight
    assert result["declared_weight_kg"] == 0.7


def test_missing_unit_returns_null_physical_characteristics():
    result = extract_net_content("Fresh apples")
    assert result["net_content_value"] is None
    assert result["extracted_weight_kg"] is None
    assert result["extracted_volume_litres"] is None


def test_malformed_and_missing_json_are_ignored():
    assert parse_products_json("not json") == []
    assert parse_products_json(None) == []
    orders = pd.DataFrame([
        {"order_id": "O-1", "products_json": "not json"},
        {"order_id": "O-2", "products_json": None},
    ])
    result = extract_products(orders)
    assert result.empty
    assert list(result.columns)[0] == "order_reference"


def test_missing_order_reference_does_not_invent_a_field():
    orders = pd.DataFrame([{"products_json": "[]"}])
    with pytest.raises(ValueError, match="order reference"):
        extract_products(orders)


@pytest.mark.parametrize(
    ("name", "unit", "weight", "volume"),
    [("Pasta 250grs", "g", 0.25, None), ("Água 1litro", "l", None, 1.0),
     ("Água 2litros", "l", None, 2.0), ("Sumo 25cl", "cl", None, 0.25),
     ("Leite 4×125g", "g", 0.5, None)],
)
def test_extract_unusual_units_and_multiplication_symbol(name, unit, weight, volume):
    result = extract_net_content(name)
    assert result["net_content_unit"] == unit
    assert result["extracted_weight_kg"] == weight
    assert result["extracted_volume_litres"] == volume
    assert result["matched_text"]


def test_ambiguous_match_has_warning_and_medium_confidence():
    result = extract_net_content("Bundle 1kg + 500ml")
    assert result["extraction_warning"]
    assert result["extraction_confidence"] == "medium"


def test_net_content_is_not_transport_weight():
    result = extract_net_content("Rice 1kg")
    assert result["net_content_weight_kg"] == 1.0
    assert result["estimated_transport_weight_kg"] is None


def test_process_orders_reports_malformed_orders():
    orders = pd.DataFrame([
        {"order_id": "valid", "products_json": json.dumps([{
            "name": "Rice 1kg", "quantity": 1, "unit_price": 2,
        }])},
        {"order_id": "broken", "products_json": "[{"},
        {"order_id": "missing", "products_json": None},
    ])
    products, malformed = process_orders(orders)
    assert len(products) == 1
    assert set(malformed["order_reference"]) == {"broken", "missing"}
    assert set(malformed["parse_error"]) == {"malformed_json", "missing_products_json"}


def test_multipack_and_explicit_total_are_not_added():
    result = extract_net_content("Lion 5x30g 150g")
    assert result["extracted_weight_kg"] == 0.15
    assert result["measurement_type"] == "ambiguous"
    assert result["extraction_method"] == "regex_multipack_total"
    assert "not added" in result["extraction_warning"]
    assert result["matched_text"] == "5x30g | 150g"


def test_weight_range_preserves_bounds_without_selecting_exact_weight():
    result = extract_net_content("Bananas 1kg - 1.3kg")
    assert result["measurement_type"] == "range"
    assert result["extracted_weight_min_kg"] == 1.0
    assert result["extracted_weight_max_kg"] == 1.3
    assert result["extracted_weight_kg"] is None
    assert result["extraction_warning"]


def test_weight_and_volume_are_preserved_separately():
    result = extract_net_content("Cat litter 10L/8,5Kg")
    assert result["measurement_type"] == "ambiguous"
    assert result["extracted_volume_litres"] == 10.0
    assert result["extracted_weight_kg"] == 8.5
    assert "10L" in result["matched_text"]
    assert "8,5Kg" in result["matched_text"]


@pytest.mark.parametrize(
    ("name", "weight", "volume"),
    [("Молоко 1кг", 1.0, None), ("Вода 500мл", None, 0.5),
     ("წყალი 1ლ", None, 1.0), ("ზეთი 500მლ", None, 0.5)],
)
def test_cyrillic_and_georgian_units(name, weight, volume):
    result = extract_net_content(name)
    assert result["extracted_weight_kg"] == weight
    assert result["extracted_volume_litres"] == volume


def test_summary_counts_orders_with_unextractable_products_once():
    orders = pd.DataFrame({"order_id": ["one", "two", "three"]})
    products = pd.DataFrame([
        {"order_reference": "one", "product_name": "Unknown", "extracted_weight_kg": None, "extracted_volume_litres": None},
        {"order_reference": "one", "product_name": "Also unknown", "extracted_weight_kg": None, "extracted_volume_litres": None},
        {"order_reference": "two", "product_name": "Rice", "extracted_weight_kg": 1.0, "extracted_volume_litres": None},
    ])
    malformed = pd.DataFrame(columns=["order_reference", "parse_error"])
    report = build_summary(orders, products, malformed)
    assert "Orders with at least one product lacking extracted weight and volume: 1" in report
    assert "Percentage of orders with at least one product lacking extracted weight and volume: 33.33%" in report