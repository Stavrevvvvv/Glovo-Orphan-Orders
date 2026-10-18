"""Load orphan orders and flatten their product payloads."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ORDER_REFERENCE_CANDIDATES = ("order_id", "order_reference", "order_ref")

UNIT_PATTERN = (
    r"kilograms?|grams?|litres?|liters?|litros?|"
	r"kg|grs|gr|g|cl|ml|l|кг|гр|г|мл|л|კგ|გ|მლ|ლ"
)

_MULTIPACK_PATTERN = re.compile(
    rf"(?P<count>\d+)\s*[xX×]\s*"
    rf"(?P<value>\d+(?:[.,]\d+)?)\s*"
    rf"(?P<unit>{UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE,
)

_CONTENT_PATTERN = re.compile(
    rf"(?P<value>\d+(?:[.,]\d+)?)\s*"
    rf"(?P<unit>{UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE,
)
OUTPUT_COLUMNS = [
	"order_reference", "product_name", "quantity", "unit_price",
	"declared_weight_kg", "multipack_quantity", "net_content_value",
	"net_content_unit", "net_content_weight_kg", "net_content_volume_litres",
	"extracted_weight_min_kg", "extracted_weight_max_kg",
	"extracted_weight_kg", "extracted_volume_litres", "extraction_method",
	"extraction_confidence", "matched_text", "measurement_sources",
	"measurement_type", "extraction_warning",
	"estimated_transport_weight_kg",
]


def load_orders(csv_path: str | Path) -> pd.DataFrame:
	"""Load an orders CSV with Pandas."""
	return pd.read_csv(csv_path)


def parse_products_json_detailed(value: Any) -> tuple[list[dict[str, Any]], str | None]:
	"""Parse product JSON and return a reason when the order cannot be parsed."""
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return [], "missing_products_json"
	if not isinstance(value, str):
		return [], "products_json_not_text"
	try:
		value = json.loads(value)
	except (json.JSONDecodeError, TypeError):
		return [], "malformed_json"
	if not isinstance(value, list):
		return [], "products_json_not_array"
	products = [product for product in value if isinstance(product, dict)]
	if len(products) != len(value):
		return [], "product_entry_not_object"
	return products, None


def parse_products_json(value: Any) -> list[dict[str, Any]]:
	"""Return product dictionaries from JSON, or an empty list if invalid."""
	return parse_products_json_detailed(value)[0]


def _normalise_unit(unit: str) -> str:
	unit = unit.lower()
	if unit in {"kg", "kilogram", "kilograms", "кг", "კგ"}:
		return "kg"
	if unit in {"g", "gr", "grs", "gram", "grams", "г", "гр", "გ"}:
		return "g"
	if unit in {"l", "litre", "litres", "liter", "liters", "litro", "litros", "л", "ლ"}:
		return "l"
	if unit == "cl":
		return "cl"
	if unit in {"мл", "მლ"}:
		return "ml"
	return "ml"


def _empty_content() -> dict[str, Any]:
	return {
		"multipack_quantity": None,
		"net_content_value": None,
		"net_content_unit": None,
		"net_content_weight_kg": None,
		"net_content_volume_litres": None,
		"extracted_weight_min_kg": None,
		"extracted_weight_max_kg": None,
		"extracted_weight_kg": None,
		"extracted_volume_litres": None,
		"extraction_method": None,
		"extraction_confidence": None,
		"matched_text": None,
		"measurement_sources": None,
		"measurement_type": None,
		"extraction_warning": None,
		"estimated_transport_weight_kg": None,
	}


def _convert_measurement(value: float, unit: str) -> tuple[str, float]:
	"""Return a normalized dimension and value without inferring other properties."""
	unit = _normalise_unit(unit)
	if unit == "kg":
		return "weight", value
	if unit == "g":
		return "weight", value / 1000
	if unit == "l":
		return "volume", value
	if unit == "cl":
		return "volume", value / 100
	return "volume", value / 1000


def _measurement_source(match: re.Match[str], total_value: float | None = None) -> dict[str, Any]:
	"""Serialize the exact source and normalized value of one regex match."""
	value = float(match.group("value").replace(",", "."))
	count = match.groupdict().get("count")
	if count:
		value *= int(count)
	dimension, normalized_value = _convert_measurement(value, match.group("unit"))
	return {
		"matched_text": match.group(0),
		"raw_value": value,
		"unit": _normalise_unit(match.group("unit")),
		"dimension": dimension,
		"normalized_value": total_value if total_value is not None else normalized_value,
		"multipack_quantity": int(count) if count else None,
	}


def extract_net_content(product_name: Any) -> dict[str, Any]:
	"""Extract net content without treating it as transport weight.

	Weight is normalized to kilograms and volume to litres in separate fields.
	The transport-weight field intentionally remains null because net content is
	not a verified shipping weight.
	"""
	if not isinstance(product_name, str) or not product_name.strip():
		return _empty_content()

	range_match = re.search(
		r"(?P<first_value>\d+(?:[.,]\d+)?)\s*(?P<first_unit>" + UNIT_PATTERN + r")"
		r"\s*[-–]\s*"
		r"(?P<second_value>\d+(?:[.,]\d+)?)\s*(?P<second_unit>" + UNIT_PATTERN + r")",
		product_name,
		re.IGNORECASE,
	)
	multipack_match = _MULTIPACK_PATTERN.search(product_name)
	content_matches = list(_CONTENT_PATTERN.finditer(product_name))
	if not content_matches:
		return _empty_content()

	if range_match:
		first_dimension, first_value = _convert_measurement(
			float(range_match.group("first_value").replace(",", ".")), range_match.group("first_unit")
		)
		second_dimension, second_value = _convert_measurement(
			float(range_match.group("second_value").replace(",", ".")), range_match.group("second_unit")
		)
		if first_dimension == second_dimension == "weight":
			return {
				**_empty_content(),
				"net_content_unit": "kg",
				"extracted_weight_min_kg": min(first_value, second_value),
				"extracted_weight_max_kg": max(first_value, second_value),
				"extraction_method": "regex_range",
				"extraction_confidence": "medium",
				"matched_text": range_match.group(0),
				"measurement_sources": json.dumps([
					{"matched_text": range_match.group(0), "range_min_kg": min(first_value, second_value),
					 "range_max_kg": max(first_value, second_value)}
				], ensure_ascii=False),
				"measurement_type": "range",
				"extraction_warning": "possible weight range; exact value not selected",
			}

	first_match = multipack_match or content_matches[0]
	if multipack_match:
		content_matches = [multipack_match] + [
			match for match in content_matches if match.start() >= multipack_match.end()
		]
	sources = [_measurement_source(match) for match in content_matches]
	weight_kg = next((source["normalized_value"] for source in sources if source["dimension"] == "weight"), None)
	volume_litres = next((source["normalized_value"] for source in sources if source["dimension"] == "volume"), None)
	multipack_quantity = int(multipack_match.group("count")) if multipack_match else None
	warning = None
	method = "regex_multipack" if multipack_match else "regex_single"
	measurement_type = "single"
	if multipack_match and len(content_matches) > 1:
		pack_source = sources[0]
		explicit_source = sources[-1]
		if pack_source["dimension"] == explicit_source["dimension"] and abs(
			pack_source["normalized_value"] - explicit_source["normalized_value"]
		) < 1e-12:
			weight_kg = explicit_source["normalized_value"] if explicit_source["dimension"] == "weight" else None
			volume_litres = explicit_source["normalized_value"] if explicit_source["dimension"] == "volume" else None
			method = "regex_multipack_total"
			warning = "multipack and explicit total may describe the same content; values not added"
		else:
			weight_kg = None if len({source["dimension"] for source in sources}) == 1 else weight_kg
			volume_litres = None if len({source["dimension"] for source in sources}) == 1 else volume_litres
			warning = "multiple measurements; interpretation is ambiguous"
		measurement_type = "ambiguous"
	elif len(content_matches) > 1:
		measurement_type = "ambiguous"
		warning = "multiple measurements; values preserved by dimension"

	return {
		"multipack_quantity": multipack_quantity,
		"net_content_value": sources[0]["raw_value"] if measurement_type == "single" else None,
		"net_content_unit": _normalise_unit(first_match.group("unit")),
		"net_content_weight_kg": weight_kg,
		"net_content_volume_litres": volume_litres,
		"extracted_weight_min_kg": None,
		"extracted_weight_max_kg": None,
		"extracted_weight_kg": weight_kg,
		"extracted_volume_litres": volume_litres,
		"extraction_method": method,
		"extraction_confidence": "medium" if warning else "high",
		"matched_text": " | ".join(source["matched_text"] for source in sources),
		"measurement_sources": json.dumps(sources, ensure_ascii=False),
		"measurement_type": measurement_type,
		"extraction_warning": warning,
		"estimated_transport_weight_kg": None,
	}


def _find_order_reference_column(columns: list[str]) -> str:
	"""Find the existing order reference column without inventing one."""
	for candidate in ORDER_REFERENCE_CANDIDATES:
		if candidate in columns:
			return candidate
	raise ValueError(
		"Could not identify an order reference column. "
		f"Expected one of: {', '.join(ORDER_REFERENCE_CANDIDATES)}"
	)


def extract_products(orders: pd.DataFrame) -> pd.DataFrame:
	"""Flatten valid product objects while preserving their order and fields."""
	return process_orders(orders)[0]


def process_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Return flattened products and a report of orders with invalid product JSON."""
	if "products_json" not in orders.columns:
		raise ValueError("CSV is missing required column: products_json")
	order_column = _find_order_reference_column(list(orders.columns))
	product_rows = []
	malformed_rows = []
	for _, order in orders.iterrows():
		products, parse_error = parse_products_json_detailed(order["products_json"])
		if parse_error:
			malformed_rows.append({"order_reference": order[order_column], "parse_error": parse_error})
			continue
		for product in products:
			name = product.get("name")
			product_rows.append({
				"order_reference": order[order_column],
				"product_name": name,
				"quantity": product.get("quantity"),
				"unit_price": product.get("unit_price"),
				"declared_weight_kg": product.get("declared_weight_kg"),
				**extract_net_content(name),
			})
	return (
		pd.DataFrame(product_rows, columns=OUTPUT_COLUMNS),
		pd.DataFrame(malformed_rows, columns=["order_reference", "parse_error"]),
	)


def create_unique_catalogue(products: pd.DataFrame) -> pd.DataFrame:
	"""Create an exact-string product-name catalogue."""
	return products[["product_name"]].drop_duplicates().reset_index(drop=True)


def create_unmatched_products(products: pd.DataFrame) -> pd.DataFrame:
	"""Create a catalogue of unique names without extracted weight or volume."""
	mask = products["extracted_weight_kg"].isna() & products["extracted_volume_litres"].isna()
	return products.loc[mask, ["product_name"]].dropna().drop_duplicates().reset_index(drop=True)


def build_summary(orders: pd.DataFrame, products: pd.DataFrame, malformed: pd.DataFrame) -> str:
	"""Build the requested human-readable preprocessing summary."""
	total_occurrences = len(products)
	weight_pct = products["extracted_weight_kg"].notna().mean() * 100 if total_occurrences else 0
	volume_pct = products["extracted_volume_litres"].notna().mean() * 100 if total_occurrences else 0
	neither_pct = (
		(products["extracted_weight_kg"].isna() & products["extracted_volume_litres"].isna()).mean() * 100
		if total_occurrences else 0
	)
	unique = products.drop_duplicates("product_name")
	unique_extractable = (
		(unique["extracted_weight_kg"].notna() | unique["extracted_volume_litres"].notna()).mean() * 100
		if len(unique) else 0
	)
	unextractable_mask = products["extracted_weight_kg"].isna() & products["extracted_volume_litres"].isna()
	orders_with_unextractable = products.loc[unextractable_mask, "order_reference"].nunique()
	orders_with_unextractable_pct = orders_with_unextractable / len(orders) * 100 if len(orders) else 0
	unmatched_examples = create_unmatched_products(products)["product_name"].head(30).tolist()
	ambiguous_examples = []
	if "extraction_warning" in products.columns:
		ambiguous_examples = products.loc[
			products["extraction_warning"].notna(), "product_name"
		].drop_duplicates().head(30).tolist()
	lines = [
		f"Total orders: {len(orders)}",
		f"Total product occurrences: {total_occurrences}",
		f"Percentage with extracted weight: {weight_pct:.2f}%",
		f"Percentage with extracted volume: {volume_pct:.2f}%",
		f"Percentage with neither: {neither_pct:.2f}%",
		f"Percentage of unique product strings with extractable information: {unique_extractable:.2f}%",
		f"Number of malformed orders: {len(malformed)}",
		f"Orders with at least one product lacking extracted weight and volume: {orders_with_unextractable}",
		f"Percentage of orders with at least one product lacking extracted weight and volume: {orders_with_unextractable_pct:.2f}%",
		"Examples of 30 unmatched product names:",
		*(f"- {name}" for name in unmatched_examples),
		"Examples of potentially ambiguous matches:",
		*(f"- {name}" for name in ambiguous_examples),
	]
	return "\n".join(lines) + "\n"


def write_outputs(
	products: pd.DataFrame,
	output_dir: str | Path,
	malformed: pd.DataFrame | None = None,
	orders: pd.DataFrame | None = None,
) -> None:
	"""Write product, catalogue, unmatched, malformed, and summary outputs."""
	malformed = malformed if malformed is not None else pd.DataFrame(
		columns=["order_reference", "parse_error"]
	)
	orders = orders if orders is not None else pd.DataFrame()
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	products.to_csv(output_dir / "processed_products.csv", index=False)
	create_unique_catalogue(products).to_csv(output_dir / "unique_products.csv", index=False)
	create_unmatched_products(products).to_csv(output_dir / "unmatched_products.csv", index=False)
	malformed.to_csv(output_dir / "malformed_orders.csv", index=False)
	(output_dir / "summary_report.txt").write_text(
		build_summary(orders, products, malformed), encoding="utf-8"
	)


def print_summary(orders: pd.DataFrame, products: pd.DataFrame) -> None:
	"""Print the requested extraction summary."""
	malformed = pd.DataFrame(columns=["order_reference", "parse_error"])
	print(build_summary(orders, products, malformed))


def main() -> None:
	"""Run preprocessing from the command line."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="data/orphan_orders_case_dataset.csv")
	parser.add_argument("--output-dir", default="data")
	args = parser.parse_args()
	orders = load_orders(args.input)
	products, malformed = process_orders(orders)
	write_outputs(products, args.output_dir, malformed, orders)
	print(build_summary(orders, products, malformed))


if __name__ == "__main__":
	main()
