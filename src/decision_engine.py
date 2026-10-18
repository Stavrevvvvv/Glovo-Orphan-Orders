"""Deterministic courier-capacity evaluation for enriched product rows."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class VehicleCapacity:
	"""Prototype capacity assumptions, not official Glovo vehicle data."""

	name: str
	max_weight_kg: float
	max_volume_litres: float
	max_item_length_cm: float
	max_item_width_cm: float
	max_item_height_cm: float
	relative_delivery_cost: float


DEFAULT_VEHICLES = (
	VehicleCapacity("Bicycle", 15.0, 50.0, 100.0, 60.0, 60.0, 1.0),
	VehicleCapacity("Motorbike", 20.0, 70.0, 120.0, 70.0, 70.0, 1.8),
	VehicleCapacity("Car", 100.0, 500.0, 200.0, 120.0, 100.0, 3.0),
)
LIQUID_VOLUME_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True)
class DecisionConfig:
	"""Adjustable safety assumptions and hypothetical relative costs."""

	vehicles: tuple[VehicleCapacity, ...] = DEFAULT_VEHICLES
	weight_safety_margin: float = 0.10
	volume_safety_margin: float = 0.15
	packing_efficiency: float = 0.85
	near_capacity_ratio: float = 0.90
	medium_confidence_multiplier: float = 1.10
	low_confidence_multiplier: float = 1.25
	max_split_couriers: int = 10

	def __post_init__(self) -> None:
		if not 0 < self.packing_efficiency <= 1:
			raise ValueError("packing_efficiency must be greater than 0 and at most 1")
		if self.weight_safety_margin < 0 or self.volume_safety_margin < 0:
			raise ValueError("safety margins cannot be negative")
		if not 0 < self.near_capacity_ratio <= 1:
			raise ValueError("near_capacity_ratio must be greater than 0 and at most 1")
		if self.max_split_couriers < 1:
			raise ValueError("max_split_couriers must be positive")
		if any(vehicle.relative_delivery_cost <= 0 for vehicle in self.vehicles):
			raise ValueError("relative delivery costs must be positive")


@dataclass(frozen=True)
class ProductEstimate:
	"""One enriched product line, expanded into whole individual units."""

	name: str
	quantity: int
	weight_kg: float | None
	length_cm: float | None
	width_cm: float | None
	height_cm: float | None
	confidence: str
	fragile_or_special_handling: bool
	manual_review_recommended: bool
	review_reasons: tuple[str, ...] = ()

	@property
	def item_volume_litres(self) -> float | None:
		if None in (self.length_cm, self.width_cm, self.height_cm):
			return None
		return self.length_cm * self.width_cm * self.height_cm / 1000

	@property
	def conservative_weight_kg(self) -> float | None:
		return self.weight_kg

	@property
	def conservative_volume_litres(self) -> float | None:
		volume = self.item_volume_litres
		if volume is None:
			return None
		return volume * (1.25 if self.confidence == "low" else 1.10 if self.confidence == "medium" else 1.0)


@dataclass(frozen=True)
class DeliveryOption:
	"""One vehicle/allocation candidate and why it is or is not feasible."""

	vehicle: str
	courier_count: int
	feasible: bool
	relative_cost: float
	rejection_reason: str | None = None
	split_required: bool = False
	heuristic_optimal: bool | None = None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(frozen=True)
class DecisionRecord:
	"""Structured recommendation kept separate from approval status."""

	order_reference: str
	estimated_total_weight_kg: float | None
	estimated_total_packaged_volume_litres: float | None
	recommended_vehicle: str | None
	recommended_courier_count: int | None
	decision_confidence: str
	requires_human_review: bool
	review_reasons: tuple[str, ...]
	decision_explanation: str
	vehicle_capacity_assumptions: tuple[str, ...]
	alternative_options: tuple[DeliveryOption, ...] = ()
	order_complete: bool = False

	def to_dict(self) -> dict[str, Any]:
		result = asdict(self)
		result["alternative_options"] = [option.to_dict() for option in self.alternative_options]
		return result


@dataclass(frozen=True)
class AllocationResult:
	courier_count: int | None
	heuristic_optimal: bool | None


def _number(row: Mapping[str, Any], field_name: str, positive: bool = False) -> float | None:
	value = row.get(field_name)
	if value is None or pd.isna(value):
		return None
	try:
		value = float(value)
	except (TypeError, ValueError):
		return None
	if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
		return None
	return value


def _text(row: Mapping[str, Any], field_name: str) -> str | None:
	value = row.get(field_name)
	return str(value) if value is not None and not pd.isna(value) else None


def _bool(row: Mapping[str, Any], field_name: str) -> bool:
	value = row.get(field_name)
	return value is True or (isinstance(value, str) and value.lower() == "true")


def _confidence_multiplier(confidence: str, config: DecisionConfig) -> float:
	if confidence == "low":
		return config.low_confidence_multiplier
	if confidence == "medium":
		return config.medium_confidence_multiplier
	return 1.0


def _special_handling_reason(row: Mapping[str, Any]) -> str | None:
	"""Require review for explicit risks, not ordinary refrigerated/perishable handling."""
	fields = (
		"product_name", "ai_normalized_product_description", "ai_product_category",
		"ai_estimate_explanation",
	)
	description = " ".join(str(row.get(field) or "") for field in fields).lower()
	risk_patterns = (
		r"\bfragile\b", r"\bbreakable\b", r"\bglass\b", r"\beggs?\b", r"\bbruis\w*\b",
		r"\bleak\w*\b", r"\bspill\w*\b", r"\bcrush\w*\b", r"\bdelicate\b", r"\bceramic\b",
		r"\bfrozen\b", r"\bfreezer\b", r"\bdry ice\b", r"\bhazard\w*\b", r"\bflammab\w*\b",
		r"\bexplos\w*\b", r"\bpressur\w*\b", r"\baerosol\b", r"\blithium battery\b",
		r"\bawkward\b", r"\boversized\b", r"\bdifficult to handle\b", r"\bunusually difficult\b",
		r"\btemperature[- ]sensitive\b", r"\bdeep[- ]frozen\b", r"\brequires? freezing\b",
	)
	if any(re.search(pattern, description) for pattern in risk_patterns):
		return "fragile, hazardous, frozen, or unusually difficult item requires review"
	cold_chain_patterns = (r"refrigerat\w*", r"cold[- ]chain", r"perishable", r"chilled", r"keep cold", r"temperature[- ]controlled transport")
	if _bool(row, "ai_fragile_or_special_handling") and any(re.search(pattern, description) for pattern in cold_chain_patterns):
		return None
	if _bool(row, "ai_manual_review_recommended"):
		uncertainty_terms = ("cannot identify", "cannot be identified", "unknown product", "ambiguous identity", "identity uncertain")
		if any(term in description for term in uncertainty_terms):
			return "product identity remains uncertain"
	return None


def _maximum_dimension_utilization(products: Iterable[ProductEstimate], vehicle: VehicleCapacity) -> float:
	"""Return worst per-item dimension utilization after allowing rotation."""
	limits = sorted((vehicle.max_item_length_cm, vehicle.max_item_width_cm, vehicle.max_item_height_cm), reverse=True)
	maximum = 0.0
	for product in products:
		if None in (product.length_cm, product.width_cm, product.height_cm):
			return math.inf
		maximum = max(maximum, *(dimension / limit for dimension, limit in zip(sorted((product.length_cm, product.width_cm, product.height_cm), reverse=True), limits)))
	return maximum


def calculate_order_weight_breakdown(
	product_rows: Iterable[Mapping[str, Any]],
	config: DecisionConfig = DecisionConfig(),
) -> tuple[float | None, float | None, float | None]:
	"""Return raw AI weight, safety-margin weight, and conservative dispatch weight."""
	total_weight = 0.0
	rows = list(product_rows)
	if not rows:
		return None, None, None
	for row in rows:
		quantity = _number(row, "quantity", positive=True)
		weight = _number(row, "ai_estimated_transport_weight_kg", positive=True)
		if quantity is None or not quantity.is_integer() or weight is None:
			return None, None, None
		total_weight += quantity * weight
	margin = total_weight * config.weight_safety_margin
	return total_weight, margin, total_weight + margin


def liquid_volume_plausibility_warning(row: Mapping[str, Any]) -> str | None:
	"""Flag materially undersized dimensions relative to reliably extracted liquid volume."""
	liquid_volume = _number(row, "extracted_volume_litres", positive=True)
	if liquid_volume is None or _text(row, "extraction_warning"):
		return None
	length = _number(row, "ai_estimated_packaged_length_cm", positive=True)
	width = _number(row, "ai_estimated_packaged_width_cm", positive=True)
	height = _number(row, "ai_estimated_packaged_height_cm", positive=True)
	if None in (length, width, height):
		return None
	external_volume = length * width * height / 1000
	if liquid_volume > external_volume * (1 + LIQUID_VOLUME_RELATIVE_TOLERANCE):
		return (
			f"liquid volume plausibility warning: {liquid_volume:.2f} L content exceeds "
			f"estimated external packaged volume of {external_volume:.2f} L"
		)
	return None


def product_estimate_from_row(row: Mapping[str, Any], config: DecisionConfig = DecisionConfig()) -> ProductEstimate:
	"""Convert one enriched CSV row into a non-zero, validated physical estimate."""
	name = _text(row, "product_name") or _text(row, "ai_original_product_name") or "Unnamed product"
	reasons: list[str] = []
	if _text(row, "ai_status") != "completed":
		reasons.append("product does not have a completed AI enrichment")
	quantity_value = _number(row, "quantity", positive=True)
	quantity = int(quantity_value) if quantity_value is not None and quantity_value.is_integer() else 0
	if quantity == 0:
		reasons.append("missing or invalid product quantity")
	weight = _number(row, "ai_estimated_transport_weight_kg", positive=True)
	length = _number(row, "ai_estimated_packaged_length_cm", positive=True)
	width = _number(row, "ai_estimated_packaged_width_cm", positive=True)
	height = _number(row, "ai_estimated_packaged_height_cm", positive=True)
	if weight is None:
		reasons.append("missing or invalid estimated transport weight")
	if any(value is None for value in (length, width, height)):
		reasons.append("missing or invalid packaged dimensions")
	confidence = (_text(row, "ai_confidence") or "unknown").lower()
	if confidence not in {"high", "medium", "low"}:
		reasons.append("missing or invalid AI confidence")
		confidence = "low"
	if confidence == "low":
		reasons.append("low-confidence physical estimate")
	special_handling_reason = _special_handling_reason(row)
	if special_handling_reason:
		reasons.append(special_handling_reason)
	regex_weight = _number(row, "extracted_weight_kg", positive=True)
	if weight is not None and regex_weight is not None and weight < regex_weight:
		reasons.append("AI transport weight is below regex net weight")
	if _text(row, "extraction_warning"):
		reasons.append("regex extraction warning: " + _text(row, "extraction_warning"))
	liquid_warning = liquid_volume_plausibility_warning(row)
	if liquid_warning:
		reasons.append(liquid_warning)
	multiplier = _confidence_multiplier(confidence, config)
	return ProductEstimate(
		name=name,
		quantity=quantity,
		weight_kg=weight * multiplier if weight is not None else None,
		length_cm=length,
		width_cm=width,
		height_cm=height,
		confidence=confidence,
		fragile_or_special_handling=_bool(row, "ai_fragile_or_special_handling"),
		manual_review_recommended=bool(reasons),
		review_reasons=tuple(dict.fromkeys(reasons)),
	)


def _dimension_fits(item: ProductEstimate, vehicle: VehicleCapacity) -> bool:
	"""Check all rotations against vehicle-specific limits without bending."""
	if None in (item.length_cm, item.width_cm, item.height_cm):
		return False
	item_dimensions = sorted((item.length_cm, item.width_cm, item.height_cm), reverse=True)
	vehicle_dimensions = sorted(
		(vehicle.max_item_length_cm, vehicle.max_item_width_cm, vehicle.max_item_height_cm), reverse=True
	)
	return all(item_value <= vehicle_value for item_value, vehicle_value in zip(item_dimensions, vehicle_dimensions))


def _expand_items(products: Iterable[ProductEstimate]) -> list[ProductEstimate]:
	items: list[ProductEstimate] = []
	for product in products:
		items.extend([product] * product.quantity)
	return items


def _fits_bin(items: list[ProductEstimate], vehicle: VehicleCapacity, config: DecisionConfig) -> bool:
	weight = sum(item.conservative_weight_kg or 0 for item in items)
	volume = sum(item.conservative_volume_litres or 0 for item in items)
	return (
		weight * (1 + config.weight_safety_margin) <= vehicle.max_weight_kg
		and volume * (1 + config.volume_safety_margin) / config.packing_efficiency <= vehicle.max_volume_litres
	)


def _exact_minimum_allocation(items: list[ProductEstimate], vehicle: VehicleCapacity, config: DecisionConfig) -> int | None:
	"""Find an exact allocation for small item sets; return null for larger sets."""
	if len(items) > 12:
		return None
	ordered = sorted(items, key=lambda item: (item.conservative_volume_litres or 0, item.conservative_weight_kg or 0), reverse=True)
	best = len(items) + 1

	def search(index: int, bins: list[list[ProductEstimate]]) -> None:
		nonlocal best
		if index == len(ordered):
			best = len(bins)
			return
		if len(bins) >= best:
			return
		item = ordered[index]
		seen_signatures: set[tuple[float, float]] = set()
		for bucket in bins:
			signature = (
				sum(entry.conservative_weight_kg or 0 for entry in bucket),
				sum(entry.conservative_volume_litres or 0 for entry in bucket),
			)
			if signature in seen_signatures:
				continue
			seen_signatures.add(signature)
			if _fits_bin(bucket + [item], vehicle, config):
				bucket.append(item)
				search(index + 1, bins)
				bucket.pop()
		if len(bins) + 1 <= best:
			bins.append([item])
			search(index + 1, bins)
			bins.pop()

	search(0, [])
	return best if best <= len(items) else None


def _allocate_items(items: list[ProductEstimate], vehicle: VehicleCapacity, config: DecisionConfig) -> AllocationResult:
	"""Return first-fit allocation and whether its count is provably minimal."""
	if any(
		item.conservative_weight_kg is None
		or item.conservative_volume_litres is None
		or not _dimension_fits(item, vehicle)
		for item in items
	):
		return AllocationResult(None, None)
	ordered = sorted(items, key=lambda item: (item.conservative_volume_litres, item.conservative_weight_kg), reverse=True)
	for courier_count in range(1, config.max_split_couriers + 1):
		buckets: list[list[ProductEstimate]] = [[] for _ in range(courier_count)]
		valid = True
		for item in ordered:
			for bucket in buckets:
				if _fits_bin(bucket + [item], vehicle, config):
					bucket.append(item)
					break
			else:
				valid = False
				break
		if valid:
			if courier_count == 1:
				return AllocationResult(1, True)
			exact = _exact_minimum_allocation(items, vehicle, config)
			if exact is None:
				return AllocationResult(courier_count, None)
			return AllocationResult(courier_count, exact == courier_count)
	return AllocationResult(None, None)


def _order_occurrences_from_json(products_json: Any) -> list[tuple[str, float | None]] | None:
	"""Parse original name/quantity occurrences while preserving duplicates."""
	if isinstance(products_json, str):
		try:
			products_json = json.loads(products_json)
		except json.JSONDecodeError:
			return None
	if not isinstance(products_json, list):
		return None
	occurences = []
	for product in products_json:
		if not isinstance(product, dict) or not isinstance(product.get("name"), str):
			return None
		quantity = product.get("quantity")
		try:
			quantity = float(quantity)
			if not math.isfinite(quantity) or quantity <= 0:
				quantity = None
		except (TypeError, ValueError):
			quantity = None
		occurences.append((product["name"], quantity))
	return occurences


def verify_order_completeness(order_reference: str, enriched_rows: Iterable[Mapping[str, Any]], products_json: Any) -> tuple[bool, tuple[str, ...]]:
	"""Compare enriched occurrence names to original JSON, including duplicates."""
	original_occurrences = _order_occurrences_from_json(products_json)
	if original_occurrences is None:
		return False, ("original products_json is missing or malformed",)
	enriched_occurrences = [
		(_text(row, "product_name"), _number(row, "quantity", positive=True))
		for row in enriched_rows
		if _text(row, "product_name") is not None
		and (row.get("order_reference") is None or str(row.get("order_reference")) == str(order_reference))
	]
	missing = list((Counter(original_occurrences) - Counter(enriched_occurrences)).elements())
	extra = list((Counter(enriched_occurrences) - Counter(original_occurrences)).elements())
	reasons = []
	if missing:
		reasons.append("missing enriched product occurrences: " + ", ".join(name for name, _ in missing[:5]))
	if extra:
		reasons.append("enrichment contains products absent from original order: " + ", ".join(name for name, _ in extra[:5]))
	return not reasons, tuple(reasons)


def evaluate_order(
	order_reference: str,
	product_rows: Iterable[Mapping[str, Any]],
	config: DecisionConfig = DecisionConfig(),
	original_products_json: Any | None = None,
) -> DecisionRecord:
	"""Evaluate every delivery option for one complete enriched order."""
	rows = list(product_rows)
	products = [product_estimate_from_row(row, config) for row in rows]
	assumptions = (
		"Vehicle capacities and relative delivery costs are hypothetical prototype assumptions, not official Glovo data.",
		f"Safety margins: {config.weight_safety_margin:.0%} weight and {config.volume_safety_margin:.0%} volume.",
		f"Packing efficiency: {config.packing_efficiency:.0%}; item dimensions allow rotation but not bending or compression.",
	)
	if original_products_json is None:
		complete = False
		completeness_reasons = ("original products_json was not supplied; completeness cannot be verified",)
	else:
		complete, completeness_reasons = verify_order_completeness(order_reference, rows, original_products_json)
	all_reasons = list(dict.fromkeys([*completeness_reasons, *(reason for product in products for reason in product.review_reasons)]))
	blocking_prefixes = (
		"original products_json", "missing enriched", "enrichment contains", "product does not have",
		"missing or invalid", "AI transport weight is below", "liquid volume plausibility warning",
	)
	blocking_reasons = [reason for reason in all_reasons if reason.startswith(blocking_prefixes)]
	if blocking_reasons:
		return DecisionRecord(
			str(order_reference), None, None, None, None, "low", True, tuple(blocking_reasons),
			"No final recommendation was made because the order is incomplete or has missing/invalid physical estimates.",
			assumptions, order_complete=complete,
		)
	items = _expand_items(products)
	total_weight = sum((item.conservative_weight_kg or 0) * item.quantity for item in products) * (1 + config.weight_safety_margin)
	total_volume = sum((item.conservative_volume_litres or 0) * item.quantity for item in products) * (1 + config.volume_safety_margin) / config.packing_efficiency
	options: list[DeliveryOption] = []
	for vehicle in config.vehicles:
		allocation = _allocate_items(items, vehicle, config)
		if allocation.courier_count is None:
			options.append(DeliveryOption(vehicle.name, 0, False, 0.0, "item dimensions or per-courier capacity cannot be satisfied"))
			continue
		cost = vehicle.relative_delivery_cost * allocation.courier_count
		options.append(DeliveryOption(vehicle.name, allocation.courier_count, True, cost, None, allocation.courier_count > 1, allocation.heuristic_optimal))
	feasible = [option for option in options if option.feasible]
	selected = min(feasible, key=lambda option: (option.relative_cost, option.courier_count)) if feasible else None
	if selected is None:
		all_reasons.append("no safe allocation across configured vehicle limits")
		explanation = "No configured vehicle can safely carry the complete order; customer intervention or human review is required."
	else:
		vehicle = next(vehicle for vehicle in config.vehicles if vehicle.name == selected.vehicle)
		weight_ratio = total_weight / (vehicle.max_weight_kg * selected.courier_count)
		volume_ratio = total_volume / (vehicle.max_volume_litres * selected.courier_count)
		if selected.courier_count > 1:
			all_reasons.append("splitting is required because one courier cannot safely carry the complete order")
		if max(weight_ratio, volume_ratio) >= config.near_capacity_ratio:
			all_reasons.append("order is close to assumed capacity limits")
		if _maximum_dimension_utilization(products, vehicle) >= config.near_capacity_ratio:
			all_reasons.append("product dimensions are close to assumed item limits")
		if selected.heuristic_optimal is False:
			all_reasons.append("a better allocation is known; human review is required")
		elif selected.heuristic_optimal is None:
			lower_cost_alternative = any(
				option.feasible and option.relative_cost < selected.relative_cost
				for option in options
				if option.vehicle != selected.vehicle or option.courier_count != selected.courier_count
			)
			if selected.courier_count > 1 or lower_cost_alternative:
				all_reasons.append("allocation optimality is unproven and could change the recommendation or cost")
		alternatives = "; ".join(
			f"{option.vehicle}: {option.relative_cost:.2f} relative cost" if option.feasible else f"{option.vehicle}: rejected ({option.rejection_reason})"
			for option in options if option.vehicle != selected.vehicle or option.courier_count != selected.courier_count
		)
		explanation = f"Selected {selected.vehicle} with {selected.courier_count} courier(s) because it has the lowest hypothetical relative cost among safe options. Alternatives: {alternatives}."
	confidence = (
		"low" if any(product.confidence == "low" for product in products)
		else "medium" if all_reasons or any(product.confidence == "medium" for product in products)
		else "high"
	)
	return DecisionRecord(
		str(order_reference), total_weight, total_volume,
		selected.vehicle if selected else None, selected.courier_count if selected else None,
		confidence, bool(all_reasons), tuple(dict.fromkeys(all_reasons)), explanation, assumptions,
		tuple(options), order_complete=complete,
	)


def main() -> None:
	"""Demonstrate one complete-order decision from the enriched sample."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="data/ai_enriched_sample.csv")
	parser.add_argument("--original-input", default="data/orphan_orders_case_dataset.csv")
	parser.add_argument("--order-reference", default=None)
	args = parser.parse_args()
	enriched = pd.read_csv(args.input)
	original = pd.read_csv(args.original_input)
	if "order_reference" not in enriched.columns or "order_id" not in original.columns:
		raise ValueError("input files are missing order reference columns")
	order_reference = args.order_reference or str(enriched.iloc[0]["order_reference"])
	rows = enriched[enriched["order_reference"].astype(str) == str(order_reference)].to_dict("records")
	original_rows = original[original["order_id"].astype(str) == str(order_reference)]
	if original_rows.empty:
		raise ValueError(f"original order not found: {order_reference}")
	decision = evaluate_order(str(order_reference), rows, original_products_json=original_rows.iloc[0]["products_json"])
	print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()