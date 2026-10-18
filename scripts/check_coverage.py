"""Read-only coverage diagnostic for cache, enrichment output, and dashboard joins."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from dashboard import build_cached_enrichment, build_order_queue
from src.ai_enrichment import (
	GeminiEnrichment,
	context_from_row,
	is_cacheable_result,
	product_identifier,
)


def _finite_positive(value: Any) -> bool:
	try:
		value = float(value)
	except (TypeError, ValueError):
		return False
	return math.isfinite(value) and value > 0


def _reliable_estimate(result: GeminiEnrichment) -> bool:
	"""Require all physical values used by the decision engine."""
	return all(
		_finite_positive(value)
		for value in (
			result.estimated_transport_weight_kg,
			result.estimated_packaged_length_cm,
			result.estimated_packaged_width_cm,
			result.estimated_packaged_height_cm,
		)
	)


def _parse_products(value: Any) -> list[dict[str, Any]]:
	if not isinstance(value, str):
		return []
	try:
		parsed = json.loads(value)
	except json.JSONDecodeError:
		return []
	return parsed if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed) else []


def _cache_states(
	processed: pd.DataFrame,
	cache: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], dict[str, Any]]:
	"""Validate cache entries against every processed product context."""
	contexts_by_name: dict[str, set[str]] = defaultdict(set)
	entries_by_key: dict[str, dict[str, Any]] = {}
	for _, row in processed.iterrows():
		context = context_from_row(row)
		key = product_identifier(context)
		contexts_by_name[context.product_name].add(key)
		entries_by_key[key] = cache.get(key)

	return contexts_by_name, entries_by_key, cache


def _classify_cache_entry(entry: Any) -> str:
	if not isinstance(entry, dict):
		return "missing"
	try:
		result = GeminiEnrichment.model_validate(entry)
	except Exception:
		return "invalid"
	if not is_cacheable_result(result):
		return "invalid"
	return "valid_reliable" if _reliable_estimate(result) else "valid_incomplete"


def _order_coverage(
	original: pd.DataFrame,
	valid_names: set[str],
) -> tuple[Counter, int, list[tuple[str, int]]]:
	missing_counts: Counter = Counter()
	complete_orders = 0
	order_missing: list[tuple[str, int]] = []
	for _, order in original.iterrows():
		occurrences = _parse_products(order.get("products_json"))
		missing = sum(1 for occurrence in occurrences if occurrence.get("name") not in valid_names)
		missing_counts[missing] += 1
		if missing == 0 and occurrences:
			complete_orders += 1
		if missing:
			order_missing.append((str(order.get("order_id")), missing))
	return missing_counts, complete_orders, order_missing


def _blocked_reason_counts(queue: pd.DataFrame) -> Counter:
	counts: Counter = Counter()
	for decision in queue["decision"]:
		for reason in decision.review_reasons:
			lower = reason.lower()
			if "missing" in lower or "does not have a completed" in lower or "incomplete" in lower:
				counts["missing or incomplete estimates"] += 1
			elif "confidence" in lower or "invalid" in lower or "below regex" in lower:
				counts["low confidence or invalid physical estimates"] += 1
	return counts


def run_diagnostic(
	processed_path: Path,
	original_path: Path,
	enriched_path: Path,
	cache_path: Path,
) -> dict[str, Any]:
	"""Compute all coverage facts without writing files or calling Gemini."""
	processed = pd.read_csv(processed_path)
	original = pd.read_csv(original_path)
	cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
	legacy_enriched = pd.read_csv(enriched_path) if enriched_path.exists() else pd.DataFrame()
	enriched = build_cached_enrichment(processed, cache)
	if enriched.empty:
		enriched = legacy_enriched
	unique_names = set(processed["product_name"].dropna().astype(str).unique())
	contexts_by_name, entries_by_key, raw_cache = _cache_states(processed, cache)
	state_by_key = {key: _classify_cache_entry(entry) for key, entry in entries_by_key.items()}
	valid_keys = {key for key, state in state_by_key.items() if state == "valid_reliable"}
	valid_names = {
		name for name, keys in contexts_by_name.items() if any(key in valid_keys for key in keys)
	}
	all_cache_states = Counter(_classify_cache_entry(entry) for entry in raw_cache.values())
	missing_counts, complete_orders, order_missing = _order_coverage(original, valid_names)
	queue = build_order_queue(original, enriched)
	complete_without_recommendation = int(
		((queue["order_complete"]) & (queue["recommended_vehicle"] == "Pending")).sum()
	)
	blocked = _blocked_reason_counts(queue)

	cache_key_set = set(raw_cache)
	known_context_keys = set(entries_by_key)
	cache_names: Counter = Counter()
	cache_id_mismatches = 0
	for key, entry in raw_cache.items():
		if not isinstance(entry, dict):
			continue
		try:
			result = GeminiEnrichment.model_validate(entry)
		except Exception:
			continue
		cache_names[result.original_product_name] += 1
		if result.product_id is not None and result.product_id != key:
			cache_id_mismatches += 1
	duplicate_name_entries = sum(max(count - 1, 0) for count in cache_names.values())
	recognized_cache_keys = len(cache_key_set & known_context_keys)
	output_names = set(enriched["product_name"].dropna().astype(str)) if not enriched.empty and "product_name" in enriched else set()
	output_order_count = int(enriched["order_reference"].astype(str).nunique()) if not enriched.empty and "order_reference" in enriched else 0

	return {
		"dataset": {
			"processed_occurrences": len(processed),
			"unique_products": len(unique_names),
			"original_orders": len(original),
		},
		"cache": {
			"raw_entries": len(raw_cache),
			"successful_valid_entries": all_cache_states["valid_reliable"] + all_cache_states["valid_incomplete"],
			"reliable_entries": all_cache_states["valid_reliable"],
			"incomplete_estimate_entries": all_cache_states["valid_incomplete"],
			"invalid_entries": all_cache_states["invalid"],
			"pending_unique_products": len(unique_names - valid_names),
			"failed_unique_products": all_cache_states["invalid"],
			"missing_unique_products": len(unique_names - valid_names),
			"valid_unique_products": len(valid_names),
			"valid_unique_percentage": round(len(valid_names) / len(unique_names) * 100, 2) if unique_names else 0,
			"recognized_cache_keys": recognized_cache_keys,
			"unrecognized_cache_entries": len(cache_key_set - known_context_keys),
			"product_id_mismatches": cache_id_mismatches,
			"duplicate_original_product_name_entries": duplicate_name_entries,
			"unique_products_with_multiple_context_ids": sum(1 for keys in contexts_by_name.values() if len(keys) > 1),
		},
		"orders": {
			"complete_with_reliable_estimates": complete_orders,
			"pending_or_incomplete": len(original) - complete_orders,
			"missing_exactly_1": missing_counts[1],
			"missing_exactly_2": missing_counts[2],
			"missing_exactly_3": missing_counts[3],
			"missing_4_or_more": sum(count for missing, count in missing_counts.items() if missing >= 4),
		},
		"dashboard": {
			"enriched_output_rows": len(enriched),
			"enriched_output_orders": output_order_count,
			"enriched_output_unique_products": len(output_names),
			"complete_orders_without_recommendation": complete_without_recommendation,
			"dashboard_reads_cache_directly": True,
			"dashboard_cache_integration_issue": None,
		},
		"blocking_reasons": dict(blocked),
		"processing_order": {
			"unique_product_selection": "processed_products.csv drop_duplicates(product_name)",
			"groups_by_order": False,
			"fixed_source_order": True,
			"prioritizes_completing_orders": False,
			"note": "The full catalogue is first-seen product order; limited samples add diversity representatives before filling remaining products.",
		},
	}


def print_summary(report: dict[str, Any]) -> None:
	"""Print a concise human-readable terminal summary."""
	dataset = report["dataset"]
	cache = report["cache"]
	orders = report["orders"]
	dashboard = report["dashboard"]
	print("Glovo enrichment coverage diagnostic")
	print(f"Unique products: {dataset['unique_products']}")
	print(f"Valid cache enrichments: {cache['valid_unique_products']} ({cache['valid_unique_percentage']:.2f}%)")
	print(f"Missing unique products: {cache['missing_unique_products']}")
	print(f"Pending/missing unique products: {cache['pending_unique_products']}")
	print(f"Failed/invalid cached products: {cache['failed_unique_products']}")
	print(f"Incomplete estimate cache entries: {cache['incomplete_estimate_entries']}")
	print(f"Invalid cache entries: {cache['invalid_entries']}")
	print(f"Orders complete with reliable estimates: {orders['complete_with_reliable_estimates']} / {dataset['original_orders']}")
	print("Orders missing occurrences: " + ", ".join(
		f"{label}={orders[key]}" for label, key in (("1", "missing_exactly_1"), ("2", "missing_exactly_2"), ("3", "missing_exactly_3"), ("4+", "missing_4_or_more"))
	))
	print(f"Dashboard output: {dashboard['enriched_output_rows']} rows across {dashboard['enriched_output_orders']} orders")
	print(f"Complete orders without recommendation: {dashboard['complete_orders_without_recommendation']}")
	print(f"Cache entries not recognized by current dataset keys: {cache['unrecognized_cache_entries']}")
	print(f"Duplicate cached product-name entries: {cache['duplicate_original_product_name_entries']}")
	print(f"Dashboard cache integration: {'direct' if dashboard['dashboard_reads_cache_directly'] else 'CSV-only; cache successes are not directly visible'}")
	if report["blocking_reasons"]:
		print("Dashboard blocking reasons: " + ", ".join(f"{key}={value}" for key, value in report["blocking_reasons"].items()))


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--processed", default="data/processed_products.csv")
	parser.add_argument("--original", default="data/orphan_orders_case_dataset.csv")
	parser.add_argument("--enriched", default="data/ai_enriched_sample.csv")
	parser.add_argument("--cache", default="data/ai_enrichment_cache.json")
	parser.add_argument("--json", dest="json_path", help="Also write a diagnostic JSON report")
	args = parser.parse_args()
	report = run_diagnostic(Path(args.processed), Path(args.original), Path(args.enriched), Path(args.cache))
	print_summary(report)
	if args.json_path:
		Path(args.json_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
	main()