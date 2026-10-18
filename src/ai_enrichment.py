"""Enrich a small product sample with Gemini structured output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_LIMIT = 20
MAX_SAMPLE_LIMIT = 100_000
DEFAULT_BATCH_SIZE = 10
DEFAULT_CACHE_PATH = Path("data/ai_enrichment_cache.json")
DEFAULT_REQUEST_INTERVAL_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0
DEFAULT_JITTER_SECONDS = 1.0


class ProductContext(BaseModel):
	"""Original product data and regex evidence sent to the model."""

	model_config = ConfigDict(extra="allow")

	product_name: str
	extracted_weight_kg: float | None = None
	extracted_volume_litres: float | None = None
	extracted_weight_min_kg: float | None = None
	extracted_weight_max_kg: float | None = None
	extraction_warning: str | None = None
	matched_text: str | None = None
	measurement_sources: str | None = None
	measurement_type: str | None = None


class GeminiEnrichment(BaseModel):
	"""Validated structured response returned by Gemini."""

	original_product_name: str
	product_id: str | None = None
	normalized_product_description: str | None = None
	product_category: str | None = None
	estimated_transport_weight_kg: float | None = Field(default=None, ge=0)
	estimated_packaged_length_cm: float | None = Field(default=None, ge=0)
	estimated_packaged_width_cm: float | None = Field(default=None, ge=0)
	estimated_packaged_height_cm: float | None = Field(default=None, ge=0)
	fragile_or_special_handling: bool | None = None
	confidence: Literal["high", "medium", "low"]
	estimate_explanation: str
	manual_review_recommended: bool


class GeminiBatchEstimate(GeminiEnrichment):
	"""One batch estimate with a required stable product identifier."""

	product_id: str


class GeminiBatchResponse(BaseModel):
	"""Structured Gemini response for a batch request."""

	estimates: list[GeminiBatchEstimate]


ModelCaller = Callable[[ProductContext], GeminiEnrichment]
BatchModelCaller = Callable[[list[ProductContext]], GeminiBatchResponse]


@dataclass(frozen=True)
class ErrorClassification:
	"""Normalized provider error information used by the retry policy."""

	kind: str
	retryable: bool
	retry_after_seconds: float | None
	reason: str


@dataclass(frozen=True)
class EnrichmentOutcome:
	"""Result of one product attempt, including its resumable status."""

	result: GeminiEnrichment | None
	status: Literal["completed", "pending", "failed"]
	error: str | None = None
	error_kind: str | None = None
	cache_hit: bool = False


def _response_code(error: Exception) -> int | None:
	"""Read an HTTP-like status code from SDK exceptions when available."""
	for value in (getattr(error, "code", None), getattr(error, "status_code", None)):
		if isinstance(value, int):
			return value
	response = getattr(error, "response", None)
	code = getattr(response, "status_code", None)
	return code if isinstance(code, int) else None


def _retry_after_seconds(error: Exception) -> float | None:
	"""Extract a bounded server retry delay from headers or SDK error text."""
	response = getattr(error, "response", None)
	headers = getattr(response, "headers", None) or getattr(error, "headers", None)
	if headers:
		value = headers.get("Retry-After") or headers.get("retry-after")
		try:
			if value is not None:
				return max(0.0, min(float(value), DEFAULT_MAX_BACKOFF_SECONDS))
		except (TypeError, ValueError):
			pass
	text = str(error)
	match = re.search(r"(?:retry\s+in\s+|retryDelay[^0-9]*)(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
	if match:
		return min(float(match.group(1)), DEFAULT_MAX_BACKOFF_SECONDS)
	return None


def classify_api_error(error: Exception) -> ErrorClassification:
	"""Distinguish temporary limits, daily quota exhaustion, server errors, and failures."""
	text = str(error)
	lower_text = text.lower()
	code = _response_code(error)
	is_rate_limit = code == 429 or "resource_exhausted" in lower_text or "rate limit" in lower_text
	if is_rate_limit:
		is_daily = bool(re.search(r"daily|per[_ -]?day|day[_ -]?quota|quota.*day", lower_text))
		kind = "daily_quota_exhausted" if is_daily else "temporary_rate_limit"
		return ErrorClassification(
			kind=kind,
			retryable=not is_daily,
			retry_after_seconds=_retry_after_seconds(error),
			reason=text,
		)
	if code is not None and 500 <= code <= 599:
		return ErrorClassification("transient_server_error", True, _retry_after_seconds(error), text)
	if isinstance(error, ValidationError):
		return ErrorClassification("invalid_response", False, None, text)
	return ErrorClassification("request_failed", False, None, text)


def load_processed_products(csv_path: str | Path) -> pd.DataFrame:
	"""Load processed product occurrences without losing order traceability."""
	products = pd.read_csv(csv_path)
	if "product_name" not in products.columns:
		raise ValueError("processed products CSV is missing product_name")
	return products.dropna(subset=["product_name"]).reset_index(drop=True)


def select_product_sample(products: pd.DataFrame, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
	"""Select a deterministic diverse sample of unique product names."""
	if limit < 1 or limit > MAX_SAMPLE_LIMIT:
		raise ValueError(f"limit must be between 1 and {MAX_SAMPLE_LIMIT}")
	products = products.drop_duplicates("product_name").reset_index(drop=True)
	if len(products) <= limit:
		return products.copy()

	selected_indices: list[int] = []
	for column in ("measurement_type", "extraction_confidence", "extraction_method"):
		if column not in products.columns:
			continue
		for index in products.groupby(column, dropna=False, sort=True).head(1).index:
			if index not in selected_indices:
				selected_indices.append(index)
				if len(selected_indices) == limit:
					return products.loc[selected_indices].reset_index(drop=True)
	for index in products.index:
		if index not in selected_indices:
			selected_indices.append(index)
			if len(selected_indices) == limit:
				break
	return products.loc[selected_indices].reset_index(drop=True)


def product_identifier(context: ProductContext) -> str:
	"""Return the legacy-compatible stable identifier for one unique product context."""
	payload = context.model_dump(mode="json", exclude_none=True)
	return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def context_from_row(row: pd.Series) -> ProductContext:
	"""Build model input from existing regex results without overwriting them."""
	available = ProductContext.model_fields
	values = {key: row[key] for key in available if key in row.index and pd.notna(row[key])}
	return ProductContext.model_validate(values)


def build_prompt(context: ProductContext) -> str:
	"""Create a cautious enrichment prompt with explicit uncertainty rules."""
	evidence = {
		"extracted_weight_kg": context.extracted_weight_kg,
		"extracted_volume_litres": context.extracted_volume_litres,
		"extracted_weight_min_kg": context.extracted_weight_min_kg,
		"extracted_weight_max_kg": context.extracted_weight_max_kg,
		"measurement_type": context.measurement_type,
		"matched_text": context.matched_text,
		"measurement_sources": context.measurement_sources,
		"extraction_warning": context.extraction_warning,
	}
	return (
		"Estimate grocery product packaging for courier transport. Return only the "
		"requested structured fields. Do not invent precise values when the product "
		"cannot be reasonably identified; use null and recommend manual review. "
		"Distinguish packaging dimensions from liquid volume and net content weight. "
		"Do not treat regex net content as verified shipping weight. A number such "
		"as 13-18kg on diapers describes the intended user, not the product. "
		f"\nOriginal product name: {context.product_name}\n"
		f"Existing regex evidence: {json.dumps(evidence, ensure_ascii=False)}"
	)


def build_batch_prompt(contexts: list[ProductContext]) -> str:
	"""Create one structured batch prompt with identifiers and regex evidence."""
	items = []
	for context in contexts:
		items.append({
			"product_id": product_identifier(context),
			"product_name": context.product_name,
			"extracted_weight_kg": context.extracted_weight_kg,
			"extracted_volume_litres": context.extracted_volume_litres,
			"extracted_weight_min_kg": context.extracted_weight_min_kg,
			"extracted_weight_max_kg": context.extracted_weight_max_kg,
			"measurement_type": context.measurement_type,
			"matched_text": context.matched_text,
			"measurement_sources": context.measurement_sources,
			"extraction_warning": context.extraction_warning,
		})
	return (
		"Estimate grocery product packaging for every item in this batch. Return one "
		"estimate per product_id in a JSON estimates array. Preserve product_id exactly; "
		"never identify products by response position. Do not invent precise values when "
		"a product cannot be reasonably identified; use null and recommend manual review. "
		"Distinguish packaging dimensions from liquid volume and net content weight. "
		"A number such as 13-18kg on diapers describes the intended user, not the product. "
		f"\nBatch products: {json.dumps(items, ensure_ascii=False)}"
	)


def _parse_model_response(response: Any) -> GeminiEnrichment:
	"""Validate a Gemini response whether the SDK exposes parsed or text output."""
	parsed = getattr(response, "parsed", None)
	if parsed is not None:
		return parsed if isinstance(parsed, GeminiEnrichment) else GeminiEnrichment.model_validate(parsed)
	text = getattr(response, "text", None)
	if not text:
		raise ValueError("Gemini response did not contain structured content")
	return GeminiEnrichment.model_validate_json(text)


def _parse_batch_model_response(response: Any) -> GeminiBatchResponse:
	"""Validate a structured batch response from the Gemini SDK or a mock."""
	parsed = getattr(response, "parsed", None)
	if parsed is not None:
		return GeminiBatchResponse.model_validate(parsed)
	if isinstance(response, (GeminiBatchResponse, dict)):
		return GeminiBatchResponse.model_validate(response)
	text = getattr(response, "text", None)
	if not text:
		raise ValueError("Gemini batch response did not contain structured content")
	return GeminiBatchResponse.model_validate_json(text)


def is_cacheable_result(result: GeminiEnrichment) -> bool:
	"""Reject placeholder narrative responses while allowing legitimate null estimates."""
	placeholder_values = {"", "unknown", "n/a", "na", "null", "none", "not available"}
	required_text = (
		result.original_product_name,
		result.estimate_explanation,
	)
	return all(value.strip().lower() not in placeholder_values for value in required_text)


def call_gemini_model(
	context: ProductContext,
	client: genai.Client,
	model_name: str,
) -> GeminiEnrichment:
	"""Call Gemini using structured JSON output; replace this boundary for another provider."""
	response = client.models.generate_content(
		model=model_name,
		contents=build_prompt(context),
		config=types.GenerateContentConfig(
			response_mime_type="application/json",
			response_schema=GeminiEnrichment,
		),
	)
	return _parse_model_response(response)


def call_gemini_batch_model(
	contexts: list[ProductContext],
	client: genai.Client,
	model_name: str,
) -> GeminiBatchResponse:
	"""Call Gemini once for a batch using structured JSON output."""
	response = client.models.generate_content(
		model=model_name,
		contents=build_batch_prompt(contexts),
		config=types.GenerateContentConfig(
			response_mime_type="application/json",
			response_schema=GeminiBatchResponse,
		),
	)
	return _parse_batch_model_response(response)


class GeminiEnricher:
	"""Retrying, cache-aware enrichment service with an injectable model caller."""

	def __init__(
		self,
		model_caller: ModelCaller,
		cache_path: str | Path = DEFAULT_CACHE_PATH,
		max_retries: int = DEFAULT_MAX_RETRIES,
		backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
		max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
		jitter_seconds: float = DEFAULT_JITTER_SECONDS,
		request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
	) -> None:
		self.model_caller = model_caller
		self.cache_path = Path(cache_path)
		self.max_retries = max_retries
		self.backoff_seconds = backoff_seconds
		self.max_backoff_seconds = max_backoff_seconds
		self.jitter_seconds = jitter_seconds
		self.request_interval_seconds = request_interval_seconds
		self._last_request_at: float | None = None
		self.cache = self._load_cache()

	def _load_cache(self) -> dict[str, dict[str, Any]]:
		if not self.cache_path.exists():
			return {}
		try:
			return json.loads(self.cache_path.read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError):
			return {}

	def _save_cache(self) -> None:
		self.cache_path.parent.mkdir(parents=True, exist_ok=True)
		self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

	@staticmethod
	def cache_key(context: ProductContext) -> str:
		return product_identifier(context)

	def _cache_result(self, context: ProductContext, result: GeminiEnrichment) -> None:
		"""Persist one validated success without replacing an existing success."""
		key = self.cache_key(context)
		if self._cached_result(key) is not None:
			return
		if result.product_id is None:
			result = result.model_copy(update={"product_id": key})
		self.cache[key] = result.model_dump(mode="json")
		self._save_cache()

	def _cached_result(self, key: str) -> GeminiEnrichment | None:
		"""Return only a valid, complete cached enrichment."""
		cached = self.cache.get(key)
		if not isinstance(cached, dict):
			return None
		try:
			result = GeminiEnrichment.model_validate(cached)
			return result if result.product_id else result.model_copy(update={"product_id": key})
		except ValidationError:
			return None

	def _wait_for_request_slot(self) -> None:
		"""Throttle actual provider calls; cache hits never wait."""
		if self._last_request_at is not None:
			remaining = self.request_interval_seconds - (time.monotonic() - self._last_request_at)
			if remaining > 0:
				time.sleep(remaining)
		self._last_request_at = time.monotonic()

	def _retry_delay(self, attempt: int, server_delay: float | None) -> float:
		"""Use a server delay when present, otherwise bounded exponential backoff plus jitter."""
		base = server_delay if server_delay is not None else self.backoff_seconds * (2**attempt)
		base = min(base, self.max_backoff_seconds)
		return min(self.max_backoff_seconds, base + random.uniform(0, self.jitter_seconds))

	def enrich_with_status(self, context: ProductContext) -> EnrichmentOutcome:
		"""Enrich one product with resumable status and no cache writes for failures."""
		key = self.cache_key(context)
		cached = self._cached_result(key)
		if cached is not None:
			return EnrichmentOutcome(cached, "completed", cache_hit=True)

		last_error: ErrorClassification | None = None
		for attempt in range(self.max_retries + 1):
			try:
				self._wait_for_request_slot()
				result = self.model_caller(context)
				result = GeminiEnrichment.model_validate(result)
				if not is_cacheable_result(result):
					raise ValueError("incomplete or placeholder enrichment response")
				self._cache_result(context, result)
				return EnrichmentOutcome(result, "completed")
			except Exception as error:  # Isolate SDK and validation failures per product.
				last_error = classify_api_error(error)
				if not last_error.retryable:
					break
				if attempt < self.max_retries:
					time.sleep(self._retry_delay(attempt, last_error.retry_after_seconds))
		status = "pending" if last_error and last_error.retryable else "failed"
		return EnrichmentOutcome(
			None,
			status,
			last_error.reason if last_error else "unknown enrichment failure",
			last_error.kind if last_error else "unknown_error",
		)

	def _call_with_retries(self, call: Callable[[], Any]) -> tuple[Any | None, ErrorClassification | None]:
		"""Run one provider call under the existing throttle and bounded retry policy."""
		last_error: ErrorClassification | None = None
		for attempt in range(self.max_retries + 1):
			try:
				self._wait_for_request_slot()
				return call(), None
			except Exception as error:
				last_error = classify_api_error(error)
				if not last_error.retryable:
					break
				if attempt < self.max_retries:
					time.sleep(self._retry_delay(attempt, last_error.retry_after_seconds))
		return None, last_error

	def enrich_batch(
		self,
		contexts: list[ProductContext],
		batch_caller: BatchModelCaller,
		batch_size: int = DEFAULT_BATCH_SIZE,
	) -> dict[str, EnrichmentOutcome]:
		"""Enrich uncached contexts in batches, shrinking invalid batches and caching per item."""
		if batch_size < 1:
			raise ValueError("batch_size must be positive")
		outcomes: dict[str, EnrichmentOutcome] = {}
		pending = []
		for context in contexts:
			key = self.cache_key(context)
			cached = self._cached_result(key)
			if cached is not None:
				outcomes[key] = EnrichmentOutcome(cached, "completed", cache_hit=True)
			else:
				pending.append(context)

		def process(batch: list[ProductContext]) -> None:
			if not batch:
				return
			response, error = self._call_with_retries(lambda: batch_caller(batch))
			if error is not None:
				if len(batch) > 1 and error.kind not in {"daily_quota_exhausted", "temporary_rate_limit"}:
					midpoint = max(1, len(batch) // 2)
					process(batch[:midpoint])
					process(batch[midpoint:])
					return
				status = "pending" if error.retryable else "failed"
				for context in batch:
					outcomes[self.cache_key(context)] = EnrichmentOutcome(None, status, error.reason, error.kind)
				return
			try:
				validated = GeminiBatchResponse.model_validate(response)
				requested = {self.cache_key(context): context for context in batch}
				seen: set[str] = set()
				valid_results: dict[str, GeminiBatchEstimate] = {}
				for result in validated.estimates:
					if result.product_id not in requested or result.product_id in seen:
						raise ValueError("batch contained an unexpected or duplicate product_id")
					seen.add(result.product_id)
					if not is_cacheable_result(result):
						continue
					valid_results[result.product_id] = result
				for key, result in valid_results.items():
					self._cache_result(requested[key], result)
					outcomes[key] = EnrichmentOutcome(result, "completed")
				unresolved = [context for key, context in requested.items() if key not in valid_results]
			except Exception:
				unresolved = batch
			if not unresolved:
				return
			if len(unresolved) > 1:
				midpoint = max(1, len(unresolved) // 2)
				process(unresolved[:midpoint])
				process(unresolved[midpoint:])
			else:
				outcome = self.enrich_with_status(unresolved[0])
				outcomes[self.cache_key(unresolved[0])] = outcome

		for start in range(0, len(pending), batch_size):
			process(pending[start:start + batch_size])
		return outcomes

	def enrich(self, context: ProductContext) -> tuple[GeminiEnrichment | None, str | None, bool]:
		"""Backward-compatible tuple wrapper around :meth:`enrich_with_status`."""
		outcome = self.enrich_with_status(context)
		return outcome.result, outcome.error, outcome.cache_hit


def get_api_key() -> str:
	"""Load the API key from .env or the process environment."""
	load_dotenv()
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is not set")
	return api_key


def create_gemini_client(api_key: str) -> genai.Client:
	"""Create a client with SDK retries disabled so this module owns retry policy."""
	return genai.Client(
		api_key=api_key,
		http_options=types.HttpOptions(
			retry_options=types.HttpRetryOptions(attempts=1),
		),
	)


def summarize_results(results: pd.DataFrame) -> str:
	"""Return final counts and grouped error reasons for a sample run."""
	counts = results["ai_status"].value_counts().to_dict() if len(results) else {}
	lines = [
		f"Successful products: {counts.get('completed', 0)}",
		f"Cached products: {int(results.get('ai_cache_hit', pd.Series(dtype=bool)).sum())}",
		f"Pending products: {counts.get('pending', 0)}",
		f"Failed products: {counts.get('failed', 0)}",
		"Error reasons:",
	]
	if "ai_error_kind" in results.columns:
		errors = results.loc[results["ai_error_kind"].notna(), "ai_error_kind"].value_counts()
		lines.extend(f"- {kind}: {count}" for kind, count in errors.items())
	if len(lines) == 5:
		lines.append("- none")
	return "\n".join(lines) + "\n"


def dry_run_summary(
	products: pd.DataFrame,
	limit: int = DEFAULT_LIMIT,
	batch_size: int = DEFAULT_BATCH_SIZE,
	cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> str:
	"""Report work and lower-bound request count without creating an API client."""
	sample = select_product_sample(products, limit)
	cache = GeminiEnricher(lambda _context: None, cache_path=cache_path)
	contexts = [context_from_row(row) for _, row in sample.iterrows()]
	cached = sum(cache._cached_result(cache.cache_key(context)) is not None for context in contexts)
	missing = len(contexts) - cached
	requests = (missing + batch_size - 1) // batch_size if batch_size > 0 else 0
	return (
		f"Unique products selected: {len(contexts)}\n"
		f"Already cached: {cached}\n"
		f"Products requiring enrichment: {missing}\n"
		f"Expected initial batch requests: {requests}\n"
		"Additional requests may be needed for partial or invalid batch responses.\n"
	)


def enrich_sample(
	products: pd.DataFrame,
	limit: int = DEFAULT_LIMIT,
	model_name: str = DEFAULT_MODEL,
	cache_path: str | Path = DEFAULT_CACHE_PATH,
	model_caller: ModelCaller | None = None,
	request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
	output_path: str | Path | None = None,
	resume: bool = True,
	batch_size: int = DEFAULT_BATCH_SIZE,
	batch_caller: BatchModelCaller | None = None,
) -> pd.DataFrame:
	"""Resume unique-product enrichment in batches and preserve all occurrences in output."""
	sample = select_product_sample(products, limit)
	if model_caller is None and batch_caller is None:
		client = create_gemini_client(get_api_key())
		model_caller = lambda context: call_gemini_model(context, client, model_name)
		batch_caller = lambda contexts: call_gemini_batch_model(contexts, client, model_name)
	elif model_caller is None:
		def unavailable_single_caller(_context: ProductContext) -> GeminiEnrichment:
			raise RuntimeError("single-product fallback is not configured")
		model_caller = unavailable_single_caller
	enricher = GeminiEnricher(
		model_caller,
		cache_path=cache_path,
		request_interval_seconds=request_interval_seconds,
	)
	if not resume:
		enricher.cache = {}
	contexts = [context_from_row(row) for _, row in sample.iterrows()]
	contexts_by_name = {context.product_name: context for context in contexts}
	if batch_caller is not None:
		outcomes = enricher.enrich_batch(contexts, batch_caller, batch_size)
	else:
		outcomes = {
			enricher.cache_key(context): enricher.enrich_with_status(context)
			for context in contexts
		}
	rows: list[dict[str, Any]] = []
	selected_names = set(contexts_by_name)
	for _, row in products[products["product_name"].isin(selected_names)].iterrows():
		context = contexts_by_name[row["product_name"]]
		outcome = outcomes[enricher.cache_key(context)]
		output = row.to_dict()
		output.update({
			"ai_status": outcome.status,
			"ai_cache_hit": outcome.cache_hit,
			"ai_error": outcome.error,
			"ai_error_kind": outcome.error_kind,
		})
		if outcome.result:
			output.update({f"ai_{key}": value for key, value in outcome.result.model_dump().items()})
		else:
			output.update({f"ai_{key}": None for key in GeminiEnrichment.model_fields})
		rows.append(output)
		if output_path is not None:
			path = Path(output_path)
			path.parent.mkdir(parents=True, exist_ok=True)
			pd.DataFrame(rows).to_csv(path, index=False)
	return pd.DataFrame(rows)


def main() -> None:
	"""Run a capped Gemini enrichment sample from the command line."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="data/processed_products.csv")
	parser.add_argument("--output", default="data/ai_enriched_sample.csv")
	parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--dry-run", action="store_true", help="Show work and request estimate without API calls")
	parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
	parser.add_argument(
		"--resume",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Reuse valid successful cache entries (default: enabled)",
	)
	args = parser.parse_args()
	products = load_processed_products(args.input)
	if args.batch_size < 1:
		parser.error("--batch-size must be positive")
	if args.dry_run:
		print(dry_run_summary(products, args.limit, args.batch_size, args.cache), end="")
		return
	result = enrich_sample(
		products,
		args.limit,
		args.model,
		args.cache,
		request_interval_seconds=args.request_interval,
		output_path=args.output,
		resume=args.resume,
		batch_size=args.batch_size,
	)
	print(f"Processed {len(result)} products; output: {args.output}")
	print(summarize_results(result), end="")


if __name__ == "__main__":
	main()
