import json

import pandas as pd
import pytest

from src.ai_enrichment import (
    GeminiBatchResponse,
    GeminiBatchEstimate,
    GeminiEnrichment,
    GeminiEnricher,
    ProductContext,
    build_prompt,
    dry_run_summary,
    enrich_sample,
    product_identifier,
    select_product_sample,
)


def valid_result(name: str, product_id: str | None = None) -> GeminiEnrichment:
    return GeminiEnrichment(
        original_product_name=name,
        product_id=product_id,
        normalized_product_description="A grocery item",
        product_category="food",
        estimated_transport_weight_kg=None,
        estimated_packaged_length_cm=None,
        estimated_packaged_width_cm=None,
        estimated_packaged_height_cm=None,
        fragile_or_special_handling=False,
        confidence="low",
        estimate_explanation="The product cannot be reliably identified from the name.",
        manual_review_recommended=True,
    )


def test_prompt_contains_regex_evidence_and_uncertainty_rules():
    prompt = build_prompt(ProductContext(
        product_name="Diapers 13-18kg",
        extraction_warning="possible consumer-size range",
        matched_text="13-18kg",
    ))
    assert "13-18kg" in prompt
    assert "intended user" in prompt
    assert "verified shipping weight" in prompt


def test_sample_is_capped_and_diverse():
    products = pd.DataFrame({
        "product_name": [f"Product {index}" for index in range(30)],
        "measurement_type": ["single", "range", "ambiguous"] * 10,
        "extraction_confidence": ["high", "medium", "low"] * 10,
    })
    sample = select_product_sample(products, 20)
    assert len(sample) == 20
    assert set(sample["measurement_type"]) == {"single", "range", "ambiguous"}
    assert len(select_product_sample(products, 21)) == 21
    with pytest.raises(ValueError):
        select_product_sample(products, 100001)


def test_mocked_enrichment_validates_and_caches_success(tmp_path):
    calls = []

    def caller(context):
        calls.append(context.product_name)
        return valid_result(context.product_name)

    products = pd.DataFrame([{
        "product_name": "Rice 1kg",
        "extracted_weight_kg": 1.0,
        "extracted_volume_litres": None,
        "matched_text": "1kg",
    }])
    cache_path = tmp_path / "cache.json"
    first = enrich_sample(products, limit=1, cache_path=cache_path, model_caller=caller)
    second = enrich_sample(products, limit=1, cache_path=cache_path, model_caller=caller)
    assert first.loc[0, "ai_status"] == "completed"
    assert first.loc[0, "ai_normalized_product_description"] == "A grocery item"
    assert bool(second.loc[0, "ai_cache_hit"]) is True
    assert calls == ["Rice 1kg"]
    assert json.loads(cache_path.read_text())


def test_invalid_response_is_failed_without_retry_or_cache(tmp_path, monkeypatch):
    calls = []

    def caller(_context):
        calls.append(1)
        raise ValueError("invalid structured response")

    monkeypatch.setattr("src.ai_enrichment.time.sleep", lambda _seconds: None)
    context = ProductContext(product_name="Unknown product")
    enricher = GeminiEnricher(caller, cache_path=tmp_path / "cache.json", max_retries=2)
    result, error, cache_hit = enricher.enrich(context)
    assert result is None
    assert "invalid structured response" in error
    assert cache_hit is False
    assert len(calls) == 1
    assert not (tmp_path / "cache.json").exists()


def test_temporary_rate_limit_retries_with_server_delay(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def caller(context):
        calls.append(context.product_name)
        if len(calls) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED; retry in 4s")
        return valid_result(context.product_name)

    monkeypatch.setattr("src.ai_enrichment.time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr("src.ai_enrichment.random.uniform", lambda _start, _end: 0)
    context = ProductContext(product_name="Rice 1kg")
    enricher = GeminiEnricher(
        caller,
        cache_path=tmp_path / "cache.json",
        request_interval_seconds=0,
        max_retries=2,
    )
    outcome = enricher.enrich_with_status(context)
    assert outcome.status == "completed"
    assert len(calls) == 2
    assert sleeps == [4.0]


def test_daily_quota_is_failed_without_retry(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def caller(_context):
        calls.append(1)
        raise RuntimeError("429 RESOURCE_EXHAUSTED daily quota exceeded")

    monkeypatch.setattr("src.ai_enrichment.time.sleep", lambda seconds: sleeps.append(seconds))
    outcome = GeminiEnricher(
        caller,
        cache_path=tmp_path / "cache.json",
        request_interval_seconds=0,
    ).enrich_with_status(ProductContext(product_name="Rice 1kg"))
    assert outcome.status == "failed"
    assert outcome.error_kind == "daily_quota_exhausted"
    assert len(calls) == 1
    assert sleeps == []


def test_transient_5xx_retries_with_bounded_backoff(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    class ServerError(RuntimeError):
        code = 503

    def caller(context):
        calls.append(context.product_name)
        if len(calls) == 1:
            raise ServerError("service unavailable")
        return valid_result(context.product_name)

    monkeypatch.setattr("src.ai_enrichment.time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr("src.ai_enrichment.random.uniform", lambda _start, _end: 0)
    outcome = GeminiEnricher(
        caller, cache_path=tmp_path / "cache.json", request_interval_seconds=0,
    ).enrich_with_status(ProductContext(product_name="Rice 1kg"))
    assert outcome.status == "completed"
    assert calls == ["Rice 1kg", "Rice 1kg"]
    assert sleeps == [2.0]


def test_placeholder_response_is_not_cached(tmp_path):
    def caller(_context):
        return GeminiEnrichment(
            original_product_name="unknown",
            confidence="low",
            estimate_explanation="unknown",
            manual_review_recommended=True,
        )

    outcome = GeminiEnricher(
        caller, cache_path=tmp_path / "cache.json", request_interval_seconds=0,
    ).enrich_with_status(ProductContext(product_name="Unknown item"))
    assert outcome.status == "failed"
    assert outcome.error_kind == "request_failed"
    assert not (tmp_path / "cache.json").exists()


def test_resume_uses_success_cache_and_preserves_pending_product(tmp_path, monkeypatch):
    products = pd.DataFrame([
        {"product_name": "Rice 1kg", "extracted_weight_kg": 1.0},
        {"product_name": "Unknown item"},
    ])
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "sample.csv"
    first_calls = []

    def first_caller(context):
        first_calls.append(context.product_name)
        if context.product_name == "Unknown item":
            raise RuntimeError("429 RESOURCE_EXHAUSTED retry in 20s")
        return valid_result(context.product_name)

    monkeypatch.setattr("src.ai_enrichment.time.sleep", lambda _seconds: None)
    first = enrich_sample(
        products, limit=2, cache_path=cache_path, output_path=output_path,
        model_caller=first_caller, request_interval_seconds=0,
    )
    assert set(first["ai_status"]) == {"completed", "pending"}
    assert first_calls == ["Rice 1kg", "Unknown item", "Unknown item", "Unknown item", "Unknown item"]

    second_calls = []

    def second_caller(context):
        second_calls.append(context.product_name)
        return valid_result(context.product_name)

    second = enrich_sample(
        products, limit=2, cache_path=cache_path, output_path=output_path,
        model_caller=second_caller, request_interval_seconds=0,
    )
    assert second_calls == ["Unknown item"]
    assert bool(second.loc[0, "ai_cache_hit"]) is True
    assert set(second["ai_status"]) == {"completed"}
    assert len(pd.read_csv(output_path)) == 2


def test_batch_matches_out_of_order_results_by_product_id(tmp_path):
    products = pd.DataFrame([
        {"product_name": "Alpha", "quantity": 1},
        {"product_name": "Beta", "quantity": 2},
    ])
    calls = []

    def batch_caller(contexts):
        calls.append([context.product_name for context in contexts])
        results = [
            GeminiBatchEstimate.model_validate(valid_result(context.product_name, product_identifier(context)).model_dump())
            for context in reversed(contexts)
        ]
        return GeminiBatchResponse(estimates=results)

    result = enrich_sample(
        products,
        limit=2,
        batch_size=2,
        cache_path=tmp_path / "cache.json",
        batch_caller=batch_caller,
        output_path=tmp_path / "out.csv",
        request_interval_seconds=0,
    )
    assert calls == [["Alpha", "Beta"]]
    assert list(result["ai_original_product_name"]) == ["Alpha", "Beta"]


def test_partial_batch_retries_only_missing_product(tmp_path, monkeypatch):
    products = pd.DataFrame([{"product_name": "Alpha", "quantity": 1}, {"product_name": "Beta", "quantity": 1}])
    batch_calls = []
    single_calls = []

    def batch_caller(contexts):
        batch_calls.append([context.product_name for context in contexts])
        context = contexts[0]
        return GeminiBatchResponse(estimates=[GeminiBatchEstimate.model_validate(valid_result(context.product_name, product_identifier(context)).model_dump())])

    def single_caller(context):
        single_calls.append(context.product_name)
        return valid_result(context.product_name)

    monkeypatch.setattr("src.ai_enrichment.time.sleep", lambda _seconds: None)
    result = enrich_sample(
        products,
        limit=2,
        batch_size=2,
        cache_path=tmp_path / "cache.json",
        batch_caller=batch_caller,
        model_caller=single_caller,
        request_interval_seconds=0,
    )
    assert set(result["ai_status"]) == {"completed"}
    assert "Beta" in single_calls or any("Beta" in call for call in batch_calls[1:])


def test_invalid_batch_estimate_falls_back_to_single_product(tmp_path):
    products = pd.DataFrame([{"product_name": "Alpha", "quantity": 1}])

    def batch_caller(contexts):
        invalid = valid_result(contexts[0].product_name, product_identifier(contexts[0])).model_dump()
        invalid["estimated_transport_weight_kg"] = -1
        return {"estimates": [invalid]}

    result = enrich_sample(
        products,
        limit=1,
        batch_size=1,
        cache_path=tmp_path / "cache.json",
        batch_caller=batch_caller,
        model_caller=lambda context: valid_result(context.product_name),
        request_interval_seconds=0,
    )
    assert result.loc[0, "ai_status"] == "completed"


def test_duplicate_or_unexpected_batch_identifier_is_not_cached(tmp_path):
    products = pd.DataFrame([{"product_name": "Alpha", "quantity": 1}, {"product_name": "Beta", "quantity": 1}])

    def batch_caller(contexts):
        first = GeminiBatchEstimate.model_validate(valid_result(contexts[0].product_name, product_identifier(contexts[0])).model_dump())
        return GeminiBatchResponse(estimates=[first, first] if len(contexts) > 1 else [first])

    result = enrich_sample(
        products,
        limit=2,
        batch_size=2,
        cache_path=tmp_path / "cache.json",
        batch_caller=batch_caller,
        model_caller=lambda _context: (_ for _ in ()).throw(RuntimeError("no single fallback")),
        request_interval_seconds=0,
    )
    assert set(result["ai_status"]) == {"completed"}
    assert len(json.loads((tmp_path / "cache.json").read_text())) == 2


def test_batch_daily_quota_fails_all_products_without_cache(tmp_path):
    products = pd.DataFrame([{"product_name": "Alpha", "quantity": 1}, {"product_name": "Beta", "quantity": 1}])

    def batch_caller(_contexts):
        raise RuntimeError("429 RESOURCE_EXHAUSTED daily quota exceeded")

    result = enrich_sample(
        products,
        limit=2,
        batch_size=2,
        cache_path=tmp_path / "cache.json",
        batch_caller=batch_caller,
        model_caller=batch_caller,
        request_interval_seconds=0,
    )
    assert set(result["ai_status"]) == {"failed"}
    assert not (tmp_path / "cache.json").exists()


def test_dry_run_reports_cached_products_and_batch_requests(tmp_path):
    products = pd.DataFrame([{"product_name": f"Product {index}", "quantity": 1} for index in range(5)])
    result = dry_run_summary(products, limit=5, batch_size=2, cache_path=tmp_path / "cache.json")
    assert "Unique products selected: 5" in result
    assert "Expected initial batch requests: 3" in result