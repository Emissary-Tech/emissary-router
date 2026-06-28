from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import re
from typing import Any

from emissary_router.catalog import CATALOG, TokenPricing
from emissary_router.config import AppConfig


DEFAULT_EXPECTED_OUTPUT_TOKENS = 1024
CCH_ATTRIBUTION_LINE_RE = re.compile(r"(?m)^.*\bcch=[^\s<>\"]+.*(?:\n|$)")


@dataclass(frozen=True)
class RequestCostFeatures:
    session_id: str | None
    prefix_hash: str
    estimated_input_tokens: int
    estimated_cacheable_prefix_tokens: int
    estimated_fresh_input_tokens: int
    expected_output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CachePrediction:
    warm: bool
    cached_tokens: int
    confidence: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EstimatedCost:
    model_name: str
    total_usd: float
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    expected_output_tokens: int
    cache_prediction: CachePrediction

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_usd"] = round(self.total_usd, 8)
        return data


def extract_request_cost_features(
    body: dict[str, Any],
    headers: dict[str, str],
    expected_output_tokens: int | None = None,
) -> RequestCostFeatures:
    session_id = _header(headers, "x-claude-code-session-id")
    prefix_material = _stable_json(
        {
            "system": _normalize_system(body.get("system")),
            "tools": body.get("tools") or [],
        }
    )
    input_material = _stable_json(
        {
            "system": _normalize_system(body.get("system")),
            "tools": body.get("tools") or [],
            "messages": body.get("messages") or [],
        }
    )
    cacheable_prefix_tokens = _estimate_tokens(prefix_material)
    input_tokens = max(_estimate_tokens(input_material), cacheable_prefix_tokens)
    fresh_input_tokens = max(input_tokens - cacheable_prefix_tokens, 0)
    return RequestCostFeatures(
        session_id=session_id,
        prefix_hash=hashlib.sha256(prefix_material.encode("utf-8")).hexdigest(),
        estimated_input_tokens=input_tokens,
        estimated_cacheable_prefix_tokens=cacheable_prefix_tokens,
        estimated_fresh_input_tokens=fresh_input_tokens,
        expected_output_tokens=_expected_output_tokens(body, expected_output_tokens),
    )


def estimate_cost(
    config: AppConfig,
    model_name: str,
    features: RequestCostFeatures,
    cache_ledger,
) -> EstimatedCost:
    model = config.resolve_model(model_name)
    prediction = cache_ledger.predict(model, features)
    price = CATALOG[model_name].pricing

    if prediction.warm:
        # Credit the cache the provider actually reported (system + tools + most of the
        # conversation history), bounded only by this request's total input. Clamping to
        # estimated_cacheable_prefix_tokens here would re-price the cached history at full
        # input rate and defeat the whole point of cache-aware routing.
        cached_tokens = min(prediction.cached_tokens, features.estimated_input_tokens)
        input_tokens = max(features.estimated_input_tokens - cached_tokens, 0)
        cache_read_tokens = cached_tokens
        cache_write_tokens = 0
    else:
        input_tokens = features.estimated_fresh_input_tokens
        cache_read_tokens = 0
        cache_write_tokens = features.estimated_cacheable_prefix_tokens

    return EstimatedCost(
        model_name=model_name,
        total_usd=_cost_usd(
            price=price,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            output_tokens=features.expected_output_tokens,
        ),
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        expected_output_tokens=features.expected_output_tokens,
        cache_prediction=prediction,
    )


def is_cheaper(candidate: EstimatedCost, baseline: EstimatedCost) -> bool:
    return candidate.total_usd < baseline.total_usd


def _cost_usd(
    price: TokenPricing,
    input_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
) -> float:
    return (
        input_tokens * price.input
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write_5m
        + output_tokens * price.output
    ) / 1_000_000


def _header(headers: dict[str, str], name: str) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    return lowered.get(name)


def _normalize_system(system: Any) -> Any:
    if isinstance(system, str):
        return CCH_ATTRIBUTION_LINE_RE.sub("", system)
    if isinstance(system, list):
        normalized = []
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                updated = dict(block)
                updated["text"] = CCH_ATTRIBUTION_LINE_RE.sub("", updated["text"])
                normalized.append(updated)
            else:
                normalized.append(block)
        return normalized
    return system


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _expected_output_tokens(body: dict[str, Any], rolling: int | None = None) -> int:
    """Expected output for cost estimation.

    Bounded above by the request's ``max_tokens`` (output can't exceed it), but driven
    by ``rolling`` — a rolling estimate of recently observed outputs — rather than a
    hard 1024 cap. The old cap understated output and skewed comparison between models
    whose output prices differ widely; ``max_tokens`` alone overstates it (real outputs
    are far below the 32k ceiling Claude Code sends), so we take the lesser of the two.
    """
    try:
        max_tokens = int(body.get("max_tokens") or DEFAULT_EXPECTED_OUTPUT_TOKENS)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_EXPECTED_OUTPUT_TOKENS
    if max_tokens <= 0:
        max_tokens = DEFAULT_EXPECTED_OUTPUT_TOKENS
    estimate = rolling if rolling and rolling > 0 else DEFAULT_EXPECTED_OUTPUT_TOKENS
    return max(1, min(max_tokens, estimate))
