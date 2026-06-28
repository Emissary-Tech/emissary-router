from __future__ import annotations

from dataclasses import dataclass
import time

from emissary_router.caching.state import CacheConfidence
from emissary_router.caching.usage import Usage
from emissary_router.config import ResolvedModel
from emissary_router.routing.cache_cost import (
    DEFAULT_EXPECTED_OUTPUT_TOKENS,
    CachePrediction,
    RequestCostFeatures,
)


DEFAULT_CACHE_TTL_SECONDS = 300
# Smoothing for the rolling output-length estimate. Small enough to track a session's
# style, large enough not to swing on a single short/long turn.
OUTPUT_EMA_ALPHA = 0.2


@dataclass(frozen=True)
class CacheLedgerKey:
    session_id: str
    provider: str
    model_id: str
    prefix_hash: str


@dataclass
class CacheLedgerEntry:
    cached_tokens: int
    expires_at: float
    confidence: CacheConfidence
    last_actual_cache_read_tokens: int = 0


class CacheLedger:
    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
        self._ttl_seconds = ttl_seconds
        self._entries: dict[CacheLedgerKey, CacheLedgerEntry] = {}
        self._output_ema: float = float(DEFAULT_EXPECTED_OUTPUT_TOKENS)

    def expected_output_tokens(self) -> int:
        """Rolling estimate of recent output length, used to weight output cost."""
        return max(1, round(self._output_ema))

    def predict(self, model: ResolvedModel, features: RequestCostFeatures) -> CachePrediction:
        key = self._key(model, features)
        if key is None:
            return CachePrediction(False, 0, None, "no_session")

        entry = self._entries.get(key)
        now = time.time()
        if entry is None:
            return CachePrediction(False, 0, None, "cold")
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return CachePrediction(False, 0, None, "expired")

        if entry.confidence == CacheConfidence.BEST_EFFORT and entry.last_actual_cache_read_tokens <= 0:
            return CachePrediction(False, 0, entry.confidence.value, "best_effort_unconfirmed")

        # entry.cached_tokens is the cache size the provider actually reported last
        # turn (~the whole warm prefix: system + tools + most of the conversation
        # history), not just the static system+tools prefix. Bound it by this
        # request's total input; do NOT clamp to estimated_cacheable_prefix_tokens or
        # the warm credit collapses to system+tools and the warm default is wildly
        # overpriced (which makes cache_aware deviate and bust the very cache it warmed).
        cached_tokens = min(entry.cached_tokens, features.estimated_input_tokens)
        if cached_tokens <= 0:
            return CachePrediction(False, 0, entry.confidence.value, "no_cached_tokens")

        return CachePrediction(True, cached_tokens, entry.confidence.value, "warm")

    def observe(
        self,
        model: ResolvedModel,
        features: RequestCostFeatures,
        usage: Usage,
        *,
        is_main: bool = True,
    ) -> None:
        # Track output length from main (interactive tool-loop) calls only. Background
        # calls — title/summary — emit short outputs that would drag the estimate down
        # and misprice the long main calls cache-aware actually routes. The per-prefix
        # cache entry below is still recorded for every call.
        if is_main and usage.output_tokens > 0:
            self._output_ema = (
                (1 - OUTPUT_EMA_ALPHA) * self._output_ema
                + OUTPUT_EMA_ALPHA * usage.output_tokens
            )

        key = self._key(model, features)
        if key is None:
            return

        observed_tokens = max(
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        if observed_tokens <= 0:
            self._entries.pop(key, None)
            return

        confidence = (
            CacheConfidence.PREDICTABLE
            if model.provider == "anthropic"
            else CacheConfidence.BEST_EFFORT
        )
        existing = self._entries.get(key)
        cached_tokens = max(
            observed_tokens,
            features.estimated_cacheable_prefix_tokens,
            existing.cached_tokens if existing else 0,
        )
        self._entries[key] = CacheLedgerEntry(
            cached_tokens=cached_tokens,
            expires_at=time.time() + self._ttl_seconds,
            confidence=confidence,
            last_actual_cache_read_tokens=usage.cache_read_input_tokens,
        )

    @staticmethod
    def _key(model: ResolvedModel, features: RequestCostFeatures) -> CacheLedgerKey | None:
        if not features.session_id:
            return None
        return CacheLedgerKey(
            session_id=features.session_id,
            provider=model.provider,
            model_id=model.model_id,
            prefix_hash=features.prefix_hash,
        )
