from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CacheConfidence(str, Enum):
    PREDICTABLE = "predictable"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class CacheObservation:
    provider: str
    model_id: str
    conversation_id: str
    prefix_hash: str
    estimated_prefix_tokens: int
    expires_at: datetime | None
    confidence: CacheConfidence
    last_actual_cache_read_tokens: int = 0


@dataclass(frozen=True)
class ExplicitCacheEntry:
    provider: str
    model_id: str
    prefix_hash: str
    resource_name: str
    token_count: int
    expires_at: datetime
