from __future__ import annotations

from enum import Enum


class CacheConfidence(str, Enum):
    PREDICTABLE = "predictable"
    BEST_EFFORT = "best_effort"
