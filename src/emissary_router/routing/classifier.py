from __future__ import annotations

import asyncio
import os

import httpx

from emissary_router.catalog import ROUTER_API_KEY_ENV
from emissary_router.config import RouterConfig


class ClassifierClient:
    def __init__(self, config: RouterConfig):
        self._config = config

    async def predict(self, classifier_input: str) -> dict[str, float]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(ROUTER_API_KEY_ENV)
        if api_key:
            headers["X-API-Key"] = api_key
        payload = {
            "model": self._config.router_model,
            "input": classifier_input,
            "data_format": "probs",
        }
        data = await self._post_with_retry(headers, payload)
        return data["data"][0]["probs"]

    async def _post_with_retry(self, headers: dict[str, str], payload: dict) -> dict:
        # Retry transient transport failures (connect/read timeouts, dropped
        # connections) and 5xx responses with exponential backoff. Client errors
        # (4xx) are not retried since they will not succeed on a repeat.
        attempts = self._config.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                    response = await client.post(self._config.url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code < 500:
                    raise
            except httpx.TransportError as exc:
                last_exc = exc

            if attempt < attempts - 1:
                delay = self._config.retry_backoff_seconds * (2 ** attempt)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
