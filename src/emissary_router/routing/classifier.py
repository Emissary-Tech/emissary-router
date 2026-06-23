from __future__ import annotations

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
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            response = await client.post(self._config.url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["data"][0]["probs"]
