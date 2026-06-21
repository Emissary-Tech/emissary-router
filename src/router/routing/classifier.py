from __future__ import annotations

import httpx

from router.config import ClassifierConfig


class ClassifierClient:
    def __init__(self, config: ClassifierConfig):
        self._config = config

    async def predict(self, classifier_input: str) -> dict[str, float]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["X-API-Key"] = self._config.api_key
        payload = {
            "model": self._config.model,
            "input": classifier_input,
            "data_format": "probs",
        }
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            response = await client.post(self._config.url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["data"][0]["probs"]
