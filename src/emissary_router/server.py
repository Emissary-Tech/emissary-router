from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from emissary_router.config import load_config, load_pricing
from emissary_router.dashboard import build_dashboard_router
from emissary_router.pipeline import RouterPipeline
from emissary_router.telemetry import SqliteStore, TurnTracker


def create_app() -> FastAPI:
    config = load_config()
    pricing = load_pricing()
    store = (
        SqliteStore(
            Path(config.telemetry.db_path),
            retention_days=config.telemetry.retention_days,
            max_events=config.telemetry.max_events,
        )
        if config.telemetry.enabled
        else None
    )
    turns = TurnTracker(store)
    pipeline = RouterPipeline(config, pricing, store=store, turns=turns)
    app = FastAPI(title="Emissary Router")
    if store is not None:
        app.include_router(
            build_dashboard_router(
                store,
                pricing,
                config.telemetry.baseline_model,
                auth_key=config.server.auth_key,
                turns_tracker=turns,
            )
        )

    @app.get("/")
    async def health() -> dict:
        return {
            "ok": True,
            "default": config.default,
            "policy": "deviate_if_confident",
        }

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        headers = dict(request.headers)
        if config.server.auth_key:
            if not _authorized(headers, config.server.auth_key):
                return JSONResponse(
                    {
                        "error": {
                            "type": "unauthorized",
                            "message": "invalid Emissary Router auth key",
                        }
                    },
                    status_code=401,
                )
            headers = _strip_router_auth(headers, config.server.auth_key)
        return await pipeline.handle_messages(body, headers)

    return app


def _authorized(headers: dict[str, str], auth_key: str) -> bool:
    lowered = {key.lower(): value for key, value in headers.items()}
    if lowered.get("x-api-key") == auth_key:
        return True
    authorization = lowered.get("authorization", "")
    return authorization == f"Bearer {auth_key}"


def _strip_router_auth(headers: dict[str, str], auth_key: str) -> dict[str, str]:
    stripped: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower == "x-api-key" and value == auth_key:
            continue
        if lower == "authorization" and value == f"Bearer {auth_key}":
            continue
        stripped[key] = value
    return stripped
