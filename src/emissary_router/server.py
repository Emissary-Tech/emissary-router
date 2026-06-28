from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from emissary_router.config import load_config, user_config_path
from emissary_router.dashboard import build_dashboard_router, build_demo_router
from emissary_router.pipeline import RouterPipeline
from emissary_router.telemetry import SqliteStore


def create_app() -> FastAPI:
    config = load_config()
    config_path = user_config_path()
    store = (
        SqliteStore(
            Path(config.telemetry.db_path).expanduser(),
            retention_days=config.telemetry.retention_days,
            max_events=config.telemetry.max_events,
        )
        if config.telemetry.enabled
        else None
    )
    app = FastAPI(title="Emissary Router")
    app.state.config = config
    app.state.pipeline = RouterPipeline(config, store=store)

    def reload_config() -> None:
        # Rebuild the pipeline from the saved config so dashboard edits apply without
        # a restart. Routing (enabled models, default, confidence, policy, provider) is
        # taken from the new pipeline; in-flight requests keep the one they already read.
        # The cache ledger is carried over so the warm-cache state cache_aware relies on
        # survives a config save (it would otherwise reset on every edit).
        new_config = load_config(config_path)
        previous = app.state.pipeline
        app.state.config = new_config
        app.state.pipeline = RouterPipeline(
            new_config, store=store, cache_ledger=previous.cache_ledger
        )

    if store is not None:
        app.include_router(
            build_dashboard_router(
                store,
                baseline_model=config.default,
                auth_key=config.server.auth_key,
                config_path=config_path,
                on_config_change=reload_config,
            )
        )

    if config.demo.enabled:
        app.include_router(
            build_demo_router(
                auth_key=config.server.auth_key,
                streaming_default=config.demo.streaming,
            )
        )

    @app.get("/")
    async def health(request: Request) -> dict:
        return {
            "ok": True,
            "default": request.app.state.config.default,
            "policy": request.app.state.config.policy,
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
        return await request.app.state.pipeline.handle_messages(body, headers)

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
