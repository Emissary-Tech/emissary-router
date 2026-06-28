from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, get_args

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from starlette.responses import HTMLResponse, JSONResponse

from emissary_router.catalog import CATALOG, cost_score
from emissary_router.config import AppConfig, load_config, user_config_path
from emissary_router.dashboard.page import dashboard_html
from emissary_router.dashboard.savings import compute_summary
from emissary_router.telemetry import SqliteStore


def _read_raw_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def _write_raw_config(path: Path, data: dict) -> None:
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2) + "\n")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False))


def _entry_enabled(entry) -> bool:
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, dict):
        return bool(entry.get("enabled", True))
    return True


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization[len("Bearer ") :]
    return None


def _make_auth_dependency(auth_key: str | None):
    async def require_auth(request: Request) -> None:
        if not auth_key:
            return
        provided = (
            request.headers.get("x-api-key")
            or _bearer(request.headers.get("authorization"))
            or request.query_params.get("key")
        )
        if provided != auth_key:
            raise HTTPException(status_code=401, detail="invalid Emissary Router auth key")

    return require_auth


def build_dashboard_router(
    store: SqliteStore,
    baseline_model: str,
    auth_key: str | None = None,
    config_path: Path | None = None,
    on_config_change: Callable[[], None] | None = None,
) -> APIRouter:
    router = APIRouter(dependencies=[Depends(_make_auth_dependency(auth_key))])
    path = config_path or user_config_path()

    @router.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(dashboard_html())

    @router.get("/api/config")
    async def get_config() -> JSONResponse:
        cfg = load_config(path)
        enabled = set(cfg.enabled_models())
        models = []
        for name in sorted(CATALOG, key=lambda name: cost_score(CATALOG[name])):
            spec = CATALOG[name]
            entry = cfg.models.get(name)
            selected = entry.provider if entry and entry.provider else spec.default_provider
            models.append(
                {
                    "name": name,
                    "enabled": name in enabled,
                    "cost_score": round(cost_score(spec), 2),
                    "providers": list(spec.providers.keys()),
                    "provider": selected,
                    "default_provider": spec.default_provider,
                }
            )
        return JSONResponse(
            {
                "default": cfg.default,
                "confidence": cfg.confidence,
                "policy": cfg.policy,
                "policies": list(get_args(AppConfig.model_fields["policy"].annotation)),
                "models": models,
            }
        )

    @router.put("/api/config")
    async def update_config(payload: dict = Body(...)) -> JSONResponse:
        raw = _read_raw_config(path)
        existing = raw.get("models") or {}
        requested = payload.get("models") or {}
        new_models: dict = {}
        for name in CATALOG:
            req = requested.get(name)
            prev = existing.get(name)
            if req is None:
                if name in existing:
                    new_models[name] = prev
                continue
            if isinstance(req, bool):
                enabled, provider = req, None
            else:
                enabled = bool(req.get("enabled", True))
                provider = req.get("provider")
            if provider is None and isinstance(prev, dict):
                provider = prev.get("provider")
            if provider:
                new_models[name] = {"enabled": enabled, "provider": provider}
            else:
                new_models[name] = enabled
        raw["models"] = new_models
        if "default" in payload:
            raw["default"] = payload["default"]
        if "confidence" in payload:
            raw["confidence"] = float(payload["confidence"])
        if "policy" in payload:
            raw["policy"] = payload["policy"]
        try:
            AppConfig.model_validate(raw)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        _write_raw_config(path, raw)
        if on_config_change is not None:
            try:
                on_config_change()
            except Exception:
                pass
        return JSONResponse({"saved": True, "restart_required": on_config_change is None})

    @router.get("/api/events")
    async def events(
        limit: int = 200, offset: int = 0, session: str | None = None
    ) -> JSONResponse:
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        return JSONResponse(
            {
                "events": store.list_events(limit=limit, offset=offset, session_id=session),
                "total": store.total_events(session_id=session),
            }
        )

    @router.get("/api/summary")
    async def summary() -> JSONResponse:
        try:
            baseline = load_config(path).default
        except Exception:
            baseline = baseline_model
        data = compute_summary(store.aggregate_by_model(), baseline)
        return JSONResponse(data)

    @router.get("/api/sessions")
    async def sessions(limit: int = 200, offset: int = 0) -> JSONResponse:
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        return JSONResponse(
            {
                "sessions": store.sessions(limit=limit, offset=offset),
                "total": store.total_sessions(),
            }
        )

    @router.delete("/api/events/{event_id}")
    async def delete_event(event_id: str) -> JSONResponse:
        deleted = store.delete_event(event_id)
        return JSONResponse({"deleted": deleted})

    @router.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        deleted = store.delete_session(session_id)
        return JSONResponse({"deleted": deleted})

    return router
