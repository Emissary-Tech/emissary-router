from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import HTMLResponse, JSONResponse

from emissary_router.config import PricingConfig
from emissary_router.dashboard.page import dashboard_html
from emissary_router.dashboard.savings import compute_summary
from emissary_router.telemetry import SqliteStore, TurnTracker


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
    pricing: PricingConfig,
    baseline_model: str,
    auth_key: str | None = None,
    turns_tracker: TurnTracker | None = None,
) -> APIRouter:
    router = APIRouter(dependencies=[Depends(_make_auth_dependency(auth_key))])

    @router.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(dashboard_html())

    @router.get("/api/events")
    async def events(limit: int = 200, session: str | None = None) -> JSONResponse:
        limit = max(1, min(limit, 1000))
        return JSONResponse({"events": store.list_events(limit=limit, session_id=session)})

    @router.get("/api/summary")
    async def summary() -> JSONResponse:
        data = compute_summary(store.aggregate_by_model(), pricing, baseline_model)
        return JSONResponse(data)

    @router.get("/api/turns")
    async def turns(limit: int = 200, session: str | None = None) -> JSONResponse:
        limit = max(1, min(limit, 1000))
        return JSONResponse({"turns": store.turns(session_id=session, limit=limit)})

    @router.delete("/api/events/{event_id}")
    async def delete_event(event_id: str) -> JSONResponse:
        deleted = store.delete_event(event_id)
        return JSONResponse({"deleted": deleted})

    @router.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        deleted = store.delete_session(session_id)
        if turns_tracker is not None:
            turns_tracker.forget(session_id)
        return JSONResponse({"deleted": deleted})

    return router
