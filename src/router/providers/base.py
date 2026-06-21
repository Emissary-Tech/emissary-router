from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from starlette.responses import Response

from router.caching.usage import Usage
from router.config import ResolvedModel
from router.schemas import AnthropicRequest, RequestContext

ProviderComplete = Callable[[Usage, dict[str, Any]], None]


class Provider(Protocol):
    name: str

    async def messages(
        self,
        request: AnthropicRequest,
        model: ResolvedModel,
        context: RequestContext,
        on_complete: ProviderComplete | None = None,
    ) -> Response:
        ...

    def usage_from_response(self, payload: dict) -> Usage:
        ...
