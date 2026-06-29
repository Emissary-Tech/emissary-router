from __future__ import annotations

import os

import httpx

# Anthropic tool definition handed to both demo sides; the gateway converts it per
# provider. Kept minimal so the model decides when a query needs the web.
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current or factual information — recent events, specific "
        "facts, or anything you are unsure about. For general knowledge you already "
        "know, answer directly without searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "the search query"}},
        "required": ["query"],
    },
}

TAVILY_URL = "https://api.tavily.com/search"


async def web_search(query: str) -> str:
    """Run a web search and return a compact text summary for the model. Uses Tavily when
    TAVILY_API_KEY is set; otherwise returns a mock note so the demo still runs offline."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return f"(no TAVILY_API_KEY set — mock search for: {query})"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                TAVILY_URL,
                json={"api_key": key, "query": query, "max_results": 4, "include_answer": True},
            )
            data = resp.json()
    except Exception as exc:  # never let a search failure break the turn
        return f"(search failed: {exc})"

    parts: list[str] = []
    if data.get("answer"):
        parts.append(str(data["answer"]))
    for result in (data.get("results") or [])[:4]:
        title = result.get("title", "")
        snippet = (result.get("content") or "")[:240]
        url = result.get("url", "")
        parts.append(f"- {title}: {snippet} ({url})")
    return "\n".join(parts) or "(no results)"
