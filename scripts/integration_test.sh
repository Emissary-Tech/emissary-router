#!/usr/bin/env bash
# End-to-end integration test for emissary-router.
#
# Verifies the real Claude Code path against live provider APIs:
#   install/CLI lifecycle -> gateway routing + telemetry (real classifier)
#   -> per-bridge caching, streaming (tool_use over SSE), and multi-tool agentic loops.
#
# Keys: copy scripts/test_env.sh.example -> scripts/test_env.sh and fill in real
# keys (gitignored; this script sources it automatically). Then just run:
#   bash scripts/integration_test.sh
# (Or export ANTHROPIC_API_KEY / OPENROUTER_API_KEY / EMISSARY_ROUTER_API_KEY
#  yourself instead of creating test_env.sh.)
#
# Optional overrides:
#   ROUTER_CLASSIFIER_URL   (default https://api.withemissary.com/v1/classification)
#   ROUTER_CLASSIFIER_MODEL (default emissary-model-router-shared)
#   ROUTER_TEST_PORT        (default 8799)
#
# Makes real (small) API calls -> costs a few cents. Exit code 0 = all checks passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- 0. load local secrets (gitignored) if present ----------------------------
if [ -f "$ROOT/scripts/test_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/test_env.sh"
fi
if [ -z "${EMISSARY_ROUTER_API_KEY:-}" ] && [ -n "${ROUTER_KEY:-}" ]; then
  export EMISSARY_ROUTER_API_KEY="$ROUTER_KEY"
fi

# --- 0b. require env ----------------------------------------------------------
missing=()
for v in ANTHROPIC_API_KEY OPENROUTER_API_KEY EMISSARY_ROUTER_API_KEY; do
  [ -n "${!v:-}" ] || missing+=("$v")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "missing required env: ${missing[*]}" >&2
  echo "hint: create scripts/test_env.sh (see test_env.sh.example) or export the keys" >&2
  exit 2
fi

CLASS_URL="${ROUTER_CLASSIFIER_URL:-https://api.withemissary.com/v1/classification}"
CLASS_MODEL="${ROUTER_CLASSIFIER_MODEL:-emissary-model-router-shared}"
PORT="${ROUTER_TEST_PORT:-8799}"

# --- isolated home (config/logs/pid/telemetry) + cleanup ----------------------
WORK="$(mktemp -d)"
export EMISSARY_ROUTER_HOME="$WORK/home"
mkdir -p "$EMISSARY_ROUTER_HOME"
TELOG="$EMISSARY_ROUTER_HOME/events.jsonl"
CFG="$WORK/config.yaml"
cleanup() { er stop >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

# --- 1. install (editable) if the CLI is not already available ----------------
if ! command -v er >/dev/null 2>&1; then
  echo "[setup] installing emissary-router (editable)..."
  python3 -m pip install -e . -q
fi

# --- 2. write test config (Gemini via OpenRouter = the supported tool path) ----
cat > "$CFG" <<YAML
models:
  claude-sonnet-4.6: true
  claude-haiku-4.5: true
  gemini-3.1-flash-lite: true
default: claude-sonnet-4.6
confidence: 0.8
server:
  host: 127.0.0.1
  port: $PORT
router:
  url: $CLASS_URL
  router_model: $CLASS_MODEL
telemetry:
  enabled: true
  log_path: $TELOG
YAML

# --- 3. CLI lifecycle ---------------------------------------------------------
echo "[lifecycle] validate-config"
er validate-config --config "$CFG" >/dev/null
echo "[lifecycle] start (background)"
er start --config "$CFG" >/dev/null
er status --config "$CFG" | grep -q '"healthy": true' \
  || { echo "  FAIL gateway did not become healthy"; exit 1; }
echo "  PASS gateway healthy on :$PORT"

# --- 4. gateway + per-bridge checks (live) ------------------------------------
python3 - "$PORT" "$TELOG" <<'PY'
import sys, os, json, asyncio, urllib.request
PORT, TELOG = sys.argv[1], sys.argv[2]
from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.schemas import AnthropicRequest, RequestContext
from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.providers.openrouter import OpenRouterProvider

fails = []
def check(name, ok, detail=""):
    print(("  PASS " if ok else "  FAIL ") + name + (f" :: {detail}" if detail else ""))
    if not ok:
        fails.append(name)

BIG = ("You are a meticulous senior engineering assistant. Follow conventions. Reason carefully. ") * 400
WTOOL = [{"name": "get_weather", "description": "Get weather for a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]

# ---- gateway: full stack (HTTP -> classifier -> provider -> telemetry) ----
print("[gateway] routing + telemetry")
def gw(task):
    body = {"model": "claude-sonnet-4-6", "max_tokens": 16,
            "system": [{"type": "text", "text": BIG, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": task}], "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

r = gw("Write a python function to compute fibonacci.")
check("gateway request returns content", bool(r.get("content")), f"model={r.get('model')}")
rows = [json.loads(l) for l in open(TELOG)] if os.path.exists(TELOG) else []
check("telemetry row written", len(rows) >= 1, f"{len(rows)} row(s)")
if rows:
    last = rows[-1]
    check("telemetry has served_model + route_reason + cost_usd",
          bool(last.get("served_model")) and bool(last.get("route_reason")) and ("cost_usd" in last),
          f"served={last.get('served_model')} reason={last.get('route_reason')}")

# ---- direct bridge checks ----
ant = AnthropicProvider(ProviderConfig(type="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"]))
orp = OpenRouterProvider(ProviderConfig(type="openrouter", api_key=os.environ["OPENROUTER_API_KEY"]))
def rm(mid, provider): return ResolvedModel(name=mid, provider=provider, model_id=mid)

async def call(prov, mid, body):
    cap = {}
    resp = await prov.messages(
        AnthropicRequest(body=body, headers={"anthropic-version": "2023-06-01"}),
        model=rm(mid, prov.name),
        context=RequestContext(request_id="t", conversation_id="c", classifier_input="", requested_model=mid),
        on_complete=lambda u, m: cap.update(u=u, m=m))
    return resp, cap

async def consume(resp):
    chunks = []
    async for c in resp.body_iterator:
        chunks.append(c if isinstance(c, (bytes, bytearray)) else str(c).encode())
    return b"".join(chunks).decode("utf-8", "replace")

async def bridges():
    # caching (Anthropic explicit = reliable): same big prefix twice -> read on 2nd
    print("[cache] anthropic-direct sonnet (2x identical prefix)")
    cbody = {"model": "x", "max_tokens": 16,
             "system": [{"type": "text", "text": BIG, "cache_control": {"type": "ephemeral"}}],
             "messages": [{"role": "user", "content": "Reply with: OK"}], "stream": False}
    await call(ant, "claude-sonnet-4-6", cbody)
    _, c2 = await call(ant, "claude-sonnet-4-6", cbody)
    check("anthropic cache hit on 2nd request", c2["u"].cache_read_input_tokens > 0,
          f"cache_read={c2['u'].cache_read_input_tokens}")

    # streaming: valid Anthropic SSE carrying a tool_use
    print("[stream] tool_use over synthesized/native SSE")
    for name, prov, mid in [("anthropic", ant, "claude-sonnet-4-6"),
                            ("OR-gemini", orp, "google/gemini-3.1-flash-lite")]:
        sbody = {"model": "x", "max_tokens": 120, "system": "Use tools when needed.", "tools": WTOOL,
                 "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}], "stream": True}
        resp, _ = await call(prov, mid, sbody)
        raw = await consume(resp)
        ok = ("message_start" in raw) and ("message_stop" in raw) and ("tool_use" in raw)
        check(f"stream {name}: valid SSE seq + tool_use", ok)

    # multi-tool agentic loop (bridge round-trips its own tool_use back through itself)
    print("[loop] multi-tool agentic loop (weather + time)")
    TOOLS = [
        {"name": "get_weather", "description": "weather",
         "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
        {"name": "get_time", "description": "time",
         "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    ]
    def execute(n, i):
        c = i.get("city", "?")
        return f"{c}: 18C, sunny" if n == "get_weather" else f"{c}: 22:30 local"

    async def loop(prov, mid):
        msgs = [{"role": "user", "content": "What's the weather in Paris and the local time in Tokyo? "
                                            "Use the tools, then give a one-line summary."}]
        calls = []
        for _ in range(5):
            body = {"model": "x", "max_tokens": 300, "system": "Use tools to get facts.",
                    "tools": TOOLS, "messages": msgs, "stream": False}
            resp, cap = await call(prov, mid, body)
            if cap["m"].get("http_status") != 200:
                return False, calls
            content = json.loads(bytes(resp.body).decode()).get("content") or []
            tus = [b for b in content if b.get("type") == "tool_use"]
            if tus:
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": t["id"], "content": execute(t["name"], t.get("input", {}))}
                    for t in tus]})
                calls += [t["name"] for t in tus]
                continue
            txt = next((b.get("text", "") for b in content if b.get("type") == "text"), "")
            ok = ("18" in txt or "sunny" in txt.lower()) and ("22" in txt or "tokyo" in txt.lower())
            return ok, calls
        return False, calls

    for name, prov, mid in [("OR-claude", orp, "anthropic/claude-haiku-4.5"),
                            ("OR-gemini", orp, "google/gemini-3.1-flash-lite")]:
        ok, calls = await loop(prov, mid)
        check(f"loop {name}: finished with tool calls + synthesis", ok, f"tools={calls}")

asyncio.run(bridges())

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")
PY

echo "[done] integration test passed"
