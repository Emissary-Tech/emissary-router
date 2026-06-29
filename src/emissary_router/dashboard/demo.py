from __future__ import annotations

import json

from fastapi import APIRouter, Body, Depends, Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

from emissary_router.dashboard.routes import _make_auth_dependency


def build_demo_router(auth_key: str | None = None, streaming_default: bool = False) -> APIRouter:
    """Conference split-screen demo: default Sonnet vs the routed system as two parallel
    chats. Mounted only when `demo.enabled`, behind the same auth as the dashboard (each
    turn makes two real model calls). The chat handler reads the live `app.state.pipeline`,
    so it tracks config hot-reloads. No conversation state is stored server-side — the
    browser holds the in-progress chat and a refresh starts a new one."""
    router = APIRouter(dependencies=[Depends(_make_auth_dependency(auth_key))])

    @router.get("/demo")
    async def demo_page() -> HTMLResponse:
        return HTMLResponse(demo_html(streaming_default))

    @router.post("/api/demo/chat")
    async def chat(request: Request, payload: dict = Body(...)) -> JSONResponse:
        args, error = _parse_payload(payload)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        try:
            result = await request.app.state.pipeline.chat(**args)
        except Exception as exc:  # surface upstream/key issues to the page instead of 500-ing silently
            return JSONResponse({"error": str(exc)}, status_code=502)
        return JSONResponse(result)

    @router.post("/api/demo/stream", response_model=None)
    async def stream(request: Request, payload: dict = Body(...)) -> StreamingResponse | JSONResponse:
        args, error = _parse_payload(payload)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        pipeline = request.app.state.pipeline
        search = bool(payload.get("search"))

        async def events():
            try:
                async for ev in pipeline.stream_chat(**args, search=search):
                    yield "data: " + json.dumps(ev) + "\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({"type": "fatal", "error": str(exc)}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return router


def _parse_payload(payload: dict) -> tuple[dict | None, str | None]:
    baseline = payload.get("baseline")
    routed = payload.get("routed")
    if not _valid_messages(baseline) or not _valid_messages(routed):
        return None, "baseline and routed message lists are required"
    if len(baseline) > 100 or len(routed) > 100:
        return None, "conversation too long"
    effort = payload.get("effort")
    policy = payload.get("policy")
    session_id = payload.get("session_id")
    return {
        "baseline_messages": baseline,
        "routed_messages": routed,
        "session_id": session_id if isinstance(session_id, str) and session_id else None,
        "max_tokens": _clamp_int(payload.get("max_tokens"), default=32000, lo=256, hi=64000),
        "effort": effort if effort in {"low", "medium", "high"} else None,
        "policy": policy if policy in {"deviate_if_confident", "cache_aware"} else None,
    }, None


def _valid_messages(msgs) -> bool:
    return (
        isinstance(msgs, list)
        and len(msgs) > 0
        and all(
            isinstance(m, dict) and m.get("role") in {"user", "assistant"} and "content" in m
            for m in msgs
        )
    )


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


PRESETS = [
    "What's the difference between TCP and UDP?",
    "Summarize what a hash map is in two sentences.",
    "If 3x + 7 = 22, what is x?",
    "Prove that the square root of 2 is irrational.",
]


def demo_html(streaming_default: bool = False) -> str:
    chips = "".join(
        f'<button class="chip" data-q="{_esc(q)}">{_esc(q)}</button>' for q in PRESETS
    )
    return _PAGE.replace("__CHIPS__", chips).replace(
        "__STREAM_CHECKED__", "checked" if streaming_default else ""
    )


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Emissary Router — live</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef; --muted:#8b93a7; --accent:#5b9cff; --good:#43c59e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:14px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:var(--muted); font-size:13px; }
  .grow { flex:1; }
  button.iconbtn { background:none; border:1px solid var(--line); color:var(--muted); border-radius:8px; cursor:pointer; padding:6px 12px; font:inherit; }
  button.iconbtn:hover { color:var(--fg); border-color:var(--accent); }
  main { padding:18px 22px; max-width:1180px; margin:0 auto; }
  .totals { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
  .metric { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 14px; }
  .metric .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .metric .val { font-size:14px; margin-top:3px; }
  .panes { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .pane-wrap { display:flex; flex-direction:column; min-width:0; }
  .pane-head { font-size:13px; font-weight:500; color:var(--muted); padding:0 4px 8px; display:flex; align-items:center; gap:8px; }
  .pane-head.routed { color:#9cc0ff; }
  .pane { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; height:56vh; overflow-y:auto; }
  .pane.routed { border-color:#27406a; }
  .msg { margin-bottom:12px; display:flex; flex-direction:column; }
  .msg.user { align-items:flex-end; }
  .bub { max-width:88%; white-space:pre-wrap; border-radius:12px; padding:9px 12px; font-size:14px; }
  .msg.user .bub { background:#23314d; color:#dbe6ff; }
  .msg.asst .bub { background:#1b1f28; color:#d6dae6; }
  .bfoot { font-size:11.5px; color:var(--muted); margin-top:4px; }
  .badge { display:inline-block; font-size:11px; border-radius:999px; padding:1px 8px; background:#1b2740; color:#9cc0ff; border:1px solid #27406a; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .save { color:var(--good); font-weight:500; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin:14px 0 8px; }
  .chip { background:var(--panel); border:1px solid var(--line); color:var(--muted); border-radius:999px; padding:5px 11px; font:12.5px inherit; cursor:pointer; }
  .chip:hover { color:var(--fg); border-color:var(--accent); }
  .composer { display:flex; gap:8px; margin-top:6px; }
  input#q { flex:1; background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:10px 12px; font:inherit; }
  button.primary { background:var(--accent); border:none; color:#fff; border-radius:8px; padding:0 18px; cursor:pointer; font:inherit; font-weight:500; }
  button.primary:disabled { opacity:.5; cursor:default; }
  .opts { display:flex; gap:18px; align-items:center; margin-top:10px; font-size:13px; color:var(--muted); }
  .opts label { display:flex; align-items:center; gap:6px; }
  .opts select { background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 6px; font:inherit; }
  .gearlink { position:fixed; right:16px; bottom:14px; background:var(--panel); border:1px solid var(--line); color:var(--muted); border-radius:999px; padding:7px 14px; font-size:13px; text-decoration:none; }
  .gearlink:hover { color:var(--fg); border-color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Emissary Router</h1>
  <span class="sub">default Sonnet vs routed — same chat, side by side</span>
  <span class="grow"></span>
  <span class="sub" id="t-turns">new chat</span>
  <button class="iconbtn" id="newchat">New chat</button>
</header>
<main>
  <div class="totals">
    <div class="metric"><div class="label">Running cost</div><div class="val" id="t-cost">—</div></div>
    <div class="metric"><div class="label">Total latency</div><div class="val" id="t-lat">—</div></div>
  </div>

  <div class="panes">
    <div class="pane-wrap">
      <div class="pane-head">Claude Sonnet <span class="badge">default</span></div>
      <div class="pane" id="pane-base"></div>
    </div>
    <div class="pane-wrap">
      <div class="pane-head routed">Emissary routed</div>
      <div class="pane routed" id="pane-routed"></div>
    </div>
  </div>

  <div class="chips">__CHIPS__</div>
  <div class="composer">
    <input id="q" placeholder="Ask anything…" autocomplete="off" />
    <button class="primary" id="send">Send</button>
  </div>
  <div class="opts">
    <label>Routing
      <select id="policy">
        <option value="cache_aware" selected>cache_aware</option>
        <option value="deviate_if_confident">deviate</option>
      </select>
    </label>
    <label>Reasoning
      <select id="effort">
        <option value="" selected>off</option>
        <option value="low">low</option>
        <option value="medium">medium</option>
        <option value="high">high</option>
      </select>
    </label>
    <label>Max tokens
      <select id="maxtok">
        <option value="32000" selected>32k</option>
        <option value="64000">64k</option>
      </select>
    </label>
    <label><input type="checkbox" id="stream" __STREAM_CHECKED__/> Stream</label>
    <label><input type="checkbox" id="search" /> Web search</label>
    <span>both sides use the same settings; converted per model</span>
  </div>
</main>
<a class="gearlink" id="gear" href="/dashboard">⚙ Settings</a>
<script>
const $ = (id) => document.getElementById(id);
const KEY = new URLSearchParams(location.search).get("key");
const usd = (v) => "$" + Number(v || 0).toFixed(4);
const ms = (v) => Math.round(Number(v || 0)) + "ms";
const short = (m) => m.replace("claude-", "").replace("-4.6", "").replace("-4.5", "");

const baselineMsgs = [], routedMsgs = [];
let cCost = 0, rCost = 0, cLat = 0, rLat = 0, turns = 0;
const newId = () => (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : "s-" + Date.now() + "-" + (turns);
let sessionId = newId();

function bubble(paneId, role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const b = document.createElement("div");
  b.className = "bub";
  b.textContent = text;
  const foot = document.createElement("div");
  foot.className = "bfoot";
  wrap.appendChild(b);
  wrap.appendChild(foot);
  const pane = $(paneId);
  pane.appendChild(wrap);
  pane.scrollTop = pane.scrollHeight;
  return { bub: b, foot, pane };
}

function fill(slot, d, routed, keepText) {
  if (!keepText) slot.bub.textContent = d.error ? ("Error: " + d.error) : (d.answer || "(no text)");
  const sx = d.searches ? ' · 🔎 ' + d.searches : "";
  if (routed) {
    slot.foot.innerHTML = '<span class="badge">' + short(d.model) + "</span> " + usd(d.cost_usd) +
      ' · <span class="mono">router ' + ms(d.router_ms) + " + model " + ms(d.model_ms) +
      " = " + ms(d.total_ms) + "</span>" + sx;
  } else {
    slot.foot.innerHTML = usd(d.cost_usd) + ' · <span class="mono">' + ms(d.total_ms) + "</span>" + sx;
  }
  slot.pane.scrollTop = slot.pane.scrollHeight;
}

function updateTotals(j) {
  cCost += j.baseline.cost_usd || 0; rCost += j.routed.cost_usd || 0;
  cLat += j.baseline.total_ms || 0; rLat += j.routed.total_ms || 0; turns++;
  const pct = cCost > 0 ? Math.round((cCost - rCost) / cCost * 100) : 0;
  $("t-cost").innerHTML = 'all-sonnet <span class="mono">' + usd(cCost) + '</span> → routed <span class="mono">' +
    usd(rCost) + '</span> · <span class="save">saved ' + pct + "%</span>";
  $("t-lat").innerHTML = 'sonnet <span class="mono">' + (cLat / 1000).toFixed(1) + 's</span> · routed <span class="mono">' +
    (rLat / 1000).toFixed(1) + "s</span>";
  $("t-turns").textContent = turns + " turn" + (turns === 1 ? "" : "s");
}

const hdrs = () => { const h = { "content-type": "application/json" }; if (KEY) h["x-api-key"] = KEY; return h; };
const reqBody = () => JSON.stringify({
  baseline: baselineMsgs, routed: routedMsgs, session_id: sessionId,
  policy: $("policy").value, max_tokens: parseInt($("maxtok").value, 10),
  effort: $("effort").value || null, search: $("search").checked,
});

async function onceTurn(bSlot, rSlot) {
  const r = await fetch("/api/demo/chat", { method: "POST", headers: hdrs(), body: reqBody() });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  fill(bSlot, j.baseline, false);
  fill(rSlot, j.routed, true);
  baselineMsgs.push({ role: "assistant", content: j.baseline.answer || "" });
  routedMsgs.push({ role: "assistant", content: j.routed.answer || "" });
  updateTotals(j);
}

async function streamTurn(bSlot, rSlot) {
  const slots = { baseline: bSlot, routed: rSlot };
  const acc = { baseline: "", routed: "" }, fin = {};
  const r = await fetch("/api/demo/stream", { method: "POST", headers: hdrs(), body: reqBody() });
  if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || r.status); }
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\\n\\n")) >= 0) {
      const dl = buf.slice(0, idx).split("\\n").find((l) => l.startsWith("data:"));
      buf = buf.slice(idx + 2);
      if (!dl) continue;
      let ev; try { ev = JSON.parse(dl.slice(5).trim()); } catch (_) { continue; }
      if (ev.type === "fatal") throw new Error(ev.error || "stream error");
      const slot = slots[ev.side];
      if (!slot) continue;
      if (ev.type === "delta") {
        if (!acc[ev.side]) slot.bub.textContent = "";
        acc[ev.side] += ev.text; slot.bub.textContent = acc[ev.side];
        slot.pane.scrollTop = slot.pane.scrollHeight;
      } else if (ev.type === "tool") {
        slot.foot.textContent = "🔎 searching: " + (ev.query || "");
      } else if (ev.type === "done") {
        fin[ev.side] = ev;
        fill(slot, { ...ev, answer: acc[ev.side] || ev.answer }, ev.side === "routed", true);
      }
    }
  }
  baselineMsgs.push({ role: "assistant", content: acc.baseline });
  routedMsgs.push({ role: "assistant", content: acc.routed });
  if (fin.baseline && fin.routed) updateTotals({ baseline: fin.baseline, routed: fin.routed });
}

async function send() {
  const q = $("q").value.trim();
  if (!q) return;
  $("q").value = ""; $("send").disabled = true;
  baselineMsgs.push({ role: "user", content: q });
  routedMsgs.push({ role: "user", content: q });
  bubble("pane-base", "user", q); bubble("pane-routed", "user", q);
  const bSlot = bubble("pane-base", "asst", "…");
  const rSlot = bubble("pane-routed", "asst", "…");
  try {
    if ($("stream").checked || $("search").checked) await streamTurn(bSlot, rSlot);
    else await onceTurn(bSlot, rSlot);
  } catch (e) {
    bSlot.bub.textContent = "Error: " + e.message;
    rSlot.bub.textContent = "";
    baselineMsgs.pop(); routedMsgs.pop();  // drop the user turn so the chat stays consistent
  } finally {
    $("send").disabled = false; $("q").focus();
  }
}

function newChat() {
  baselineMsgs.length = 0; routedMsgs.length = 0;
  cCost = rCost = cLat = rLat = turns = 0;
  sessionId = newId();  // fresh cache-ledger scope, so policy comparisons start clean
  $("pane-base").innerHTML = ""; $("pane-routed").innerHTML = "";
  $("t-cost").textContent = "—"; $("t-lat").textContent = "—"; $("t-turns").textContent = "new chat";
}

$("send").onclick = send;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
$("newchat").onclick = newChat;
document.querySelectorAll(".chip").forEach((c) => c.onclick = () => { $("q").value = c.dataset.q; send(); });
$("gear").href = "/dashboard" + (KEY ? "?key=" + encodeURIComponent(KEY) : "") + "#settings";
</script>
</body>
</html>
"""
