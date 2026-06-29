from __future__ import annotations

import json
import os

from fastapi import APIRouter, Body, Depends, Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

from emissary_router.config import read_env_file, user_env_path, write_env_file
from emissary_router.dashboard.routes import _make_auth_dependency


def build_demo_router(auth_key: str | None = None) -> APIRouter:
    """Conference split-screen demo: default Sonnet vs the routed system as two parallel
    chats. Mounted only when `demo.enabled`, behind the same auth as the dashboard (each
    turn makes two real model calls). The chat handler reads the live `app.state.pipeline`,
    so it tracks config hot-reloads. No conversation state is stored server-side — the
    browser holds the in-progress chat and a refresh starts a new one."""
    router = APIRouter(dependencies=[Depends(_make_auth_dependency(auth_key))])

    @router.get("/demo")
    async def demo_page() -> HTMLResponse:
        return HTMLResponse(demo_html())

    @router.post("/api/demo/chat")
    async def chat(request: Request, payload: dict = Body(...)) -> JSONResponse:
        args, error = _parse_payload(payload)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        side = payload.get("side")
        try:
            pipeline = request.app.state.pipeline
            if side in ("baseline", "routed"):
                result = await pipeline.chat_side(side, **args)  # one side, rendered as soon as it's done
            else:
                result = await pipeline.chat(**args)
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

    @router.get("/api/demo/search-key")
    async def get_search_key() -> JSONResponse:
        key = os.environ.get("TAVILY_API_KEY", "")
        return JSONResponse({"set": bool(key), "hint": _key_hint(key)})

    @router.put("/api/demo/search-key")
    async def set_search_key(payload: dict = Body(...)) -> JSONResponse:
        key = (payload.get("key") or "").strip()
        if not key:
            return JSONResponse({"error": "empty key"}, status_code=400)
        if len(key) > 200 or any(c in key for c in "\"'\n\r"):
            return JSONResponse({"error": "invalid key"}, status_code=400)
        try:
            # Stored in .env (chmod 600), never in config.json — keeps keys out of the
            # forkable config. Applied to the running process immediately.
            path = user_env_path()
            values = read_env_file(path)
            values["TAVILY_API_KEY"] = key
            write_env_file(path, values)
            os.environ["TAVILY_API_KEY"] = key
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"set": True, "hint": _key_hint(key)})

    return router


def _key_hint(key: str) -> str:
    if not key:
        return ""
    return "••••" + key[-4:] if len(key) >= 4 else "••••"


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
    'On an island, A says "B is a liar," B says "C is a liar," C says "A and B are both liars." Who tells the truth?',
    "Two quantum states with energies E1 and E2 have lifetimes 10⁻⁹ s and 10⁻⁸ s. What energy difference lets them be clearly resolved?",
    "Find the median of two sorted arrays in O(log(m+n)) time, with code and a correctness argument.",
    "Prove that 7 divides 3^(2n+1) + 2^(n+2) for every positive integer n.",
]


def demo_html() -> str:
    chips = "".join(
        f'<button class="chip" data-q="{_esc(q)}">{_esc(q)}</button>' for q in PRESETS
    )
    return _PAGE.replace("__CHIPS__", chips)


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
  .bub.md { white-space:normal; }
  .bub strong { font-weight:500; color:#fff; }
  .bub em { font-style:italic; }
  .bub code { background:#11151c; padding:1px 5px; border-radius:4px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
  .bub pre { background:#11151c; padding:8px 10px; border-radius:6px; overflow-x:auto; margin:6px 0; }
  .bub pre code { background:none; padding:0; }
  .bub ul, .bub ol { margin:6px 0; padding-left:20px; }
  .bub li { margin:2px 0; }
  .bub h3, .bub h4 { font-size:14px; font-weight:500; margin:8px 0 4px; }
  .bub a { color:var(--accent); }
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
</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  .opts { display:flex; gap:18px; align-items:center; margin-top:10px; font-size:13px; color:var(--muted); }
  .opts label { display:flex; align-items:center; gap:6px; }
  .opts select { background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 6px; font:inherit; }
  .bub table { border-collapse:collapse; margin:8px 0; font-size:13px; }
  .bub th, .bub td { border:1px solid var(--line); padding:4px 9px; text-align:left; }
  .bub th { background:#1b1f28; font-weight:500; }
  .bub blockquote { border-left:3px solid var(--line); margin:6px 0; padding:2px 12px; color:var(--muted); }
  .bub hr { border:none; border-top:1px solid var(--line); margin:10px 0; }
  .bub .katex-display { margin:8px 0; overflow-x:auto; overflow-y:hidden; }
  .gearlink { position:fixed; right:16px; bottom:14px; background:var(--panel); border:1px solid var(--line); color:var(--muted); border-radius:999px; padding:7px 14px; font-size:13px; text-decoration:none; }
  .gearlink:hover { color:var(--fg); border-color:var(--accent); }
  .keyrow { display:flex; gap:10px; align-items:center; margin-top:10px; font-size:13px; color:var(--muted); }
  .keyrow input { background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:5px 8px; font:inherit; width:170px; }
</style>
</head>
<body>
<header>
  <h1>Emissary Router</h1>
  <span class="sub">default Sonnet vs routed — same chat, side by side</span>
  <span class="grow"></span>
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
    <label><input type="checkbox" id="search" /> Web search</label>
    <span>both sides use the same settings; converted per model</span>
  </div>
  <div class="keyrow">
    <span>Web search key (Tavily)</span>
    <input type="password" id="tavily" placeholder="paste key" autocomplete="off" />
    <button class="iconbtn" id="savekey">Save</button>
    <span id="keystat"></span>
  </div>
</main>
<a class="gearlink" id="gear" href="/dashboard">⚙ Settings</a>
<script>
const $ = (id) => document.getElementById(id);
const KEY = new URLSearchParams(location.search).get("key");
const usd = (v) => "$" + Number(v || 0).toFixed(4);
const ms = (v) => Math.round(Number(v || 0)) + "ms";
const short = (m) => m.replace("claude-", "").replace("-4.6", "").replace("-4.5", "");

const mdEsc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function md(src) {
  const ph = [];
  const stash = (h) => { ph.push(h); return "%%PH" + (ph.length - 1) + "%%"; };
  let s = String(src);
  s = s.replace(/```([\\s\\S]*?)```/g, (_, c) => stash("<pre><code>" + mdEsc(c.replace(/^\\n/, "")) + "</code></pre>"));
  s = s.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (_, m) => stash('<span class="math">$$' + mdEsc(m) + '$$</span>'));
  s = s.replace(/\\$([^$\\n]+?)\\$/g, (_, m) => stash('<span class="math">$' + mdEsc(m) + '$</span>'));
  s = mdEsc(s);
  s = s.replace(/`([^`]+)`/g, (_, c) => "<code>" + c + "</code>");
  s = s.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\\*([^*\\n]+)\\*/g, "$1<em>$2</em>");
  s = s.replace(/\\[([^\\]]+)\\]\\((https?:[^\\s)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/^###\\s+(.+)$/gm, "<h4>$1</h4>").replace(/^##\\s+(.+)$/gm, "<h3>$1</h3>").replace(/^#\\s+(.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/(?:^|\\n)(\\|.+\\|)\\n\\|[\\s:|-]+\\|\\n((?:\\|.+\\|\\n?)*)/g, (_, head, rows) => {
    const cells = (r) => r.split("|").slice(1, -1).map((c) => c.trim());
    const th = cells(head).map((c) => "<th>" + c + "</th>").join("");
    const trs = rows.trim().split(/\\n/).map((r) => "<tr>" + cells(r).map((c) => "<td>" + c + "</td>").join("") + "</tr>").join("");
    return stash("<table><thead><tr>" + th + "</tr></thead><tbody>" + trs + "</tbody></table>");
  });
  s = s.replace(/(?:^|\\n)((?:&gt;\\s?.*(?:\\n|$))+)/g, (_, b) => stash("<blockquote>" + b.trim().split(/\\n/).map((l) => l.replace(/^&gt;\\s?/, "")).join("<br>") + "</blockquote>"));
  s = s.replace(/(?:^|\\n)---+(?:\\n|$)/g, () => stash("<hr>"));
  s = s.replace(/(?:^|\\n)((?:[-*]\\s+.+(?:\\n|$))+)/g, (_, it) => stash("<ul>" + it.trim().split(/\\n/).map((l) => "<li>" + l.replace(/^[-*]\\s+/, "") + "</li>").join("") + "</ul>"));
  s = s.replace(/(?:^|\\n)((?:\\d+\\.\\s+.+(?:\\n|$))+)/g, (_, it) => stash("<ol>" + it.trim().split(/\\n/).map((l) => "<li>" + l.replace(/^\\d+\\.\\s+/, "") + "</li>").join("") + "</ol>"));
  s = s.replace(/\\n{2,}/g, "<br><br>").replace(/\\n/g, "<br>");
  s = s.replace(/%%PH(\\d+)%%/g, (_, i) => ph[+i]);
  return s;
}
function typeset(el) {
  if (window.renderMathInElement) {
    try { renderMathInElement(el, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false }); } catch (_) {}
  }
}
function renderMd(bub, text) { bub.className = "bub md"; bub.innerHTML = md(text); }

const baselineMsgs = [], routedMsgs = [];
let cCost = 0, rCost = 0, cLat = 0, rLat = 0, turns = 0;
const newId = () => (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : "s-" + Date.now() + "-" + (turns);
let sessionId = newId();
let inFlight = false;

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
  if (!keepText) {
    if (d.error) slot.bub.textContent = "Error: " + d.error;
    else renderMd(slot.bub, d.answer || "(no text)");
  }
  const sx = d.searches ? ' · 🔎 ' + d.searches : "";
  if (routed) {
    slot.foot.innerHTML = '<span class="badge">' + short(d.model) + "</span> " + usd(d.cost_usd) +
      ' · <span class="mono">router ' + ms(d.router_ms) + " + model " + ms(d.model_ms) +
      " = " + ms(d.total_ms) + "</span>" + sx;
  } else {
    slot.foot.innerHTML = usd(d.cost_usd) + ' · <span class="mono">' + ms(d.total_ms) + "</span>" + sx;
  }
  typeset(slot.bub);  // render any LaTeX math once the answer is final
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
}

const hdrs = () => { const h = { "content-type": "application/json" }; if (KEY) h["x-api-key"] = KEY; return h; };
const hdrsBody = (side) => JSON.stringify({
  baseline: baselineMsgs, routed: routedMsgs, session_id: sessionId,
  policy: $("policy").value, max_tokens: parseInt($("maxtok").value, 10),
  effort: $("effort").value || null, search: $("search").checked, side,
});
const reqBody = () => hdrsBody(undefined);

async function fetchT(url, opts, ms) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try { return await fetch(url, { ...opts, signal: ctrl.signal }); }
  finally { clearTimeout(t); }
}

async function twoSidedTurn(bSlot, rSlot) {
  // Fire each side as its own request and render whichever finishes first.
  const res = {};
  async function one(name, slot, routed) {
    try {
      const r = await fetchT("/api/demo/chat", { method: "POST", headers: hdrs(), body: hdrsBody(name) }, 180000);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.status);
      res[name] = j;
      fill(slot, j, routed);  // show this side immediately, don't wait for the other
    } catch (e) {
      slot.bub.textContent = "Error: " + (e && e.name === "AbortError" ? "timed out" : (e && e.message ? e.message : e));
    }
  }
  await Promise.all([one("baseline", bSlot, false), one("routed", rSlot, true)]);
  if (res.baseline && res.routed) {
    baselineMsgs.push({ role: "assistant", content: res.baseline.answer || "" });
    routedMsgs.push({ role: "assistant", content: res.routed.answer || "" });
    updateTotals({ baseline: res.baseline, routed: res.routed });
  } else {
    baselineMsgs.pop(); routedMsgs.pop();  // a side failed — roll back the user turn
  }
}

async function streamTurn(bSlot, rSlot) {
  const slots = { baseline: bSlot, routed: rSlot };
  const acc = { baseline: "", routed: "" }, fin = {};
  const ctrl = new AbortController();
  let timer = setTimeout(() => ctrl.abort(), 150000);  // idle guard: abort if no data for a while
  const bump = () => { clearTimeout(timer); timer = setTimeout(() => ctrl.abort(), 150000); };
  try {
  const r = await fetch("/api/demo/stream", { method: "POST", headers: hdrs(), body: reqBody(), signal: ctrl.signal });
  if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || r.status); }
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bump();
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
        acc[ev.side] += ev.text; renderMd(slot.bub, acc[ev.side]);
        slot.pane.scrollTop = slot.pane.scrollHeight;
      } else if (ev.type === "tool") {
        slot.foot.textContent = "🔎 searching: " + (ev.query || "");
      } else if (ev.type === "done") {
        fin[ev.side] = ev;
        if (ev.error && !acc[ev.side]) slot.bub.textContent = "Error: " + ev.error;
        fill(slot, { ...ev, answer: acc[ev.side] || ev.answer }, ev.side === "routed", true);
      }
    }
  }
  baselineMsgs.push({ role: "assistant", content: acc.baseline });
  routedMsgs.push({ role: "assistant", content: acc.routed });
  if (fin.baseline && fin.routed) updateTotals({ baseline: fin.baseline, routed: fin.routed });
  } finally { clearTimeout(timer); }
}

async function send() {
  if (inFlight) return;  // one turn at a time — both sides must finish before the next
  const q = $("q").value.trim();
  if (!q) return;
  inFlight = true;
  $("q").value = ""; $("send").disabled = true; $("q").disabled = true;
  let bSlot = null, rSlot = null, added = false;
  try {
    baselineMsgs.push({ role: "user", content: q });
    routedMsgs.push({ role: "user", content: q });
    added = true;
    bubble("pane-base", "user", q); bubble("pane-routed", "user", q);
    bSlot = bubble("pane-base", "asst", "…");
    rSlot = bubble("pane-routed", "asst", "…");
    if ($("search").checked) await streamTurn(bSlot, rSlot);  // search uses the agent/tool loop
    else await twoSidedTurn(bSlot, rSlot);
  } catch (e) {
    const msg = (e && e.name === "AbortError") ? "timed out" : (e && e.message ? e.message : e);
    if (bSlot) bSlot.bub.textContent = "Error: " + msg;
    if (rSlot) rSlot.bub.textContent = "";
    if (added) { baselineMsgs.pop(); routedMsgs.pop(); }  // drop the user turn so the chat stays consistent
  } finally {
    inFlight = false;
    $("send").disabled = false; $("q").disabled = false; $("q").focus();
  }
}

function newChat() {
  baselineMsgs.length = 0; routedMsgs.length = 0;
  cCost = rCost = cLat = rLat = turns = 0;
  sessionId = newId();  // fresh cache-ledger scope, so policy comparisons start clean
  $("pane-base").innerHTML = ""; $("pane-routed").innerHTML = "";
  $("t-cost").textContent = "—"; $("t-lat").textContent = "—";
}

$("send").onclick = send;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
$("newchat").onclick = newChat;
document.querySelectorAll(".chip").forEach((c) => c.onclick = () => { $("q").value = c.dataset.q; send(); });
$("gear").href = "/dashboard" + (KEY ? "?key=" + encodeURIComponent(KEY) : "") + "#settings";
window.addEventListener("load", () => document.querySelectorAll(".bub.md").forEach(typeset));

async function loadKey() {
  try {
    const j = await (await fetch("/api/demo/search-key", { headers: hdrs() })).json();
    $("keystat").textContent = j.set ? ("key set " + j.hint) : "no key — search uses mock results";
  } catch (_) {}
}
$("savekey").onclick = async () => {
  const k = $("tavily").value.trim();
  if (!k) return;
  const r = await fetch("/api/demo/search-key", { method: "PUT", headers: hdrs(), body: JSON.stringify({ key: k }) });
  const j = await r.json();
  if (r.ok) { $("tavily").value = ""; $("keystat").textContent = "key set " + j.hint; }
  else $("keystat").textContent = "error: " + (j.error || r.status);
};
loadKey();
</script>
</body>
</html>
"""
