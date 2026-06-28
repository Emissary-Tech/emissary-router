from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Request
from starlette.responses import HTMLResponse, JSONResponse

from emissary_router.dashboard.routes import _make_auth_dependency


def build_demo_router(auth_key: str | None = None, streaming_default: bool = False) -> APIRouter:
    """Conference split-screen demo: default Sonnet vs the routed system, side by side.

    Mounted only when `demo.enabled`, behind the same auth as the dashboard (each
    comparison makes two real model calls). The compare handler reads the live
    `app.state.pipeline`, so it tracks config hot-reloads.
    """
    router = APIRouter(dependencies=[Depends(_make_auth_dependency(auth_key))])

    @router.get("/demo")
    async def demo_page() -> HTMLResponse:
        return HTMLResponse(demo_html(streaming_default))

    @router.post("/api/demo/compare")
    async def compare(request: Request, payload: dict = Body(...)) -> JSONResponse:
        query = (payload.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "empty query"}, status_code=400)
        if len(query) > 4000:
            return JSONResponse({"error": "query too long"}, status_code=400)

        max_tokens = _clamp_int(payload.get("max_tokens"), default=32000, lo=256, hi=64000)
        effort = payload.get("effort")
        effort = effort if effort in {"low", "medium", "high"} else None

        try:
            result = await request.app.state.pipeline.compare(
                query, max_tokens=max_tokens, effort=effort
            )
        except Exception as exc:  # surface upstream/key issues to the page instead of 500-ing silently
            return JSONResponse({"error": str(exc)}, status_code=502)
        return JSONResponse(result)

    return router


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
<title>Emissary Router — live comparison</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef; --muted:#8b93a7; --accent:#5b9cff; --good:#43c59e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:var(--muted); font-size:13px; }
  main { padding:24px; max-width:1000px; margin:0 auto; }
  .qrow { display:flex; gap:8px; margin-bottom:12px; }
  input#q { flex:1; background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:10px 12px; font:inherit; }
  button.primary { background:var(--accent); border:none; color:#fff; border-radius:8px; padding:0 18px; cursor:pointer; font:inherit; font-weight:500; }
  button.primary:disabled { opacity:.5; cursor:default; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:20px; }
  .chip { background:var(--panel); border:1px solid var(--line); color:var(--muted); border-radius:999px; padding:5px 12px; font:13px inherit; cursor:pointer; }
  .chip:hover { color:var(--fg); border-color:var(--accent); }
  .opts { display:flex; gap:20px; align-items:center; margin-bottom:18px; font-size:13px; color:var(--muted); }
  .opts label { display:flex; align-items:center; gap:6px; cursor:pointer; }
  .opts select { background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 6px; font:inherit; }
  .panes { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .pane { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; min-height:200px; display:flex; flex-direction:column; }
  .pane.routed { border-color:var(--accent); }
  .pane h2 { font-size:14px; font-weight:500; margin:0 0 10px; display:flex; align-items:center; gap:8px; }
  .badge { font-size:11px; border-radius:999px; padding:1px 9px; border:1px solid var(--line); color:var(--muted); }
  .badge.model { background:#1b2740; color:#9cc0ff; border-color:#27406a; }
  .answer { flex:1; white-space:pre-wrap; color:#d6dae6; font-size:14px; }
  .answer.placeholder { color:var(--muted); }
  .foot { display:flex; align-items:center; justify-content:space-between; border-top:1px solid var(--line); margin-top:14px; padding-top:10px; font-size:13px; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }
  .save { color:var(--good); font-weight:500; }
  .total { display:flex; align-items:center; justify-content:space-between; margin-top:16px; background:#12151c; border:1px solid var(--line); border-radius:8px; padding:10px 14px; font-size:13px; color:var(--muted); }
  .err { color:#ff6b6b; }
  .spin { color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>Emissary Router</h1>
  <span class="sub">type a query — default Sonnet vs routed, side by side</span>
</header>
<main>
  <div class="qrow">
    <input id="q" placeholder="Ask anything…" autocomplete="off" />
    <button class="primary" id="go">Compare</button>
  </div>
  <div class="chips">__CHIPS__</div>

  <div class="opts">
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
    <span class="sub">both sides use the same settings; converted per model</span>
  </div>

  <div class="panes">
    <div class="pane">
      <h2>Claude Sonnet <span class="badge">default</span></h2>
      <div class="answer placeholder" id="a-base">Answer appears here.</div>
      <div class="foot"><span class="mono" id="m-base"></span><span class="mono" id="c-base"></span></div>
    </div>
    <div class="pane routed">
      <h2>Emissary routed <span class="badge model" id="r-badge">—</span></h2>
      <div class="answer placeholder" id="a-routed">Answer appears here.</div>
      <div class="foot"><span class="mono" id="m-routed"></span><span id="c-routed"></span></div>
    </div>
  </div>

  <div class="total">
    <span id="t-count">no queries yet</span>
    <span id="t-savings"></span>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const KEY = new URLSearchParams(location.search).get("key");
const usd = (v) => "$" + Number(v || 0).toFixed(4);
let totBase = 0, totRouted = 0, count = 0;

async function compare(query) {
  $("go").disabled = true;
  $("a-base").className = "answer spin"; $("a-base").textContent = "thinking…";
  $("a-routed").className = "answer spin"; $("a-routed").textContent = "thinking…";
  $("m-base").textContent = ""; $("c-base").textContent = "";
  $("m-routed").textContent = ""; $("c-routed").textContent = ""; $("r-badge").textContent = "—";
  try {
    const headers = { "content-type": "application/json" };
    if (KEY) headers["x-api-key"] = KEY;
    const opts = { query, max_tokens: parseInt($("maxtok").value, 10), effort: $("effort").value || null };
    const r = await fetch("/api/demo/compare", { method: "POST", headers, body: JSON.stringify(opts) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    render(j);
  } catch (e) {
    $("a-base").className = "answer err"; $("a-base").textContent = "Error: " + e.message;
    $("a-routed").className = "answer"; $("a-routed").textContent = "";
  } finally {
    $("go").disabled = false;
  }
}

function side(prefix, d) {
  $("a-" + prefix).className = "answer";
  $("a-" + prefix).textContent = d.error ? ("Error: " + d.error) : (d.answer || "(no text)");
  $("m-" + prefix).textContent = d.model + " · " + Math.round(d.latency_ms) + "ms";
}

function render(j) {
  side("base", j.baseline);
  side("routed", j.routed);
  $("c-base").textContent = usd(j.baseline.cost_usd);
  $("r-badge").textContent = j.routed.model.replace("claude-", "").replace("-4.6", "").replace("-4.5", "");
  const escalated = j.routed.model === j.baseline_model;
  $("c-routed").innerHTML = '<span class="mono">' + usd(j.routed.cost_usd) + '</span>' +
    (escalated ? ' <span class="badge">escalated → sonnet</span>'
               : ' <span class="save">−' + j.savings_pct + '%</span>');
  totBase += j.baseline.cost_usd || 0; totRouted += j.routed.cost_usd || 0; count++;
  $("t-count").textContent = count + " quer" + (count === 1 ? "y" : "ies");
  const pct = totBase > 0 ? Math.round((totBase - totRouted) / totBase * 100) : 0;
  $("t-savings").innerHTML = "all-sonnet <span class='mono'>" + usd(totBase) + "</span> → routed <span class='mono'>" +
    usd(totRouted) + "</span> · <span class='save'>saved " + pct + "%</span>";
}

$("go").onclick = () => { const q = $("q").value.trim(); if (q) compare(q); };
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") $("go").click(); });
document.querySelectorAll(".chip").forEach((c) => c.onclick = () => { $("q").value = c.dataset.q; $("go").click(); });
</script>
</body>
</html>
"""
