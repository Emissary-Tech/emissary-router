from __future__ import annotations

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Emissary Router</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef; --muted:#8b93a7; --accent:#5b9cff; --good:#43c59e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:16px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .tabs { display:flex; gap:4px; margin-left:auto; }
  .tab { padding:6px 14px; border-radius:8px; cursor:pointer; color:var(--muted); }
  .tab.active { background:var(--panel); color:var(--fg); }
  main { padding:24px; max-width:1100px; margin:0 auto; }
  .view { display:none; } .view.active { display:block; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:24px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
  .card .label { color:var(--muted); font-size:12px; }
  .card .value { font-size:24px; font-weight:600; margin-top:6px; }
  .card .value.good { color:var(--good); }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
  th { color:var(--muted); font-weight:500; font-size:12px; }
  tr:hover td { background:#1c2029; }
  .pill { display:inline-block; padding:1px 8px; border-radius:999px; background:#222836; color:var(--fg); font-size:12px; }
  .muted { color:var(--muted); }
  .bar { height:10px; background:var(--accent); border-radius:5px; }
  button.del { background:none; border:1px solid var(--line); color:var(--muted); border-radius:6px; cursor:pointer; padding:2px 8px; }
  button.del:hover { color:#ff6b6b; border-color:#ff6b6b; }
  button.iconbtn { background:none; border:1px solid var(--line); color:var(--muted); border-radius:8px; cursor:pointer; padding:5px 12px; font:inherit; }
  button.iconbtn:hover { color:var(--fg); border-color:var(--accent); }
  .note { color:var(--muted); font-size:12px; margin-top:8px; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
  select, input[type=number] { background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:6px 8px; font:inherit; }
  input[type=checkbox] { width:16px; height:16px; accent-color:var(--accent); }
  .set-row { display:flex; align-items:center; gap:10px; padding:6px 0; }
  button.primary { background:var(--accent); border:none; color:#fff; border-radius:8px; padding:8px 16px; cursor:pointer; font:inherit; }
</style>
</head>
<body>
<header>
  <h1>Emissary Router</h1>
  <div class="tabs">
    <div class="tab active" data-view="savings">Savings</div>
    <div class="tab" data-view="requests">Requests</div>
    <div class="tab" data-view="sessions">Sessions</div>
    <div class="tab" data-view="settings">Settings</div>
  </div>
  <button id="refresh" class="iconbtn" title="Refresh">↻ Refresh</button>
</header>
<main>
  <section id="savings" class="view active"></section>
  <section id="requests" class="view"></section>
  <section id="sessions" class="view"></section>
  <section id="settings" class="view"></section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const KEY = new URLSearchParams(location.search).get("key");
const usd = (v) => (v == null ? "-" : "$" + Number(v).toFixed(4));
const tokfmt = (v) => (v == null ? "0" : Number(v).toLocaleString());
const when = (ts) => new Date(ts * 1000).toLocaleString();
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (KEY) headers["x-api-key"] = KEY;
  const r = await fetch(path, { ...opts, headers });
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}

async function renderSavings() {
  const s = await api("/api/summary");
  const saved = s.baseline_available
    ? `<div class="card"><div class="label">Estimated saved vs all-${esc(s.baseline_model)}</div>
         <div class="value good">${usd(s.savings_usd)} (${s.savings_pct}%)</div></div>`
    : `<div class="card"><div class="label">Savings</div><div class="value muted">baseline_unavailable</div>
         <div class="note">no price for ${esc(s.baseline_model)}</div></div>`;
  const maxN = Math.max(1, ...s.by_model.map(m => m.n));
  const bars = s.by_model.map(m => `
    <tr><td>${esc(m.served_model)}</td><td>${m.n}</td><td>${usd(m.cost_usd)}</td>
      <td style="width:40%"><div class="bar" style="width:${(m.n / maxN * 100).toFixed(0)}%"></div></td></tr>`).join("");
  $("savings").innerHTML = `
    <div class="cards">
      <div class="card"><div class="label">Total calls</div><div class="value">${s.total_events}</div></div>
      <div class="card"><div class="label">Actual spend</div><div class="value">${usd(s.total_cost_usd)}</div></div>
      <div class="card"><div class="label">All-${esc(s.baseline_model)} estimate</div><div class="value">${usd(s.baseline_cost_usd)}</div></div>
      ${saved}
    </div>
    <table><thead><tr><th>Model</th><th>Calls</th><th>Cost</th><th>Share</th></tr></thead><tbody>${bars}</tbody></table>
    <div class="note">Savings is an estimate: baseline prices applied to each call's actual token counts.</div>`;
}

async function renderRequests() {
  const { events } = await api("/api/events?limit=300");
  if (!events.length) { $("requests").innerHTML = '<div class="empty">No requests yet.</div>'; return; }
  const st = (e) => {
    if (e.http_status == null) return '<td class="muted">-</td>';
    const bad = e.http_status >= 400;
    return `<td style="color:${bad ? "#ff6b6b" : "var(--good)"}">${e.http_status}</td>`;
  };
  const rows = events.map(e => `
    <tr>
      <td class="muted">${when(e.ts)}</td>
      <td><span class="pill">${esc(e.served_model)}</span></td>
      ${st(e)}
      <td class="muted">${esc(e.requested_model) || "-"}</td>
      <td>${esc(e.provider)}</td>
      <td>${esc(e.call_kind)}</td>
      <td>${usd(e.cost_usd)}</td>
      <td class="muted">${e.cache_read_tokens > 0 ? tokfmt(e.cache_read_tokens) + " read" : (e.cache_creation_tokens > 0 ? tokfmt(e.cache_creation_tokens) + " write" : "-")}</td>
      <td class="muted">${tokfmt(e.input_tokens + e.cache_read_tokens + e.cache_creation_tokens)}/${tokfmt(e.output_tokens)}</td>
      <td><button class="del" data-id="${esc(e.id)}">delete</button></td>
    </tr>`).join("");
  $("requests").innerHTML = `
    <table><thead><tr><th>Time</th><th>Served</th><th>Status</th><th>Requested</th><th>Provider</th><th>Kind</th>
      <th>Cost</th><th>Cached</th><th>Prompt/Out tok</th><th></th></tr></thead><tbody>${rows}</tbody></table>
    <div class="note">Prompt = full input incl. cached tokens (matches the provider's view); Cached shows the read/written cache.</div>`;
  $("requests").querySelectorAll("button.del").forEach(b => b.onclick = async () => {
    await api("/api/events/" + encodeURIComponent(b.dataset.id), { method: "DELETE" });
    renderRequests(); renderSavings();
  });
}

async function renderSessions() {
  const { sessions } = await api("/api/sessions?limit=300");
  if (!sessions.length) { $("sessions").innerHTML = '<div class="empty">No sessions yet.</div>'; return; }
  const rows = sessions.map(s => {
    const models = Object.entries(s.models).map(([m, n]) => `<span class="pill">${esc(m)}×${n}</span>`).join(" ");
    return `<tr>
      <td class="muted">${when(s.last_ts)}</td>
      <td class="muted" title="${esc(s.session_id)}">${esc((s.session_id || "?").slice(0, 8))}</td>
      <td>${s.n_calls} <span class="muted">(${s.n_main} main / ${s.n_background} bg)</span></td>
      <td>${models}</td>
      <td>${usd(s.cost_usd)}</td>
      <td><button class="del" data-session="${esc(s.session_id)}">delete</button></td>
    </tr>`;
  }).join("");
  $("sessions").innerHTML = `
    <table><thead><tr><th>Last activity</th><th>Session</th><th>Calls</th><th>Models</th><th>Cost</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table>
    <div class="note">Each row is one Claude Code session — all of its calls grouped together.</div>`;
  $("sessions").querySelectorAll("button.del").forEach(b => b.onclick = async () => {
    if (!b.dataset.session) return;
    await api("/api/sessions/" + encodeURIComponent(b.dataset.session), { method: "DELETE" });
    renderSessions(); renderSavings();
  });
}

async function renderSettings() {
  const cfg = await api("/api/config");
  const toggles = cfg.models.map(m => {
    const prov = m.providers.length > 1
      ? `<select data-provider="${esc(m.name)}" style="margin-left:auto">${m.providers.map(p => `<option ${p === m.provider ? "selected" : ""}>${esc(p)}</option>`).join("")}</select>`
      : `<span class="muted" style="margin-left:auto">${esc(m.provider)}</span>`;
    return `<label class="set-row">
      <input type="checkbox" data-model="${esc(m.name)}" ${m.enabled ? "checked" : ""}>
      <span>${esc(m.name)}</span>
      <span class="muted">$${m.cost_score.toFixed(2)}/Mtok</span>
      ${prov}
    </label>`;
  }).join("");
  $("settings").innerHTML = `
    <div class="cards" style="grid-template-columns:1fr 1fr">
      <div class="card"><div class="label">Enabled models</div><div id="toggles">${toggles}</div></div>
      <div class="card">
        <div class="label">Default model (stay here unless confident)</div>
        <div class="set-row"><select id="default-select"></select></div>
        <div class="label" style="margin-top:14px">Confidence — deviate to a cheaper model when p ≥ this</div>
        <div class="set-row"><input type="number" id="confidence" min="0" max="1" step="0.05" value="${cfg.confidence}"></div>
      </div>
    </div>
    <button class="primary" id="save-config">Save</button>
    <span id="save-msg" class="note" style="margin-left:10px"></span>
    <div class="note">Changes are written to your config file and applied to the running gateway.</div>`;
  const rebuildDefault = () => {
    const enabled = [...document.querySelectorAll("#settings [data-model]")].filter(c => c.checked).map(c => c.dataset.model);
    const cur = $("default-select").value || cfg.default;
    $("default-select").innerHTML = enabled.length
      ? enabled.map(n => `<option ${n === cur ? "selected" : ""}>${esc(n)}</option>`).join("")
      : `<option value="">(enable a model)</option>`;
  };
  document.querySelectorAll("#settings [data-model]").forEach(c => c.onchange = rebuildDefault);
  rebuildDefault();
  $("save-config").onclick = async () => {
    const models = {};
    document.querySelectorAll("#settings [data-model]").forEach(c => {
      const sel = document.querySelector(`#settings [data-provider="${c.dataset.model}"]`);
      models[c.dataset.model] = { enabled: c.checked, provider: sel ? sel.value : null };
    });
    const headers = { "content-type": "application/json" };
    if (KEY) headers["x-api-key"] = KEY;
    const body = JSON.stringify({ models, default: $("default-select").value, confidence: parseFloat($("confidence").value) });
    const r = await fetch("/api/config", { method: "PUT", headers, body });
    const j = await r.json().catch(() => ({}));
    const msg = $("save-msg");
    if (r.ok) {
      msg.textContent = j.restart_required ? "Saved — run `er restart` to apply." : "Saved and applied.";
      msg.style.color = "var(--good)";
      renderSavings();
    } else {
      msg.textContent = "Error: " + (j.error || r.status);
      msg.style.color = "#ff6b6b";
    }
  };
}

const renderers = { savings: renderSavings, requests: renderRequests, sessions: renderSessions, settings: renderSettings };
document.querySelectorAll(".tab").forEach(tab => tab.onclick = () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  tab.classList.add("active");
  const view = tab.dataset.view;
  $(view).classList.add("active");
  renderers[view]();
});
document.getElementById("refresh").onclick = () => {
  const active = document.querySelector(".tab.active");
  if (active) renderers[active.dataset.view]();
};
renderSavings();
</script>
</body>
</html>
"""


def dashboard_html() -> str:
    return _PAGE
