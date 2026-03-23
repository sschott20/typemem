"""Frontend HTML for memory visualization."""

FRONTEND_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>typemem viz</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0d1117;
  --panel: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-secondary: #8b949e;
  --accent: #58a6ff;
  --glow: #f78166;
  --tier-m0: #484f58;
  --tier-m1: #8b949e;
  --tier-m2: #58a6ff;
  --tier-m3: #d2a8ff;
  --font-mono: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
  --font-ui: system-ui, -apple-system, sans-serif;
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-ui);
  font-size: 14px;
  line-height: 1.5;
  overflow: hidden;
}

.app { display: grid; grid-template-rows: auto 1fr; height: 100vh; }

.stats-bar {
  display: flex; align-items: center; gap: 20px;
  padding: 10px 20px; background: var(--panel);
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.stats-bar .title { font-family: var(--font-mono); font-size: 16px; font-weight: 700; color: var(--accent); margin-right: 12px; }
.stat { font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary); }
.stat .val { color: var(--text); font-weight: 600; }
.connection-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }
.connection-dot.connected { background: #3fb950; }
.connection-dot.disconnected { background: #f85149; }
.connection-label { font-size: 11px; color: var(--text-secondary); }

.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 0; overflow: hidden; }
.panel { display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid var(--border); }
.panel:last-child { border-right: none; }
.panel-header { padding: 12px 16px; background: var(--panel); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.panel-header h2 { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); margin-bottom: 10px; }
.panel-body { flex: 1; overflow-y: auto; padding: 0; }
.panel-body::-webkit-scrollbar { width: 6px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.controls { display: flex; gap: 8px; flex-wrap: wrap; }
.search-input {
  flex: 1; min-width: 150px; padding: 6px 10px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; color: var(--text);
  font-family: var(--font-mono); font-size: 12px; outline: none;
}
.search-input:focus { border-color: var(--accent); }
.search-input::placeholder { color: var(--tier-m0); }
.filter-select {
  padding: 6px 8px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 12px; outline: none; cursor: pointer;
}

.entry-row {
  padding: 8px 16px; border-bottom: 1px solid var(--border); cursor: pointer;
  transition: background 0.15s; font-family: var(--font-mono); font-size: 12px;
}
.entry-row:hover { background: rgba(88,166,255,0.06); }
.entry-row.expanded { background: rgba(88,166,255,0.04); }
.entry-summary { display: flex; align-items: center; gap: 10px; }
.entry-id { color: var(--text-secondary); font-size: 11px; flex-shrink: 0; width: 68px; }
.entry-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.entry-links { color: var(--text-secondary); font-size: 10px; flex-shrink: 0; }

.tier-badge {
  font-size: 10px; padding: 1px 8px; border-radius: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.3px; flex-shrink: 0;
}
.tier-badge.M0 { background: rgba(72,79,88,0.15); color: var(--tier-m0); }
.tier-badge.M1 { background: rgba(139,148,158,0.15); color: var(--tier-m1); }
.tier-badge.M2 { background: rgba(88,166,255,0.15); color: var(--tier-m2); }
.tier-badge.M3 { background: rgba(210,168,255,0.15); color: var(--tier-m3); }

.entry-time { color: var(--text-secondary); font-size: 11px; flex-shrink: 0; text-align: right; min-width: 60px; }
.entry-detail { max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }
.entry-row.expanded .entry-detail { max-height: 600px; }
.detail-content { margin-top: 10px; padding: 10px 12px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
.detail-field { margin-bottom: 6px; font-size: 12px; line-height: 1.6; }
.detail-field .label { color: var(--text-secondary); margin-right: 6px; }
.detail-field pre { margin-top: 4px; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px; overflow-x: auto; font-size: 11px; white-space: pre-wrap; word-break: break-word; }

@keyframes entry-glow { 0% { background: rgba(248,129,102,0.2); } 100% { background: transparent; } }
.entry-row.new-entry { animation: entry-glow 2s ease-out; }

.tab-group { display: flex; gap: 4px; }
.tab-btn {
  padding: 5px 14px; border: none; border-radius: 16px; background: transparent;
  color: var(--text-secondary); font-size: 12px; font-weight: 500; cursor: pointer;
}
.tab-btn:hover { background: rgba(88,166,255,0.1); color: var(--text); }
.tab-btn.active { background: var(--accent); color: #fff; }

.op-row { padding: 10px 16px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.15s; }
.op-row:hover { background: rgba(88,166,255,0.06); }
.op-row.expanded { background: rgba(88,166,255,0.04); }
.op-summary { display: flex; align-items: center; gap: 10px; font-size: 12px; }
.op-type-icon { flex-shrink: 0; width: 20px; text-align: center; }
.op-time { color: var(--text-secondary); font-family: var(--font-mono); font-size: 11px; flex-shrink: 0; }
.op-desc { flex: 1; font-family: var(--font-mono); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.op-detail { max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }
.op-row.expanded .op-detail { max-height: 1200px; }
.op-detail-content {
  margin-top: 10px; padding: 10px 12px; background: var(--bg); border-radius: 6px;
  border: 1px solid var(--border); font-family: var(--font-mono); font-size: 12px;
}
.op-detail-content .label { color: var(--text-secondary); }
.op-detail-content pre { margin-top: 4px; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px; overflow-x: auto; font-size: 11px; white-space: pre-wrap; word-break: break-word; }

.consol-badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; background: rgba(88,166,255,0.15); color: var(--accent); margin-left: 4px; }
.inject-badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; background: rgba(210,168,255,0.15); color: var(--tier-m3); margin-left: 4px; }
.empty-state { padding: 40px 20px; text-align: center; color: var(--text-secondary); font-size: 13px; }

@media (max-width: 900px) {
  .panels { grid-template-columns: 1fr; }
  .panel { border-right: none; border-bottom: 1px solid var(--border); }
}
</style>
</head>
<body>
<div class="app">
  <div class="stats-bar">
    <span class="title">typemem viz</span>
    <span class="stat">Total: <span class="val" id="stat-total">0</span></span>
    <span class="stat">M0: <span class="val" id="stat-M0">0</span></span>
    <span class="stat">M1: <span class="val" id="stat-M1">0</span></span>
    <span class="stat">M2: <span class="val" id="stat-M2">0</span></span>
    <span class="stat">M3: <span class="val" id="stat-M3">0</span></span>
    <span class="stat">Consol: <span class="val" id="stat-consol">0</span></span>
    <span class="stat">Injects: <span class="val" id="stat-injects">0</span></span>
    <span style="margin-left:auto; display:flex; align-items:center; gap:4px;">
      <span class="connection-dot disconnected" id="conn-dot"></span>
      <span class="connection-label" id="conn-label">Connecting...</span>
    </span>
  </div>

  <div class="panels">
    <div class="panel">
      <div class="panel-header">
        <h2>Memory Entries</h2>
        <div class="controls">
          <input type="text" class="search-input" id="search-input" placeholder="Search entries...">
          <select class="filter-select" id="tier-filter">
            <option value="all">All Tiers</option>
            <option value="M0">M0 (raw)</option>
            <option value="M1">M1 (observations)</option>
            <option value="M2">M2 (summaries)</option>
            <option value="M3">M3 (knowledge)</option>
          </select>
          <select class="filter-select" id="sort-select">
            <option value="newest">Newest</option>
            <option value="oldest">Oldest</option>
            <option value="text">By Text</option>
          </select>
        </div>
      </div>
      <div class="panel-body" id="entries-panel"></div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2>Operations</h2>
        <div class="tab-group" id="tab-group">
          <button class="tab-btn active" data-tab="all">All</button>
          <button class="tab-btn" data-tab="consolidation">Consolidations</button>
          <button class="tab-btn" data-tab="injection">Injections</button>
        </div>
      </div>
      <div class="panel-body" id="ops-panel"></div>
    </div>
  </div>
</div>

<script>
(function() {
  var entries = [], operations = [], stats = {};
  var currentTab = "all", searchText = "", filterTier = "all", sortBy = "newest";
  var expandedEntryId = null, expandedOpIdx = null, knownEntryIds = {};

  var entriesPanel = document.getElementById("entries-panel");
  var opsPanel = document.getElementById("ops-panel");

  function esc(text) {
    if (text == null) return "";
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(String(text)));
    return div.innerHTML;
  }
  function trunc(t, n) { return !t ? "" : t.length <= n ? t : t.slice(0, n) + "..."; }
  function timeAgo(ts) {
    if (!ts) return "";
    var d = Math.max(0, Date.now()/1000 - ts);
    if (d < 5) return "just now";
    if (d < 60) return Math.floor(d) + "s ago";
    if (d < 3600) return Math.floor(d/60) + "m ago";
    return Math.floor(d/3600) + "h ago";
  }
  function fmtTime(ts) { return ts ? new Date(ts*1000).toLocaleTimeString() : ""; }

  function renderStats() {
    document.getElementById("stat-total").textContent = stats.total || 0;
    ["M0","M1","M2","M3"].forEach(function(t) {
      var el = document.getElementById("stat-" + t);
      if (el) el.textContent = (stats.tiers && stats.tiers[t]) || 0;
    });
    document.getElementById("stat-consol").textContent = stats.consolidations_count || 0;
    document.getElementById("stat-injects").textContent = stats.injections_count || 0;
  }

  function renderEntries(newIds) {
    if (!newIds) newIds = {};
    var filtered = entries.filter(function(e) {
      if (filterTier !== "all" && e.tier !== filterTier) return false;
      if (searchText) {
        var q = searchText.toLowerCase();
        if ((e.text||"").toLowerCase().indexOf(q) === -1 && (e.id||"").toLowerCase().indexOf(q) === -1) return false;
      }
      return true;
    });
    filtered.sort(function(a, b) {
      if (sortBy === "newest") return (b.timestamp||0) - (a.timestamp||0);
      if (sortBy === "oldest") return (a.timestamp||0) - (b.timestamp||0);
      return (a.text||"").localeCompare(b.text||"");
    });
    if (!filtered.length) { entriesPanel.innerHTML = '<div class="empty-state">No memory entries</div>'; return; }

    var html = "";
    filtered.forEach(function(e) {
      var exp = expandedEntryId === e.id;
      var cls = "entry-row" + (exp ? " expanded" : "") + (newIds[e.id] ? " new-entry" : "");
      html += '<div class="' + cls + '" data-id="' + esc(e.id) + '">';
      html += '<div class="entry-summary">';
      html += '<span class="entry-id">' + esc(trunc(e.id, 8)) + '</span>';
      html += '<span class="entry-text">' + esc(trunc(e.text, 60)) + '</span>';
      html += '<span class="tier-badge ' + esc(e.tier) + '">' + esc(e.tier) + '</span>';
      if (e.links) html += '<span class="entry-links">' + e.links + ' links</span>';
      html += '<span class="entry-time">' + esc(timeAgo(e.timestamp)) + '</span>';
      html += '</div>';
      html += '<div class="entry-detail">';
      if (exp) {
        html += '<div class="detail-content">';
        html += '<div class="detail-field"><span class="label">ID:</span> ' + esc(e.id) + '</div>';
        html += '<div class="detail-field"><span class="label">Text:</span> ' + esc(e.text) + '</div>';
        html += '<div class="detail-field"><span class="label">Tier:</span> ' + esc(e.tier) + '</div>';
        html += '<div class="detail-field"><span class="label">Type:</span> ' + esc(e.memory_type) + '</div>';
        html += '<div class="detail-field"><span class="label">Time:</span> ' + esc(fmtTime(e.timestamp)) + '</div>';
        html += '<div class="detail-field"><span class="label">Links:</span> ' + (e.links || 0) + '</div>';
        if (e.keywords) html += '<div class="detail-field"><span class="label">Keywords:</span> ' + esc(e.keywords) + '</div>';
        if (e.source) html += '<div class="detail-field"><span class="label">Source:</span> ' + esc(e.source) + '</div>';
        html += '</div>';
      }
      html += '</div></div>';
    });
    entriesPanel.innerHTML = html;
  }

  function renderOperations() {
    var ops = operations.filter(function(o) {
      if (currentTab === "all") return true;
      return o.event_type === currentTab;
    });
    ops.sort(function(a, b) { return (b.timestamp||0) - (a.timestamp||0); });

    if (!ops.length) { opsPanel.innerHTML = '<div class="empty-state">No operations recorded</div>'; return; }

    var html = "";
    ops.forEach(function(op, i) {
      var exp = expandedOpIdx === i;
      var d = op.details || {};
      html += '<div class="op-row' + (exp ? " expanded" : "") + '" data-op-idx="' + i + '">';
      html += '<div class="op-summary">';
      if (op.event_type === "consolidation") {
        html += '<span class="op-type-icon" title="Consolidation">&#9670;</span>';
        html += '<span class="op-time">' + esc(fmtTime(op.timestamp)) + '</span>';
        html += '<span class="op-desc">' + esc(d.name || "consolidation");
        html += '<span class="consol-badge">' + (d.input_count||0) + ' in &rarr; ' + (d.output_count||0) + ' out</span>';
        html += '</span>';
      } else if (op.event_type === "injection") {
        html += '<span class="op-type-icon" title="Injection" style="color:var(--tier-m3);">&#9672;</span>';
        html += '<span class="op-time">' + esc(fmtTime(op.timestamp)) + '</span>';
        html += '<span class="op-desc">' + esc(trunc(d.query || "", 40));
        html += '<span class="inject-badge">' + (d.result_count||0) + ' results</span>';
        html += '</span>';
      } else {
        html += '<span class="op-type-icon">&#8226;</span>';
        html += '<span class="op-time">' + esc(fmtTime(op.timestamp)) + '</span>';
        html += '<span class="op-desc">' + esc(op.event_type) + '</span>';
      }
      html += '</div>';
      html += '<div class="op-detail">';
      if (exp) {
        html += '<div class="op-detail-content">';
        if (d.duration_ms != null) html += '<div class="detail-field"><span class="label">Duration:</span> ' + d.duration_ms + ' ms</div>';
        if (op.event_type === "consolidation") {
          if (d.name) html += '<div class="detail-field"><span class="label">Strategy:</span> ' + esc(d.name) + '</div>';
          if (d.inputs && d.inputs.length) {
            html += '<div class="detail-field"><span class="label">Inputs (' + d.inputs.length + '):</span></div>';
            d.inputs.forEach(function(inp) {
              html += '<div style="margin-left:12px;margin-bottom:4px;font-size:11px;">';
              html += '<span class="tier-badge ' + esc(inp.tier) + '" style="font-size:9px;margin-right:6px;">' + esc(inp.tier) + '</span>';
              html += esc(trunc(inp.text, 80));
              html += '</div>';
            });
          }
          if (d.outputs && d.outputs.length) {
            html += '<div class="detail-field" style="margin-top:8px;"><span class="label">Outputs (' + d.outputs.length + '):</span></div>';
            d.outputs.forEach(function(out) {
              html += '<div style="margin-left:12px;margin-bottom:4px;font-size:11px;">';
              html += '<span class="tier-badge ' + esc(out.tier) + '" style="font-size:9px;margin-right:6px;">' + esc(out.tier) + '</span>';
              html += esc(trunc(out.text, 120));
              html += '</div>';
            });
          }
        } else if (op.event_type === "injection") {
          if (d.stage) html += '<div class="detail-field"><span class="label">Stage:</span> ' + esc(d.stage) + '</div>';
          if (d.query) html += '<div class="detail-field"><span class="label">Query:</span> ' + esc(d.query) + '</div>';
          if (d.results && d.results.length) {
            html += '<div class="detail-field"><span class="label">Results (' + d.results.length + '):</span></div>';
            d.results.forEach(function(r) {
              html += '<div style="margin-left:12px;margin-bottom:4px;font-size:11px;">';
              html += '<span class="tier-badge ' + esc(r.tier) + '" style="font-size:9px;margin-right:6px;">' + esc(r.tier) + '</span>';
              html += '<span style="color:var(--accent);margin-right:6px;">' + (r.score != null ? r.score.toFixed(3) : '') + '</span>';
              html += esc(trunc(r.text, 80));
              html += '</div>';
            });
          }
          if (d.context) html += '<div class="detail-field"><span class="label">Context:</span><pre>' + esc(d.context) + '</pre></div>';
        } else {
          html += '<div class="detail-field"><span class="label">Details:</span><pre>' + esc(JSON.stringify(d, null, 2)) + '</pre></div>';
        }
        html += '</div>';
      }
      html += '</div></div>';
    });
    opsPanel.innerHTML = html;
  }

  // SSE
  function connectSSE() {
    var es = new EventSource("/api/stream");
    es.onopen = function() {
      document.getElementById("conn-dot").className = "connection-dot connected";
      document.getElementById("conn-label").textContent = "Live";
    };
    es.onmessage = function(ev) {
      var data; try { data = JSON.parse(ev.data); } catch(e) { return; }
      if (!data) return;
      operations.push(data);
      if (data.type === "add" || data.type === "consolidation") fetchEntries();
      fetchStats();
      renderOperations();
      opsPanel.scrollTop = 0;
    };
    es.onerror = function() {
      document.getElementById("conn-dot").className = "connection-dot disconnected";
      document.getElementById("conn-label").textContent = "Reconnecting...";
    };
  }

  // Events
  document.getElementById("search-input").addEventListener("input", function() { searchText = this.value; renderEntries(); });
  document.getElementById("tier-filter").addEventListener("change", function() { filterTier = this.value; renderEntries(); });
  document.getElementById("sort-select").addEventListener("change", function() { sortBy = this.value; renderEntries(); });
  document.getElementById("tab-group").addEventListener("click", function(ev) {
    var btn = ev.target.closest(".tab-btn");
    if (!btn) return;
    currentTab = btn.getAttribute("data-tab");
    this.querySelectorAll(".tab-btn").forEach(function(b) { b.classList.remove("active"); });
    btn.classList.add("active");
    expandedOpIdx = null;
    renderOperations();
  });
  entriesPanel.addEventListener("click", function(ev) {
    var row = ev.target.closest(".entry-row");
    if (!row) return;
    var id = row.getAttribute("data-id");
    expandedEntryId = expandedEntryId === id ? null : id;
    renderEntries();
  });
  opsPanel.addEventListener("click", function(ev) {
    var row = ev.target.closest(".op-row");
    if (!row) return;
    var idx = parseInt(row.getAttribute("data-op-idx"));
    expandedOpIdx = expandedOpIdx === idx ? null : idx;
    renderOperations();
  });

  function fetchJSON(url) { return fetch(url).then(function(r) { return r.json(); }); }
  function fetchEntries() {
    return fetchJSON("/api/entries").then(function(data) {
      var newIds = {};
      data.forEach(function(e) { if (!knownEntryIds[e.id]) newIds[e.id] = true; knownEntryIds[e.id] = true; });
      entries = data;
      renderEntries(newIds);
    });
  }
  function fetchStats() { return fetchJSON("/api/stats").then(function(d) { stats = d; renderStats(); }); }
  function fetchOps() {
    return fetchJSON("/api/events").then(function(data) {
      operations = data;
      renderOperations();
    });
  }

  // Init
  fetchJSON("/api/entries").then(function(data) {
    data.forEach(function(e) { knownEntryIds[e.id] = true; });
    entries = data; renderEntries();
  }).catch(function(){});
  fetchStats().catch(function(){});
  fetchOps().catch(function(){});
  connectSSE();
})();
</script>
</body>
</html>'''
