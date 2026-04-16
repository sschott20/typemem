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

.app { display: grid; grid-template-rows: auto auto auto 1fr; height: 100vh; }

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

.stats-toggle-btn {
  padding: 4px 12px; border: 1px solid var(--border); border-radius: 6px;
  background: transparent; color: var(--text-secondary); font-size: 11px;
  cursor: pointer; font-family: var(--font-mono);
}
.stats-toggle-btn:hover { border-color: var(--accent); color: var(--text); }

.stats-dashboard {
  display: none; padding: 16px 20px; background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.stats-dashboard.open { display: block; }
.live-panel {
  padding: 8px 20px; background: var(--panel);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono); font-size: 12px;
  display: flex; align-items: center; gap: 12px;
}
.live-indicator {
  width: 8px; height: 8px; border-radius: 50%;
  background: #3fb950; animation: live-pulse 2s infinite;
  flex-shrink: 0;
}
@keyframes live-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.live-label { color: var(--text-secondary); font-size: 11px; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.live-content { color: var(--text); white-space: pre; overflow-x: auto; flex: 1; }
.live-panel.empty { display: none; }
.stats-charts { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.chart-section h3 {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px; color: var(--text-secondary); margin-bottom: 10px;
}
.bar-chart-h { display: flex; flex-direction: column; gap: 4px; }
.bar-row { display: flex; align-items: center; gap: 8px; font-family: var(--font-mono); font-size: 11px; }
.bar-label { width: 80px; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-secondary); flex-shrink: 0; }
.bar-track { flex: 1; height: 14px; background: rgba(88,166,255,0.06); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
.bar-value { width: 36px; font-size: 10px; color: var(--text-secondary); flex-shrink: 0; }
.timeline-chart { display: flex; align-items: flex-end; gap: 1px; height: 60px; }
.timeline-bar {
  flex: 1; background: var(--accent); border-radius: 1px 1px 0 0;
  min-width: 2px; transition: height 0.3s; opacity: 0.7;
}
.timeline-bar:hover { opacity: 1; }

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
.search-input.semantic-active { border-color: var(--glow); box-shadow: 0 0 0 1px var(--glow); }
.filter-select {
  padding: 6px 8px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 12px; outline: none; cursor: pointer;
}

.col-headers {
  display: flex; align-items: center; gap: 10px; padding: 6px 16px;
  border-bottom: 1px solid var(--border); background: var(--panel);
  font-family: var(--font-mono); font-size: 11px; flex-shrink: 0;
}
.col-header {
  cursor: pointer; color: var(--text-secondary); user-select: none;
  display: flex; align-items: center; gap: 3px;
}
.col-header:hover { color: var(--text); }
.col-header.active { color: var(--accent); }
.col-header .arrow { font-size: 9px; }
.col-tier { width: 44px; flex-shrink: 0; }
.col-time { width: 60px; flex-shrink: 0; text-align: right; }
.col-source { width: 80px; flex-shrink: 0; }
.col-text { flex: 1; }

.entry-row {
  padding: 8px 16px; border-bottom: 1px solid var(--border); cursor: pointer;
  transition: background 0.15s; font-family: var(--font-mono); font-size: 12px;
}
.entry-row:hover { background: rgba(88,166,255,0.06); }
.entry-row.expanded { background: rgba(88,166,255,0.04); }
.entry-summary { display: flex; align-items: center; gap: 10px; }
.entry-id { color: var(--text-secondary); font-size: 11px; flex-shrink: 0; width: 68px; }
.entry-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.entry-source { color: var(--text-secondary); font-size: 11px; flex-shrink: 0; width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.entry-links { color: var(--text-secondary); font-size: 10px; flex-shrink: 0; }
.entry-distance { color: var(--glow); font-size: 10px; flex-shrink: 0; font-weight: 600; }

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
.entry-row.expanded .entry-detail { max-height: 800px; }
.detail-content { margin-top: 10px; padding: 10px 12px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
.detail-field { margin-bottom: 6px; font-size: 12px; line-height: 1.6; }
.detail-field .label { color: var(--text-secondary); margin-right: 6px; }
.detail-field pre { margin-top: 4px; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px; overflow-x: auto; font-size: 11px; white-space: pre-wrap; word-break: break-word; }
.detail-links { margin-top: 8px; }
.detail-links .link-item {
  display: flex; align-items: center; gap: 6px; padding: 4px 0;
  font-size: 11px; color: var(--text-secondary);
}
.detail-links .link-item .tier-badge { font-size: 9px; }

@keyframes entry-glow { 0% { background: rgba(248,129,102,0.2); } 100% { background: transparent; } }
.entry-row.new-entry { animation: entry-glow 2s ease-out; }

.pagination {
  display: flex; align-items: center; justify-content: center; gap: 12px;
  padding: 8px 16px; border-top: 1px solid var(--border); background: var(--panel);
  font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary); flex-shrink: 0;
}
.page-btn {
  padding: 4px 10px; border: 1px solid var(--border); border-radius: 4px;
  background: transparent; color: var(--text-secondary); font-size: 11px;
  cursor: pointer; font-family: var(--font-mono);
}
.page-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--text); }
.page-btn:disabled { opacity: 0.3; cursor: default; }

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
  .stats-charts { grid-template-columns: 1fr; }
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
    <button class="stats-toggle-btn" id="stats-toggle">Stats &#9660;</button>
    <span style="margin-left:auto; display:flex; align-items:center; gap:12px;">
      <span style="display:flex; align-items:center; gap:4px;">
        <select class="filter-select" id="db-select" style="max-width:180px;font-size:11px;"></select>
        <input type="text" class="search-input" id="db-path-input" placeholder="Custom path..." style="width:200px;font-size:11px;display:none;">
        <button class="page-btn" id="db-switch-btn" style="font-size:11px;display:none;">Load</button>
        <button class="page-btn" id="db-clear-btn" style="font-size:11px;color:#f85149;border-color:#f85149;">Clear DB</button>
      </span>
      <span style="display:flex; align-items:center; gap:4px;">
        <span class="connection-dot disconnected" id="conn-dot"></span>
        <span class="connection-label" id="conn-label">Connecting...</span>
      </span>
    </span>
  </div>

  <div class="stats-dashboard" id="stats-dashboard">
    <div class="stats-charts">
      <div class="chart-section">
        <h3>Keyword Frequency</h3>
        <div class="bar-chart-h" id="keyword-chart"></div>
      </div>
      <div class="chart-section">
        <h3>Timeline</h3>
        <div class="timeline-chart" id="timeline-chart"></div>
      </div>
    </div>
  </div>

  <div class="live-panel empty" id="live-panel">
    <span class="live-indicator"></span>
    <span class="live-label">System Observation</span>
    <span class="live-content" id="live-content"></span>
  </div>

  <div class="panels">
    <div class="panel">
      <div class="panel-header">
        <h2>Memory Entries</h2>
        <div class="controls">
          <input type="text" class="search-input" id="search-input" placeholder="Search (Enter for semantic)...">
          <select class="filter-select" id="tier-filter">
            <option value="all">All Tiers</option>
            <option value="M0">M0 (raw)</option>
            <option value="M1">M1 (observations)</option>
            <option value="M2">M2 (summaries)</option>
            <option value="M3">M3 (knowledge)</option>
          </select>
          <select class="filter-select" id="type-filter">
            <option value="all">All Types</option>
            <option value="observation">observation</option>
            <option value="summary">summary</option>
            <option value="instruction">instruction</option>
            <option value="action">action</option>
            <option value="lesson">lesson</option>
          </select>
          <select class="filter-select" id="source-filter">
            <option value="all">All Sources</option>
          </select>
          <select class="filter-select" id="time-filter">
            <option value="all">All Time</option>
            <option value="300">Last 5m</option>
            <option value="900">Last 15m</option>
            <option value="3600">Last 1h</option>
            <option value="21600">Last 6h</option>
            <option value="86400">Last 24h</option>
          </select>
        </div>
      </div>
      <div class="col-headers" id="col-headers">
        <span class="col-header active col-tier" data-sort="tier">Tier</span>
        <span class="col-header col-text" style="cursor:default;">Text</span>
        <span class="col-header col-source" data-sort="source">Source</span>
        <span class="col-header active col-time" data-sort="newest">Time <span class="arrow">&#9660;</span></span>
      </div>
      <div class="panel-body" id="entries-panel"></div>
      <div class="pagination" id="pagination">
        <button class="page-btn" id="page-prev" disabled>&#8249; Prev</button>
        <span id="page-info">Page 1 of 1 (0 total)</span>
        <button class="page-btn" id="page-next" disabled>Next &#8250;</button>
      </div>
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
  var currentTab = "all";
  var filterTier = "all", filterType = "all", filterSource = "all", filterTime = "all";
  var sortCol = "newest", sortAsc = false;
  var expandedEntryId = null, expandedOpIdx = null, knownEntryIds = {};
  var currentPage = 0, pageLimit = 100, totalEntries = 0;
  var semanticMode = false, semanticResults = null;
  var linksCache = {};
  var statsOpen = false;

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
  function fetchJSON(url) { return fetch(url).then(function(r) { return r.json(); }); }

  // Stats bar
  function renderStats() {
    document.getElementById("stat-total").textContent = stats.total || 0;
    ["M0","M1","M2","M3"].forEach(function(t) {
      var el = document.getElementById("stat-" + t);
      if (el) el.textContent = (stats.tiers && stats.tiers[t]) || 0;
    });
    document.getElementById("stat-consol").textContent = stats.consolidations_count || 0;
    document.getElementById("stat-injects").textContent = stats.injections_count || 0;
    updateSourceFilter();
    renderCharts();
  }

  // Source filter dynamic population
  function updateSourceFilter() {
    var sel = document.getElementById("source-filter");
    var sc = (stats.source_counts) || {};
    var current = filterSource;
    var opts = '<option value="all">All Sources</option>';
    Object.keys(sc).sort().forEach(function(s) {
      opts += '<option value="' + esc(s) + '"' + (current === s ? ' selected' : '') + '>' + esc(s) + ' (' + sc[s] + ')</option>';
    });
    sel.innerHTML = opts;
  }

  // Charts
  function renderCharts() {
    var kwEl = document.getElementById("keyword-chart");
    var tlEl = document.getElementById("timeline-chart");
    // Tag frequency (renamed from keywords)
    var tags = stats.tags || stats.keywords || [];
    var kwPairs = Array.isArray(tags)
      ? tags.map(function(t) { return {k:t.tag || t.keyword, v:t.count}; })
      : Object.keys(tags).map(function(k) { return {k:k, v:tags[k]}; });
    kwPairs.sort(function(a,b) { return b.v - a.v; });
    kwPairs = kwPairs.slice(0, 10);
    var maxKw = kwPairs.length ? kwPairs[0].v : 1;
    var kwHtml = "";
    kwPairs.forEach(function(p) {
      var pct = Math.round((p.v / maxKw) * 100);
      kwHtml += '<div class="bar-row"><span class="bar-label" title="' + esc(p.k) + '">' + esc(p.k) + '</span>';
      kwHtml += '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%"></div></div>';
      kwHtml += '<span class="bar-value">' + p.v + '</span></div>';
    });
    kwEl.innerHTML = kwHtml || '<div style="color:var(--text-secondary);font-size:11px;">No keyword data</div>';

    // Timeline
    var tl = stats.timeline || [];
    if (!tl.length) { tlEl.innerHTML = '<div style="color:var(--text-secondary);font-size:11px;align-self:center;">No timeline data</div>'; return; }
    var maxTl = 1;
    tl.forEach(function(v) { if (v > maxTl) maxTl = v; });
    var tlHtml = "";
    tl.forEach(function(v) {
      var h = Math.max(2, Math.round((v / maxTl) * 60));
      tlHtml += '<div class="timeline-bar" style="height:' + h + 'px" title="' + v + ' items"></div>';
    });
    tlEl.innerHTML = tlHtml;
  }

  // Stats toggle
  document.getElementById("stats-toggle").addEventListener("click", function() {
    statsOpen = !statsOpen;
    document.getElementById("stats-dashboard").classList.toggle("open", statsOpen);
    this.innerHTML = "Stats " + (statsOpen ? "&#9650;" : "&#9660;");
  });

  // Column headers sort
  function updateColHeaders() {
    document.querySelectorAll("#col-headers .col-header").forEach(function(h) {
      var s = h.getAttribute("data-sort");
      var isActive = (s === sortCol) || (s === "newest" && (sortCol === "newest" || sortCol === "oldest"));
      h.classList.toggle("active", isActive);
      var arrow = h.querySelector(".arrow");
      if (arrow) arrow.remove();
      if (isActive) {
        var ar = document.createElement("span");
        ar.className = "arrow";
        ar.innerHTML = sortAsc ? "&#9650;" : "&#9660;";
        h.appendChild(ar);
      }
    });
  }
  document.getElementById("col-headers").addEventListener("click", function(ev) {
    var h = ev.target.closest(".col-header");
    if (!h) return;
    var s = h.getAttribute("data-sort");
    if (s === "newest") {
      if (sortCol === "newest") { sortCol = "oldest"; sortAsc = true; }
      else if (sortCol === "oldest") { sortCol = "newest"; sortAsc = false; }
      else { sortCol = "newest"; sortAsc = false; }
    } else {
      if (sortCol === s) { sortAsc = !sortAsc; }
      else { sortCol = s; sortAsc = true; }
    }
    updateColHeaders();
    currentPage = 0;
    if (semanticMode) return;
    fetchEntries();
  });

  // Entries rendering
  function renderEntries(newIds) {
    if (!newIds) newIds = {};
    var list = semanticMode ? (semanticResults || []) : entries;
    if (!list.length) {
      entriesPanel.innerHTML = '<div class="empty-state">' + (semanticMode ? "No search results" : "No memory entries") + '</div>';
      return;
    }
    var html = "";
    list.forEach(function(e) {
      var exp = expandedEntryId === e.id;
      var cls = "entry-row" + (exp ? " expanded" : "") + (newIds[e.id] ? " new-entry" : "");
      html += '<div class="' + cls + '" data-id="' + esc(e.id) + '">';
      html += '<div class="entry-summary">';
      html += '<span class="tier-badge ' + esc(e.tier) + '">' + esc(e.tier) + '</span>';
      html += '<span class="entry-text">' + esc(trunc(e.text, 60)) + '</span>';
      html += '<span class="entry-source">' + esc(e.source || "") + '</span>';
      if (semanticMode && e.distance != null) html += '<span class="entry-distance">' + e.distance.toFixed(3) + '</span>';
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
        if (e.tags && e.tags.length) html += '<div class="detail-field"><span class="label">Tags:</span> ' + esc(e.tags.join(", ")) + '</div>';
        if (e.source) html += '<div class="detail-field"><span class="label">Source:</span> ' + esc(e.source) + '</div>';
        if (semanticMode && e.distance != null) html += '<div class="detail-field"><span class="label">Distance:</span> ' + e.distance.toFixed(4) + '</div>';
        html += '<div class="detail-links" id="links-' + esc(e.id) + '"></div>';
        html += '</div>';
      }
      html += '</div></div>';
    });
    entriesPanel.innerHTML = html;
    // Fetch links for expanded entry
    if (expandedEntryId) {
      var expandedEntry = list.find(function(e) { return e.id === expandedEntryId; });
      if (expandedEntry && expandedEntry.links > 0) fetchLinks(expandedEntryId);
    }
  }

  function fetchLinks(id) {
    if (linksCache[id]) { renderLinks(id, linksCache[id]); return; }
    fetchJSON("/api/links/" + encodeURIComponent(id)).then(function(data) {
      linksCache[id] = data;
      renderLinks(id, data);
    }).catch(function(){});
  }
  function renderLinks(id, links) {
    var el = document.getElementById("links-" + id);
    if (!el || !links.length) return;
    var html = '<div class="detail-field" style="margin-top:8px;"><span class="label">Linked items (' + links.length + '):</span></div>';
    links.forEach(function(l) {
      html += '<div class="link-item">';
      html += '<span class="tier-badge ' + esc(l.tier) + '" style="font-size:9px;">' + esc(l.tier) + '</span>';
      html += '<span>' + esc(trunc(l.text, 80)) + '</span>';
      html += '</div>';
    });
    el.innerHTML = html;
  }

  // Pagination
  function renderPagination() {
    var totalPages = Math.max(1, Math.ceil(totalEntries / pageLimit));
    var page = currentPage + 1;
    document.getElementById("page-info").textContent = "Page " + page + " of " + totalPages + " (" + totalEntries + " total)";
    document.getElementById("page-prev").disabled = currentPage <= 0;
    document.getElementById("page-next").disabled = page >= totalPages;
  }
  document.getElementById("page-prev").addEventListener("click", function() {
    if (currentPage > 0) { currentPage--; fetchEntries(); }
  });
  document.getElementById("page-next").addEventListener("click", function() {
    var totalPages = Math.ceil(totalEntries / pageLimit);
    if (currentPage + 1 < totalPages) { currentPage++; fetchEntries(); }
  });

  // Fetch entries with filters and pagination
  function fetchEntries() {
    var params = [];
    if (filterTier !== "all") params.push("tier=" + encodeURIComponent(filterTier));
    if (filterType !== "all") params.push("type=" + encodeURIComponent(filterType));
    if (filterSource !== "all") params.push("source=" + encodeURIComponent(filterSource));
    if (filterTime !== "all") {
      var since = Math.floor(Date.now()/1000) - parseInt(filterTime);
      params.push("since=" + since);
    }
    var apiSort = sortCol;
    if (sortCol === "oldest") apiSort = "oldest";
    else if (sortCol === "newest") apiSort = "newest";
    else apiSort = sortCol + (sortAsc ? "_asc" : "_desc");
    params.push("sort=" + apiSort);
    params.push("offset=" + (currentPage * pageLimit));
    params.push("limit=" + pageLimit);
    var url = "/api/entries?" + params.join("&");
    return fetchJSON(url).then(function(data) {
      var items, total;
      if (Array.isArray(data)) {
        items = data; total = data.length;
      } else {
        items = data.items || []; total = data.total || items.length;
      }
      var newIds = {};
      items.forEach(function(e) { if (!knownEntryIds[e.id]) newIds[e.id] = true; knownEntryIds[e.id] = true; });
      entries = items;
      totalEntries = total;
      renderEntries(newIds);
      renderPagination();
    }).catch(function(){});
  }

  // Operations rendering
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

  // Live sources polling
  async function pollLiveSources() {
    try {
      const res = await fetch('/api/live_sources');
      const data = await res.json();
      const panel = document.getElementById('live-panel');
      const content = document.getElementById('live-content');
      const values = Object.values(data);
      if (values.length > 0) {
        content.textContent = values.join('\n');
        panel.classList.remove('empty');
      } else {
        panel.classList.add('empty');
      }
    } catch (e) {}
  }
  setInterval(pollLiveSources, 1000);
  pollLiveSources();

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
      if (data.type === "add" || data.type === "consolidation") {
        if (!semanticMode) fetchEntries();
      }
      fetchStats();
      renderOperations();
      opsPanel.scrollTop = 0;
    };
    es.onerror = function() {
      document.getElementById("conn-dot").className = "connection-dot disconnected";
      document.getElementById("conn-label").textContent = "Reconnecting...";
    };
  }

  // Search - semantic on Enter, text filter on input
  var searchInput = document.getElementById("search-input");
  searchInput.addEventListener("keydown", function(ev) {
    if (ev.key === "Enter" && this.value.trim()) {
      ev.preventDefault();
      semanticMode = true;
      this.classList.add("semantic-active");
      fetchJSON("/api/search?q=" + encodeURIComponent(this.value.trim())).then(function(data) {
        semanticResults = data.items || data;
        totalEntries = semanticResults.length;
        currentPage = 0;
        renderEntries();
        renderPagination();
      }).catch(function() {
        semanticResults = [];
        renderEntries();
      });
    } else if (ev.key === "Escape") {
      exitSemanticMode();
    }
  });
  searchInput.addEventListener("input", function() {
    if (!this.value && semanticMode) {
      exitSemanticMode();
    }
  });
  function exitSemanticMode() {
    semanticMode = false;
    semanticResults = null;
    searchInput.classList.remove("semantic-active");
    searchInput.value = "";
    currentPage = 0;
    fetchEntries();
  }

  // Filter events
  document.getElementById("tier-filter").addEventListener("change", function() {
    filterTier = this.value; currentPage = 0;
    if (!semanticMode) fetchEntries();
  });
  document.getElementById("type-filter").addEventListener("change", function() {
    filterType = this.value; currentPage = 0;
    if (!semanticMode) fetchEntries();
  });
  document.getElementById("source-filter").addEventListener("change", function() {
    filterSource = this.value; currentPage = 0;
    if (!semanticMode) fetchEntries();
  });
  document.getElementById("time-filter").addEventListener("change", function() {
    filterTime = this.value; currentPage = 0;
    if (!semanticMode) fetchEntries();
  });

  // Tab group
  document.getElementById("tab-group").addEventListener("click", function(ev) {
    var btn = ev.target.closest(".tab-btn");
    if (!btn) return;
    currentTab = btn.getAttribute("data-tab");
    this.querySelectorAll(".tab-btn").forEach(function(b) { b.classList.remove("active"); });
    btn.classList.add("active");
    expandedOpIdx = null;
    renderOperations();
  });

  // Entry expand
  entriesPanel.addEventListener("click", function(ev) {
    var row = ev.target.closest(".entry-row");
    if (!row) return;
    var id = row.getAttribute("data-id");
    expandedEntryId = expandedEntryId === id ? null : id;
    linksCache = {};
    renderEntries();
  });

  // Op expand
  opsPanel.addEventListener("click", function(ev) {
    var row = ev.target.closest(".op-row");
    if (!row) return;
    var idx = parseInt(row.getAttribute("data-op-idx"));
    expandedOpIdx = expandedOpIdx === idx ? null : idx;
    renderOperations();
  });

  function fetchStats() { return fetchJSON("/api/stats").then(function(d) { stats = d; renderStats(); }); }
  function fetchOps() {
    return fetchJSON("/api/events").then(function(data) {
      operations = data;
      renderOperations();
    });
  }

  // Database selector
  var dbSelect = document.getElementById("db-select");
  var dbPathInput = document.getElementById("db-path-input");
  var dbSwitchBtn = document.getElementById("db-switch-btn");

  function initDbSelector() {
    fetchJSON("/api/databases").then(function(data) {
      var current = data.current || "";
      var presets = data.presets || {};
      var opts = '';
      var foundCurrent = false;
      Object.keys(presets).forEach(function(name) {
        var path = presets[name];
        var sel = (path === current) ? " selected" : "";
        if (path === current) foundCurrent = true;
        opts += '<option value="' + esc(path) + '"' + sel + '>' + esc(name) + '</option>';
      });
      if (!foundCurrent && current) {
        opts = '<option value="' + esc(current) + '" selected>' + esc(current.split("/").pop() || current) + '</option>' + opts;
      }
      opts += '<option value="__custom__">Custom path...</option>';
      dbSelect.innerHTML = opts;
      dbPathInput.style.display = "none";
      dbSwitchBtn.style.display = "none";
    }).catch(function(){});
  }

  dbSelect.addEventListener("change", function() {
    if (this.value === "__custom__") {
      dbPathInput.style.display = "";
      dbSwitchBtn.style.display = "";
      dbPathInput.focus();
    } else {
      dbPathInput.style.display = "none";
      dbSwitchBtn.style.display = "none";
      switchDatabase(this.value);
    }
  });

  dbSwitchBtn.addEventListener("click", function() {
    var path = dbPathInput.value.trim();
    if (path) switchDatabase(path);
  });

  document.getElementById("db-clear-btn").addEventListener("click", function() {
    if (!confirm("Delete ALL memories from the database? This cannot be undone.")) return;
    fetch("/api/database/clear", { method: "POST" })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.error) { alert("Clear failed: " + data.error); return; }
        knownEntryIds = {};
        entries = [];
        operations = [];
        stats = {};
        currentPage = 0;
        fetchEntries();
        fetchStats().catch(function(){});
      })
      .catch(function(err) { alert("Clear failed: " + err); });
  });
  dbPathInput.addEventListener("keydown", function(ev) {
    if (ev.key === "Enter") {
      var path = this.value.trim();
      if (path) switchDatabase(path);
    }
  });

  function switchDatabase(path) {
    fetch("/api/database", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: path})
    }).then(function(r) { return r.json(); }).then(function(data) {
      if (data.error) {
        alert("Failed to load database: " + data.error);
        return;
      }
      knownEntryIds = {};
      entries = [];
      operations = [];
      stats = {};
      currentPage = 0;
      expandedEntryId = null;
      expandedOpIdx = null;
      linksCache = {};
      semanticMode = false;
      semanticResults = null;
      searchInput.classList.remove("semantic-active");
      searchInput.value = "";
      fetchEntries();
      fetchStats().catch(function(){});
      fetchOps().catch(function(){});
      initDbSelector();
    }).catch(function(e) {
      alert("Error switching database: " + e);
    });
  }

  // Init
  fetchEntries();
  fetchStats().catch(function(){});
  fetchOps().catch(function(){});
  connectSSE();
  updateColHeaders();
  initDbSelector();

  // Periodic stats refresh
  setInterval(function() { fetchStats().catch(function(){}); }, 10000);
})();
</script>
</body>
</html>'''
