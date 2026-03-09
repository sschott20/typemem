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
  --tier-raw: #8b949e;
  --tier-summary: #58a6ff;
  --tier-knowledge: #d2a8ff;
  --tier-none: #484f58;
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

/* Layout */
.app {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100vh;
}

.stats-bar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 10px 20px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.stats-bar .title {
  font-family: var(--font-mono);
  font-size: 16px;
  font-weight: 700;
  color: var(--accent);
  margin-right: 12px;
  white-space: nowrap;
}

.stat {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.stat .val {
  color: var(--text);
  font-weight: 600;
}

.connection-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 4px;
  vertical-align: middle;
}

.connection-dot.connected { background: #3fb950; }
.connection-dot.disconnected { background: #f85149; }

.connection-label {
  font-size: 11px;
  color: var(--text-secondary);
}

.panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  overflow: hidden;
}

.panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
}

.panel:last-child { border-right: none; }

.panel-header {
  padding: 12px 16px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.panel-header h2 {
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  margin-bottom: 10px;
}

.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 0;
}

/* Scrollbar */
.panel-body::-webkit-scrollbar { width: 6px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
.panel-body::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* Search & Filters */
.controls {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.search-input {
  flex: 1;
  min-width: 150px;
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus { border-color: var(--accent); }

.search-input::placeholder { color: var(--tier-none); }

.filter-select {
  padding: 6px 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-family: var(--font-ui);
  font-size: 12px;
  outline: none;
  cursor: pointer;
}

.filter-select:focus { border-color: var(--accent); }

/* Entry Rows */
.entry-row {
  padding: 8px 16px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.15s;
  font-family: var(--font-mono);
  font-size: 12px;
}

.entry-row:hover { background: rgba(88, 166, 255, 0.06); }

.entry-row.expanded { background: rgba(88, 166, 255, 0.04); }

.entry-summary {
  display: flex;
  align-items: center;
  gap: 10px;
}

.entry-id {
  color: var(--text-secondary);
  font-size: 11px;
  flex-shrink: 0;
  width: 68px;
}

.entry-text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text);
}

.tier-badge {
  font-size: 10px;
  padding: 1px 8px;
  border-radius: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  flex-shrink: 0;
}

.tier-badge.raw { background: rgba(139,148,158,0.15); color: var(--tier-raw); }
.tier-badge.summary { background: rgba(88,166,255,0.15); color: var(--tier-summary); }
.tier-badge.knowledge { background: rgba(210,168,255,0.15); color: var(--tier-knowledge); }
.tier-badge.none { background: rgba(72,79,88,0.15); color: var(--tier-none); }

.entry-time {
  color: var(--text-secondary);
  font-size: 11px;
  flex-shrink: 0;
  text-align: right;
  min-width: 60px;
}

.entry-detail {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.entry-row.expanded .entry-detail {
  max-height: 600px;
}

.detail-content {
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--bg);
  border-radius: 6px;
  border: 1px solid var(--border);
}

.detail-field {
  margin-bottom: 6px;
  font-size: 12px;
  line-height: 1.6;
}

.detail-field .label {
  color: var(--text-secondary);
  margin-right: 6px;
}

.detail-field pre {
  margin-top: 4px;
  padding: 8px;
  background: rgba(0,0,0,0.3);
  border-radius: 4px;
  overflow-x: auto;
  font-size: 11px;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}

/* New entry highlight */
@keyframes entry-glow {
  0% { background: rgba(248,129,102,0.2); }
  100% { background: transparent; }
}

.entry-row.new-entry {
  animation: entry-glow 2s ease-out;
}

/* Tab Buttons */
.tab-group {
  display: flex;
  gap: 4px;
}

.tab-btn {
  padding: 5px 14px;
  border: none;
  border-radius: 16px;
  background: transparent;
  color: var(--text-secondary);
  font-family: var(--font-ui);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.tab-btn:hover { background: rgba(88,166,255,0.1); color: var(--text); }

.tab-btn.active { background: var(--accent); color: #fff; }

/* Operation Rows */
.op-row {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.15s;
}

.op-row:hover { background: rgba(88,166,255,0.06); }

.op-row.expanded { background: rgba(88,166,255,0.04); }

.op-summary {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
}

.op-type-icon {
  flex-shrink: 0;
  width: 20px;
  text-align: center;
}

.op-time {
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 11px;
  flex-shrink: 0;
}

.op-desc {
  flex: 1;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.op-detail {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.op-row.expanded .op-detail {
  max-height: 800px;
}

.op-detail-content {
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--bg);
  border-radius: 6px;
  border: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 12px;
}

.op-detail-content .label {
  color: var(--text-secondary);
}

.op-detail-content pre {
  margin-top: 4px;
  padding: 8px;
  background: rgba(0,0,0,0.3);
  border-radius: 4px;
  overflow-x: auto;
  font-size: 11px;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}

.op-detail-content table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 6px;
}

.op-detail-content th {
  text-align: left;
  color: var(--text-secondary);
  font-weight: 500;
  padding: 4px 8px 4px 0;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
}

.op-detail-content td {
  padding: 4px 8px 4px 0;
  border-bottom: 1px solid rgba(48,54,61,0.5);
  font-size: 11px;
  vertical-align: top;
}

.empty-state {
  padding: 40px 20px;
  text-align: center;
  color: var(--text-secondary);
  font-size: 13px;
}

.consol-badge {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 10px;
  font-weight: 600;
  background: rgba(88,166,255,0.15);
  color: var(--accent);
  margin-left: 4px;
}

.inject-badge {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 10px;
  font-weight: 600;
  background: rgba(210,168,255,0.15);
  color: var(--tier-knowledge);
  margin-left: 4px;
}

/* Responsive */
@media (max-width: 900px) {
  .panels { grid-template-columns: 1fr; }
  .panel { border-right: none; border-bottom: 1px solid var(--border); }
  .panel:last-child { border-bottom: none; }
}
</style>
</head>
<body>
<div class="app">
  <div class="stats-bar" id="stats-bar">
    <span class="title">typemem viz</span>
    <span class="stat">Total: <span class="val" id="stat-total">0</span></span>
    <span class="stat">Raw: <span class="val" id="stat-raw">0</span></span>
    <span class="stat">Summary: <span class="val" id="stat-summary">0</span></span>
    <span class="stat">Knowledge: <span class="val" id="stat-knowledge">0</span></span>
    <span class="stat">Events: <span class="val" id="stat-events">0</span></span>
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
            <option value="raw">Raw</option>
            <option value="summary">Summary</option>
            <option value="knowledge">Knowledge</option>
            <option value="none">None</option>
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
  // State
  var entries = [];
  var consolidations = [];
  var injections = [];
  var stats = { total: 0, tiers: {}, events_count: 0, consolidations_count: 0, injections_count: 0 };
  var currentTab = "all";
  var searchText = "";
  var filterTier = "all";
  var sortBy = "newest";
  var expandedEntryId = null;
  var expandedOpIdx = null;
  var knownEntryIds = {};
  var sseConnected = false;

  // DOM refs
  var entriesPanel = document.getElementById("entries-panel");
  var opsPanel = document.getElementById("ops-panel");
  var searchInput = document.getElementById("search-input");
  var tierFilter = document.getElementById("tier-filter");
  var sortSelect = document.getElementById("sort-select");
  var tabGroup = document.getElementById("tab-group");
  var connDot = document.getElementById("conn-dot");
  var connLabel = document.getElementById("conn-label");

  // Helpers
  function escapeHtml(text) {
    if (text == null) return "";
    var s = String(text);
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(s));
    return div.innerHTML;
  }

  function truncate(text, maxLen) {
    if (!text) return "";
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + "...";
  }

  function timeAgo(ts) {
    if (!ts) return "";
    var now = Date.now() / 1000;
    var diff = Math.max(0, now - ts);
    if (diff < 5) return "just now";
    if (diff < 60) return Math.floor(diff) + "s ago";
    if (diff < 3600) return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    return Math.floor(diff / 86400) + "d ago";
  }

  function formatTime(ts) {
    if (!ts) return "";
    var d = new Date(ts * 1000);
    return d.toLocaleString();
  }

  function tierClass(tier) {
    if (tier === "raw" || tier === "summary" || tier === "knowledge") return tier;
    return "none";
  }

  function tierLabel(tier) {
    return tier || "none";
  }

  // API
  function fetchJSON(url) {
    return fetch(url).then(function(r) { return r.json(); });
  }

  function fetchEntries() {
    return fetchJSON("/api/entries").then(function(data) {
      var newIds = {};
      data.forEach(function(e) {
        if (!knownEntryIds[e.id]) newIds[e.id] = true;
        knownEntryIds[e.id] = true;
      });
      entries = data;
      renderEntries(newIds);
    });
  }

  function fetchStats() {
    return fetchJSON("/api/stats").then(function(data) {
      stats = data;
      renderStats();
    });
  }

  function fetchConsolidations() {
    return fetchJSON("/api/consolidations").then(function(data) {
      consolidations = data;
      renderOperations();
    });
  }

  function fetchInjections() {
    return fetchJSON("/api/injections").then(function(data) {
      injections = data;
      renderOperations();
    });
  }

  // Rendering: Stats
  function renderStats() {
    document.getElementById("stat-total").textContent = stats.total || 0;
    document.getElementById("stat-raw").textContent = (stats.tiers && stats.tiers.raw) || 0;
    document.getElementById("stat-summary").textContent = (stats.tiers && stats.tiers.summary) || 0;
    document.getElementById("stat-knowledge").textContent = (stats.tiers && stats.tiers.knowledge) || 0;
    document.getElementById("stat-events").textContent = stats.events_count || 0;
    document.getElementById("stat-consol").textContent = stats.consolidations_count || 0;
    document.getElementById("stat-injects").textContent = stats.injections_count || 0;
  }

  // Rendering: Entries
  function renderEntries(newIds) {
    if (!newIds) newIds = {};
    var filtered = entries.filter(function(e) {
      if (filterTier !== "all") {
        var eTier = (e.metadata && e.metadata._tier) || "none";
        if (eTier !== filterTier) return false;
      }
      if (searchText) {
        var q = searchText.toLowerCase();
        var match = (e.text && e.text.toLowerCase().indexOf(q) !== -1) ||
                    (e.id && e.id.toLowerCase().indexOf(q) !== -1);
        if (!match) return false;
      }
      return true;
    });

    filtered.sort(function(a, b) {
      if (sortBy === "newest") return (b.timestamp || 0) - (a.timestamp || 0);
      if (sortBy === "oldest") return (a.timestamp || 0) - (b.timestamp || 0);
      return (a.text || "").localeCompare(b.text || "");
    });

    if (filtered.length === 0) {
      entriesPanel.innerHTML = '<div class="empty-state">No memory entries</div>';
      return;
    }

    var html = "";
    filtered.forEach(function(e) {
      var tier = (e.metadata && e.metadata._tier) || "none";
      var isExpanded = expandedEntryId === e.id;
      var isNew = newIds[e.id] ? " new-entry" : "";
      html += '<div class="entry-row' + (isExpanded ? " expanded" : "") + isNew + '" data-id="' + escapeHtml(e.id) + '">';
      html += '<div class="entry-summary">';
      html += '<span class="entry-id">' + escapeHtml(truncate(e.id, 8)) + '</span>';
      html += '<span class="entry-text">' + escapeHtml(truncate(e.text, 60)) + '</span>';
      html += '<span class="tier-badge ' + tierClass(tier) + '">' + escapeHtml(tierLabel(tier)) + '</span>';
      html += '<span class="entry-time">' + escapeHtml(timeAgo(e.timestamp)) + '</span>';
      html += '</div>';
      html += '<div class="entry-detail">';
      if (isExpanded) {
        html += renderEntryDetail(e);
      }
      html += '</div>';
      html += '</div>';
    });
    entriesPanel.innerHTML = html;
  }

  function renderEntryDetail(e) {
    var meta = e.metadata || {};
    var html = '<div class="detail-content">';
    html += '<div class="detail-field"><span class="label">ID:</span> ' + escapeHtml(e.id) + '</div>';
    html += '<div class="detail-field"><span class="label">Text:</span> ' + escapeHtml(e.text) + '</div>';
    html += '<div class="detail-field"><span class="label">Tier:</span> ' + escapeHtml((meta._tier) || "none") + '</div>';
    html += '<div class="detail-field"><span class="label">Time:</span> ' + escapeHtml(formatTime(e.timestamp)) + '</div>';
    html += '<div class="detail-field"><span class="label">Metadata:</span><pre>' + escapeHtml(JSON.stringify(meta, null, 2)) + '</pre></div>';
    html += '</div>';
    return html;
  }

  // Rendering: Operations
  function renderOperations() {
    var ops = [];

    if (currentTab === "all" || currentTab === "consolidation") {
      consolidations.forEach(function(c, i) {
        ops.push({ type: "consolidation", data: c, idx: "c" + i, timestamp: c.timestamp });
      });
    }

    if (currentTab === "all" || currentTab === "injection") {
      injections.forEach(function(inj, i) {
        ops.push({ type: "injection", data: inj, idx: "i" + i, timestamp: inj.timestamp });
      });
    }

    ops.sort(function(a, b) { return (b.timestamp || 0) - (a.timestamp || 0); });

    if (ops.length === 0) {
      opsPanel.innerHTML = '<div class="empty-state">No operations recorded</div>';
      return;
    }

    var html = "";
    ops.forEach(function(op) {
      if (op.type === "consolidation") {
        html += renderConsolidationItem(op.data, op.idx);
      } else {
        html += renderInjectionItem(op.data, op.idx);
      }
    });
    opsPanel.innerHTML = html;
  }

  function renderConsolidationItem(c, idx) {
    var isExpanded = expandedOpIdx === idx;
    var numOutputs = c.outputs ? c.outputs.length : 0;
    var numDeletions = c.deletions ? c.deletions.length : 0;
    var html = '<div class="op-row' + (isExpanded ? " expanded" : "") + '" data-op-idx="' + idx + '">';
    html += '<div class="op-summary">';
    html += '<span class="op-type-icon" title="Consolidation">&#9670;</span>';
    html += '<span class="op-time">' + escapeHtml(formatTime(c.timestamp)) + '</span>';
    html += '<span class="op-desc">';
    html += escapeHtml(c.name || "consolidation");
    html += '<span class="consol-badge">' + numOutputs + ' output' + (numOutputs !== 1 ? 's' : '') + '</span>';
    if (numDeletions > 0) {
      html += '<span class="consol-badge" style="background:rgba(248,129,102,0.15);color:#f78166;">' + numDeletions + ' deleted</span>';
    }
    html += '</span>';
    html += '</div>';
    html += '<div class="op-detail">';
    if (isExpanded) {
      html += renderConsolidationDetail(c);
    }
    html += '</div>';
    html += '</div>';
    return html;
  }

  function renderConsolidationDetail(c) {
    var html = '<div class="op-detail-content">';
    html += '<div class="detail-field"><span class="label">Duration:</span> ' + escapeHtml((c.duration_ms || 0).toFixed(1)) + ' ms</div>';
    if (c.outputs && c.outputs.length > 0) {
      html += '<div class="detail-field"><span class="label">Outputs:</span></div>';
      c.outputs.forEach(function(o) {
        html += '<div style="margin:4px 0 4px 12px;font-size:11px;">';
        html += '<span style="color:var(--text-secondary);">' + escapeHtml(truncate(o.id || "", 12)) + '</span> ';
        html += escapeHtml(truncate(o.text || "(no text)", 80));
        html += '</div>';
      });
    }
    if (c.deletions && c.deletions.length > 0) {
      html += '<div class="detail-field" style="margin-top:8px;"><span class="label">Deleted IDs:</span></div>';
      html += '<pre>' + escapeHtml(c.deletions.join("\n")) + '</pre>';
    }
    html += '</div>';
    return html;
  }

  function renderInjectionItem(inj, idx) {
    var isExpanded = expandedOpIdx === idx;
    var numResults = inj.search_results ? inj.search_results.length : 0;
    var html = '<div class="op-row' + (isExpanded ? " expanded" : "") + '" data-op-idx="' + idx + '">';
    html += '<div class="op-summary">';
    html += '<span class="op-type-icon" title="Injection" style="color:var(--tier-knowledge);">&#9672;</span>';
    html += '<span class="op-time">' + escapeHtml(formatTime(inj.timestamp)) + '</span>';
    html += '<span class="op-desc">';
    html += escapeHtml(truncate(inj.query || "", 40));
    html += '<span class="inject-badge">' + numResults + ' result' + (numResults !== 1 ? 's' : '') + '</span>';
    html += '</span>';
    html += '</div>';
    html += '<div class="op-detail">';
    if (isExpanded) {
      html += renderInjectionDetail(inj);
    }
    html += '</div>';
    html += '</div>';
    return html;
  }

  function renderInjectionDetail(inj) {
    var html = '<div class="op-detail-content">';
    html += '<div class="detail-field"><span class="label">Query:</span> ' + escapeHtml(inj.query) + '</div>';
    html += '<div class="detail-field"><span class="label">Token Budget:</span> ' + escapeHtml(inj.token_budget) + '</div>';
    html += '<div class="detail-field"><span class="label">Duration:</span> ' + escapeHtml((inj.duration_ms || 0).toFixed(1)) + ' ms</div>';
    if (inj.search_results && inj.search_results.length > 0) {
      html += '<div class="detail-field" style="margin-top:8px;"><span class="label">Search Results:</span></div>';
      html += '<table><thead><tr><th>ID</th><th>Text</th><th>Distance</th></tr></thead><tbody>';
      inj.search_results.forEach(function(r) {
        html += '<tr>';
        html += '<td>' + escapeHtml(truncate(r.id || "", 10)) + '</td>';
        html += '<td>' + escapeHtml(truncate(r.text || "", 50)) + '</td>';
        html += '<td>' + escapeHtml(typeof r.distance === "number" ? r.distance.toFixed(4) : "") + '</td>';
        html += '</tr>';
      });
      html += '</tbody></table>';
    }
    if (inj.context) {
      html += '<div class="detail-field" style="margin-top:8px;"><span class="label">Context:</span><pre>' + escapeHtml(inj.context) + '</pre></div>';
    }
    html += '</div>';
    return html;
  }

  // SSE
  function connectSSE() {
    var es = new EventSource("/api/stream");

    es.onopen = function() {
      sseConnected = true;
      connDot.className = "connection-dot connected";
      connLabel.textContent = "Connected";
    };

    es.onmessage = function(ev) {
      var data;
      try { data = JSON.parse(ev.data); } catch(e) { return; }
      if (!data || !data.type) return;

      if (data.type === "store") {
        fetchStats();
        if (data.operation === "add" || data.operation === "add_batch" || data.operation === "delete" || data.operation === "update") {
          fetchEntries();
        }
      } else if (data.type === "consolidation") {
        consolidations.push(data);
        fetchStats();
        renderOperations();
        opsPanel.scrollTop = 0;
      } else if (data.type === "injection") {
        injections.push(data);
        fetchStats();
        renderOperations();
        opsPanel.scrollTop = 0;
      }
    };

    es.onerror = function() {
      sseConnected = false;
      connDot.className = "connection-dot disconnected";
      connLabel.textContent = "Disconnected";
      // EventSource auto-reconnects
    };
  }

  // Event Handlers
  searchInput.addEventListener("input", function() {
    searchText = this.value;
    renderEntries();
  });

  tierFilter.addEventListener("change", function() {
    filterTier = this.value;
    renderEntries();
  });

  sortSelect.addEventListener("change", function() {
    sortBy = this.value;
    renderEntries();
  });

  tabGroup.addEventListener("click", function(ev) {
    var btn = ev.target.closest(".tab-btn");
    if (!btn) return;
    currentTab = btn.getAttribute("data-tab");
    tabGroup.querySelectorAll(".tab-btn").forEach(function(b) { b.classList.remove("active"); });
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
    var idx = row.getAttribute("data-op-idx");
    expandedOpIdx = expandedOpIdx === idx ? null : idx;
    renderOperations();
  });

  // Init
  function init() {
    // Mark all existing entries as known (no glow animation on first load)
    fetchJSON("/api/entries").then(function(data) {
      data.forEach(function(e) { knownEntryIds[e.id] = true; });
      entries = data;
      renderEntries();
    }).catch(function() {});

    fetchStats().catch(function() {});
    fetchConsolidations().catch(function() {});
    fetchInjections().catch(function() {});
    connectSSE();
  }

  init();
})();
</script>
</body>
</html>'''
