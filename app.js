const DETAIL_LOCK_MS = 2500;

const state = {
  selectedTraceId: null,
  traces: [],
  detail: null,
  detailSignature: "",
  selectedOverviewSignature: "",
  pendingDetail: null,
  pendingDetailSignature: "",
  pendingOverviewSignature: "",
  timer: null,
  interactionLockUntil: 0,
  lastUpdatedAt: null,
  newTraceCount: 0,
  newTraceThresholdEpoch: 0,
};

const traceListEl = document.getElementById("traceList");
const traceCountEl = document.getElementById("traceCount");
const pollStateEl = document.getElementById("pollState");
const refreshBtnEl = document.getElementById("refreshBtn");
const heroTitleEl = document.getElementById("heroTitle");
const heroSubtitleEl = document.getElementById("heroSubtitle");
const traceStatusEl = document.getElementById("traceStatus");
const traceReasonEl = document.getElementById("traceReason");
const traceSimTimeEl = document.getElementById("traceSimTime");
const detailUpdateBadgeEl = document.getElementById("detailUpdateBadge");
const summaryContextEl = document.getElementById("summaryContext");
const summaryGuardEl = document.getElementById("summaryGuard");
const nodeGridEl = document.getElementById("nodeGrid");
const llmCallListEl = document.getElementById("llmCallList");
const rawSectionsEl = document.getElementById("rawSections");
const traceItemTpl = document.getElementById("traceItemTpl");
const nodeCardTpl = document.getElementById("nodeCardTpl");
const mainEl = document.querySelector(".main");

traceListEl.dataset.scrollKey = "trace-list";
llmCallListEl.dataset.scrollKey = "llm-call-list";
rawSectionsEl.dataset.scrollKey = "raw-sections";

function badgeClass(status) {
  const value = String(status || "").toLowerCase();
  if (["ok", "completed", "running"].includes(value)) {
    return "badge-ok";
  }
  if (["retry", "semantic_replan", "graph_error", "no_actions", "warning", "rule_only"].includes(value)) {
    return value === "graph_error" ? "badge-danger" : "badge-warning";
  }
  if (["error", "failed", "aborted"].includes(value)) {
    return "badge-danger";
  }
  return "badge-neutral";
}

function humanTime(isoText) {
  if (!isoText) {
    return "-";
  }
  const dt = new Date(isoText);
  if (Number.isNaN(dt.getTime())) {
    return String(isoText);
  }
  return dt.toLocaleString();
}

function shortText(value, limit = 120) {
  const text = String(value || "");
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, Math.max(limit - 3, 0))}...`;
}

function formatMs(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const amount = Number(value);
  if (Number.isNaN(amount)) {
    return "-";
  }
  return `${amount}ms`;
}

async function fetchJSON(url) {
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}

function setEmpty(container, text) {
  container.innerHTML = "";
  container.classList.add("empty");
  container.textContent = text;
}

function addKvBlock(container, title, lines) {
  const block = document.createElement("div");
  block.className = "kv-item";
  const strong = document.createElement("strong");
  strong.textContent = title;
  block.appendChild(strong);
  lines.forEach((line) => {
    const p = document.createElement("div");
    p.textContent = line;
    block.appendChild(p);
  });
  container.appendChild(block);
}

function captureScrollState() {
  const elements = {};
  document.querySelectorAll("[data-scroll-key]").forEach((el) => {
    elements[el.dataset.scrollKey] = {
      top: el.scrollTop,
      left: el.scrollLeft,
    };
  });
  return {
    windowX: window.scrollX,
    windowY: window.scrollY,
    elements,
  };
}

function restoreScrollState(snapshot) {
  if (!snapshot) {
    return;
  }
  window.requestAnimationFrame(() => {
    window.scrollTo(snapshot.windowX || 0, snapshot.windowY || 0);
    Object.entries(snapshot.elements || {}).forEach(([key, position]) => {
      const selector = `[data-scroll-key="${CSS.escape(key)}"]`;
      const el = document.querySelector(selector);
      if (!el) {
        return;
      }
      el.scrollTop = position.top || 0;
      el.scrollLeft = position.left || 0;
    });
  });
}

function isInteractionLocked() {
  return Date.now() < state.interactionLockUntil;
}

function bumpInteractionLock() {
  state.interactionLockUntil = Date.now() + DETAIL_LOCK_MS;
}

function traceOverviewSignature(item) {
  if (!item) {
    return "";
  }
  const nodeParts = (item.nodes || []).map((node) => `${node.node}:${node.status}:${node.duration_ms ?? "-"}`).join("|");
  return [item.trace_id, item.status, item.started_at_epoch ?? 0, item.total_duration_ms ?? "-", nodeParts].join("::");
}

function detailSignature(detail) {
  if (!detail) {
    return "";
  }
  return JSON.stringify({
    trace_id: detail.trace_id,
    status: detail.status,
    started_at_epoch: detail.started_at_epoch,
    total_duration_ms: detail.total_duration_ms,
    graph_nodes: detail.sections_text?.["Graph Nodes"] || [],
    llm_calls: detail.sections_text?.["LLM Calls"] || [],
    parsed_actions: detail.sections_text?.["Parsed Actions"] || [],
    guard: detail.sections_text?.["Guard"] || [],
    submit: detail.sections_text?.["Submit"] || [],
    errors: detail.errors || [],
  });
}

function clearPendingDetail() {
  state.pendingDetail = null;
  state.pendingDetailSignature = "";
  state.pendingOverviewSignature = "";
}

function updateChromeStatus(prefix = "") {
  const parts = [];
  if (prefix) {
    parts.push(prefix);
  } else if (state.lastUpdatedAt) {
    parts.push(`Updated ${state.lastUpdatedAt.toLocaleTimeString()}`);
  } else {
    parts.push("Polling...");
  }
  if (state.newTraceCount > 0) {
    parts.push(`${state.newTraceCount} new trace${state.newTraceCount > 1 ? "s" : ""}`);
  }
  if (state.pendingDetail) {
    parts.push("detail update ready");
  }
  pollStateEl.textContent = parts.join(" | ");
  refreshBtnEl.textContent = state.pendingDetail ? "Apply update" : "Refresh now";
  detailUpdateBadgeEl.classList.toggle("hidden", !state.pendingDetail);
}

function renderTraceList() {
  const snapshot = captureScrollState();
  traceListEl.innerHTML = "";
  traceListEl.classList.remove("empty");
  traceCountEl.textContent = `${state.traces.length} traces`;
  if (!state.traces.length) {
    setEmpty(traceListEl, "No trace data yet.");
    restoreScrollState(snapshot);
    return;
  }

  const latestEpoch = Number(state.traces[0]?.started_at_epoch || 0);
  state.traces.forEach((item) => {
    const node = traceItemTpl.content.firstElementChild.cloneNode(true);
    node.dataset.traceId = item.trace_id;
    if (item.trace_id === state.selectedTraceId) {
      node.classList.add("active");
    }
    node.querySelector(".trace-id").textContent = item.trace_id;
    const statusEl = node.querySelector(".trace-status");
    statusEl.textContent = item.status || "unknown";
    statusEl.classList.add(badgeClass(item.status));
    node.querySelector(".trace-item-meta").textContent =
      `t=${item.sim_time} | ${item.reason || "-"} | ${humanTime(item.started_at)}`;

    const markEl = node.querySelector(".trace-item-mark");
    const isNewTrace = state.newTraceCount > 0 && Number(item.started_at_epoch || 0) > state.newTraceThresholdEpoch;
    markEl.classList.toggle("hidden", !isNewTrace);

    const chipWrap = node.querySelector(".trace-nodes");
    (item.nodes || []).forEach((nodeItem) => {
      const chip = document.createElement("span");
      chip.className = "trace-node-chip";
      chip.textContent = `${nodeItem.node}:${nodeItem.status || "pending"}`;
      chipWrap.appendChild(chip);
    });

    node.addEventListener("click", () => {
      state.selectedTraceId = item.trace_id;
      if (Number(item.started_at_epoch || 0) >= latestEpoch) {
        state.newTraceCount = 0;
        state.newTraceThresholdEpoch = 0;
      }
      clearPendingDetail();
      renderTraceList();
      updateChromeStatus();
      void refreshDetail({ force: true });
    });
    traceListEl.appendChild(node);
  });
  restoreScrollState(snapshot);
}

function renderSummary(detail) {
  heroTitleEl.textContent = detail.trace_id || "Trace Detail";
  heroSubtitleEl.textContent = `started_at: ${humanTime(detail.started_at)} | total_ms: ${detail.total_duration_ms ?? "-"}`;
  traceStatusEl.textContent = detail.status || "unknown";
  traceStatusEl.className = `badge ${badgeClass(detail.status)}`;
  traceReasonEl.textContent = detail.reason || "reason:-";
  traceReasonEl.className = "badge badge-neutral";
  traceSimTimeEl.textContent = `sim_time:${detail.sim_time ?? "-"}`;
  traceSimTimeEl.className = "badge badge-neutral";

  summaryContextEl.innerHTML = "";
  summaryContextEl.classList.remove("empty");
  const stmLines = detail.sections_text?.STM || [];
  const ltmLines = detail.sections_text?.LTM || [];
  if (!stmLines.length && !ltmLines.length) {
    setEmpty(summaryContextEl, "No STM/LTM details.");
  } else {
    if (stmLines.length) {
      addKvBlock(summaryContextEl, "STM", stmLines);
    }
    if (ltmLines.length) {
      addKvBlock(summaryContextEl, "LTM", ltmLines);
    }
  }

  summaryGuardEl.innerHTML = "";
  summaryGuardEl.classList.remove("empty");
  const guardLines = detail.sections_text?.Guard || [];
  const submitLines = detail.sections_text?.Submit || [];
  const errorLines = detail.errors || [];
  if (!guardLines.length && !submitLines.length && !errorLines.length) {
    setEmpty(summaryGuardEl, "No guard or submit details.");
  } else {
    if (guardLines.length) {
      addKvBlock(summaryGuardEl, "Guard", guardLines);
    }
    if (submitLines.length) {
      addKvBlock(summaryGuardEl, "Submit", submitLines);
    }
    if (errorLines.length) {
      addKvBlock(summaryGuardEl, "Errors", errorLines);
    }
  }
}

function renderNodeViews(detail) {
  nodeGridEl.innerHTML = "";
  const nodes = detail.node_views || [];
  if (!nodes.length) {
    setEmpty(nodeGridEl, "No node view data.");
    return;
  }

  nodes.forEach((view) => {
    const card = nodeCardTpl.content.firstElementChild.cloneNode(true);
    const timing = view.timing || {};
    card.querySelector(".node-name").textContent = view.node;
    card.querySelector(".node-meta").textContent = [
      `total=${formatMs(timing.total_ms)}`,
      `prompt=${formatMs(timing.prompt_build_ms)}`,
      `llm=${formatMs(timing.llm_ms)}`,
      `parse=${formatMs(timing.parse_ms)}`,
    ].join(" | ");

    const stats = [
      `retry=${view.retry_count ?? 0}`,
      view.operator_mode ? `mode=${view.operator_mode}` : "",
      timing.postprocess_ms !== null && timing.postprocess_ms !== undefined ? `post=${formatMs(timing.postprocess_ms)}` : "",
      timing.deterministic_translate_ms !== null && timing.deterministic_translate_ms !== undefined
        ? `deterministic=${formatMs(timing.deterministic_translate_ms)}`
        : "",
      timing.rule_check_ms !== null && timing.rule_check_ms !== undefined ? `rule=${formatMs(timing.rule_check_ms)}` : "",
      view.model ? `model=${view.model}` : "",
    ].filter(Boolean);
    card.querySelector(".node-stats").textContent = stats.join(" | ");

    const statusEl = card.querySelector(".node-status");
    statusEl.textContent = view.status || "pending";
    statusEl.classList.add(badgeClass(view.status));

    const outputEl = card.querySelector(".node-output");
    outputEl.dataset.scrollKey = `node:${view.node}:output`;
    outputEl.textContent = view.output_summary || "No output summary.";

    const promptEl = card.querySelector(".node-prompt");
    promptEl.dataset.scrollKey = `node:${view.node}:prompt`;
    if (view.prompt) {
      promptEl.textContent = view.prompt;
      promptEl.classList.remove("empty");
    }

    const responseEl = card.querySelector(".node-response");
    responseEl.dataset.scrollKey = `node:${view.node}:response`;
    if (view.raw_response) {
      responseEl.textContent = view.raw_response;
      responseEl.classList.remove("empty");
    }

    nodeGridEl.appendChild(card);
  });
}

function renderLlmCalls(detail) {
  llmCallListEl.innerHTML = "";
  llmCallListEl.classList.remove("empty");
  const llmCalls = detail.sections?.["LLM Calls"] || [];
  if (!llmCalls.length) {
    setEmpty(llmCallListEl, "No LLM call records.");
    return;
  }

  llmCalls.forEach((item, index) => {
    const block = document.createElement("div");
    block.className = "call-item";
    const strong = document.createElement("strong");
    strong.textContent = `${item.component || "llm"} | ${formatMs(item.duration_ms)} | success=${item.success}`;
    block.appendChild(strong);

    const meta = document.createElement("div");
    meta.className = "call-meta muted";
    meta.textContent = `sim_time=${item.sim_time ?? "-"} | prompt_chars=${item.prompt_chars ?? 0} | response_chars=${item.response_chars ?? 0}`;
    block.appendChild(meta);

    const summary = document.createElement("div");
    summary.textContent = shortText(item.summary || "", 220);
    block.appendChild(summary);

    if (item.file_path) {
      const fileLine = document.createElement("div");
      fileLine.className = "muted";
      fileLine.textContent = item.file_path;
      block.appendChild(fileLine);
    }

    block.dataset.scrollKey = `llm-call:${index}`;
    llmCallListEl.appendChild(block);
  });
}

function renderRawSections(detail) {
  rawSectionsEl.innerHTML = "";
  rawSectionsEl.classList.remove("empty");
  const blocks = [
    ["Parsed Actions", detail.sections_text?.["Parsed Actions"] || []],
    ["Graph Nodes", detail.sections_text?.["Graph Nodes"] || []],
    ["LLM Calls", detail.sections_text?.["LLM Calls"] || []],
  ].filter(([, lines]) => lines.length);

  if (!blocks.length) {
    setEmpty(rawSectionsEl, "No raw trace sections.");
    return;
  }

  blocks.forEach(([title, lines]) => {
    const block = document.createElement("div");
    block.className = "raw-block";
    const strong = document.createElement("strong");
    strong.textContent = title;
    block.appendChild(strong);
    const pre = document.createElement("pre");
    pre.dataset.scrollKey = `raw:${title}`;
    pre.textContent = lines.join("\n");
    block.appendChild(pre);
    rawSectionsEl.appendChild(block);
  });
}

function applyDetail(detail, signature, overviewSignature) {
  const snapshot = captureScrollState();
  state.detail = detail;
  state.detailSignature = signature;
  state.selectedOverviewSignature = overviewSignature;
  renderSummary(detail);
  renderNodeViews(detail);
  renderLlmCalls(detail);
  renderRawSections(detail);
  clearPendingDetail();
  updateChromeStatus();
  restoreScrollState(snapshot);
}

async function refreshDetail({ force = false } = {}) {
  const selectedItem = state.traces.find((item) => item.trace_id === state.selectedTraceId);
  if (!selectedItem) {
    return;
  }

  const overviewSignature = traceOverviewSignature(selectedItem);
  const needsFetch =
    force ||
    !state.detail ||
    state.detail.trace_id !== selectedItem.trace_id ||
    overviewSignature !== state.selectedOverviewSignature;

  if (!needsFetch) {
    if (state.pendingDetail && !isInteractionLocked()) {
      applyDetail(state.pendingDetail, state.pendingDetailSignature, state.pendingOverviewSignature || overviewSignature);
    }
    return;
  }

  const detail = await fetchJSON(`/api/trace/${encodeURIComponent(selectedItem.trace_id)}`);
  const signature = detailSignature(detail);
  if (force || !isInteractionLocked()) {
    applyDetail(detail, signature, overviewSignature);
    return;
  }
  if (signature !== state.detailSignature) {
    state.pendingDetail = detail;
    state.pendingDetailSignature = signature;
    state.pendingOverviewSignature = overviewSignature;
    updateChromeStatus();
  }
}

async function refreshAll({ force = false } = {}) {
  updateChromeStatus("Refreshing...");
  try {
    const previousLatestEpoch = Number(state.traces[0]?.started_at_epoch || 0);
    const payload = await fetchJSON("/api/traces");
    state.traces = payload.items || [];

    if (!state.selectedTraceId && state.traces.length) {
      state.selectedTraceId = state.traces[0].trace_id;
    } else if (state.selectedTraceId && !state.traces.some((item) => item.trace_id === state.selectedTraceId)) {
      state.selectedTraceId = state.traces[0]?.trace_id || null;
      clearPendingDetail();
      state.detail = null;
      state.detailSignature = "";
      state.selectedOverviewSignature = "";
    }

    const newTraceCount =
      previousLatestEpoch > 0
        ? state.traces.filter((item) => Number(item.started_at_epoch || 0) > previousLatestEpoch).length
        : 0;
    state.newTraceCount = force ? 0 : newTraceCount;
    state.newTraceThresholdEpoch = force ? 0 : previousLatestEpoch;
    state.lastUpdatedAt = new Date();

    renderTraceList();
    if (state.selectedTraceId) {
      await refreshDetail({ force });
    } else {
      updateChromeStatus();
    }
  } catch (err) {
    console.error(err);
    updateChromeStatus(`Refresh failed: ${err.message}`);
  }
}

function bindInteractionGuards() {
  mainEl.addEventListener("wheel", bumpInteractionLock, { passive: true });
  mainEl.addEventListener("pointerdown", bumpInteractionLock, true);
  mainEl.addEventListener("keydown", bumpInteractionLock, true);
  document.addEventListener(
    "scroll",
    (event) => {
      const target = event.target;
      if (target instanceof Element && mainEl.contains(target)) {
        bumpInteractionLock();
      }
    },
    true
  );
  document.addEventListener("selectionchange", () => {
    const selection = window.getSelection();
    if (!selection || !selection.anchorNode) {
      return;
    }
    const anchor =
      selection.anchorNode.nodeType === Node.TEXT_NODE ? selection.anchorNode.parentElement : selection.anchorNode;
    if (anchor instanceof Element && mainEl.contains(anchor)) {
      bumpInteractionLock();
    }
  });
}

refreshBtnEl.addEventListener("click", () => {
  if (state.pendingDetail && state.pendingDetail.trace_id === state.selectedTraceId) {
    applyDetail(
      state.pendingDetail,
      state.pendingDetailSignature,
      state.pendingOverviewSignature || state.selectedOverviewSignature
    );
  }
  void refreshAll({ force: true });
});

bindInteractionGuards();
void refreshAll({ force: true });
state.timer = window.setInterval(() => {
  void refreshAll();
}, 2000);
