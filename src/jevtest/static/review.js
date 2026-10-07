(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const state = {
    csrfToken: null,
    queue: [],
    requestId: null,
    envelope: null,
    etag: null,
    facts: null,
    evidence: [],
    reviews: [],
    ephemeralInputs: new Map(),
    reports: new Map(),
    executionMetadata: new Map(),
    submissionItems: [],
    submittingBatch: false,
    stopBatchRequested: false,
    loadingBatchFiles: false,
    batchProgress: null,
    pendingReview: null,
    loadSequence: 0,
    reviewMode: null,
    executionProfile: null,
  };

  const resultLabels = {
    PRODUCT_DEFECT: "产品缺陷",
    TEST_SCRIPT: "测试脚本",
    ENVIRONMENT: "环境问题",
    CONFIGURATION: "配置问题",
    FLAKY: "偶发问题",
    UNKNOWN: "需要人工分诊",
  };
  const actionLabels = {
    RERUN: "重跑",
    COLLECT_MORE_LOGS: "补充日志",
    CHECK_ENVIRONMENT: "检查环境",
    ASSIGN_TEST_OWNER: "交由测试负责人",
    ASSIGN_PRODUCT_DEVELOPER: "交由产品开发",
    HUMAN_TRIAGE: "人工分诊",
  };
  const fieldLabels = {
    result: "测试结果",
    previous_runs: "历史运行",
    device: "设备型号",
    sw_version: "软件版本",
    assertion: "失败断言",
    syslog_excerpt: "提交的日志摘录",
    env: "环境事实",
    retry: "重试状态",
  };
  const reviewLabels = { accept: "接受建议", correct: "修改建议", defer: "暂缓补证" };
  const statusLabels = {
    pending: "待审核",
    deferred: "暂缓",
    accepted: "已接受建议",
    corrected: "已修订并生成新版本",
    not_requested: "未请求审核",
    result_ready: "结果就绪",
    degraded: "部分执行失败 / 降级",
    failed: "执行失败",
    cancelled: "已取消",
    local_queued: "本机队列已记录",
    pending_delivery: "等待本地登记",
    not_requested_delivery: "无需本地登记",
    failed_delivery: "本地登记失败",
  };
  const stageLabels = { rule: "规则", jev: "Jev", llm: "LLM", human: "人工" };
  const routeStatusLabels = { completed: "已完成", failed: "执行失败", skipped: "未执行" };
  const routeReasonLabels = {
    escalate_rule_result: "规则未形成可直接采纳的结论，继续判断。",
    unknown_escalation_requested: "Jev 未能确认，继续升级判断。",
    unknown_requires_human_review: "仍无法确认，转交人工判断。",
    review_required: "需要人工审核。",
    deadline_exceeded: "达到本次运行的时间限制。",
    provider_failure: "该阶段执行失败。",
    provider_failure_human_review: "Provider 调用失败或输出无效，结果已标记降级并转交人工。",
    low_confidence_escalation_requested: "Jev 建议置信度较低，升级给 Strong LLM 继续判断。",
    mid_confidence_escalation_requested: "Jev 尚未达到免复核标准，升级给 Strong LLM 继续判断。",
    high_confidence_result: "该阶段返回结果并结束后续判断。",
  };

  function make(tag, className, value) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== undefined && value !== null) item.textContent = String(value);
    return item;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function label(map, value) {
    return map[value] || String(value == null ? "未提供" : value);
  }

  function formatDate(value) {
    if (!value) return "未提供";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
  }

  function numeric(value, digits) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "未提供";
  }

  function newKey() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    const bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  class ApiError extends Error {
    constructor(status, code) {
      super(code || "request_rejected");
      this.status = status;
      this.code = code || "request_rejected";
    }
  }

  async function request(path, options) {
    const config = Object.assign({ credentials: "same-origin" }, options || {});
    const headers = new Headers(config.headers || {});
    if (config.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    config.headers = headers;
    const response = await fetch(path, config);
    let payload = null;
    try { payload = await response.json(); } catch (_error) { payload = null; }
    if (!response.ok) {
      const code = payload && payload.error && payload.error.code;
      throw new ApiError(response.status, code);
    }
    return { response, payload };
  }

  function explainError(error) {
    const messages = {
      result_version_conflict: "当前结果版本已变化。请刷新并核对新版本后再审核。",
      review_not_pending: "此版本已不处于待审核状态，请刷新队列。",
      confidence_reassessment_not_supported: "当前合同不允许在人工修订中重评 confidence。请保留现有值。",
      idempotency_conflict: "相同幂等键已有不同操作内容，请刷新后重试。",
      state_conflict: "本地审核状态已变化，请刷新后核对。",
      invalid_request: "服务端拒绝了请求内容。请检查 FailureState 字段和审核表单。",
      request_too_large: "提交内容超过本地服务的大小限制。",
      csrf_rejected: "本地会话已失效，请刷新页面重新建立会话。",
      origin_rejected: "页面来源不符合本地服务要求；请使用本机 127.0.0.1 地址打开。",
      host_rejected: "请求 Host 与本地服务地址不一致；请使用当前 127.0.0.1 地址。",
      storage_unavailable: "本地数据库暂不可用。",
      not_found: "本地服务没有找到该记录或页面资源。",
      live_confirmation_required: "真实 API 请求必须先在页面确认。",
    };
    if (error instanceof ApiError) return messages[error.code] || ("本地服务拒绝请求：" + error.code + " (HTTP " + error.status + ")");
    return "无法连接本地服务。请确认服务进程仍在运行，然后刷新页面。";
  }

  let toastTimer;
  function toast(message) {
    const node = byId("toast");
    node.textContent = message;
    node.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => { node.hidden = true; }, 4200);
  }

  function setBadge(node, text, style) {
    node.textContent = text;
    node.className = "badge " + (style || "neutral");
  }

  function reviewBadge(status) {
    if (status === "pending" || status === "deferred") return "waiting";
    if (status === "accepted" || status === "corrected") return "ok";
    return "neutral";
  }

  function renderQueue() {
    const list = byId("queue-list");
    clear(list);
    byId("queue-count").textContent = String(state.queue.length);
    byId("queue-empty").hidden = state.queue.length !== 0;
    state.queue.forEach((item) => {
      const envelope = item.envelope;
      const button = make("button", "queue-item" + (envelope.request_id === state.requestId ? " active" : ""));
      button.type = "button";
      button.setAttribute("aria-current", envelope.request_id === state.requestId ? "true" : "false");
      const top = make("span", "queue-item-top");
      const status = make("span", "badge " + reviewBadge(envelope.review_status), label(statusLabels, envelope.review_status));
      top.append(status);
      const imported = state.ephemeralInputs.get(envelope.request_id);
      const titleText = imported && imported.assertion
        ? imported.assertion
        : envelope.result ? label(resultLabels, envelope.result.category) : label(statusLabels, envelope.run_status);
      const title = make("strong", "", titleText);
      const meta = make("span", "queue-item-meta");
      const priority = envelope.result && envelope.result.priority != null ? "优先级 P" + envelope.result.priority : "优先级未提供";
      meta.append(make("span", "", priority));
      button.append(top, title, meta);
      button.addEventListener("click", () => loadCase(envelope.request_id));
      list.appendChild(button);
    });
  }

  async function refreshQueue(options) {
    const config = options || {};
    const result = await request("/v1/review-queue");
    state.queue = Array.isArray(result.payload.items) ? result.payload.items : [];
    renderQueue();
    if (!config.keepCurrent && !state.requestId && state.queue.length) await loadCase(state.queue[0].envelope.request_id);
    return state.queue;
  }

  function setRequestInAddress(requestId) {
    const url = new URL(window.location.href);
    if (requestId) url.searchParams.set("request", requestId);
    else url.searchParams.delete("request");
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  }

  async function loadCase(requestId) {
    const sequence = ++state.loadSequence;
    state.requestId = requestId;
    state.envelope = null;
    byId("detail-empty").hidden = true;
    byId("case-content").hidden = false;
    byId("review-bar").hidden = true;
    byId("case-title").textContent = "读取本地结果…";
    renderQueue();
    setRequestInAddress(requestId);
    try {
      const resultPath = "/v1/triage/" + encodeURIComponent(requestId);
      const detailPromise = request(resultPath);
      const factsPromise = request(resultPath + "/facts").catch((error) => error);
      const reviewsPromise = request(resultPath + "/reviews").catch((error) => error);
      const reportPromise = request(resultPath + "/report").catch((error) => error);
      const [detail, factsResult, historyResult, reportResult] = await Promise.all([detailPromise, factsPromise, reviewsPromise, reportPromise]);
      if (sequence !== state.loadSequence) return;
      state.envelope = detail.payload.envelope;
      state.etag = detail.response.headers.get("ETag") || ('"' + state.envelope.result_version + '"');
      state.facts = factsResult instanceof Error ? null : factsResult.payload.facts;
      state.reviews = historyResult instanceof Error ? [] : historyResult.payload.reviews || [];
      const reportRecord = reportResult instanceof Error ? null : reportResult.payload;
      state.reports.set(requestId, reportRecord);
      const metadataVersion = reportRecord ? reportRecord.report_result_version : state.envelope.result_version;
      const metadataQuery = metadataVersion ? "?result_version=" + encodeURIComponent(metadataVersion) : "";
      const metadataResult = await request(resultPath + "/execution-metadata" + metadataQuery).catch((error) => error);
      if (sequence !== state.loadSequence) return;
      state.executionMetadata.set(requestId, metadataResult instanceof Error ? null : metadataResult.payload);
      const factsVersion = factsResult instanceof Error ? null : factsResult.payload.result_version;
      if (factsVersion && factsVersion !== state.envelope.result_version) state.facts = null;
      state.evidence = await loadEvidence(state.envelope, state.facts, sequence);
      if (sequence !== state.loadSequence) return;
      renderCase();
    } catch (error) {
      if (sequence !== state.loadSequence) return;
      byId("case-title").textContent = explainError(error);
      toast(explainError(error));
    }
  }

  async function loadEvidence(envelope, facts, sequence) {
    if (!facts || !envelope) return [];
    const referencedFields = (envelope.evidence_refs || []).map((item) => item.field_path);
    const fields = referencedFields.length
      ? referencedFields
      : Object.keys(facts).filter((key) => key !== "schema_version");
    const rows = await Promise.all(fields.map(async (fieldPath) => {
      const query = new URLSearchParams({ result_version: envelope.result_version, field_path: fieldPath });
      try {
        const response = await request("/v1/triage/" + encodeURIComponent(envelope.request_id) + "/evidence?" + query.toString());
        if (sequence !== state.loadSequence) return null;
        return response.payload;
      } catch (_error) {
        return null;
      }
    }));
    return rows.filter(Boolean);
  }

  function renderCase() {
    const envelope = state.envelope;
    if (!envelope) return;
    const result = envelope.result || {};
    const input = state.ephemeralInputs.get(envelope.request_id);
    const assertion = (state.facts && state.facts.assertion) || (input && input.assertion);
    byId("case-kicker").textContent = "故障摘要";
    const displayAssertion = String(assertion || "失败摘要未持久化").trim();
    byId("case-title").textContent = displayAssertion.charAt(0).toUpperCase() + displayAssertion.slice(1);
    byId("request-id").textContent = envelope.request_id;
    byId("result-category").textContent = result.category ? label(resultLabels, result.category) : "系统未提供建议分类";
    byId("result-priority").textContent = result.priority == null ? "优先级未提供" : "优先级 P" + String(result.priority);
    const action = byId("result-action");
    action.hidden = !result.action || (result.category === "UNKNOWN" && result.action === "HUMAN_TRIAGE");
    if (!action.hidden) action.textContent = "建议动作：" + label(actionLabels, result.action);
    const failedProviders = (envelope.route_steps || []).filter(
      (step) => ["jev", "llm"].includes(step.stage) && step.status === "failed",
    );
    const failureNote = byId("provider-failure-note");
    failureNote.hidden = failedProviders.length === 0;
    if (!failureNote.hidden) {
      failureNote.textContent = "Provider 执行失败（" + failedProviders.map((step) => label(stageLabels, step.stage)).join("、") + "）；结果已降级，需人工审核。";
    }
    const rawReason = result.reason || "本结果未提供分诊依据。";
    const reviewerReasons = {
      "offline LLM stub abstained; human review required": "离线判断未能确认分类，建议由人工核对。",
    };
    byId("result-reason").textContent = reviewerReasons[rawReason] || rawReason;
    byId("result-reason-raw").textContent = rawReason;
    renderReasonEvidence(envelope.evidence_refs || []);
    renderMissingEvidence(envelope.missing_evidence || []);
    setBadge(byId("review-status"), label(statusLabels, envelope.review_status), reviewBadge(envelope.review_status));
    byId("result-version").textContent = envelope.result_version;
    byId("schema-value").textContent = String(envelope.schema_version || "未提供");
    byId("policy-value").textContent = String(envelope.policy_id) + " · v" + String(envelope.policy_version);
    byId("mode-value").textContent = envelope.execution_mode;
    byId("delivery-value").textContent = label(statusLabels, envelope.delivery_status);
    byId("created-value").textContent = formatDate(envelope.created_at);
    byId("case-provenance-value").textContent = input
      ? [input.case_id, input.provenance && input.provenance.dataset_version ? "dataset " + input.provenance.dataset_version : null].filter(Boolean).join(" · ") || "case_id / provenance 未提供"
      : "case_id 与 provenance 未由当前持久化投影保存";
    byId("input-hash-value").textContent = String(envelope.input_hash || "未提供");
    byId("projection-value").textContent = envelope.input_projection_version + " · " + envelope.projection_hash;
    byId("run-status").textContent = label(statusLabels, envelope.run_status);
    byId("category-action-value").textContent = [result.category, result.action].filter(Boolean).join(" · ") || "未提供";
    byId("confidence-value").textContent = result.confidence == null ? "未提供" : String(result.confidence) + "（沿用系统值，未由审核人重评）";
    byId("terminal-stage").textContent = envelope.terminal_stage ? "终点：" + label(stageLabels, envelope.terminal_stage) : "终点未提供";
    renderRoute(envelope.route_steps || []);
    renderAccounting(envelope);
    renderProviderExecution(state.executionMetadata.get(envelope.request_id));
    renderFacts(envelope, state.facts, state.evidence, input);
    renderEvidencePreview(state.facts, input);
    renderHistory(state.reviews, envelope);
    byId("decision-run-status").textContent = "整体运行状态：" + label(statusLabels, envelope.run_status) + " · " + label({ offline: "离线运行" }, envelope.execution_mode);
    const canReview = Boolean(result.category) && ["pending", "deferred"].includes(envelope.review_status);
    byId("review-bar").hidden = !canReview;
    byId("review-bar-title").textContent = canReview ? "审核当前结果版本" : "此版本当前不可审核";
    byId("review-bar-copy").textContent = canReview ? "接受只记录建议审核；修改会生成新版本；暂缓保留在待办中。" : "审核历史仍可查看；本地审核不会把故障标记为已解决。";
    renderReport(state.reports.get(envelope.request_id), envelope);
  }

  function renderReasonEvidence(refs) {
    const holder = byId("reason-evidence");
    const list = byId("reason-evidence-list");
    clear(list);
    holder.hidden = !refs.length;
    refs.forEach((ref) => {
      const item = make("li");
      item.append(make("code", "", ref.field_path), make("small", "", ref.capture_scope === "provided_excerpt" ? "提交摘录" : "提交字段"));
      list.appendChild(item);
    });
  }

  function renderMissingEvidence(items) {
    const panel = byId("missing-evidence-panel");
    const list = byId("missing-evidence-list");
    clear(list);
    panel.hidden = !items.length;
    items.forEach((item) => list.appendChild(make("li", "", item)));
  }

  function renderRoute(routeSteps) {
    const list = byId("route-list");
    clear(list);
    byId("route-raw-data").textContent = JSON.stringify(routeSteps, null, 2);
    byId("route-summary").textContent = routeSteps.length
      ? routeSteps.map((step) => label(stageLabels, step.stage) + (step.provider_mock ? "（离线模拟）" : "")).join(" → ")
      : "服务没有返回判断路径记录";
    if (!routeSteps.length) {
      list.appendChild(make("li", "inline-note", "后端没有记录 route_steps。"));
      return;
    }
    routeSteps.forEach((step) => {
      const row = make("li", "route-step" + (step.stage === "human" ? " active" : ""));
      row.append(make("strong", "", label(stageLabels, step.stage) + " · " + label(routeStatusLabels, step.status)));
      const detail = String(step.attempts) + " 次尝试" + (step.provider_mock ? " · 离线模拟" : "");
      row.append(make("span", "", detail));
      const reason = routeReasonLabels[step.reason_code] || (step.reason_code ? "该阶段已记录判断原因。" : "该阶段未提供补充说明。");
      row.append(make("p", "route-explanation", reason));
      list.appendChild(row);
    });
  }

  function renderAccounting(envelope) {
    byId("total-latency").textContent = numeric(envelope.total_latency_ms, 2) + " ms";
    byId("total-cost").textContent = envelope.total_cost == null ? "未提供" : String(envelope.total_cost) + (envelope.cost_currency ? " " + envelope.cost_currency : "");
    byId("pricing-status").textContent = String(envelope.pricing_status);
    const body = byId("attempt-rows");
    clear(body);
    (envelope.attempts || []).forEach((attempt) => {
      const row = document.createElement("tr");
      const provider = String(attempt.provider_id || attempt.stage) + (attempt.provider_mock ? " · mock/stub" : "");
      const tokens = "in " + (attempt.input_tokens == null ? "未提供" : attempt.input_tokens) + " / out " + (attempt.output_tokens == null ? "未提供" : attempt.output_tokens);
      const cost = attempt.cost == null ? "未提供" : String(attempt.cost) + (attempt.cost_currency ? " " + attempt.cost_currency : "");
      [String(attempt.stage) + " / " + provider, String(attempt.status), String(attempt.attempt_index), tokens, numeric(attempt.latency_ms, 2) + " ms", cost].forEach((value) => row.appendChild(make("td", "", value)));
      body.appendChild(row);
    });
    if (!(envelope.attempts || []).length) {
      const row = make("tr");
      row.appendChild(make("td", "", "没有 attempt 记录"));
      body.appendChild(row);
    }
    byId("raw-attempts").textContent = JSON.stringify(envelope.attempts || [], null, 2);
  }

  function renderProviderExecution(record) {
    const holder = byId("provider-model-list");
    clear(holder);
    const rows = record && Array.isArray(record.items) ? record.items : [];
    if (!rows.length) {
      holder.appendChild(make("p", "inline-note", "该版本没有独立的 Provider 模型版本记录。"));
    }
    rows.forEach((item) => {
      const row = make("div", "provider-model-row");
      const resolutionLabel = item.resolution_source === "provider_response"
        ? "版本字段来自 Provider 响应"
        : "Provider 未返回版本字段";
      row.append(
        make("strong", "", label(stageLabels, item.stage) + " · " + String(item.provider_id)),
        make("span", "", "请求：" + String(item.requested_model || "未提供")),
        make("span", "", "响应：" + String(item.resolved_model || "Provider 未返回")),
        make("small", "", "第 " + String(item.attempt_index) + " 次应用调用 · " + resolutionLabel),
      );
      holder.appendChild(row);
    });
    byId("raw-provider-metadata").textContent = JSON.stringify(record || [], null, 2);
  }

  function displayFact(value) {
    if (value == null) return "未提供";
    if (typeof value === "object") return JSON.stringify(value, null, 2);
    return String(value);
  }

  function renderFacts(envelope, facts, evidence, input) {
    const grid = byId("facts-grid");
    clear(grid);
    byId("case-id-note").hidden = false;
    if (input && input.case_id) {
      byId("case-id-note").textContent = "case_id：" + input.case_id + "（本次提交后保留在当前页面内存；SQLite 的 common-facts-v1 投影不含 case_id 或 provenance）。";
    } else {
      byId("case-id-note").textContent = "case_id 与 provenance 不在持久化的 common-facts-v1 投影中；页面不会从模型结果推造它们。";
    }
    if (facts) {
      Object.keys(facts).filter((key) => key !== "schema_version").forEach((key) => {
        const cell = make("div", "fact" + (key === "syslog_excerpt" || key === "env" ? " wide" : ""));
        cell.append(make("dt", "", fieldLabels[key] || key), make("dd", "", displayFact(facts[key])));
        grid.appendChild(cell);
      });
    } else {
      grid.appendChild(make("div", "inline-note", "服务没有返回该版本的 facts 投影。"));
    }
    const list = byId("evidence-list");
    clear(list);
    const audit = byId("evidence-audit");
    clear(audit);
    audit.parentElement.hidden = !(evidence && evidence.length);
    byId("evidence-empty").hidden = Boolean(evidence && evidence.length);
    (evidence || []).forEach((item) => {
      const row = make("div", "evidence-item");
      row.append(
        make("code", "", "/" + item.field_path),
        make("blockquote", "", displayFact(item.value)),
      );
      list.appendChild(row);
      const auditRow = make("div", "evidence-audit-row");
      auditRow.append(make("code", "", "/" + item.field_path), make("span", "", String(item.capture_scope) + " · root cause verified: " + String(item.root_cause_verified) + " · sha256: " + String(item.content_sha256 || "未提供")));
      audit.appendChild(auditRow);
    });
  }

  function renderEvidencePreview(facts, input) {
    const holder = byId("evidence-preview");
    clear(holder);
    const source = facts || input || {};
    const excerpt = source.syslog_excerpt;
    if (typeof excerpt !== "string" || !excerpt.trim()) {
      const fallback = source.assertion
        ? "未提交日志摘录；失败断言见上方摘要。"
        : "该版本没有可展示的提交证据文本。";
      holder.appendChild(make("p", "evidence-preview-empty", fallback));
      return;
    }
    const item = make("blockquote", "evidence-preview-item");
    item.append(make("strong", "", "提交日志摘录"), make("span", "", excerpt));
    holder.appendChild(item);
  }

  function renderHistory(reviews, envelope) {
    const list = byId("history-list");
    clear(list);
    byId("history-count").textContent = String(reviews.length) + " 条审核记录";
    byId("history-empty").hidden = reviews.length !== 0;
    reviews.forEach((review) => {
      const card = make("article", "history-event");
      const header = make("header");
      const title = make("strong", "", label(reviewLabels, review.decision) + " · " + review.result_version);
      const date = make("time", "", formatDate(review.reviewed_at));
      header.append(title, date);
      card.appendChild(header);
      card.appendChild(make("small", "", String(review.actor_origin) + " · " + String(review.actor_type) + " · 身份未验证"));
      if (review.reason) card.appendChild(make("p", "", review.reason));
      if (review.after_result) {
        card.appendChild(make("p", "", "修订：" + review.after_result.category + " / " + review.after_result.action + " / P" + review.after_result.priority + " · " + review.after_result.reason));
        const confidenceText = review.confidence_origin === "inherited_unreassessed"
          ? "Confidence=" + String(review.after_result.confidence) + "，沿用修订前值，审核人未重评。"
          : "Confidence 来源未在此审核事件中记录。";
        card.appendChild(make("small", "", confidenceText));
      }
      list.appendChild(card);
    });
  }

  function renderReport(record, envelope) {
    const holder = byId("run-report-summary");
    if (!holder) return;
    if (!record || !record.report) {
      holder.hidden = true;
      return;
    }
    const report = record.report;
    holder.hidden = false;
    const accuracy = report.accuracy == null ? "null · " + String(report.accuracy_null_reason || "无标签") : String(report.accuracy);
    const versionNote = record.report_result_version === envelope.result_version ? "" : " · 绑定原始结果版本 " + String(record.report_result_version);
    holder.textContent = "已保存的运行报告：accuracy " + accuracy + " · eligibility " + String(report.eligibility && report.eligibility.status) + " · 已评估 " + String(report.evaluated_count) + "/" + String(report.sample_count) + versionNote + "。";
  }

  function validateFailureState(value) {
    const errors = [];
    if (!value || typeof value !== "object" || Array.isArray(value)) return ["顶层必须是 JSON 对象。"];
    const required = ["case_id", "result", "previous_runs", "device", "sw_version", "assertion", "syslog_excerpt", "env", "retry", "provenance"];
    required.filter((key) => !Object.prototype.hasOwnProperty.call(value, key)).forEach((key) => errors.push("缺少必填字段：" + key));
    const allowed = new Set(required);
    Object.keys(value).filter((key) => !allowed.has(key)).forEach((key) => errors.push("未知字段：" + key));
    ["case_id", "device", "sw_version", "assertion"].forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(value, key) && (typeof value[key] !== "string" || !value[key].trim())) errors.push(key + " 必须是非空字符串。");
    });
    if (value.result !== "FAIL") errors.push("result 必须为 FAIL。");
    if (!Array.isArray(value.previous_runs) || value.previous_runs.length > 10 || value.previous_runs.some((item) => item !== "PASS" && item !== "FAIL")) errors.push("previous_runs 必须是最多 10 项的 PASS/FAIL 数组。");
    if (typeof value.syslog_excerpt !== "string" || value.syslog_excerpt.length > 8192) errors.push("syslog_excerpt 必须是字符串且不超过 8192 字符。");
    if (!value.env || typeof value.env !== "object" || Array.isArray(value.env) || Object.values(value.env).some((item) => typeof item !== "boolean" && typeof item !== "string")) errors.push("env 必须是值为布尔值或字符串的对象。");
    if (!["not_executed", "passed", "failed"].includes(value.retry)) errors.push("retry 值不符合 FailureState 合同。");
    if (!value.provenance || typeof value.provenance !== "object" || Array.isArray(value.provenance)) {
      errors.push("provenance 必须是对象。");
    } else {
      if (typeof value.provenance.dataset_version !== "string" || !value.provenance.dataset_version.trim()) errors.push("provenance.dataset_version 必须是非空字符串。");
      if (typeof value.provenance.synthetic !== "boolean") errors.push("provenance.synthetic 必须是布尔值。");
      Object.keys(value.provenance).filter((key) => !["dataset_version", "synthetic", "source"].includes(key)).forEach((key) => errors.push("未知 provenance 字段：" + key));
      if (value.provenance.source != null && typeof value.provenance.source !== "string") errors.push("provenance.source 必须是字符串或 null。");
    }
    const bytes = new TextEncoder().encode(JSON.stringify(value)).length;
    if (bytes > 64 * 1024) errors.push("FailureState 超过本地服务 64 KiB 请求上限。");
    return errors;
  }

  function submissionCandidates() {
    return state.submissionItems.filter((item) => item.status === "ready" || item.status === "failed");
  }

  function uuidSignature(value) {
    return JSON.stringify(value);
  }

  function renderSubmissionBatch() {
    const list = byId("batch-list");
    const items = state.submissionItems;
    clear(list);
    list.hidden = items.length === 0;
    byId("clear-batch").hidden = items.length === 0;
    const counts = { ready: 0, invalid: 0, submitting: 0, succeeded: 0, failed: 0 };
    items.forEach((item) => { counts[item.status] = (counts[item.status] || 0) + 1; });
    const summary = byId("batch-summary");
    summary.hidden = items.length === 0;
    if (items.length) {
      summary.textContent = `${items.length} 个文件：待提交 ${counts.ready}，格式错误 ${counts.invalid}，处理中 ${counts.submitting}，已完成 ${counts.succeeded}，失败 ${counts.failed}`;
    }

    const statusText = {
      ready: "待提交",
      invalid: "格式错误，已跳过",
      submitting: state.executionProfile === "live-advisory" ? "真实分诊中" : "离线分诊中",
      succeeded: "已完成",
      failed: "失败，可重试",
    };
    items.forEach((item) => {
      const row = make("li", "batch-item");
      row.dataset.status = item.status;
      const heading = make("div", "batch-item-heading");
      heading.append(make("strong", "", item.filename));
      const tools = make("div", "batch-item-tools");
      tools.append(make("span", "batch-item-status", statusText[item.status] || item.status));
      const remove = make("button", "button quiet batch-remove", "移除");
      remove.type = "button";
      remove.disabled = state.submittingBatch || state.loadingBatchFiles;
      remove.addEventListener("click", () => {
        state.submissionItems = state.submissionItems.filter((candidate) => candidate.id !== item.id);
        renderSubmissionBatch();
      });
      tools.append(remove);
      heading.append(tools);
      row.append(heading);
      const detail = item.errors.length
        ? item.errors.join(" ")
        : item.requestId
          ? `请求 ${item.requestId}${item.resultSummary ? " · " + item.resultSummary : ""}`
          : item.status === "submitting" ? "正在按文件顺序提交。" : "FailureState 结构完整。";
      row.append(make("small", "batch-item-meta", detail));
      list.append(row);
    });

    const hasFiles = items.length > 0;
    byId("failure-file").disabled = state.submittingBatch || state.loadingBatchFiles;
    byId("clear-batch").disabled = state.submittingBatch || state.loadingBatchFiles;
    byId("stop-batch").hidden = !state.submittingBatch;
    byId("stop-batch").disabled = state.stopBatchRequested;
    byId("stop-batch").textContent = state.stopBatchRequested ? "正在等待当前请求结束" : "当前请求结束后停止";
    byId("failure-json").disabled = hasFiles || state.submittingBatch || state.loadingBatchFiles;
    byId("input-mode-note").textContent = hasFiles
      ? "当前按文件列表分诊；格式错误的文件会跳过。需要粘贴单条内容时，先清空文件列表。"
      : "可一次选择多个 JSON 文件；不选文件时，也可在下方粘贴单条 FailureState JSON。";

    const candidates = submissionCandidates();
    const manualText = byId("failure-json").value.trim();
    const canSubmitManual = !hasFiles && manualText.length > 0;
    const button = byId("submit-button");
    if (state.submittingBatch) {
      const progress = state.batchProgress || { completed: 0, total: 0 };
      button.textContent = `正在分诊 ${progress.completed}/${progress.total}…`;
    } else if (candidates.length) {
      const live = state.executionProfile === "live-advisory";
      const retryOnly = candidates.every((item) => item.status === "failed");
      button.textContent = retryOnly
        ? `${live ? "真实 API" : "离线"}重试失败项（${candidates.length}）`
        : candidates.length > 1
          ? `${live ? "真实 API " : "离线"}批量分诊（${candidates.length}）`
          : live ? "真实 API 分诊" : "离线分诊并入队";
    } else if (items.length && counts.succeeded === items.length) {
      button.textContent = "本批次已完成";
    } else if (items.length) {
      button.textContent = "没有可提交文件";
    } else {
      button.textContent = state.executionProfile === "live-advisory" ? "真实 API 分诊" : "离线分诊并入队";
    }
    button.disabled = !state.executionProfile || state.submittingBatch || state.loadingBatchFiles || (!candidates.length && !canSubmitManual);
  }

  function makeSubmissionItem(filename, raw, failureState, errors) {
    return {
      id: newKey(),
      filename,
      raw,
      failureState,
      errors,
      status: errors.length ? "invalid" : "ready",
      idempotencyKey: null,
      requestId: null,
      resultSummary: "",
    };
  }

  async function submitFailureState() {
    const errorNode = byId("submit-error");
    errorNode.textContent = "";
    if (state.submittingBatch || state.loadingBatchFiles) return;

    if (!state.submissionItems.length) {
      const raw = byId("failure-json").value;
      let failureState;
      try { failureState = JSON.parse(raw); }
      catch (_error) { errorNode.textContent = "JSON 格式错误，请检查逗号和引号。"; return; }
      const errors = validateFailureState(failureState);
      if (errors.length) { errorNode.textContent = errors.join(" "); return; }
      state.submissionItems.push(makeSubmissionItem("粘贴内容", raw, failureState, []));
      byId("failure-json").value = "";
      renderSubmissionBatch();
    }

    const candidates = submissionCandidates();
    if (!candidates.length) {
      errorNode.textContent = "没有结构有效且待提交的文件。请移除错误文件或重新选择。";
      return;
    }
    const liveConfirmed = state.executionProfile === "live-advisory";
    const invalidCount = state.submissionItems.filter((item) => item.status === "invalid").length;
    if (liveConfirmed) {
      const confirmText = `本次将依次对 ${candidates.length} 个 FailureState 调用真实分诊。每个文件的本次服务端请求最多触发一次 Jev 和一次 Strong LLM 应用层尝试，具体阶段由路由决定，可能产生费用。${invalidCount ? `另有 ${invalidCount} 个格式错误文件会跳过。` : ""}结果仅作为建议，仍需人工审核。是否继续？`;
      if (!window.confirm(confirmText)) return;
    }

    state.submittingBatch = true;
    state.stopBatchRequested = false;
    state.batchProgress = { completed: 0, total: candidates.length };
    renderSubmissionBatch();
    let succeeded = 0;
    let failed = 0;
    let attempted = 0;
    let lastSuccess = null;
    for (const item of candidates) {
      if (state.stopBatchRequested) break;
      item.status = "submitting";
      item.errors = [];
      item.idempotencyKey = item.idempotencyKey || newKey();
      renderSubmissionBatch();
      attempted += 1;
      try {
        const result = await request("/v1/triage", {
          method: "POST",
          headers: { "X-CSRF-Token": state.csrfToken, "Idempotency-Key": item.idempotencyKey },
          body: JSON.stringify({ failure_state: item.failureState, live_confirmed: liveConfirmed }),
        });
        const envelope = result.payload.envelope;
        item.status = "succeeded";
        item.requestId = envelope.request_id;
        item.resultSummary = envelope.result
          ? `${label(resultLabels, envelope.result.category)} / ${label(actionLabels, envelope.result.action)}`
          : label(statusLabels, envelope.run_status);
        state.ephemeralInputs.set(envelope.request_id, item.failureState);
        succeeded += 1;
        lastSuccess = envelope.request_id;
      } catch (error) {
        item.status = "failed";
        item.errors = [explainError(error)];
        failed += 1;
      }
      state.batchProgress.completed += 1;
      renderSubmissionBatch();
    }

    state.submittingBatch = false;
    state.batchProgress = null;
    const stopped = state.stopBatchRequested && attempted < candidates.length;
    state.stopBatchRequested = false;
    renderSubmissionBatch();
    if (succeeded) {
      try {
        await refreshQueue({ keepCurrent: true });
        if (lastSuccess) await loadCase(lastSuccess);
      } catch (error) {
        errorNode.textContent = "分诊结果已返回，但刷新本地队列失败：" + explainError(error);
      }
    }
    const skipped = state.submissionItems.filter((item) => item.status === "invalid").length;
    toast(stopped
      ? `已停止后续提交：本轮成功 ${succeeded}，失败 ${failed}，尚未开始 ${candidates.length - attempted}。当前请求已完成。`
      : `批次完成：成功 ${succeeded}，失败 ${failed}，格式错误跳过 ${skipped}。`);
  }

  function showReviewDialog(mode) {
    if (!state.envelope || !state.envelope.result) return;
    state.reviewMode = mode;
    state.pendingReview = null;
    const idPrefix = state.envelope.request_id.slice(0, 8);
    const expected = mode + " " + idPrefix;
    byId("dialog-title").textContent = label(reviewLabels, mode);
    byId("dialog-copy").textContent = mode === "accept"
      ? "接受只记录本地审核决定。故障是否解决仍需独立确认。"
      : mode === "correct"
        ? "修订分类、动作、优先级和理由。此操作会基于当前 If-Match 版本生成新的待审核版本。"
        : "说明需要补充的证据或暂缓原因。此操作会保留本地待办。";
    byId("correct-fields").hidden = mode !== "correct";
    byId("correct-category").required = mode === "correct";
    byId("correct-action").required = mode === "correct";
    byId("correct-priority").required = mode === "correct";
    byId("review-reason").required = mode === "correct" || mode === "defer";
    byId("operator-label").value = "";
    byId("review-reason").value = "";
    byId("confirmation").value = "";
    byId("confirmation-prompt").textContent = expected;
    byId("dialog-error").textContent = "";
    if (mode === "correct") {
      byId("correct-category").value = state.envelope.result.category;
      byId("correct-action").value = state.envelope.result.action;
      byId("correct-priority").value = String(state.envelope.result.priority);
      byId("confidence-copy").textContent = String(state.envelope.result.confidence) + " · 从当前系统结果保留；没有人工重评。";
      byId("review-reason").maxLength = 200;
    } else {
      byId("review-reason").maxLength = 2000;
    }
    byId("review-dialog").showModal();
    byId("operator-label").focus();
  }

  async function submitReview(event) {
    event.preventDefault();
    const mode = state.reviewMode;
    const envelope = state.envelope;
    if (!mode || !envelope) return;
    const operator = byId("operator-label").value.trim();
    const reason = byId("review-reason").value.trim();
    const confirmation = byId("confirmation").value;
    const expected = mode + " " + envelope.request_id.slice(0, 8);
    if (!operator) { byId("dialog-error").textContent = "请填写审核人自述标签。"; return; }
    if (confirmation !== expected) { byId("dialog-error").textContent = "确认文本必须逐字匹配提示。"; return; }
    if ((mode === "correct" || mode === "defer") && !reason) { byId("dialog-error").textContent = "修改和暂缓必须填写理由。"; return; }
    if (mode === "correct" && reason.length > 200) { byId("dialog-error").textContent = "修订结果理由最多 200 字。"; return; }
    const body = { decision: mode, operator_label: operator, confirmation: confirmation };
    if (mode === "correct") {
      body.reason = reason;
      body.after_result = {
        category: byId("correct-category").value,
        action: byId("correct-action").value,
        priority: Number(byId("correct-priority").value),
        confidence: envelope.result.confidence,
        reason: reason,
      };
    } else if (reason) body.reason = reason;
    const signature = uuidSignature({ request_id: envelope.request_id, etag: state.etag, body: body });
    if (!state.pendingReview || state.pendingReview.signature !== signature) state.pendingReview = { signature, key: newKey() };
    const submit = byId("review-submit");
    submit.disabled = true;
    byId("dialog-error").textContent = "";
    try {
      const result = await request("/v1/triage/" + encodeURIComponent(envelope.request_id) + "/reviews", {
        method: "POST",
        headers: {
          "X-CSRF-Token": state.csrfToken,
          "Idempotency-Key": state.pendingReview.key,
          "If-Match": state.etag,
        },
        body: JSON.stringify(body),
      });
      state.pendingReview = null;
      state.envelope = result.payload.envelope;
      state.etag = result.response.headers.get("ETag") || ('"' + state.envelope.result_version + '"');
      byId("review-dialog").close();
      await refreshQueue({ keepCurrent: true });
      await loadCase(envelope.request_id);
      toast(label(reviewLabels, mode) + "已写入本地审核历史；故障解决状态未由此动作确认。");
    } catch (error) {
      byId("dialog-error").textContent = explainError(error);
      if (error instanceof ApiError && ["result_version_conflict", "state_conflict", "review_not_pending"].includes(error.code)) {
        await refreshQueue({ keepCurrent: true }).catch(() => {});
        await loadCase(envelope.request_id);
      }
    } finally {
      submit.disabled = false;
    }
  }

  function readFileText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("无法读取文件。"));
      reader.readAsText(file);
    });
  }

  async function importFiles(files) {
    const selected = Array.from(files || []);
    if (!selected.length || state.submittingBatch || state.loadingBatchFiles) return;
    byId("submit-error").textContent = "";
    state.loadingBatchFiles = true;
    renderSubmissionBatch();
    const imported = await Promise.all(selected.map(async (file) => {
      if (file.size > 64 * 1024) {
        return makeSubmissionItem(file.name, "", null, ["文件超过本地服务 64 KiB 请求上限。"]);
      }
      try {
        const raw = await readFileText(file);
        let failureState;
        try { failureState = JSON.parse(raw); }
        catch (_error) { return makeSubmissionItem(file.name, raw, null, ["JSON 格式错误，请检查逗号和引号。"]); }
        return makeSubmissionItem(file.name, raw, failureState, validateFailureState(failureState));
      } catch (error) {
        return makeSubmissionItem(file.name, "", null, [error.message || "无法读取文件。"]);
      }
    }));
    state.submissionItems.push(...imported);
    state.loadingBatchFiles = false;
    renderSubmissionBatch();
    const validCount = imported.filter((item) => item.status === "ready").length;
    const invalidCount = imported.length - validCount;
    byId("submit-error").textContent = `已添加 ${imported.length} 个文件：有效 ${validCount}，格式错误 ${invalidCount}。选择批量分诊后才会发送。`;
    byId("failure-file").value = "";
  }

  async function boot() {
    try {
      const session = await request("/v1/session");
      state.csrfToken = session.payload.csrf_token;
      const health = await request("/v1/health");
      const status = health.payload;
      if (status.profile === "live-advisory" && status.execution_mode === "live" && status.run_purpose === "advisory" && status.provider_mode === "live") {
        state.executionProfile = "live-advisory";
        setBadge(byId("execution-badge"), "真实 API 测试", "waiting");
        byId("live-warning").hidden = false;
        byId("submit-mode-note").textContent = "完整输入在本页内存中显示到刷新；SQLite 只持久化当前合同定义的 facts 投影。";
      } else if (status.profile === "offline" && status.execution_mode === "offline" && status.provider_mode === "mock") {
        state.executionProfile = "offline";
      } else {
        throw new Error("服务运行模式与 Provider 状态不一致");
      }
      renderSubmissionBatch();
      await refreshQueue({ keepCurrent: true });
      const queryId = new URL(window.location.href).searchParams.get("request");
      if (queryId) await loadCase(queryId);
      else if (state.queue.length) await loadCase(state.queue[0].envelope.request_id);
    } catch (error) {
      setBadge(byId("execution-badge"), "本地服务不可用", "waiting");
      byId("queue-empty").hidden = false;
      byId("queue-empty").querySelector("strong").textContent = "无法连接本地服务";
      byId("queue-empty").querySelector("span").textContent = explainError(error);
      byId("submit-button").disabled = true;
      toast(explainError(error));
    }
  }

  byId("refresh-button").addEventListener("click", async () => {
    try {
      await refreshQueue({ keepCurrent: true });
      if (state.requestId) await loadCase(state.requestId);
      toast("已从本地服务刷新队列和当前版本。");
    } catch (error) { toast(explainError(error)); }
  });
  byId("submit-button").addEventListener("click", submitFailureState);
  byId("failure-file").addEventListener("change", (event) => importFiles(event.target.files));
  byId("clear-batch").addEventListener("click", () => {
    if (state.submittingBatch || state.loadingBatchFiles) return;
    state.submissionItems = [];
    byId("submit-error").textContent = "";
    renderSubmissionBatch();
  });
  byId("stop-batch").addEventListener("click", () => {
    if (!state.submittingBatch) return;
    state.stopBatchRequested = true;
    renderSubmissionBatch();
  });
  byId("failure-json").addEventListener("input", renderSubmissionBatch);
  byId("review-form").addEventListener("submit", submitReview);
  byId("dialog-cancel").addEventListener("click", () => byId("review-dialog").close());
  byId("review-dialog").addEventListener("click", (event) => { if (event.target === byId("review-dialog")) byId("review-dialog").close(); });
  document.querySelectorAll("[data-review]").forEach((button) => button.addEventListener("click", () => showReviewDialog(button.dataset.review)));
  boot();
})();
