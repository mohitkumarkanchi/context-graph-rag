/**
 * Main application logic.
 *
 * Handles tab switching, chat interactions (sends to both
 * pipelines simultaneously), scenario execution, and
 * WebSocket connection for streaming updates.
 */

const API = `${window.location.origin}/api`;
let sessionId = crypto.randomUUID();
let turnNumber = 0;
let ws = null;

console.log("App initialized, API base:", API);

// ─── Tab switching ──────────────────────────────────

document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
    });
});

// ─── Chat ───────────────────────────────────────────

const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const btnReset = document.getElementById("btn-reset");
const chatBasic = document.getElementById("chat-basic");
const chatContext = document.getElementById("chat-context");

/**
 * Send a chat query to both pipelines simultaneously.
 */
async function sendChat() {
    const query = chatInput.value.trim();
    if (!query) return;

    turnNumber++;
    chatInput.value = "";
    btnSend.disabled = true;

    // Show user message in both panes
    appendMessage(chatBasic, query, "user");
    appendMessage(chatContext, query, "user");

    // Show loading indicators
    const basicLoading = appendLoading(chatBasic);
    const contextLoading = appendLoading(chatContext);

    try {
        // Fire both requests in parallel
        const [basicRes, contextRes] = await Promise.all([
            fetchChat(query, "basic"),
            fetchChat(query, "context"),
        ]);

        // Remove loading indicators
        basicLoading.remove();
        contextLoading.remove();

        // Show basic response
        appendBotMessage(chatBasic, basicRes);

        // Show context response (with augmented query if present)
        appendBotMessage(chatContext, contextRes);

        // Update context info bar
        updateContextInfo(contextRes);
    } catch (err) {
        basicLoading.remove();
        contextLoading.remove();
        appendMessage(chatBasic, `Error: ${err.message}`, "bot");
        appendMessage(chatContext, `Error: ${err.message}`, "bot");
    }

    btnSend.disabled = false;
    chatInput.focus();
}

/**
 * Call the /chat endpoint for a specific mode.
 */
async function fetchChat(query, mode) {
    const url = `${API}/chat`;
    const body = {
        query,
        mode,
        session_id: `${sessionId}_${mode}`,
        turn_number: turnNumber,
    };

    console.log(`[${mode}] POST ${url}`, body);

    let resp;
    try {
        resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
    } catch (networkErr) {
        console.error(`[${mode}] Network error:`, networkErr);
        throw new Error(`Cannot reach server. Is it running at ${window.location.origin}?`);
    }

    if (!resp.ok) {
        const errBody = await resp.text();
        console.error(`[${mode}] HTTP ${resp.status}:`, errBody);
        throw new Error(`HTTP ${resp.status}: ${errBody}`);
    }

    return resp.json();
}

/**
 * Append a user or plain bot message bubble.
 */
function appendMessage(container, text, role) {
    const div = document.createElement("div");
    div.className = `msg msg-${role}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

/**
 * Append a bot message with metadata.
 */
function appendBotMessage(container, data) {
    const wrapper = document.createElement("div");

    // Show augmented query if different from original
    if (data.augmented_query && data.augmented_query !== data.response) {
        const aug = document.createElement("div");
        aug.className = "msg-augmented";
        aug.textContent = `Augmented: "${data.augmented_query}"`;
        wrapper.appendChild(aug);
    }

    // Response bubble
    const msg = document.createElement("div");
    msg.className = "msg msg-bot";
    msg.textContent = data.response;
    wrapper.appendChild(msg);

    // Timing metadata
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const totalMs = Math.round(data.retrieval_time_ms + data.generation_time_ms);
    meta.textContent = `${totalMs}ms · ${data.extracted_entities.length} entities · ${data.sources.length} sources`;
    wrapper.appendChild(meta);

    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}

/**
 * Append a loading indicator.
 */
function appendLoading(container) {
    const div = document.createElement("div");
    div.className = "msg msg-bot msg-loading";
    div.innerHTML = '<span class="spinner"></span>Thinking...';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

/**
 * Update the context info bar below the chat.
 */
function updateContextInfo(contextRes) {
    const bar = document.getElementById("context-info");
    bar.classList.remove("hidden");
    document.getElementById("context-turn").textContent = `Turn: ${turnNumber}`;
    document.getElementById("context-entities").textContent =
        `Entities: ${contextRes.extracted_entities.length}`;
    document.getElementById("context-refs").textContent =
        `Sources: ${contextRes.sources.length}`;
}

/**
 * Reset both sessions.
 */
async function resetChat() {
    // Reset server-side sessions
    await fetch(`${API}/context/${sessionId}_context`, { method: "DELETE" }).catch(() => {});

    // Reset client state
    sessionId = crypto.randomUUID();
    turnNumber = 0;
    chatBasic.innerHTML = "";
    chatContext.innerHTML = "";
    document.getElementById("context-info").classList.add("hidden");
    chatInput.focus();
}

// Event listeners
btnSend.addEventListener("click", sendChat);
btnReset.addEventListener("click", resetChat);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChat();
    }
});

// ─── Scenario ───────────────────────────────────────

const btnScenario = document.getElementById("btn-run-scenario");
const scenarioResults = document.getElementById("scenario-results");
const scenarioSummary = document.getElementById("scenario-summary");

/**
 * Run the full 6-turn RCA scenario evaluation.
 */
async function runScenario() {
    btnScenario.disabled = true;
    btnScenario.textContent = "Running...";
    scenarioResults.innerHTML = '<p class="scenario-placeholder"><span class="spinner"></span>Running 6-turn RCA scenario through both pipelines...</p>';
    scenarioSummary.classList.add("hidden");

    try {
        const resp = await fetch(`${API}/evaluate/scenario`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const report = await resp.json();
        renderScenarioReport(report);
    } catch (err) {
        scenarioResults.innerHTML = `<p class="scenario-placeholder">Error: ${err.message}</p>`;
    }

    btnScenario.disabled = false;
    btnScenario.textContent = "Run scenario";
}

/**
 * Render the scenario report as turn cards.
 */
function renderScenarioReport(report) {
    scenarioResults.innerHTML = "";

    for (const turn of report.turns) {
        const card = document.createElement("div");
        card.className = "turn-card";

        const basicTime = Math.round(turn.basic.retrieval_time_ms + turn.basic.generation_time_ms);
        const ctxTime = Math.round(turn.context.retrieval_time_ms + turn.context.generation_time_ms);

        const augNote = turn.context.augmented_query && turn.context.augmented_query !== turn.query
            ? `<div class="augmented-note">Augmented: "${escapeHtml(turn.context.augmented_query)}"</div>`
            : "";

        card.innerHTML = `
            <div class="turn-header">
                <span class="turn-number">Turn ${turn.turn_number}</span>
                <span class="turn-query">${escapeHtml(turn.query)}</span>
            </div>
            <div class="turn-body">
                <div class="turn-response">
                    <h4 class="basic-label">Basic RAG</h4>
                    <p>${escapeHtml(turn.basic.response)}</p>
                    <div class="timing">${basicTime}ms · ${turn.basic.extracted_entities.length} entities</div>
                </div>
                <div class="turn-response">
                    <h4 class="context-label">Context RAG</h4>
                    ${augNote}
                    <p>${escapeHtml(turn.context.response)}</p>
                    <div class="timing">${ctxTime}ms · ${turn.context.extracted_entities.length} entities</div>
                </div>
            </div>
        `;

        scenarioResults.appendChild(card);
    }

    // Show summary
    if (report.summary) {
        scenarioSummary.textContent = report.summary;
        scenarioSummary.classList.remove("hidden");
    }
}

btnScenario.addEventListener("click", runScenario);

// ─── Graph tab buttons ──────────────────────────────

document.getElementById("btn-load-kg").addEventListener("click", async () => {
    try {
        const resp = await fetch(`${API}/graph`);
        const data = await resp.json();
        renderForceGraph("#kg-svg", data.nodes, data.links);
    } catch (err) {
        console.error("Failed to load KG:", err);
    }
});

document.getElementById("btn-load-ctx").addEventListener("click", async () => {
    try {
        const resp = await fetch(`${API}/context/${sessionId}_context/graph`);
        const data = await resp.json();
        renderForceGraph("#ctx-svg", data.nodes, data.links);
    } catch (err) {
        console.error("Failed to load context graph:", err);
    }
});

// ─── WebSocket (optional — for status updates) ──────

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);

    ws.onopen = () => {
        document.getElementById("status-dot").className = "dot dot-connected";
        document.getElementById("status-text").textContent = "Connected";
    };

    ws.onclose = () => {
        document.getElementById("status-dot").className = "dot dot-connecting";
        document.getElementById("status-text").textContent = "Reconnecting...";
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        document.getElementById("status-dot").className = "dot dot-error";
        document.getElementById("status-text").textContent = "Connection error";
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === "context_update" && msg.data?.graph) {
                renderForceGraph("#ctx-svg", msg.data.graph.nodes, msg.data.graph.links);
            }
        } catch (err) {
            console.error("WS message error:", err);
        }
    };
}

// ─── Utilities ──────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ─── Health check on load ───────────────────────────

async function checkHealth() {
    const url = `${API}/health`;
    console.log("Health check:", url);
    try {
        const resp = await fetch(url);
        const data = await resp.json();
        console.log("Health response:", data);
        if (data.status === "healthy") {
            document.getElementById("status-dot").className = "dot dot-connected";
            document.getElementById("status-text").textContent =
                `${data.llm_model} · ${data.graph_node_count} nodes`;
        } else {
            document.getElementById("status-dot").className = "dot dot-connecting";
            document.getElementById("status-text").textContent = `Degraded: ${data.llm_status}`;
        }
    } catch (err) {
        console.error("Health check failed:", err);
        document.getElementById("status-dot").className = "dot dot-error";
        document.getElementById("status-text").textContent = "API unreachable";
    }
}

// ─── Init ───────────────────────────────────────────

checkHealth();
connectWebSocket();