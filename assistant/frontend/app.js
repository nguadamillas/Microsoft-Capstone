/* Procurement Assistant — frontend logic */

const API = "http://localhost:8010/chat";
const HEALTH = "http://localhost:8010/health";

const QUICK_QUESTIONS = [
  "What are the top 5 countries by number of contract awards?",
  "Which CPV categories have the highest average savings %?",
  "What % of awards went to SMEs?",
  "Show the most competitive procurement categories",
  "What is the total estimated value of open tenders?",
  "Compare savings % across Services, Supplies and Works",
  "Which buyer countries publish the largest tenders on average?",
  "What are the top 10 CPV categories by total awarded value?",
];

const MS_COLORS = ["#00a4ef", "#7fba00", "#f25022", "#ffb900", "#2f6fed"];

const els = {
  thread: document.getElementById("thread"),
  welcome: document.getElementById("welcome"),
  quickGrid: document.getElementById("quickGrid"),
  input: document.getElementById("input"),
  send: document.getElementById("sendBtn"),
  clear: document.getElementById("clearBtn"),
  theme: document.getElementById("themeToggle"),
  banner: document.getElementById("sampleBanner"),
};

let history = [];   // [{role, content}]
let busy = false;

/* ---------- setup ---------- */
function buildQuickGrid() {
  els.quickGrid.innerHTML = "";
  QUICK_QUESTIONS.forEach((q) => {
    const b = document.createElement("button");
    b.className = "quick-btn";
    b.textContent = q;
    b.onclick = () => send(q);
    els.quickGrid.appendChild(b);
  });
}

async function checkHealth() {
  try {
    const r = await fetch(HEALTH);
    const j = await r.json();
    if (j.using_sample_data) els.banner.hidden = false;
  } catch { /* backend not up yet; ignore */ }
}

/* ---------- rendering ---------- */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe markdown: escape first, then bold / inline-code / bullet lists / paragraphs.
function miniMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    let line = raw
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
    if (/^\s*[-*]\s+/.test(raw)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + line.replace(/^\s*[-*]\s+/, "") + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim()) html += "<p>" + line + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function avatarSVG() {
  return `<span class="avatar" aria-hidden="true"><svg viewBox="0 0 22 22" width="18" height="18">
    <rect x="0" y="0" width="10" height="10" fill="#F25022"/><rect x="12" y="0" width="10" height="10" fill="#7FBA00"/>
    <rect x="0" y="12" width="10" height="10" fill="#00A4EF"/><rect x="12" y="12" width="10" height="10" fill="#FFB900"/>
  </svg></span>`;
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  els.thread.appendChild(el);
  scrollDown();
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.id = "typing";
  el.innerHTML = `<div class="msg-avatar">${avatarSVG()}</div>
    <div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>`;
  els.thread.appendChild(el);
  scrollDown();
}
function removeTyping() { document.getElementById("typing")?.remove(); }

function addAssistantAnswer(answer) {
  const el = document.createElement("div");
  el.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (answer.error) {
    bubble.classList.add("error-bubble");
    bubble.innerHTML = miniMarkdown(answer.error);
  } else {
    bubble.innerHTML = miniMarkdown(answer.text || "");
    if (Array.isArray(answer.kpis) && answer.kpis.length) bubble.appendChild(renderKpis(answer.kpis));
    if (answer.chart && answer.chart.labels) bubble.appendChild(renderChart(answer.chart));
    if (answer.source) {
      const src = document.createElement("div");
      src.className = "source-line";
      src.textContent = "Source: " + answer.source;
      bubble.appendChild(src);
    }
  }
  el.innerHTML = `<div class="msg-avatar">${avatarSVG()}</div>`;
  el.appendChild(bubble);
  els.thread.appendChild(el);
  scrollDown();
}

function renderKpis(kpis) {
  const wrap = document.createElement("div");
  wrap.className = "kpis";
  kpis.slice(0, 4).forEach((k) => {
    const c = document.createElement("div");
    c.className = "kpi";
    c.innerHTML = `<div class="kpi-value">${escapeHtml(String(k.value ?? ""))}</div>
                   <div class="kpi-label">${escapeHtml(String(k.label ?? ""))}</div>`;
    wrap.appendChild(c);
  });
  return wrap;
}

function renderChart(spec) {
  const card = document.createElement("div");
  card.className = "chart-card";
  card.innerHTML = `<p class="chart-title">${escapeHtml(spec.title || "")}</p>
                    <div class="chart-wrap"><canvas></canvas></div>`;
  const canvas = card.querySelector("canvas");

  // Defer until Chart.js (deferred script) is ready.
  const draw = () => {
    const css = getComputedStyle(document.documentElement);
    const text = css.getPropertyValue("--text-dim").trim();
    const grid = css.getPropertyValue("--border").trim();
    const datasets = (spec.series || []).map((s, i) => ({
      label: s.name,
      data: s.data,
      backgroundColor: spec.type === "doughnut"
        ? spec.labels.map((_, j) => MS_COLORS[j % MS_COLORS.length])
        : MS_COLORS[i % MS_COLORS.length],
      borderColor: MS_COLORS[i % MS_COLORS.length],
      borderWidth: spec.type === "line" ? 2 : 0,
      borderRadius: spec.type === "bar" ? 6 : 0,
      tension: .35,
      fill: false,
    }));
    const single = (spec.series || []).length <= 1 && spec.type !== "line";
    new Chart(canvas, {
      type: spec.type || "bar",
      data: { labels: spec.labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: !single, labels: { color: text } } },
        scales: spec.type === "doughnut" ? {} : {
          x: { ticks: { color: text }, grid: { color: grid } },
          y: { ticks: { color: text }, grid: { color: grid } },
        },
      },
    });
  };
  if (window.Chart) draw(); else window.addEventListener("load", draw, { once: true });
  return card;
}

function scrollDown() { els.thread.scrollTop = els.thread.scrollHeight; }

/* ---------- send ---------- */
async function send(text) {
  text = (text || els.input.value).trim();
  if (!text || busy) return;
  busy = true; els.send.disabled = true;
  els.welcome?.remove();
  els.input.value = "";

  addUserMessage(text);
  history.push({ role: "user", content: text });
  addTyping();

  let answer;
  try {
    const r = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    answer = await r.json();
  } catch {
    answer = { error: "Can't reach the assistant. Is the backend running on port 8000?" };
  }
  removeTyping();
  addAssistantAnswer(answer);
  if (answer.text) history.push({ role: "assistant", content: answer.text });

  busy = false; els.send.disabled = false; els.input.focus();
}

/* ---------- controls ---------- */
els.send.onclick = () => send();
els.input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
els.clear.onclick = () => {
  history = [];
  els.thread.innerHTML = "";
  els.thread.appendChild(els.welcome);
  location.reload();
};
els.theme.onclick = () => {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === "dark" ? "light" : "dark";
};

buildQuickGrid();
checkHealth();
els.input.focus();
