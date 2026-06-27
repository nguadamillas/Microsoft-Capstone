"""
app/project_chatbot.py
──────────────────────
Project Assistant chatbot — grounded in this repository only.

Public API:
    render_project_chatbot(T)

    T  — the active theme palette dict from dashboard.py (dark or light).
         Required keys: page, card, surface, border, heading, muted,
                        accent, positive, negative, card_shadow, categorical.

The chatbot reads a curated set of repo files once (cached), sends them as a
system-prompt context to the Anthropic API, and streams responses.  Anything
not in the project context is explicitly refused.
"""

import os
from pathlib import Path

import streamlit as st

ROOT  = Path(__file__).parent.parent
MODEL = "claude-sonnet-4-6"

# ── Grounding context files ───────────────────────────────────────────────────
# (fname relative to ROOT, max_lines: None = full file)
_KB_FILES: list[tuple[str, int | None]] = [
    ("README.md",                  None),
    ("config.py",                  None),
    ("pipeline/ingest.py",          80),
    ("pipeline/bronze.py",         110),
    ("pipeline/silver.py",         110),
    ("pipeline/gold.py",           100),
    ("pipeline/run_pipeline.py",    60),
    ("models/train_models.py",     130),
]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are the Project Assistant for the TED Procurement Intelligence dashboard
(IE University × Microsoft — Capstone Group 2).

Answer ONLY using the PROJECT CONTEXT supplied below.

Rules:
- If something is not in the context, say exactly: "That isn't covered in this \
project's documentation." and, if helpful, say where the user might find it.
- Never invent table names, column names, metrics, model details, paths, or commands.
- Be concise, factual, and professional — suitable for a Microsoft procurement audience.
- Use plain English; avoid unnecessary jargon.
- End every answer with a single line formatted as:
  Source: <name the specific file or section, e.g. "README — Gold tables" or \
"models/train_models.py">

=== PROJECT CONTEXT ===
{kb}
=== END CONTEXT ==="""

# ── Suggestion chips ──────────────────────────────────────────────────────────
_SUGGESTIONS = [
    "What Gold tables are available?",
    "How do the three ML models work?",
    "How do I run the full pipeline?",
    "What data source does this use?",
]

# ── Microsoft 4-square mark (inline SVG) ─────────────────────────────────────
_MS_MARK = """\
<svg width="18" height="18" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" \
style="flex-shrink:0">
  <rect x="0" y="0" width="9" height="9" fill="#F25022"/>
  <rect x="11" y="0" width="9" height="9" fill="#7FBA00"/>
  <rect x="0" y="11" width="9" height="9" fill="#00A4EF"/>
  <rect x="11" y="11" width="9" height="9" fill="#FFB900"/>
</svg>"""


# ── Knowledge-base builder ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _build_kb() -> str:
    """Read curated repo files; return a single concatenated grounding string."""
    parts: list[str] = []
    for fname, max_lines in _KB_FILES:
        p = ROOT / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if max_lines is not None:
            text = "\n".join(text.splitlines()[:max_lines])
        parts.append(f"\n\n### {fname}\n{text}")
    return "".join(parts)


# ── Streaming answer ──────────────────────────────────────────────────────────

def _stream_answer(history: list[dict], kb: str):
    """Yield text chunks from the Anthropic Messages streaming API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=_SYSTEM.format(kb=kb),
        messages=history,
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


# ── CSS helper ────────────────────────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _inject_chat_css(T: dict) -> None:
    accent_a12 = _hex_rgba(T["accent"], .12)
    accent_a25 = _hex_rgba(T["accent"], .25)
    accent_a10 = _hex_rgba(T["accent"], .10)
    accent_a20 = _hex_rgba(T["accent"], .20)

    st.markdown(f"""
<style>
/* ── Chat header ── */
.chat-header {{
    background: {T["card"]};
    border: 0.5px solid {T["border"]};
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 4px;
    box-shadow: {T["card_shadow"]};
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}}
.chat-header-left {{
    display: flex; align-items: center; gap: 12px;
}}
.chat-bot-icon {{
    width: 36px; height: 36px;
    background: {accent_a12};
    border: 1px solid {accent_a25};
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}}
.chat-header-title {{
    font-size: 15px; font-weight: 700;
    color: {T["heading"]}; line-height: 1.2;
}}
.chat-header-sub {{
    font-size: 11px; color: {T["muted"]}; margin-top: 1px;
}}
.chat-header-brand {{
    display: flex; align-items: center; gap: 7px; flex-shrink: 0;
}}
.chat-header-brand-text {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-weight: 600; font-size: 13px;
    color: {T["muted"]};
    letter-spacing: -.1px;
}}

/* ── Welcome card ── */
.chat-welcome {{
    background: {T["card"]};
    border: 0.5px solid {T["border"]};
    border-radius: 10px;
    padding: 18px 20px;
    margin: 12px 0 8px 0;
    color: {T["heading"]};
    font-size: 14px;
    line-height: 1.65;
    box-shadow: {T["card_shadow"]};
}}
.chat-welcome b {{ color: {T["accent"]}; font-weight: 600; }}
.chat-welcome-hint {{
    font-size: 11px; color: {T["muted"]};
    margin-top: 6px; line-height: 1.5;
}}
.chip-label {{
    font-size: 10.5px; font-weight: 600; letter-spacing: .7px;
    text-transform: uppercase; color: {T["muted"]};
    margin: 14px 0 6px 0;
}}

/* ── Suggestion chip buttons ── */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button,
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {{
    border-radius: 18px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    border: 1px solid {accent_a25} !important;
    color: {T["accent"]} !important;
    background: {accent_a10} !important;
    transition: background .15s !important;
    white-space: normal !important;
    text-align: left !important;
    height: auto !important;
    min-height: 36px !important;
    line-height: 1.4 !important;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover,
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button:hover {{
    background: {accent_a20} !important;
    border-color: {T["accent"]} !important;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    border-radius: 10px !important;
    padding: 4px 0 !important;
    margin-bottom: 2px !important;
}}

/* ── Source line styling ── */
.chat-source {{
    display: inline-block;
    margin-top: 8px;
    font-size: 11px;
    color: {T["muted"]};
    border-top: 1px solid {T["border"]};
    padding-top: 6px;
    width: 100%;
}}

/* ── No-key notice ── */
.chat-nokey {{
    background: {_hex_rgba(T["negative"], .08)};
    border: 1px solid {_hex_rgba(T["negative"], .25)};
    border-radius: 8px;
    padding: 14px 18px;
    color: {T["heading"]};
    font-size: 13px;
    margin: 12px 0;
}}
.chat-nokey code {{
    background: {T["surface"]};
    border: 1px solid {T["border"]};
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 12px;
    color: {T["accent"]};
}}

/* ── Error card ── */
.chat-error {{
    background: {_hex_rgba(T["negative"], .07)};
    border: 1px solid {_hex_rgba(T["negative"], .2)};
    border-radius: 8px;
    padding: 10px 14px;
    color: {T["heading"]};
    font-size: 12px;
    margin-top: 6px;
}}

/* ── Powered-by footer ── */
.chat-powered {{
    display: flex; align-items: center; gap: 6px;
    font-size: 10px; color: {T["muted"]};
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid {T["border"]};
    justify-content: flex-end;
}}
</style>
""", unsafe_allow_html=True)


# ── Header HTML ───────────────────────────────────────────────────────────────

def _render_header(T: dict) -> None:
    st.markdown(f"""
<div class="chat-header">
  <div class="chat-header-left">
    <div class="chat-bot-icon">🤖</div>
    <div>
      <div class="chat-header-title">Project Assistant</div>
      <div class="chat-header-sub">Answers grounded in this repository</div>
    </div>
  </div>
  <div class="chat-header-brand">
    {_MS_MARK}
    <span class="chat-header-brand-text">Microsoft</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main render function ──────────────────────────────────────────────────────

def render_project_chatbot(T: dict) -> None:
    """Render the full Project Assistant UI into the active Streamlit context."""

    # ── Init ──────────────────────────────────────────────────────────────────
    if "proj_chat" not in st.session_state:
        st.session_state.proj_chat: list[dict] = []
    history: list[dict] = st.session_state.proj_chat

    # Pop any chip-triggered input before rendering (avoids showing empty-state
    # on the same run the chip was clicked)
    pending = st.session_state.pop("proj_pending", None)

    # ── CSS + header ──────────────────────────────────────────────────────────
    _inject_chat_css(T)

    hdr_col, clr_col = st.columns([9, 1])
    with hdr_col:
        _render_header(T)
    with clr_col:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="proj_chat_clear", help="Clear conversation",
                     use_container_width=True):
            st.session_state.proj_chat = []
            st.rerun()

    # ── API-key guard ─────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.markdown(f"""
<div class="chat-nokey">
  ⚠️ <strong>No API key found.</strong>
  Add <code>ANTHROPIC_API_KEY=sk-ant-...</code> to your
  <code>.env</code> file in the project root and restart the app.
</div>
""", unsafe_allow_html=True)
        return

    # ── Build knowledge base (cached) ─────────────────────────────────────────
    with st.spinner("Loading project context…"):
        kb = _build_kb()

    # ── Existing message history ──────────────────────────────────────────────
    for msg in history:
        avatar = "🤖" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Welcome / empty state ─────────────────────────────────────────────────
    if not history and pending is None:
        st.markdown(f"""
<div class="chat-welcome">
  Welcome! Ask anything about the <b>TED Procurement Intelligence</b> project —
  the data pipeline, Gold tables, ML models, CPV categories, or how to run the code.
  <div class="chat-welcome-hint">
    Answers come only from the repository. Out-of-scope questions are politely declined.
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="chip-label">Suggested questions</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, sugg in enumerate(_SUGGESTIONS):
            with (c1 if i < 2 else c2):
                if st.button(sugg, key=f"proj_chip_{i}", use_container_width=True):
                    st.session_state.proj_pending = sugg
                    st.rerun()

        st.markdown(f"""
<div class="chat-powered">
  {_MS_MARK}
  <span>Powered by Microsoft · claude-sonnet-4-6</span>
</div>
""", unsafe_allow_html=True)

    # ── Resolve user input ────────────────────────────────────────────────────
    chat_input = st.chat_input("Ask about the TED Procurement project…")
    user_input: str | None = chat_input or pending

    if not user_input:
        return

    # ── Process user message ──────────────────────────────────────────────────
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build API payload — pass full history (user message already appended)
    api_messages = [{"role": m["role"], "content": m["content"]} for m in history]

    with st.chat_message("assistant", avatar="🤖"):
        try:
            response_text = st.write_stream(_stream_answer(api_messages, kb))
        except Exception as exc:
            response_text = (
                f"*Sorry, there was an error calling the API:* `{exc}`\n\n"
                f"Check that your `ANTHROPIC_API_KEY` is valid and the `anthropic` "
                f"package is up to date (`pip install 'anthropic>=0.40.0'`)."
            )
            st.markdown(response_text)

    history.append({"role": "assistant", "content": str(response_text)})
