"""
app/chatbot_app.py
───────────────────
Standalone TED Procurement Chatbot — a dedicated, ChatGPT-style page that uses the
text-to-pandas engine in `app/chatbot.py`. Kept separate from the dashboard so it
owns its own clean full-screen layout (input bar docked at the bottom) and doesn't
touch teammates' files.

Run:
    streamlit run app/chatbot_app.py

Reads the real Gold tables from data/gold/ if present, otherwise falls back to the
synthetic fixtures so it runs anywhere. Needs an LLM credential in .env
(GITHUB_TOKEN / Azure OpenAI / Anthropic) — see app/chatbot.py.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GOLD_DIR  # noqa: E402
from app.chatbot import answer_question, DEFAULT_OPENAI_MODEL  # noqa: E402

st.set_page_config(page_title="TED Procurement Chatbot", page_icon="🇪🇺", layout="centered")

# file in data/gold  →  variable name exposed to the engine
TABLE_MAP = {
    "gold_opportunities":       "opportunities",
    "gold_awards":              "awards",
    "gold_market_summary":      "market_summary",
    "gold_cpv_analysis":        "cpv_analysis",
    "gold_notice_enrichment":   "notice_enrichment",
    "gold_bid_win_probability": "bid_win_probability",
    "gold_country_kpis":        "country_kpis",
    "gold_cpv_kpis":            "cpv_kpis",
}

SAMPLE_QUESTIONS = [
    "Top 5 buyer countries by number of awards",
    "Total estimated value of open tenders",
    "Top 10 CPV categories by total awarded value",
    "Which categories are most competitive (avg bids per lot)?",
    "Average savings % by procurement type",
    "What's the average predicted award value for open tenders?",
    "How many notices are flagged for CPV review?",
    "For one lot, rank its bidders by win probability",
]


@st.cache_data(show_spinner=False)
def load_dfs() -> tuple[dict[str, pd.DataFrame], bool]:
    real = {}
    for fname, key in TABLE_MAP.items():
        p = GOLD_DIR / f"{fname}.parquet"
        if p.exists():
            real[key] = pd.read_parquet(p)
    if real:
        return real, False
    from tests.fixtures.make_synthetic_gold import make_synthetic_gold
    return make_synthetic_gold(), True


dfs, is_demo = load_dfs()

st.title("🇪🇺 TED Procurement Chatbot")
_src = "⚠️ synthetic demo data" if is_demo else f"real Gold data — {len(dfs.get('awards', [])):,} award rows"
st.caption(
    f"Ask in plain English — I write pandas, **run it** on the Gold tables, and answer "
    f"with the real result. Model: `{DEFAULT_OPENAI_MODEL}` (GitHub Models) · Data: {_src}."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_result(res):
    if res.result is not None:
        if isinstance(res.result, pd.DataFrame) and not res.result.empty:
            st.dataframe(res.result.head(50), use_container_width=True)
        elif isinstance(res.result, pd.Series) and len(res.result):
            st.dataframe(res.result.head(50).reset_index(), use_container_width=True)
        elif not isinstance(res.result, (pd.DataFrame, pd.Series)):
            st.info(f"**Result:** {res.result}")
    if res.code:
        with st.expander("Show the pandas code that was run"):
            st.code(res.code, language="python")


# Sample-question chips (only show before the first message, to stay clean)
if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"s{i}", use_container_width=True):
            st.session_state["prefill"] = q

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("res") is not None:
            render_result(m["res"])

# Top-level chat_input → Streamlit docks it to the bottom of the window (ChatGPT-style)
prompt = st.chat_input("Ask about the procurement data…")
# A sample-question chip stashes its text in `prefill`; consume it here.
if not prompt and st.session_state.get("prefill"):
    prompt = st.session_state.pop("prefill")

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Writing and running pandas…"):
        res = answer_question(prompt, dfs, history=history)
    st.session_state.messages.append({"role": "assistant", "content": res.answer, "res": res})
    st.rerun()

if st.session_state.messages and st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
