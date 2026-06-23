"""
scripts/try_chatbot.py
───────────────────────
Standalone local test runner for the chatbot engine — needs NO Streamlit and
touches none of the dashboard code. Uses real Gold parquet from data/gold/ if
present, otherwise synthetic fixtures.

Usage:
    python scripts/try_chatbot.py                         # runs a few sample questions
    python scripts/try_chatbot.py "Your question here"    # one question
    python scripts/try_chatbot.py --provider github "..." # force a provider

Credentials (set in a .env at project root, or your shell env) — pick ONE:
    Azure OpenAI:  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    GitHub Models: GITHUB_TOKEN            (free; optional GITHUB_MODEL=gpt-4o)
    Anthropic:     ANTHROPIC_API_KEY
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GOLD_DIR  # noqa: E402
from app.chatbot import answer_question  # noqa: E402
from tests.fixtures.make_synthetic_gold import make_synthetic_gold  # noqa: E402

SAMPLES = [
    "What are the top 5 countries by number of contract awards?",
    "What % of awards went to SMEs?",
    "Which CPV categories have the highest average savings %?",
]


def load_dfs() -> tuple[dict[str, pd.DataFrame], bool]:
    file_map = {
        "opportunities":  "gold_opportunities",
        "awards":         "gold_awards",
        "market_summary": "gold_market_summary",
        "cpv_analysis":   "gold_cpv_analysis",
    }
    real = {}
    for key, fname in file_map.items():
        p = GOLD_DIR / f"{fname}.parquet"
        real[key] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if any(len(v) for v in real.values()):
        return real, False
    return make_synthetic_gold(), True


def main(argv: list[str]) -> int:
    provider = None
    if "--provider" in argv:
        i = argv.index("--provider")
        provider = argv[i + 1]
        del argv[i:i + 2]

    questions = argv or SAMPLES
    dfs, is_demo = load_dfs()
    print(f"Data: {'SYNTHETIC fixtures' if is_demo else 'real Gold parquet'} "
          f"(awards={len(dfs['awards']):,}, opportunities={len(dfs['opportunities']):,})\n")

    for q in questions:
        print("=" * 70)
        print(f"Q: {q}")
        res = answer_question(q, dfs, provider=provider)
        print(f"[provider={res.provider}  steps={res.steps}  ok={res.ok}]")
        if res.code:
            print(f"\n  code:\n{res.code}")
        print(f"\n  answer:\n{res.answer}\n")
        if res.error:
            print(f"  ERROR: {res.error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
