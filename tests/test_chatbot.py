"""
tests/test_chatbot.py
──────────────────────
Headless tests for the provider-agnostic text-to-pandas engine (app/chatbot.py).

Most tests inject a STUB backend so they run offline with no credentials — they
exercise the real machinery: schema introspection, sandboxed execution, the
agentic tool-use loop, and self-correction. One live smoke test runs against a
real provider and is skipped unless credentials are present.

Run:  pytest tests/test_chatbot.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chatbot import (  # noqa: E402
    answer_question,
    build_schema_context,
    _execute,
    ToolCall,
    Turn,
)
from tests.fixtures.make_synthetic_gold import make_synthetic_gold  # noqa: E402


@pytest.fixture(scope="module")
def dfs():
    return make_synthetic_gold(seed=7)


# ── Stub backend ─────────────────────────────────────────────────────────────────

# Maps a keyword in the question → the pandas code the "model" emits.
GOLDEN_CODE = {
    "top 5 countries": "result = awards['buyer_country'].value_counts().head(5)",
    "average savings": "result = awards.groupby('cpv_division_name')['savings_pct'].mean().sort_values(ascending=False).head(5)",
    "sme": "result = awards['sme_winner'].mean() * 100",
    "total estimated value of open": "result = opportunities['estimated'].sum()",
    "savings % across": "result = awards.groupby('proc_type')['savings_pct'].mean()",
    "most competitive": "result = awards.groupby('cpv_division_name')['avg_tenders_per_lot'].mean().sort_values(ascending=False).head(5)",
    "top 10 cpv": "result = awards.groupby('cpv_division_name')['awarded_eur'].sum().sort_values(ascending=False).head(10)",
    "largest tenders on average": "result = opportunities.groupby('buyer_country')['estimated'].mean().sort_values(ascending=False).head(5)",
}


class StubBackend:
    """Implements the backend protocol (begin/next_turn/submit_tool_result)
    deterministically. If `retry_bad_first`, the first code is broken (missing
    column) so the engine must self-correct."""
    name = "stub"

    def __init__(self, retry_bad_first=False):
        self.retry_bad_first = retry_bad_first
        self.calls = 0
        self.q = ""
        self.last_error = False

    def begin(self, system, history, question):
        self.q = question.lower()
        self.calls = 0
        self.last_error = False

    def next_turn(self):
        self.calls += 1
        if self.calls == 1:
            code = ("result = awards['column_that_does_not_exist'].sum()"
                    if self.retry_bad_first else self._good())
            return Turn(None, ToolCall("t1", code))
        if self.last_error:                       # self-correct after an error
            self.last_error = False
            return Turn(None, ToolCall("t2", self._good()))
        return Turn("Here is the answer based on the computed result.", None)

    def submit_tool_result(self, tool_call, content, is_error):
        self.last_error = is_error

    def _good(self):
        for kw, code in GOLDEN_CODE.items():
            if kw in self.q:
                return code
        return "result = len(awards)"


# ── Golden question tests (offline) ──────────────────────────────────────────────

GOLDEN_QUESTIONS = [
    "What are the top 5 countries by number of contract awards?",
    "Which CPV categories have the highest average savings %?",
    "What % of awards went to SMEs?",
    "What is the total estimated value of open tenders?",
    "Compare savings % across Services, Supplies and Works",
    "Show the most competitive procurement categories",
    "What are the top 10 CPV categories by total awarded value?",
    "Which buyer countries publish the largest tenders on average?",
]


@pytest.mark.parametrize("question", GOLDEN_QUESTIONS)
def test_golden_question_executes(question, dfs):
    res = answer_question(question, dfs, backend=StubBackend())
    assert res.ok, f"engine errored: {res.error}"
    assert res.code, "no code was executed"
    assert res.result is not None, "no executed result"
    assert res.answer, "no final answer text"


def test_results_match_direct_pandas(dfs):
    res = answer_question("What % of awards went to SMEs?", dfs, backend=StubBackend())
    expected = dfs["awards"]["sme_winner"].mean() * 100
    assert abs(float(res.result) - float(expected)) < 1e-9

    res2 = answer_question(
        "What are the top 5 countries by number of contract awards?", dfs, backend=StubBackend()
    )
    expected2 = dfs["awards"]["buyer_country"].value_counts().head(5)
    assert res2.result.equals(expected2)


def test_self_correction_loop(dfs):
    """First code errors (bad column); engine feeds the traceback back and retries."""
    res = answer_question(
        "What % of awards went to SMEs?", dfs, backend=StubBackend(retry_bad_first=True), max_steps=4
    )
    assert res.ok, f"engine did not recover: {res.error}"
    assert res.steps >= 2, "expected at least one retry"
    assert len(res.attempts) >= 2
    assert res.attempts[0]["error"] is not None      # first attempt failed
    assert res.attempts[-1]["error"] is None          # final attempt succeeded


# ── Sandbox / safety unit tests ───────────────────────────────────────────────────

def test_execute_forbids_imports(dfs):
    result, err = _execute("import os\nresult = 1", dfs)
    assert result is None and err is not None


def test_execute_forbids_file_access(dfs):
    result, err = _execute("result = open('/etc/passwd').read()", dfs)
    assert result is None and "Disallowed" in err


def test_execute_requires_result_var(dfs):
    result, err = _execute("x = awards.shape", dfs)
    assert result is None and "result" in err


def test_execute_happy_path(dfs):
    result, err = _execute("result = len(awards)", dfs)
    assert err is None and result == len(dfs["awards"])


def test_schema_context_is_schema_only(dfs):
    ctx = build_schema_context(dfs)
    assert "opportunities" in ctx and "awards" in ctx
    assert "savings_pct" in ctx and "buyer_country" in ctx


def test_missing_credentials_is_graceful(dfs, monkeypatch):
    for var in ["LLM_PROVIDER", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                "GITHUB_TOKEN", "GITHUB_MODELS_TOKEN", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("app.chatbot.ANTHROPIC_API_KEY", "")
    res = answer_question("anything", dfs)        # no backend, no creds
    assert not res.ok and res.error == "missing_api_key"


# ── Live smoke test (requires real credentials for any provider) ──────────────────

_HAS_CREDS = bool(
    (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
    or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")
    or os.getenv("ANTHROPIC_API_KEY")
)


@pytest.mark.skipif(not _HAS_CREDS, reason="no LLM credentials set; skipping live smoke test")
def test_live_smoke(dfs):
    res = answer_question("What are the top 3 countries by total awarded value?", dfs)
    assert res.ok, f"live call failed: {res.error}"
    assert res.result is not None
    assert res.code and "awarded_eur" in res.code.lower()
