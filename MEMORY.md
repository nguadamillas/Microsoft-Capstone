# MEMORY — Chatbot Work Log (Role 6)

> Living record of **my part** of the TED Procurement Intelligence capstone (the
> chatbot + repo/PM). Read this at the start of every working session to pick up
> where we left off. Update it whenever something meaningful changes.

---

## Project in one paragraph

**TED Procurement Intelligence** (IE University × Microsoft capstone). Takes one
month (January 2026) of EU public-procurement data — **71,738 raw TED XML notices** —
and turns it into business intelligence via a medallion pipeline
(**Bronze → Silver → Gold**), then **ML models**, a **Streamlit dashboard**, and a
**chatbot**. Team of 6. **I am Role 6: the text-to-pandas chatbot, plus the GitHub
repo and the report/slide deck.**

## My deliverable

A **text-to-pandas natural-language query system**: the user asks a question in plain
English, the system writes pandas code, **actually executes it** against the Gold
tables, and answers from the real computed result. (Not the weak "summarise stats in
the prompt" approach — that one can hallucinate numbers.)

## Where the team stands (as of last session)

- Bronze ✅ done · Silver ✅ done (PASS 68 / WARN 8 / FAIL 0).
- Gold ⚠️ **only a smoke-test starter** — NOT final. Owned by Role 2.
- ML models 🔸 code exists, only placeholder artifacts.
- Dashboard 🔸 big WIP on the `dev` branch (`app/dashboard.py`).
- Chatbot (mine) 🔸 real engine built this session — see below.

## Branch & safety

- Working branch: **`feature/chatbot`** (off `origin/dev`).
- **Nothing pushed yet.** Only teammate file touched is `app/dashboard.py` (local only).
- `.env` is git-ignored — secrets never get committed.

---

## What I've built so far

| File | Status | What it is |
|---|---|---|
| `app/chatbot.py` | ✅ | The engine. Provider-agnostic (Azure OpenAI / GitHub Models / Anthropic). Generates pandas → executes in a sandbox → self-corrects on error → returns the real result + the code + a grounded answer. Schema-agnostic. |
| `tests/fixtures/make_synthetic_gold.py` | ✅ | Synthetic Gold data so everything runs with zero real data. |
| `tests/test_chatbot.py` | ✅ | 16 tests pass (engine, sandbox safety, retry loop) + 1 live test gated on creds. |
| `docs/gold_contract.md` | ✅ | The exact Gold tables/columns the chatbot needs — handoff doc for Role 2. |
| `scripts/try_chatbot.py` | ✅ | Standalone CLI runner — test the chatbot WITHOUT the dashboard. |
| `app/dashboard.py` (Tab 4) | ✅ edited | Wired to call the engine; shows the executed result + the code; respects sidebar filters. |
| `requirements.txt` | ✅ edited | Added `openai`, `pytest`. |

## Key decisions made

- **Scope:** Gold-query only (no ML coupling) — keeps me decoupled from Role 3.
- **Engine:** robust text-to-pandas with sandboxed execution + self-correction.
- **LLM backend:** **GitHub Models (free)** chosen for testing (Microsoft-aligned, OpenAI-compatible, runs gpt-4o). Engine also supports Azure OpenAI and Anthropic via env vars (`LLM_PROVIDER`).
- **Decoupling strategy:** build against a schema contract + synthetic data now; swap in real Gold later with no code change (drop parquet into `data/gold/`).

## Gotchas / things to remember

- **Gold schema mismatch:** `pipeline/gold.py`'s `run()` writes `gold_notices / gold_lots /
  gold_awards / gold_country_kpis / gold_cpv_kpis`, but the dashboard + chatbot read
  `gold_opportunities / gold_awards / gold_market_summary / gold_cpv_analysis`. The
  `build_opportunities / build_market_summary / build_cpv_analysis` functions exist but
  aren't called in `run()`. **Role 2 must fix this** (see `docs/gold_contract.md`).
- **No real Gold parquet locally** — we test on synthetic fixtures.
- **Open question:** keep the `dashboard.py` integration, or split the chatbot into its
  own file so the teammate's dashboard stays untouched? (I flagged not wanting to edit
  teammates' work.)

## Credentials to run it live

Set ONE in `.env` at project root (then `python scripts/try_chatbot.py`):
- GitHub Models: `GITHUB_TOKEN=github_pat_...` (free; optional `GITHUB_MODEL=gpt-4o`)
- Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- Anthropic: `ANTHROPIC_API_KEY`

---

## Session log

### 2026-06-23
- Synced repo to remote (was stale after a force-push); branched `feature/chatbot` off `dev`.
- Read all 3 report sections + full codebase; mapped team status.
- Built the engine, fixtures, tests (16 pass), contract doc, standalone runner.
- Wired engine into dashboard Tab 4.
- Refactored engine to be provider-agnostic; chose **GitHub Models** for live testing.
- Installed `openai` SDK; confirmed `.env` git-ignored.
- Added graceful API-error handling so a bad/failed LLM call never crashes the app.
- **Live end-to-end test PASSED** via GitHub Models (gpt-4o): chatbot wrote pandas,
  executed it, and returned correct answers (verified against direct pandas — e.g.
  SME=46.67%, top country Romania=14). The engine genuinely computes, no hallucination.
- **Next:** preview in the Streamlit UI; decide keep-vs-split of the dashboard
  integration; then polish (harder questions, prompt tuning, result formatting).
