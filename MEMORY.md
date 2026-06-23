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
- Gold ✅ **real tables now delivered** (from the capstone Google Drive) — see real-schema
  notes below. Base tables: `gold_opportunities, gold_awards, gold_market_summary,
  gold_cpv_analysis` (+ `gold_notices, gold_lots, gold_country_kpis, gold_cpv_kpis`).
- ML models ✅ **batch-scored into Gold** as precomputed columns — tables
  `gold_notice_enrichment` (cpv_pred, predicted_award_value, cpv_review_flag) and
  `gold_bid_win_probability` (win_probability, is_winner_actual). NOT in the local zip yet
  (separate files in the Drive `Gold/` folder) — need to download.
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

## Real Gold schema (verified 2026-06-23 from the downloaded parquet)

Real data lives in `data/gold/*.parquet` (unzipped from the Drive; the nested
`data/gold/data/...` was flattened; bronze/silver moved to `data/bronze`, `data/silver`).
`data/` is fully git-ignored.

- `gold_opportunities` (29,537 rows) — ✅ **matches our contract exactly**.
- `gold_market_summary` (41) and `gold_cpv_analysis` (38) — ✅ match the contract.
- `gold_awards` (97,913 rows) — ⚠️ **DIFFERENT from our contract**: it is **lot-grained**
  (has `lot_id`, `lot_result_id`), and uses different names:
  `awarded_amount`/`total_awarded` (NOT `awarded_eur`), **no `savings_pct`**,
  `tenders_count` (NOT `avg_tenders_per_lot`), `winner_name`/`is_sme` (NOT
  `winner_names`/`sme_winner`).
- **Empty columns in `gold_awards`:** `is_sme`, `winner_country`, `winner_name` are **100%
  null**. So SME / winner-identity questions are NOT answerable on current Gold (matches the
  data-dictionary "bidders anonymized" caveat). `winner_org_id` has anonymized `TPA-*` ids.
- Savings is available via `gold_market_summary.avg_savings_pct` / `gold_cpv_analysis.avg_savings`,
  or computed from `estimated` vs `awarded_amount`/`total_awarded`.
- Also available (clean aggregates): `gold_country_kpis` (63), `gold_cpv_kpis` (46),
  `gold_notices` (71,432), `gold_lots` (227,017).

## Key behaviour confirmed
- The schema-agnostic engine **adapts to the real columns automatically** and **truthfully
  reports empty columns** instead of hallucinating (verified live on real data).

## Resolved
- **Two ways to use the chatbot, both on our engine:**
  - `app/chatbot_app.py` — standalone ChatGPT-style page (input docked at bottom).
  - `app/dashboard.py` **Tab 4** — re-wired to our engine with a **minimal, Tab-4-only edit**
    (rest of the teammate's dashboard untouched), so the full dashboard shows the good chatbot.
- **Synthetic fixtures realigned** to the real schema (lot-grained awards, real names, empty
  SME/winner columns, + the 2 ML tables). Tests pass (19).
- **Engine aligned:** schema context shows per-column fill rate (marks EMPTY columns);
  system prompt declines off-topic questions, prefers actuals over predictions, and honors
  the prediction caveats.

## Known minor polish item
- The model sometimes reads "open tenders" literally as `proc_type=='open'` (proc_type is
  Services/Supplies/Works). "Open tenders" really means Contract Notices (CN). Could clarify
  in the prompt. Low priority — it reports "no data" rather than hallucinating.

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
- Fixed the chatbot UI: scrollable message box with the input bar pinned beneath it;
  relabelled pill to "GPT-4o · text-to-pandas".
- **Read the capstone Google Drive** (via Chrome connector) → found the real Gold tables +
  `GOLD_DATA_DICTIONARY.md` + ML-output tables + `final_bronze_silver_gold_parquets.zip`.
- **Integrated real Gold data** into `data/gold/` (flattened the nested unzip). Verified the
  chatbot runs on real data and adapts to the real schema. Documented real-schema findings
  above (esp. lot-grained `gold_awards`, empty SME/winner columns).
- **Next / pending:** (1) download the 2 ML tables from Drive `Gold/`; (2) realign
  contract + fixtures + sample questions + prompt to the real schema; (3) add column-coverage
  to the schema context so the bot avoids empty columns; (4) load the extra tables into the chatbot.

### 2026-06-23 (continued — alignment + standalone page)
- Downloaded the 2 ML tables from Drive into `data/gold/` (notice_enrichment via CSV→parquet
  after a transient Google 503 on the parquet; bid_win_probability direct).
- Aligned the engine (per-column fill-rate in schema context, prediction caveats, off-topic
  decline) and realigned fixtures/tests/contract to the real schema.
- **Built standalone `app/chatbot_app.py`; reverted `app/dashboard.py` to origin/dev.**
- **Verified live on real data:** "CPV review flags → 1,524" (matches dictionary), SME question
  now honestly declines (empty column), off-topic politely redirected. 19 tests pass.
- **Done so far this phase:** steps 1–4 (align, ML answers, own page, polish). Next: commit + push.
