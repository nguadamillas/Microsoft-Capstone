# TODO — Complete the Chatbot Feature (Role 6)

Ordered steps to take the chatbot from "built & tested offline" to "shipped and
demo-ready." Check items off as we go. See `MEMORY.md` for context.

---

## ✅ Done
- [x] Provider-agnostic text-to-pandas engine (`app/chatbot.py`)
- [x] Synthetic Gold fixtures (`tests/fixtures/make_synthetic_gold.py`)
- [x] Offline test suite — 16 passing (`tests/test_chatbot.py`)
- [x] Gold schema contract for Role 2 (`docs/gold_contract.md`)
- [x] Standalone CLI runner (`scripts/try_chatbot.py`)
- [x] Engine wired into dashboard Tab 4 (`app/dashboard.py`)
- [x] `openai` SDK installed; `.env` confirmed git-ignored

## ✅ Live local test — DONE
- [x] **Got a GitHub token** (fine-grained, **Models: Read-only**)
- [x] Added it to `.env` (`LLM_PROVIDER=github`, `GITHUB_TOKEN=...`, `GITHUB_MODEL=gpt-4o`)
- [x] `python scripts/try_chatbot.py` → real answers, verified correct vs direct pandas
- [x] Added graceful API-error handling in the engine
- [ ] (optional) rotate the GitHub token, since it was shared in chat

## Next — verify in the UI
- [ ] `streamlit run app/dashboard.py` → open the **Chatbot** tab
- [ ] Ask a sample question; confirm: real answer, result table, "show code" expander,
      and that sidebar filters change the answer

## Decision before pushing
- [ ] Decide: **keep** the `dashboard.py` Tab-4 integration, OR **split** the chatbot
      into its own standalone page so the teammate's dashboard stays untouched
      (resolves the "no edits to teammates' work" concern)

## Hardening / polish (optional but improves the demo)
- [ ] Add a few more golden questions + edge cases (empty filters, ambiguous question)
- [ ] Confirm graceful behaviour when Gold data is empty or a column is missing
- [ ] Tune the system prompt if the model picks wrong columns
- [ ] Cap/format large result tables nicely in the UI

## Integrate real data (depends on Role 2)
- [ ] Send `docs/gold_contract.md` to Role 2; confirm `gold.py` `run()` will emit the
      4 contract tables (`gold_opportunities / gold_awards / gold_market_summary /
      gold_cpv_analysis`)
- [ ] When real Gold parquet arrives, drop it into `data/gold/` and re-run the runner +
      dashboard (no code change expected)

## Ship
- [ ] Commit `feature/chatbot`
- [ ] Open PR into `dev` (only when ready; coordinate with the dashboard owner)
- [ ] Update `MEMORY.md` session log

## My other Role-6 deliverables (not the chatbot, but mine)
- [ ] GitHub repo housekeeping (README section on the chatbot, run instructions)
- [ ] Written report — chatbot architecture section (Section 5: business analytics layer)
- [ ] Slide deck contribution
