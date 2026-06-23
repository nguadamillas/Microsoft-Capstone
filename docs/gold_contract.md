# Gold Layer — tables the chatbot consumes

**Maintainer:** Role 6 (Chatbot/PM) · **Producers:** Role 2 (Gold) + Role 3 (ML)

This documents the **real** Gold tables (verified 2026-06-23 from the capstone Drive)
that the chatbot reads from `data/gold/*.parquet`. The chatbot engine is
**schema-agnostic** — it introspects whatever columns exist at runtime — so adding
columns is safe; renaming/removing the ones marked **key** below changes answers.

All monetary values are EUR. Country codes are ISO-3 in `gold_notice_enrichment`
(e.g. `DEU`) and may differ in other tables — don't assume cross-table joins on country.

---

## `gold_opportunities` — open Contract Notices (CN) · 29,537 rows · grain: notice
Key cols: `notice_id`, `buyer_country`, `proc_type`, `cpv_division_name`, `estimated`
(~43% filled), `num_lots`, `submission_deadline`. Also: `pub_date`, `buyer_name`,
`procedure_type`, `cpv_main`, `cpv_division`, `avg_lot_duration`, `framework`.

## `gold_awards` — Contract Award rows (CAN) · 97,913 rows · grain: **lot/award row**
> ⚠️ Lot-grained, NOT one row per notice. Different names from the old assumption.

Key cols: `notice_id`, `lot_id`, `buyer_country`, `proc_type`, `cpv_division_name`,
`awarded_amount` (EUR, ~60% filled), `total_awarded`, `estimated` (~35% filled),
`tenders_count` (competition; NOT `avg_tenders_per_lot`), `contract_value`,
`tender_value`, `winner_org_id` (anonymized `TPA-*`), `is_awarded`, `result_code`.

> **Empty columns (100% null — do NOT use):** `is_sme`, `winner_name`, `winner_country`.
> So **SME analysis and winner-company questions are not answerable** until the pipeline
> links tendering parties to org name + SME flag.
> **No `savings_pct`** column — compute as `(estimated - awarded_amount)/estimated*100`,
> or read `gold_market_summary.avg_savings_pct` / `gold_cpv_analysis.avg_savings`.

## `gold_market_summary` — KPIs by dimension · 41 rows · long format
Cols: `dimension` (`buyer_country`/`cpv_division_name`/`proc_type`), `dimension_value`,
`notice_count`, `total_estimated`, `total_awarded`, `avg_savings_pct`, `avg_tenders`,
`sme_winner_count`.

## `gold_cpv_analysis` — per-CPV stats · 38 rows
Cols: `cpv_division`, `cpv_division_name`, `awards_count`, `total_awarded`, `avg_savings`,
`avg_competition`, `sme_wins`, `open_notices`, `total_open_est`.

---

## ML-output tables (Role 3 — batch-scored predictions, read-only)

### `gold_notice_enrichment` · 71,432 rows · grain: notice
Key cols: `notice_id`, `notice_type`, `buyer_country`, `proc_type`,
**`cpv_pred`** + `cpv_pred_name` (predicted CPV, 100% filled — use to fill/clean categories),
`cpv_confidence`, **`cpv_review_flag`** (likely mis-coded — a *candidate*, not proof),
**`predicted_award_value`** (benchmark range, best for open tenders),
`total_awarded_actual` (~43% — **prefer when present**), `estimated_value` (~43%),
`cpv_division_original` / `cpv_division_name_original` (buyer-entered; may be wrong/missing).

### `gold_bid_win_probability` · 134,516 rows · grain: bid (competitive lots only)
Key cols: `notice_id`+`lot_id`+`tender_id`, `tendering_party_id` (anonymized),
`tender_value`, `n_bids`, **`win_probability`** (0–1 triage/ranking score, AUC≈0.74 — not a
guaranteed winner), `is_winner_actual` (0/1 — **prefer when present**).

---

## Also present in `data/gold/` (available if useful)
`gold_notices` (71,432), `gold_lots` (227,017), `gold_country_kpis` (63),
`gold_cpv_kpis` (46) — clean per-notice / per-lot / aggregate tables.

## Prediction caveats the chatbot must honor
- Prefer **actuals** (`total_awarded_actual`, `is_winner_actual`) over predictions.
- `predicted_award_value` = range/benchmark, not a quote. `win_probability` = triage score.
  `cpv_review_flag` = review candidate. Never present a prediction as a certainty.
