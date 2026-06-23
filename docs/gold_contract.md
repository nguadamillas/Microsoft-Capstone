# Gold Layer Contract — Chatbot & Dashboard interface

**Owner of this contract:** Role 6 (Chatbot/PM) · **Producer:** Role 2 (Gold layer)

This document pins the **exact Gold tables and columns** the chatbot (`app/chatbot.py`) and
dashboard (`app/dashboard.py`) consume. It is the interface between the analytics interface
(Role 6 / Role 5) and the Gold producer (Role 2).

> **Status note (2026-06-23):** `pipeline/gold.py`'s `run()` currently writes a *smoke-test*
> set of tables (`gold_notices, gold_lots, gold_awards, gold_country_kpis, gold_cpv_kpis`).
> The four tables below are produced by the `build_opportunities / build_awards /
> build_market_summary / build_cpv_analysis` functions that already exist in `gold.py` but are
> **not yet called in `run()`**. **Action for Role 2:** wire those four `build_*` functions into
> `run()` so the files below are written to `data/gold/`. Until then, the interface runs against
> synthetic fixtures (`tests/fixtures/make_synthetic_gold.py`).

The chatbot engine is **schema-agnostic** — it introspects whatever columns exist at runtime, so
adding columns is safe. Removing or renaming the columns marked **required** below will break
specific dashboard charts / sample questions.

---

## `gold_opportunities.parquet` — open Contract Notices (CN)

One row per open tender. Source: `build_opportunities()`.

| Column | Type | Required | Notes |
|---|---|---|---|
| `notice_id` | str | ✅ | Primary key |
| `pub_date` | date-str | | Publication date |
| `buyer_name` | str | | Contracting authority |
| `buyer_country` | str | ✅ | Country filter / grouping |
| `buyer_org_id` | str | | |
| `project_title` | str | | |
| `proc_type` | str | ✅ | Services / Supplies / Works |
| `procedure_type` | str | | Open / restricted / etc. |
| `cpv_main` | str | | Full CPV code |
| `cpv_division` | str | | 2-digit CPV division |
| `cpv_division_name` | str | ✅ | Human-readable category |
| `estimated` | float (EUR) | ✅ | Estimated value; ~43% populated (legal optionality) |
| `num_lots` | int | | |
| `avg_lot_duration` | float | | |
| `submission_deadline` | date-str | | |
| `framework` | bool | | |
| `source_file` | str | | Provenance |

## `gold_awards.parquet` — Contract Award Notices (CAN)

One row per awarded notice. Source: `build_awards()`.

| Column | Type | Required | Notes |
|---|---|---|---|
| `notice_id` | str | ✅ | Primary key |
| `pub_date` | date-str | | |
| `buyer_name` | str | | |
| `buyer_country` | str | ✅ | |
| `buyer_org_id` | str | | |
| `project_title` | str | | |
| `proc_type` | str | ✅ | |
| `cpv_main` | str | | |
| `cpv_division` | str | | |
| `cpv_division_name` | str | ✅ | |
| `estimated` | float (EUR) | | |
| `awarded_eur` | float (EUR) | ✅ | Final awarded value |
| `savings_pct` | float | ✅ | `(estimated - awarded) / estimated * 100`, rounded 1dp |
| `num_lots` | int | | |
| `num_awarded_lots` | int | | |
| `avg_tenders_per_lot` | float | ✅ | Competition intensity |
| `total_tenders` | int | | |
| `winner_names` | str | | Comma-joined |
| `winner_countries` | str | | Comma-joined |
| `sme_winner` | bool/int | ✅ | True if any winner is an SME |
| `avg_lot_duration` | float | | |
| `source_file` | str | | |

## `gold_market_summary.parquet` — aggregated KPIs

Long-format: one row per (dimension, dimension_value). Source: `build_market_summary()`.
Dimensions: `buyer_country`, `cpv_division_name`, `proc_type`.

| Column | Type | Required | Notes |
|---|---|---|---|
| `dimension` | str | ✅ | Which grouping (country / cpv / proc_type) |
| `dimension_value` | str | ✅ | The group label |
| `notice_count` | int | | |
| `total_estimated` | float (EUR) | | |
| `total_awarded` | float (EUR) | | |
| `avg_savings_pct` | float | | |
| `avg_tenders` | float | | |
| `sme_winner_count` | int | | |

## `gold_cpv_analysis.parquet` — CPV-level stats

One row per CPV division. Source: `build_cpv_analysis()`.

| Column | Type | Required | Notes |
|---|---|---|---|
| `cpv_division` | str | ✅ | 2-digit code |
| `cpv_division_name` | str | ✅ | |
| `awards_count` | int | | |
| `total_awarded` | float (EUR) | | |
| `avg_savings` | float | | |
| `avg_competition` | float | | Avg tenders per lot |
| `sme_wins` | int | | |
| `open_notices` | int | | From opportunities |
| `total_open_est` | float (EUR) | | |

---

## Conventions

- All monetary values are in **EUR**.
- `proc_type` ∈ {`Services`, `Supplies`, `Works`} (plus possibly `Other`).
- Missing numeric values are `NaN`, not 0 — aggregations must use `min_count`/`dropna` as needed.
- Variable names exposed to the chatbot engine: `opportunities`, `awards`, `market_summary`,
  `cpv_analysis` (mapped from the four files above).
