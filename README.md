# TED Procurement Intelligence
**IE University × Microsoft — Capstone 2026**

Transform 1 month of European public procurement data into strategic intelligence — from raw XML notices to ML models, dashboards, and a conversational chatbot.

---

## Architecture

```
TED packages
    ↓  pipeline/ingest.py
Raw XML (data/raw/)
    ↓  pipeline/bronze.py
Bronze  (8 Parquet tables · raw strings)
    ↓  pipeline/silver.py
Silver  (8 cleaned, typed, CPV-enriched Parquet tables)
    ↓  pipeline/validate_silver.py
Silver validation report
    ↓  pipeline/gold.py
Gold    (committed Parquet datasets · starter/smoke tables + dashboard analytics)
    ↓  models/train_models.py
ML Models  (win probability · competition · bid estimation)
    ↓  app/dashboard.py
Dashboard + Chatbot
```

---

## Quick start

### 1. Clone and install

Use Python 3.11, matching `runtime.txt`.

```bash
git clone https://github.com/nguadamillas/Microsoft-Capstone.git
cd Microsoft-Capstone
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
git clone https://github.com/nguadamillas/Microsoft-Capstone.git
cd Microsoft-Capstone
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Bronze, Silver, and Gold Parquet datasets are included in GitHub. `data/raw/` is intentionally git-ignored because it contains downloaded TED XML packages.

### 2. Add your Anthropic API key

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### 3. Configure date range

Edit `config.py` → `TED_PACKAGES` list.  
Default: **January 2026** (21 publication days, ~30k notices, ~1.2 GB raw XML).

To find package IDs for other months, check the [TED release calendar](https://ted.europa.eu/release-calendar) — package IDs follow `YYYYNNNNN` where `NNNNN` is the OJ issue number.

### 4. Run the pipeline

```bash
# Rebuild Silver/Gold from the committed Parquet layers
python -m pipeline.silver
python -m pipeline.validate_silver
python -m pipeline.gold

# Full pipeline, including raw XML download
python -m pipeline.run_pipeline

# Skip download if data/raw/ already exists locally
python -m pipeline.run_pipeline --skip-ingest

# Run only one step
python -m pipeline.run_pipeline --only silver
```

### 5. Train the ML models

```bash
python -m models.train_models
```

### 6. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open http://localhost:8501 in your browser.

---

## Data source

**TED — Tenders Electronic Daily**  
Supplement to the Official Journal of the European Union  
Published by the Publications Office of the EU  
Format: eForms UBL 2.3 XML  
URL: `https://ted.europa.eu/packages/daily/{YYYYNNNNN}`

---

## Gold tables reference

The repository includes eight Gold Parquet files. `pipeline.gold` writes the starter/smoke-test Gold tables (`gold_notices`, `gold_lots`, `gold_awards`, `gold_country_kpis`, `gold_cpv_kpis`) and a smoke report. The dashboard analytical tables (`gold_opportunities`, `gold_market_summary`, `gold_cpv_analysis`) are also committed for immediate app use.

| Table | Description | Key columns |
|---|---|---|
| `gold_notices` | Notice-level joined summary | `notice_id`, `buyer_country`, `estimated`, `awarded_amount`, `contract_value` |
| `gold_lots` | Lot-level joined summary | `notice_id`, `lot_id`, `winner_org_id`, `tenders_count`, `awarded_amount` |
| `gold_awards` | Awarded lot/contract rows | `notice_id`, `lot_result_id`, `winner_org_id`, `awarded_amount`, `contract_value` |
| `gold_country_kpis` | Country-level KPIs | `buyer_country`, `notice_count`, `total_contract_value`, `avg_tenders_count` |
| `gold_cpv_kpis` | CPV division KPIs | `cpv_division`, `cpv_division_name`, `notice_count`, `total_contract_value` |
| `gold_opportunities` | Dashboard opportunity table for open Contract Notices (CN) | `notice_id`, `buyer_country`, `estimated`, `cpv_division_name`, `num_lots` |
| `gold_market_summary` | Dashboard KPIs by country / CPV / procedure type | `dimension`, `dimension_value`, `total_awarded`, `avg_savings_pct` |
| `gold_cpv_analysis` | Dashboard CPV-level competition and value stats | `cpv_division_name`, `avg_competition`, `sme_wins` |

---

## ML models

| Model | Task | Algorithm | Target |
|---|---|---|---|
| Win Probability | Binary classification | LightGBM | `sme_winner` (1/0) |
| Competition Intensity | Regression | LightGBM | `avg_tenders_per_lot` |
| Bid Estimation | Regression | LightGBM | `awarded_eur` (log-scale) |

Models saved to `models/saved/` as `.joblib` files with accompanying feature importance JSON.

---

## Project structure

```
Microsoft-Capstone/
├── runtime.txt                ← deployment Python runtime
├── config.py                  ← date ranges, paths, API keys, CPV lookup
├── requirements.txt
├── .env                       ← ANTHROPIC_API_KEY (git-ignored)
├── analytics/                 ← business analysis scripts
├── assistant/                 ← standalone chatbot prototype
├── ml/                        ← standalone EDA and model scripts
├── pipeline/
│   ├── ingest.py              ← download TED XML packages
│   ├── bronze.py              ← XML → 8 Parquet tables
│   ├── silver.py              ← type casting, deduplication, CPV enrichment
│   ├── validate_silver.py     ← Silver quality checks
│   ├── gold.py                ← starter/smoke Gold tables and report
│   └── run_pipeline.py        ← orchestrator
├── models/
│   ├── train_models.py        ← feature engineering + 3 LightGBM models
│   └── saved/                 ← .joblib model files + metrics JSON
├── app/
│   ├── dashboard.py           ← main Streamlit dashboard
│   ├── streamlit_app.py       ← legacy/simple Streamlit dashboard
│   ├── procurement_assistant.py
│   └── project_chatbot.py
└── data/
    ├── raw/                   ← extracted XML files per package (git-ignored)
    ├── bronze/                ← 8 committed Parquet tables
    ├── silver/                ← 8 committed Parquet tables
    └── gold/                  ← 8 committed Parquet tables
```

---

## Submission checklist

- [ ] Public GitHub repo with code, committed Parquet datasets, and this README
- [ ] Executive slide deck (see report outline below)
- [ ] Written report (20 pages max)

---

## Report outline (20 pages)

1. **Executive summary** (1p) — problem, approach, key findings
2. **Data source** (2p) — TED/eForms, schema, volume, quality
3. **Pipeline architecture** (4p) — medallion design, Bronze/Silver/Gold decisions
4. **ML models** (6p) — feature engineering, model selection, results per model
5. **Business analytics layer** (4p) — dashboard design, chatbot architecture
6. **Key insights** (5p) — market patterns, savings distribution, competition intensity
7. **Microsoft value proposition** (4p) — how this system helps procurement strategy
8. **Limitations & future work** (2p)
9. **Appendices** (2p) — data dictionary, CPV lookup, model metrics
