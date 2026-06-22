# TED Procurement Intelligence
**IE University × Microsoft — Capstone 2026**

Transform 1 month of European public procurement data into strategic intelligence — from raw XML notices to ML models, dashboards, and a conversational chatbot.

---

## Architecture

```
Raw XML (TED)
    ↓  pipeline/ingest.py
Bronze  (8 Parquet tables · raw strings)
    ↓  pipeline/bronze.py
Silver  (cleaned, typed, CPV-enriched)
    ↓  pipeline/silver.py
Gold    (4 analytical tables · joined · computed KPIs)
    ↓  pipeline/gold.py
ML Models  (win probability · competition · bid estimation)
    ↓  models/train_models.py
Dashboard + Chatbot
    ↓  app/streamlit_app.py
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/your-org/ted-procurement.git
cd ted-procurement
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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
# Full pipeline (download + transform + build Gold)
python -m pipeline.run_pipeline

# Skip download if you already have data/raw/
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
streamlit run app/streamlit_designed.py
```

Open http://localhost:8501 in your browser.

> Note: this repo installs the visual dashboard dependencies out of the box. If you want to train or evaluate the LightGBM models from `models/train_models.py`, install `lightgbm` separately with OpenMP support. On macOS, that typically requires:
>
> ```bash
> brew install libomp
> source .venv/bin/activate
> pip install lightgbm
> ```
>
> Then run:
>
> ```bash
> python -m models.train_models
> ```

---

## Data source

**TED — Tenders Electronic Daily**  
Supplement to the Official Journal of the European Union  
Published by the Publications Office of the EU  
Format: eForms UBL 2.3 XML  
URL: `https://ted.europa.eu/packages/daily/{YYYYNNNNN}`

---

## Gold tables reference

| Table | Description | Key columns |
|---|---|---|
| `gold_opportunities` | Open Contract Notices (CN) | `notice_id`, `buyer_country`, `estimated`, `cpv_division_name`, `num_lots` |
| `gold_awards` | Contract Award Notices (CAN) | `awarded_eur`, `savings_pct`, `avg_tenders_per_lot`, `sme_winner` |
| `gold_market_summary` | Aggregated KPIs by country / CPV / type | `dimension`, `dimension_value`, `total_awarded`, `avg_savings_pct` |
| `gold_cpv_analysis` | CPV-level competition & value stats | `cpv_division_name`, `avg_competition`, `sme_wins` |

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
ted_procurement/
├── config.py                  ← date ranges, paths, API keys, CPV lookup
├── requirements.txt
├── .env                       ← ANTHROPIC_API_KEY (git-ignored)
├── pipeline/
│   ├── ingest.py              ← download TED XML packages
│   ├── bronze.py              ← XML → 8 Parquet tables
│   ├── silver.py              ← type casting, deduplication, CPV enrichment
│   ├── gold.py                ← joins, aggregations, computed KPIs
│   └── run_pipeline.py        ← orchestrator
├── models/
│   ├── train_models.py        ← feature engineering + 3 LightGBM models
│   └── saved/                 ← .joblib model files + metrics JSON
├── app/
│   └── streamlit_app.py       ← dashboard (opportunities, awards) + chatbot
└── data/
    ├── raw/                   ← extracted XML files per package
    ├── bronze/                ← 8 Parquet tables (raw strings)
    ├── silver/                ← 8 cleaned Parquet tables
    └── gold/                  ← 4 analytical Parquet tables
```

---

## Submission checklist

- [ ] Public GitHub repo with all code, notebooks, and this README
- [ ] Executive slide deck (see report outline below)
- [ ] Written report (30 pages max)

---

## Report outline (30 pages)

1. **Executive summary** (1p) — problem, approach, key findings
2. **Data source** (2p) — TED/eForms, schema, volume, quality
3. **Pipeline architecture** (4p) — medallion design, Bronze/Silver/Gold decisions
4. **ML models** (6p) — feature engineering, model selection, results per model
5. **Business analytics layer** (4p) — dashboard design, chatbot architecture
6. **Key insights** (5p) — market patterns, savings distribution, competition intensity
7. **Microsoft value proposition** (4p) — how this system helps procurement strategy
8. **Limitations & future work** (2p)
9. **Appendices** (2p) — data dictionary, CPV lookup, model metrics
