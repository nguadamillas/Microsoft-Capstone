# Procurement Assistant (standalone)

A custom chat assistant for the **TED Procurement Intelligence** product (IE University × Microsoft).
It answers business questions about the European public-procurement market using the project's **Gold
tables**, and renders **KPI cards and charts** when an answer is quantitative. Built separately from the
Streamlit dashboard so the design is fully controllable; it can be embedded back into the dashboard via an
iframe.

```
assistant/
├── backend/   FastAPI · loads Gold parquet · Anthropic tool-use · /chat
└── frontend/  vanilla HTML/CSS/JS chat UI · Microsoft branding · Chart.js
```

## Run it

**1. Backend** (terminal 1)
```bash
cd assistant/backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then put your real ANTHROPIC_API_KEY in .env
uvicorn app:app --reload --port 8000
```

**2. Frontend** (terminal 2)
```bash
cd assistant/frontend
python -m http.server 5500
```
Open **http://localhost:5500**.

## Data

The backend reads `gold_opportunities`, `gold_awards`, `gold_market_summary`, `gold_cpv_analysis` from
`data/gold/*.parquet`. If those files aren't found it falls back to **illustrative sample data** (a yellow
banner appears in the UI) so you can design against a working app immediately. Point it at the real tables
by running from the repo root or setting `GOLD_DIR`:
```bash
GOLD_DIR=../../data/gold uvicorn app:app --reload --port 8000
```

## How answers are built

`/chat` runs an Anthropic tool-use loop: the model calls a sandboxed `query_data(pandas_expression)` tool to
compute exact figures from the Gold DataFrames, then returns a small JSON object:
```json
{ "text": "...", "kpis": [{"label": "...", "value": "..."}],
  "chart": {"type": "bar", "title": "...", "labels": [...], "series": [{"name": "...", "data": [...]}]},
  "source": "gold_awards" }
```
The frontend renders `text` (markdown), `kpis` (cards), and `chart` (Chart.js). `kpis`/`chart` appear only
when relevant.

## Embed in the Streamlit dashboard (optional)

With the backend + frontend running, add to the dashboard:
```python
import streamlit.components.v1 as components
components.iframe("http://localhost:5500", height=720)
```
For a deployed demo, host the API and the static frontend and point `API`/`HEALTH` in `frontend/app.js` at
the hosted backend URL.

## Notes

- Model is set in `backend/app.py` (`ASSISTANT_MODEL`, default `claude-sonnet-4-6`) — confirm your team's string.
- `query_data` runs in a restricted namespace (no imports/files/network); only the model's pandas expression
  is evaluated, never raw user text.
- Theme toggle (◐) switches dark/light; both are tuned for readable contrast.
