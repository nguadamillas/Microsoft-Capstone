"""
app/dashboard.py
─────────────────────────────────────────────────────────────
Comprehensive Streamlit dashboard — run with:
    streamlit run app/dashboard.py

Tabs:
  📊 Opportunities  — open Contract Notices with charts, map & export
  🏆 Awards         — Contract Award Notices with analytics & export
  🤖 ML Predictions — interactive bid / competition / win-prob estimator
  💬 Chatbot        — natural-language queries via Claude API
  🔧 Pipeline       — ETL health panel & model artefacts
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOLD_DIR, MODEL_DIR, ANTHROPIC_API_KEY, CPV_DIVISIONS

try:
    import joblib
    JOBLIB_OK = True
except ModuleNotFoundError:
    JOBLIB_OK = False

try:
    import pycountry
except ModuleNotFoundError:
    pycountry = None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TED Procurement Intelligence",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, .stApp {
    background: #0d1117 !important;
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e6edf3;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid='stSidebar'] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
    min-width: 260px !important;
}
section[data-testid='stSidebar'] > div { padding-top: 0 !important; }

/* ── Sidebar text overrides ── */
section[data-testid='stSidebar'] label,
section[data-testid='stSidebar'] .stMarkdown p,
section[data-testid='stSidebar'] .stCaption { color: #8b949e !important; }
section[data-testid='stSidebar'] h1,
section[data-testid='stSidebar'] h2,
section[data-testid='stSidebar'] h3 { color: #e6edf3 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #0078D4 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.45rem 1.1rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover { background: #106EBE !important; }

.stDownloadButton > button {
    background: transparent !important;
    color: #58a6ff !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover { border-color: #58a6ff !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
    gap: 4px !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #0078D4 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px !important;
}

/* ── Native metric ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 22px 16px !important;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}
[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── DataFrames ── */
.stDataFrame { border-radius: 8px !important; border: 1px solid #21262d !important; }

/* ── Inputs ── */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(0,120,212,0.25) !important;
    border: 1px solid rgba(0,120,212,0.4) !important;
    border-radius: 3px !important;
    color: #58a6ff !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.stSlider [data-baseweb="slider"] { background: #21262d !important; }

/* ── Expanders ── */
div[data-testid="stExpander"] details {
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    background: #161b22 !important;
}
div[data-testid="stExpander"] summary {
    color: #e6edf3 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* ── Chat ── */
.stChatMessage {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Dividers ── */
hr { border-color: #21262d !important; }

/* ── Alert/info boxes ── */
.stAlert { border-radius: 6px !important; }

/* ── Code blocks ── */
.stCode, code { border-radius: 6px !important; }

/* ── Custom: header bar ── */
.ms-topbar {
    background: linear-gradient(90deg, #0d1117 60%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 18px 0 20px;
    margin-bottom: 20px;
}
.ms-brand-row {
    display: flex;
    align-items: center;
    gap: 14px;
}
.ms-icon-box {
    width: 38px; height: 38px;
    background: #0078D4;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.ms-product-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.ms-product-sub {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 2px;
    font-weight: 400;
}
.ms-tag {
    display: inline-block;
    background: rgba(0,120,212,0.12);
    border: 1px solid rgba(0,120,212,0.25);
    color: #58a6ff;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 9px;
    border-radius: 20px;
    margin-left: 10px;
    vertical-align: middle;
}
.ms-demo-bar {
    background: rgba(255,152,0,0.07);
    border: 1px solid rgba(255,152,0,0.2);
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 0.82rem;
    color: #ffa657;
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Custom: KPI cards ── */
.ms-kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 4px;
}
.ms-kpi {
    background: #161b22;
    border: 1px solid #21262d;
    border-top: 3px solid var(--c, #0078D4);
    border-radius: 8px;
    padding: 18px 20px 14px;
}
.ms-kpi-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
}
.ms-kpi-lbl {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

/* ── Custom: section heading ── */
.ms-sh {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
    margin: 20px 0 14px;
}
.ms-sh-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e6edf3;
}
.ms-sh-pill {
    background: #21262d;
    color: #8b949e;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
}

/* ── Custom: sidebar nav ── */
.ms-sb-header {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
    padding: 16px;
    margin: -1rem -1rem 0;
}
.ms-sb-brand { display: flex; align-items: center; gap: 10px; }
.ms-sb-icon {
    width: 30px; height: 30px;
    background: #0078D4;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}
.ms-sb-name { font-size: 0.9rem; font-weight: 700; color: #e6edf3; line-height: 1.2; }
.ms-sb-sub  { font-size: 0.68rem; color: #8b949e; }
.ms-sb-section {
    font-size: 0.65rem;
    font-weight: 700;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 16px 0 6px;
}
.ms-sb-footer {
    font-size: 0.7rem;
    color: #8b949e;
    padding: 14px 0 4px;
    border-top: 1px solid #21262d;
    margin-top: 8px;
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Palettes ──────────────────────────────────────────────────────────────────
PALETTES: dict[str, dict] = {
    "Microsoft Blue": {
        "cont": ["#00338D", "#0078D4", "#00BCF2", "#06D6A0", "#FFD166"],
        "disc": ["#0078D4", "#06D6A0", "#FF6A00", "#9B5DE5", "#FFD166", "#EF476F"],
    },
    "Teal-Purple": {
        "cont": ["#005F73", "#0A9396", "#94D2BD", "#5A2D81", "#E9C46A"],
        "disc": ["#06D6A0", "#118AB2", "#5A2D81", "#9B5DE5", "#E9C46A", "#B0E3E6"],
    },
    "Warm Sunset": {
        "cont": ["#7F2704", "#E85D04", "#FF9E3B", "#FFD166"],
        "disc": ["#FF6A00", "#FF9E3B", "#FFD166", "#E85D04", "#FFB4A2", "#FB8500"],
    },
    "High Contrast": {
        "cont": ["#001219", "#005F73", "#0A9396", "#FB8500", "#F28482"],
        "disc": ["#0078D4", "#FB8500", "#F28482", "#FFD166", "#06D6A0", "#9B5DE5"],
    },
}

# ── Utility helpers ───────────────────────────────────────────────────────────

def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", size=12,
                  family="Inter, Segoe UI, -apple-system, sans-serif"),
        legend=dict(font=dict(color="#8b949e", size=11),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="#21262d"),
        margin=dict(l=12, r=12, t=36, b=12),
        title_font=dict(color="#e6edf3", size=13, family="Inter, Segoe UI, sans-serif"),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#8b949e"),
            titlefont=dict(color="#8b949e"),
            bgcolor="#161b22",
            outlinecolor="#21262d",
        ),
    )
    fig.update_xaxes(
        gridcolor="#21262d",
        zeroline=False,
        tickfont=dict(color="#8b949e", size=11),
        linecolor="#21262d",
    )
    fig.update_yaxes(
        gridcolor="#21262d",
        zeroline=False,
        tickfont=dict(color="#8b949e", size=11),
        linecolor="#21262d",
    )
    return fig


def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="data")
    return buf.getvalue()


def _iso3(name: str) -> str:
    if pycountry is None:
        return name
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        return name


def _fmt_eur(v) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1e6:
        return f"€{v / 1e6:.2f}M"
    return f"€{v:,.0f}"


# ── Demo data (fallback when no Gold parquet files exist) ─────────────────────

@st.cache_data
def _demo_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    countries = [
        "Germany", "France", "Spain", "Italy", "Netherlands",
        "Poland", "Sweden", "Denmark", "Belgium", "Austria",
        "Portugal", "Finland", "Ireland", "Czech Republic", "Romania",
    ]
    proc_types = ["Services", "Supplies", "Works"]
    cpv_names = list(CPV_DIVISIONS.values())
    cpv_codes = list(CPV_DIVISIONS.keys())

    n_o, n_a = 120, 90

    def rnd_dates(start, n, freq="6h"):
        return pd.date_range(start, periods=n, freq=freq).strftime("%Y-%m-%d").tolist()

    opp = pd.DataFrame({
        "notice_id":           [f"CN-2026-{i:03d}" for i in range(n_o)],
        "pub_date":            rnd_dates("2026-01-02", n_o, "6h"),
        "buyer_name":          [f"Authority {i}" for i in range(n_o)],
        "buyer_country":       rng.choice(countries, n_o),
        "estimated":           rng.lognormal(14, 1.8, n_o).clip(50_000, 60_000_000),
        "num_lots":            rng.integers(1, 9, n_o),
        "proc_type":           rng.choice(proc_types, n_o),
        "cpv_division_name":   rng.choice(cpv_names, n_o),
        "cpv_division":        rng.choice(cpv_codes, n_o),
        "project_title":       [f"Project {chr(65+i%26)}{i//26}" for i in range(n_o)],
        "submission_deadline": rnd_dates("2026-02-01", n_o, "8h"),
    })

    est = rng.lognormal(14, 1.8, n_a).clip(50_000, 60_000_000)
    awd = pd.DataFrame({
        "notice_id":            [f"CAN-2026-{i:03d}" for i in range(n_a)],
        "award_date":           rnd_dates("2026-01-05", n_a, "9h"),
        "buyer_name":           [f"Authority {i}" for i in range(n_a)],
        "buyer_country":        rng.choice(countries, n_a),
        "estimated":            est,
        "awarded_eur":          est * rng.uniform(0.72, 1.05, n_a),
        "savings_pct":          rng.normal(7.5, 5, n_a).clip(-5, 32),
        "avg_tenders_per_lot":  rng.exponential(4.5, n_a).clip(1, 22),
        "sme_winner":           rng.choice([0, 1], n_a, p=[0.55, 0.45]),
        "proc_type":            rng.choice(proc_types, n_a),
        "cpv_division_name":    rng.choice(cpv_names, n_a),
        "cpv_division":         rng.choice(cpv_codes, n_a),
        "project_title":        [f"Award Project {i}" for i in range(n_a)],
        "winner_names":         [f"Supplier {chr(65+i%26)}" for i in range(n_a)],
        "winner_countries":     rng.choice(countries, n_a),
        "num_awarded_lots":     rng.integers(1, 6, n_a),
    })

    market = pd.DataFrame({
        "dimension":       ["country"] * len(countries),
        "dimension_value": countries,
        "total_awarded":   rng.lognormal(15, 1.2, len(countries)),
        "avg_savings_pct": rng.uniform(3, 15, len(countries)),
    })

    cpv_df = pd.DataFrame({
        "cpv_division_name": cpv_names[:18],
        "avg_competition":   rng.exponential(4.5, 18).clip(1, 16),
        "sme_wins":          rng.integers(8, 80, 18),
    })

    return {"opp": opp, "awd": awd, "market": market, "cpv": cpv_df}


@st.cache_data
def _load_data() -> tuple[dict[str, pd.DataFrame], bool]:
    def _p(name):
        p = GOLD_DIR / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    real = {
        "opp":    _p("gold_opportunities"),
        "awd":    _p("gold_awards"),
        "market": _p("gold_market_summary"),
        "cpv":    _p("gold_cpv_analysis"),
    }
    if any(len(v) > 0 for v in real.values()):
        return real, False
    return _demo_data(), True


data, IS_DEMO = _load_data()
opp    = data["opp"]
awards = data["awd"]
market = data["market"]
cpv_df = data["cpv"]


def _ensure_demo_models() -> None:
    """Create and fit demo sklearn Pipeline models if missing or stale (< 2 KB)."""
    if not JOBLIB_OK:
        return

    stems = ["demo_win_probability", "demo_competition_intensity", "demo_bid_estimation"]
    # Fitted models are always > 2 KB; unfitted shells are ~900 bytes
    all_ok = all(
        (MODEL_DIR / f"{s}.joblib").exists()
        and (MODEL_DIR / f"{s}.joblib").stat().st_size > 2000
        for s in stems
    )
    if all_ok:
        return

    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    # Build inline demo data — avoids calling @st.cache_data outside Streamlit context
    _rng = np.random.default_rng(42)
    _n = 90
    _countries = ["Germany", "France", "Spain", "Italy", "Netherlands",
                  "Poland", "Sweden", "Denmark", "Belgium", "Austria"]
    _types = ["Services", "Supplies", "Works"]
    _est = _rng.lognormal(14, 1.8, _n).clip(50_000, 60_000_000)
    _df = pd.DataFrame({
        "estimated":            _est,
        "num_lots":             _rng.integers(1, 9, _n).astype(float),
        "buyer_country":        _rng.choice(_countries, _n),
        "proc_type":            _rng.choice(_types, _n),
        "awarded_eur":          _est * _rng.uniform(0.72, 1.05, _n),
        "avg_tenders_per_lot":  _rng.exponential(4.5, _n).clip(1, 22),
        "sme_winner":           _rng.choice([0, 1], _n, p=[0.55, 0.45]),
    })

    NUM = ["estimated", "num_lots"]
    CAT = ["buyer_country", "proc_type"]

    def _preproc():
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc",  StandardScaler())]), NUM),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), CAT),
        ], remainder="drop")

    X = _df[NUM + CAT]

    pipe_clf = Pipeline([("preproc", _preproc()), ("model", DummyClassifier(strategy="prior"))])
    pipe_clf.fit(X, _df["sme_winner"].astype(int))
    joblib.dump({"model": pipe_clf, "preprocessor": None, "num_feats": NUM, "cat_feats": CAT},
                MODEL_DIR / "demo_win_probability.joblib")

    pipe_comp = Pipeline([("preproc", _preproc()), ("model", DummyRegressor(strategy="mean"))])
    pipe_comp.fit(X, _df["avg_tenders_per_lot"])
    joblib.dump({"model": pipe_comp, "preprocessor": None, "num_feats": NUM, "cat_feats": CAT},
                MODEL_DIR / "demo_competition_intensity.joblib")

    pipe_bid = Pipeline([("preproc", _preproc()), ("model", DummyRegressor(strategy="mean"))])
    pipe_bid.fit(X, np.log1p(_df["awarded_eur"]))
    joblib.dump({"model": pipe_bid, "preprocessor": None, "num_feats": NUM, "cat_feats": CAT},
                MODEL_DIR / "demo_bid_estimation.joblib")

    uniform = {f: round(1.0 / len(NUM + CAT), 3) for f in NUM + CAT}
    for stem in stems:
        with open(MODEL_DIR / f"{stem}_features.json", "w") as f:
            json.dump(uniform, f)


_ensure_demo_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="ms-sb-header">
          <div class="ms-sb-brand">
            <div class="ms-sb-icon">🇪🇺</div>
            <div>
              <div class="ms-sb-name">TED Procurement</div>
              <div class="ms-sb-sub">Intelligence Platform</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ms-sb-section">Visual theme</div>', unsafe_allow_html=True)
    vis_theme = st.selectbox("", list(PALETTES.keys()), index=0, label_visibility="collapsed")
    palette = PALETTES[vis_theme]

    st.markdown('<div class="ms-sb-section">Filters</div>', unsafe_allow_html=True)

    all_countries = sorted(set(
        (opp["buyer_country"].dropna().tolist() if "buyer_country" in opp.columns else []) +
        (awards["buyer_country"].dropna().tolist() if "buyer_country" in awards.columns else [])
    ))
    sel_countries = st.multiselect("Countries", all_countries, placeholder="All countries")

    all_types = sorted(set(
        (opp["proc_type"].dropna().tolist() if "proc_type" in opp.columns else []) +
        (awards["proc_type"].dropna().tolist() if "proc_type" in awards.columns else [])
    ))
    sel_types = st.multiselect("Procurement type", all_types, placeholder="All types")

    all_cpv = sorted(set(
        (opp["cpv_division_name"].dropna().tolist() if "cpv_division_name" in opp.columns else []) +
        (awards["cpv_division_name"].dropna().tolist() if "cpv_division_name" in awards.columns else [])
    ))
    sel_cpv = st.multiselect("CPV category", all_cpv, placeholder="All categories")

    st.markdown('<div class="ms-sb-section">Date range</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    start_date = dc1.date_input("From", value=None)
    end_date   = dc2.date_input("To",   value=None)

    st.markdown('<div class="ms-sb-section">Display</div>', unsafe_allow_html=True)
    rows_show = st.slider("Table rows", 10, 200, 50, label_visibility="collapsed")
    st.caption(f"Showing up to {rows_show} rows per table")

    st.markdown(
        '<div class="ms-sb-footer">IE University × Microsoft<br>Capstone 2026</div>',
        unsafe_allow_html=True,
    )


def _apply_filters(df: pd.DataFrame, date_cols: list[str] | None = None) -> pd.DataFrame:
    if sel_countries and "buyer_country" in df.columns:
        df = df[df["buyer_country"].isin(sel_countries)]
    if sel_types and "proc_type" in df.columns:
        df = df[df["proc_type"].isin(sel_types)]
    if sel_cpv and "cpv_division_name" in df.columns:
        df = df[df["cpv_division_name"].isin(sel_cpv)]
    if start_date and end_date and date_cols:
        for col in date_cols:
            if col in df.columns:
                try:
                    s = pd.to_datetime(df[col], errors="coerce")
                    df = df[(s >= pd.to_datetime(start_date)) & (s <= pd.to_datetime(end_date))]
                    break
                except Exception:
                    pass
    return df


opp_f    = _apply_filters(opp,    ["pub_date", "publication_date", "notice_date"])
awards_f = _apply_filters(awards, ["award_date", "publication_date", "notice_date"])

# ── Header bar ────────────────────────────────────────────────────────────────
_total_est = opp["estimated"].sum()      if "estimated"   in opp.columns    else 0
_total_awd = awards["awarded_eur"].sum() if "awarded_eur" in awards.columns else 0
_sme_pct   = (
    awards["sme_winner"].sum() / len(awards) * 100
    if ("sme_winner" in awards.columns and len(awards) > 0) else 0
)
_avg_sav = awards["savings_pct"].mean() if "savings_pct" in awards.columns else 0
_n_ctry  = len(set(
    (opp["buyer_country"].dropna().tolist() if "buyer_country" in opp.columns else []) +
    (awards["buyer_country"].dropna().tolist() if "buyer_country" in awards.columns else [])
))

_demo_html = (
    '<div class="ms-demo-bar">⚠ Demo data — run <code>python -m pipeline.run_pipeline</code> '
    "to load real TED procurement tables.</div>"
    if IS_DEMO else ""
)

st.markdown(
    f"""
    <div class="ms-topbar">
      <div class="ms-brand-row">
        <div class="ms-icon-box">📋</div>
        <div>
          <div class="ms-product-name">
            TED Procurement Intelligence
            <span class="ms-tag">Capstone 2026</span>
          </div>
          <div class="ms-product-sub">
            European tender opportunities &nbsp;·&nbsp; Award analytics
            &nbsp;·&nbsp; ML forecasting &nbsp;·&nbsp; AI chatbot
          </div>
        </div>
      </div>
      {_demo_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="ms-kpi-grid">
      <div class="ms-kpi" style="--c:#0078D4">
        <div class="ms-kpi-val">{len(opp):,}</div>
        <div class="ms-kpi-lbl">Open tenders</div>
      </div>
      <div class="ms-kpi" style="--c:#0A9396">
        <div class="ms-kpi-val">{len(awards):,}</div>
        <div class="ms-kpi-lbl">Award records</div>
      </div>
      <div class="ms-kpi" style="--c:#6E40C9">
        <div class="ms-kpi-val">{_n_ctry}</div>
        <div class="ms-kpi-lbl">Countries</div>
      </div>
      <div class="ms-kpi" style="--c:#FF8C00">
        <div class="ms-kpi-val">€{_total_est/1e6:.1f}M</div>
        <div class="ms-kpi-lbl">Total estimated</div>
      </div>
      <div class="ms-kpi" style="--c:#2EA043">
        <div class="ms-kpi-val">{_avg_sav:.1f}%</div>
        <div class="ms-kpi-lbl">Avg savings</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_opp, tab_awd, tab_pred, tab_chat, tab_etl = st.tabs(
    ["📊 Opportunities", "🏆 Awards", "🤖 ML Predictions", "💬 Chatbot", "🔧 Pipeline"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPPORTUNITIES
# ════════════════════════════════════════════════════════════════════════════
with tab_opp:
    st.markdown(
        f'<div class="ms-sh"><span class="ms-sh-title">Open Contract Notices</span>'
        f'<span class="ms-sh-pill">{len(opp_f):,} records</span></div>',
        unsafe_allow_html=True,
    )

    if opp_f.empty:
        st.info("No opportunities match the current filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tenders",         f"{len(opp_f):,}")
        c2.metric("Countries",       f"{opp_f['buyer_country'].nunique() if 'buyer_country' in opp_f.columns else 0}")
        c3.metric("Total estimated", f"€{opp_f['estimated'].sum() / 1e6:.1f}M" if "estimated" in opp_f.columns else "—")
        c4.metric("Avg lots",        f"{opp_f['num_lots'].mean():.1f}" if "num_lots" in opp_f.columns else "—")

        col_l, col_r = st.columns([3, 2])

        # Top countries bar
        with col_l:
            st.markdown("##### Notices by country")
            ctry = (
                opp_f.groupby("buyer_country")["notice_id"]
                .count().reset_index(name="Notices")
                .sort_values("Notices", ascending=False).head(12)
            )
            fig = px.bar(
                ctry, x="Notices", y="buyer_country", orientation="h",
                color="Notices", color_continuous_scale=palette["cont"],
                text="Notices",
            )
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(yaxis_title="", xaxis_title="Notices", coloraxis_showscale=False)
            st.plotly_chart(_style(fig), use_container_width=True)

        # Procurement type donut
        with col_r:
            st.markdown("##### Procurement type split")
            pt = opp_f["proc_type"].value_counts().reset_index() if "proc_type" in opp_f.columns else pd.DataFrame()
            if not pt.empty:
                pt.columns = ["type", "count"]
                fig = px.pie(
                    pt, names="type", values="count", hole=0.45,
                    color_discrete_sequence=palette["disc"],
                )
                fig.update_traces(textinfo="percent+label", textfont_size=12)
                fig.update_layout(showlegend=False)
                st.plotly_chart(_style(fig), use_container_width=True)
            else:
                st.info("No procurement type data.")

        # CPV bar
        st.markdown("##### Top CPV categories by estimated value")
        if "cpv_division_name" in opp_f.columns and "estimated" in opp_f.columns:
            cpv_opp = (
                opp_f.dropna(subset=["cpv_division_name"])
                .groupby("cpv_division_name")
                .agg(Notices=("notice_id", "count"), Total=("estimated", "sum"))
                .reset_index().sort_values("Total", ascending=False).head(14)
            )
            fig = px.bar(
                cpv_opp, x="cpv_division_name", y="Total",
                color="Notices", color_continuous_scale=palette["cont"],
                labels={"cpv_division_name": "CPV category", "Total": "Total estimated (€)"},
                text="Notices",
            )
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=True)
            st.plotly_chart(_style(fig), use_container_width=True)

        # Choropleth map
        st.markdown("##### Geographic distribution")
        map_col1, map_col2 = st.columns([2, 1])
        with map_col2:
            map_metric  = st.selectbox("Map metric",  ["Count", "Total estimated €"], key="opp_map_metric")
            map_top_n   = st.slider("Top N countries", 5, 30, 15, key="opp_map_n")

        if "buyer_country" in opp_f.columns:
            grp = opp_f.dropna(subset=["buyer_country"])
            if map_metric == "Count":
                agg = grp.groupby("buyer_country")["notice_id"].count().reset_index(name="value")
            else:
                agg = grp.groupby("buyer_country")["estimated"].sum().reset_index(name="value") if "estimated" in grp.columns else pd.DataFrame()

            if not agg.empty and agg["value"].sum() > 0:
                agg["iso3"] = agg["buyer_country"].apply(_iso3)
                n_iso3 = agg["iso3"].apply(lambda x: isinstance(x, str) and len(x) == 3).sum()
                lmode  = "ISO-3" if n_iso3 >= max(3, len(agg) // 2) else "country names"
                loc_col = "iso3" if lmode == "ISO-3" else "buyer_country"

                fig = px.choropleth(
                    agg, locations=loc_col, locationmode=lmode,
                    color="value", hover_name="buyer_country",
                    color_continuous_scale=palette["cont"],
                    labels={"value": map_metric},
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    geo_bgcolor="rgba(0,26,51,0.5)",
                    coloraxis_colorbar=dict(
                        tickfont=dict(color="#ffffff"),
                        titlefont=dict(color="#ffffff"),
                        bgcolor="rgba(0,0,0,0)",
                        outlinecolor="rgba(255,255,255,0.1)",
                    ),
                    font=dict(color="#ffffff"),
                )
                with map_col1:
                    st.plotly_chart(fig, use_container_width=True)

        # Table & export
        st.markdown("##### Opportunity details")
        show_cols = [c for c in ["project_title", "buyer_country", "proc_type",
                                  "cpv_division_name", "estimated", "num_lots",
                                  "submission_deadline"] if c in opp_f.columns]
        disp = opp_f[show_cols].sort_values("estimated", ascending=False).head(rows_show).copy() if "estimated" in opp_f.columns else opp_f[show_cols].head(rows_show).copy()
        if "estimated" in disp.columns:
            disp["estimated"] = disp["estimated"].apply(_fmt_eur)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        ex1, ex2 = st.columns(2)
        ex1.download_button("⬇ CSV",   _to_csv(opp_f),   file_name="opportunities.csv",   mime="text/csv")
        try:
            ex2.download_button("⬇ Excel", _to_excel(opp_f), file_name="opportunities.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — AWARDS
# ════════════════════════════════════════════════════════════════════════════
with tab_awd:
    st.markdown(
        f'<div class="ms-sh"><span class="ms-sh-title">Contract Award Notices</span>'
        f'<span class="ms-sh-pill">{len(awards_f):,} records</span></div>',
        unsafe_allow_html=True,
    )

    if awards_f.empty:
        st.info("No awards match the current filters.")
    else:
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Awards",       f"{len(awards_f):,}")
        a2.metric("Countries",    f"{awards_f['buyer_country'].nunique() if 'buyer_country' in awards_f.columns else 0}")
        a3.metric("Total awarded",f"€{awards_f['awarded_eur'].sum() / 1e6:.1f}M" if "awarded_eur" in awards_f.columns else "—")
        a4.metric("Avg savings",  f"{awards_f['savings_pct'].mean():.1f}%" if "savings_pct" in awards_f.columns else "—")
        sme_f = (awards_f["sme_winner"].sum() / len(awards_f) * 100) if "sme_winner" in awards_f.columns else 0
        a5.metric("SME wins",     f"{sme_f:.1f}%")

        col_l, col_r = st.columns(2)

        # Savings histogram
        with col_l:
            st.markdown("##### Savings % distribution")
            if "savings_pct" in awards_f.columns:
                df_sav = awards_f.dropna(subset=["savings_pct"])
                fig = px.histogram(
                    df_sav, x="savings_pct", nbins=35,
                    color_discrete_sequence=[palette["disc"][1]],
                    labels={"savings_pct": "Savings (%)"},
                )
                mean_s = df_sav["savings_pct"].mean()
                fig.add_vline(
                    x=mean_s, line_dash="dash", line_color="#FFD166",
                    annotation_text=f"Mean {mean_s:.1f}%",
                    annotation_font_color="#FFD166",
                )
                st.plotly_chart(_style(fig), use_container_width=True)

        # Competition by best available grouping column
        with col_r:
            if "avg_tenders_per_lot" in awards_f.columns:
                # pick the most descriptive available grouping column
                group_col = next(
                    (c for c in ["cpv_division_name", "proc_type", "buyer_country"]
                     if c in awards_f.columns),
                    None,
                )
                label_map = {
                    "cpv_division_name": "CPV category",
                    "proc_type":         "Procurement type",
                    "buyer_country":     "Country",
                }
                group_label = label_map.get(group_col, group_col) if group_col else ""
                st.markdown(f"##### Avg tenders per lot by {group_label.lower()}")

                if group_col:
                    comp = (
                        awards_f.dropna(subset=[group_col, "avg_tenders_per_lot"])
                        .groupby(group_col)["avg_tenders_per_lot"]
                        .mean().reset_index()
                        .sort_values("avg_tenders_per_lot", ascending=False).head(12)
                    )
                    if not comp.empty:
                        fig = px.bar(
                            comp, x="avg_tenders_per_lot", y=group_col, orientation="h",
                            color="avg_tenders_per_lot", color_continuous_scale=palette["cont"],
                            text="avg_tenders_per_lot",
                            labels={"avg_tenders_per_lot": "Avg tenders / lot", group_col: ""},
                        )
                        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                        fig.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(_style(fig), use_container_width=True)
                    else:
                        st.info("No competition data available.")
                else:
                    # no grouping column — show overall distribution as histogram
                    st.markdown("##### Avg tenders per lot distribution")
                    fig = px.histogram(
                        awards_f.dropna(subset=["avg_tenders_per_lot"]),
                        x="avg_tenders_per_lot", nbins=20,
                        color_discrete_sequence=[palette["disc"][2]],
                        labels={"avg_tenders_per_lot": "Avg tenders / lot"},
                    )
                    st.plotly_chart(_style(fig), use_container_width=True)
            else:
                st.markdown("##### Avg tenders per lot by category")
                st.info("No competition intensity data in this dataset.")

        # Savings vs contract size scatter
        st.markdown("##### Savings % vs contract size")
        if all(c in awards_f.columns for c in ["estimated", "savings_pct", "proc_type"]):
            sc_df = awards_f.dropna(subset=["estimated", "savings_pct"])
            hover = [c for c in ["project_title", "buyer_country", "winner_names"] if c in sc_df.columns]
            fig = px.scatter(
                sc_df, x="estimated", y="savings_pct",
                color="proc_type" if "proc_type" in sc_df.columns else None,
                color_discrete_sequence=palette["disc"],
                hover_data=hover or None,
                log_x=True, opacity=0.65,
                labels={"estimated": "Estimated value (€, log)", "savings_pct": "Savings (%)"},
                trendline="lowess",
            )
            st.plotly_chart(_style(fig), use_container_width=True)

        # Awards by country (bar)
        st.markdown("##### Total awarded by country")
        if "buyer_country" in awards_f.columns and "awarded_eur" in awards_f.columns:
            ctry_awd = (
                awards_f.groupby("buyer_country")["awarded_eur"]
                .sum().reset_index(name="Awarded €")
                .sort_values("Awarded €", ascending=False).head(12)
            )
            fig = px.bar(
                ctry_awd, x="Awarded €", y="buyer_country", orientation="h",
                color="Awarded €", color_continuous_scale=palette["cont"],
                text="Awarded €",
            )
            fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
            fig.update_layout(yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(_style(fig), use_container_width=True)

        # SME pie
        if "sme_winner" in awards_f.columns:
            sme_col1, _ = st.columns([1, 2])
            with sme_col1:
                st.markdown("##### SME vs large company wins")
                sme_counts = awards_f["sme_winner"].map({1: "SME", 0: "Large"}).value_counts().reset_index()
                sme_counts.columns = ["Type", "Count"]
                fig = px.pie(sme_counts, names="Type", values="Count", hole=0.5,
                             color_discrete_sequence=[palette["disc"][1], palette["disc"][0]])
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(showlegend=False)
                st.plotly_chart(_style(fig), use_container_width=True)

        # Table & export
        st.markdown("##### Award details")
        show_cols = [c for c in ["project_title", "buyer_country", "proc_type",
                                  "cpv_division_name", "estimated", "awarded_eur",
                                  "savings_pct", "avg_tenders_per_lot",
                                  "winner_names", "sme_winner"] if c in awards_f.columns]
        disp = awards_f[show_cols].sort_values("awarded_eur", ascending=False).head(rows_show).copy() if "awarded_eur" in awards_f.columns else awards_f[show_cols].head(rows_show).copy()
        for col in ["estimated", "awarded_eur"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(_fmt_eur)
        if "savings_pct" in disp.columns:
            disp["savings_pct"] = disp["savings_pct"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
            )
        st.dataframe(disp, use_container_width=True, hide_index=True)

        ex1, ex2 = st.columns(2)
        ex1.download_button("⬇ CSV",   _to_csv(awards_f),   file_name="awards.csv",   mime="text/csv")
        try:
            ex2.download_button("⬇ Excel", _to_excel(awards_f), file_name="awards.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown(
        '<div class="ms-sh"><span class="ms-sh-title">ML Prediction Playground</span>'
        '<span class="ms-sh-pill">3 models</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Enter contract parameters below and run any of the three trained models. "
        "Models are loaded from `models/saved/`."
    )

    # ── Input form ────────────────────────────────────────────────────────
    with st.form("pred_form"):
        st.markdown("**Contract parameters**")
        f1, f2 = st.columns(2)
        estimated_val = f1.number_input(
            "Estimated contract value (€)", min_value=1_000, max_value=500_000_000,
            value=500_000, step=10_000,
        )
        num_lots_val = f2.number_input("Number of lots", min_value=1, max_value=50, value=2)

        f3, f4, f5 = st.columns(3)
        known_countries = sorted(set(
            (list(opp["buyer_country"].dropna().unique()) if "buyer_country" in opp.columns else []) +
            (list(awards["buyer_country"].dropna().unique()) if "buyer_country" in awards.columns else [])
        )) or ["Germany", "France", "Spain"]
        buyer_country_val = f3.selectbox("Buyer country", known_countries)
        proc_type_val = f4.selectbox("Procurement type", ["Services", "Supplies", "Works"])

        cpv_options = sorted(set(
            (list(opp["cpv_division_name"].dropna().unique()) if "cpv_division_name" in opp.columns else []) +
            (list(awards["cpv_division_name"].dropna().unique()) if "cpv_division_name" in awards.columns else [])
        )) or list(CPV_DIVISIONS.values())
        cpv_name_val = f5.selectbox("CPV category", cpv_options)

        f6, f7 = st.columns(2)
        import datetime
        pub_date_val = f6.date_input("Publication date", value=datetime.date.today())
        is_framework_val = f7.checkbox("Framework agreement", value=False)

        submitted = st.form_submit_button("Run predictions")

    if submitted and JOBLIB_OK:
        # derive features
        cpv_div_code = ""
        for code, name in CPV_DIVISIONS.items():
            if name == cpv_name_val:
                cpv_div_code = code
                break

        log_est_val  = float(np.log1p(estimated_val))
        cpv_div_int  = int(cpv_div_code) if cpv_div_code.isdigit() else 0
        pub_month    = pub_date_val.month
        pub_weekday  = pub_date_val.weekday()
        is_fw_int    = int(is_framework_val)

        base_row = pd.DataFrame([{
            "estimated":         estimated_val,
            "log_estimated":     log_est_val,
            "cpv_div_int":       cpv_div_int,
            "pub_month":         pub_month,
            "pub_weekday":       pub_weekday,
            "num_lots":          float(num_lots_val),
            "is_framework":      is_fw_int,
            "buyer_country":     buyer_country_val,
            "proc_type":         proc_type_val,
            "cpv_division_name": cpv_name_val,
            "cpv_division":      cpv_div_code,
            "avg_tenders_per_lot": 5.0,
            "num_awarded_lots":    float(num_lots_val),
            "sme_winner_int":      0,
        }])

        MODEL_FILES = {
            "Win Probability":      "win_probability.joblib",
            "Competition Intensity": "competition_intensity.joblib",
            "Bid Estimation":       "bid_estimation.joblib",
        }
        DEMO_MODEL_FILES = {
            "Win Probability":      "demo_win_probability.joblib",
            "Competition Intensity": "demo_competition_intensity.joblib",
            "Bid Estimation":       "demo_bid_estimation.joblib",
        }

        def _try_load(name: str):
            p = MODEL_DIR / name
            if p.exists():
                return joblib.load(p)
            return None

        results_cols = st.columns(3)
        model_labels = list(MODEL_FILES.keys())

        for idx, label in enumerate(model_labels):
            obj = _try_load(MODEL_FILES[label]) or _try_load(DEMO_MODEL_FILES[label])
            with results_cols[idx]:
                st.markdown(f"**{label}**")
                if obj is None:
                    st.warning("Model not found")
                    continue
                try:
                    model   = obj["model"]           if isinstance(obj, dict) else obj
                    preproc = obj.get("preprocessor") if isinstance(obj, dict) else None
                    num_feats = obj.get("num_feats")  if isinstance(obj, dict) else None
                    cat_feats = obj.get("cat_feats")  if isinstance(obj, dict) else None

                    X = base_row.copy()
                    X_t = None

                    if preproc is not None:
                        # Separate preprocessor — transform then predict
                        needed: list[str] = []
                        try:
                            for t in preproc.transformers_:
                                needed += list(t[2])
                        except Exception:
                            if num_feats:
                                needed = num_feats + (cat_feats or [])
                        avail = [c for c in needed if c in X.columns]
                        X_t = preproc.transform(X[avail] if avail else X)
                        pred = model.predict(X_t)
                    else:
                        # Model is a full Pipeline (preproc baked in) — feed only its columns
                        pipe_cols = (num_feats or []) + (cat_feats or [])
                        if pipe_cols:
                            avail = [c for c in pipe_cols if c in X.columns]
                            X_in = X[avail]
                        else:
                            X_in = X
                        pred = model.predict(X_in)

                    if label == "Win Probability":
                        if hasattr(model, "predict_proba"):
                            X_for_proba = X_t if X_t is not None else X_in
                            proba = model.predict_proba(X_for_proba)
                            p_sme = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
                        else:
                            p_sme = float(pred[0])
                        st.metric("P(SME wins)", f"{p_sme:.1%}")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=round(p_sme * 100, 1),
                            domain={"x": [0, 1], "y": [0, 1]},
                            number={"suffix": "%", "font": {"color": "#fff"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#aaa"},
                                "bar":  {"color": palette["disc"][1]},
                                "bgcolor": "rgba(0,0,0,0)",
                                "bordercolor": "rgba(255,255,255,0.1)",
                                "steps": [
                                    {"range": [0,  40],  "color": "rgba(239,71,111,0.25)"},
                                    {"range": [40, 60],  "color": "rgba(255,209,102,0.25)"},
                                    {"range": [60, 100], "color": "rgba(6,214,160,0.25)"},
                                ],
                                "threshold": {"line": {"color": "#FFD166", "width": 2}, "value": 50},
                            },
                            title={"text": "SME win probability", "font": {"color": "#aaa"}},
                        ))
                        fig.update_layout(
                            height=220, paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#fff"), margin=dict(t=30, b=0, l=20, r=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif label == "Competition Intensity":
                        comp_val = float(pred[0])
                        st.metric("Avg tenders / lot", f"{comp_val:.1f}")
                        level = "Low" if comp_val < 3 else ("Medium" if comp_val < 6 else "High")
                        color = {"Low": "green", "Medium": "orange", "High": "red"}[level]
                        st.markdown(
                            f"<span style='color:{color};font-weight:700'>Competition: {level}</span>",
                            unsafe_allow_html=True,
                        )

                    elif label == "Bid Estimation":
                        raw = float(pred[0])
                        bid_eur = float(np.expm1(raw)) if raw < 30 else raw
                        savings_est = (estimated_val - bid_eur) / estimated_val * 100
                        st.metric("Predicted awarded (€)", _fmt_eur(bid_eur))
                        st.metric("Implied savings", f"{savings_est:.1f}%")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # Feature importances
        st.markdown("---")
        st.markdown("##### Feature importances")
        feat_cols = st.columns(3)
        for idx, label in enumerate(model_labels):
            stem_real = MODEL_FILES[label].replace(".joblib", "")
            stem_demo = DEMO_MODEL_FILES[label].replace(".joblib", "")
            feat_path = MODEL_DIR / f"{stem_real}_features.json"
            if not feat_path.exists():
                feat_path = MODEL_DIR / f"{stem_demo}_features.json"
            with feat_cols[idx]:
                st.markdown(f"**{label}**")
                if feat_path.exists():
                    with open(feat_path) as f:
                        fi = json.load(f)
                    fi_df = (
                        pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                        .sort_values("Importance", ascending=False).head(10)
                    )
                    fig = px.bar(
                        fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=palette["cont"],
                    )
                    fig.update_layout(coloraxis_showscale=False, height=300)
                    st.plotly_chart(_style(fig), use_container_width=True)
                else:
                    st.info("No feature importance file found.")

    elif submitted and not JOBLIB_OK:
        st.error("joblib not installed. Run `pip install joblib` to enable predictions.")

    elif not submitted:
        st.info("Fill in the contract parameters above and click **Run predictions**.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHATBOT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    from app.chatbot import answer_question

    st.markdown(
        '<div class="ms-sh"><span class="ms-sh-title">Procurement AI Chatbot</span>'
        '<span class="ms-sh-pill">Powered by Claude · text-to-pandas</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Ask questions in plain English. The assistant writes pandas code, **runs it** "
        "against the Gold tables, and answers using the real computed result. "
        "Sidebar filters apply to the data it queries."
    )

    # Engine queries the *filtered* data, keyed by the names it expects.
    ENGINE_DFS = {
        "opportunities":  opp_f,
        "awards":         awards_f,
        "market_summary": market,
        "cpv_analysis":   cpv_df,
    }

    SAMPLE_QUESTIONS = [
        "What are the top 5 countries by number of contract awards?",
        "Which CPV categories have the highest average savings %?",
        "What % of awards went to SMEs?",
        "Show the most competitive procurement categories",
        "What is the total estimated value of open tenders?",
        "Compare savings % across Services, Supplies and Works",
        "Which buyer countries publish the largest tenders on average?",
        "What are the top 10 CPV categories by total awarded value?",
    ]

    st.markdown("**Quick questions:**")
    sq1, sq2 = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with (sq1 if i % 2 == 0 else sq2):
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["chat_prefill"] = q

    st.markdown("---")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    def _render_result(res):
        """Render the engine's executed result + the code it ran."""
        if res.result is not None:
            if isinstance(res.result, pd.DataFrame) and not res.result.empty:
                st.dataframe(res.result.head(50), use_container_width=True)
            elif isinstance(res.result, pd.Series) and len(res.result):
                st.dataframe(res.result.head(50).reset_index(), use_container_width=True)
            elif not isinstance(res.result, (pd.DataFrame, pd.Series)):
                st.info(f"**Result:** {res.result}")
        if res.code:
            with st.expander("Show the pandas code that was run"):
                st.code(res.code, language="python")

    # Replay prior turns (and re-render stored results, if any).
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("res") is not None:
                _render_result(msg["res"])

    user_input = st.chat_input("Ask about procurement data…")
    if "chat_prefill" in st.session_state and st.session_state["chat_prefill"]:
        user_input = st.session_state.pop("chat_prefill")

    if user_input:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Writing and running pandas…"):
                res = answer_question(user_input, ENGINE_DFS, history=history)
            st.markdown(res.answer)
            _render_result(res)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": res.answer, "res": res}
        )

    if st.session_state.chat_messages:
        if st.button("Clear chat history"):
            st.session_state.chat_messages = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PIPELINE / ETL HEALTH
# ════════════════════════════════════════════════════════════════════════════
with tab_etl:
    st.markdown(
        '<div class="ms-sh"><span class="ms-sh-title">Pipeline & ETL Health</span>'
        '<span class="ms-sh-pill">Medallion architecture</span></div>',
        unsafe_allow_html=True,
    )

    BASE = Path(__file__).parent.parent / "data"

    def _parquet_info(folder: Path) -> pd.DataFrame:
        if not folder.exists():
            return pd.DataFrame()
        rows = []
        for p in sorted(folder.glob("*.parquet")):
            try:
                n = len(pd.read_parquet(p))
            except Exception:
                n = None
            rows.append({
                "File":      p.name,
                "Rows":      n,
                "Size (KB)": round(p.stat().st_size / 1024, 1),
                "Modified":  pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            })
        return pd.DataFrame(rows)

    def _empty_layer(name: str) -> None:
        st.markdown(f"⚪ **{name} layer** — directory is empty, pipeline not yet run for this layer.")

    # ── Gold + Silver ─────────────────────────────────────────────────────
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("##### Gold files")
        gdf = _parquet_info(BASE / "gold")
        if gdf.empty:
            st.info("No Gold parquet files found.")
        else:
            st.dataframe(gdf, use_container_width=True, hide_index=True)

    with e2:
        st.markdown("##### Silver / Bronze / Raw")
        sdf = _parquet_info(BASE / "silver")
        if sdf.empty:
            _empty_layer("Silver")
        else:
            st.markdown("**Silver:**")
            st.dataframe(sdf, use_container_width=True, hide_index=True)

        bdf = _parquet_info(BASE / "bronze")
        if bdf.empty:
            _empty_layer("Bronze")
        else:
            st.markdown("**Bronze:**")
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        raw_dir = BASE / "raw"
        raw_files = [f for f in raw_dir.iterdir() if not f.name.startswith(".")] \
            if raw_dir.exists() else []
        if not raw_files:
            _empty_layer("Raw")
        else:
            st.markdown(f"**Raw:** {len(raw_files)} files in `data/raw/`")

    st.markdown("---")

    # ── Saved models ──────────────────────────────────────────────────────
    st.markdown("##### Saved models")
    if MODEL_DIR.exists():
        model_files = sorted(MODEL_DIR.iterdir())
        if not model_files:
            st.info("No model files found. Run `python -m models.train_models` to train.")
        else:
            for p in model_files:
                with st.expander(p.name):
                    st.write(f"Size: {p.stat().st_size:,} bytes")
                    with open(p, "rb") as fh:
                        raw_bytes = fh.read()
                    st.download_button(
                        f"⬇ Download {p.name}", raw_bytes,
                        file_name=p.name, key=f"dl_{p.name}",
                    )
    else:
        st.info("Model directory not found.")

    st.markdown("---")
    st.markdown("##### Pipeline commands")
    st.code(
        "# Full pipeline (download + bronze + silver + gold)\n"
        "python -m pipeline.run_pipeline\n\n"
        "# Skip download if data/raw/ already populated\n"
        "python -m pipeline.run_pipeline --skip-ingest\n\n"
        "# Run a single step\n"
        "python -m pipeline.run_pipeline --only silver\n\n"
        "# Train ML models (requires gold_awards.parquet)\n"
        "python -m models.train_models",
        language="bash",
    )
