"""
app/streamlit_designed.py
────────────────────────
Enhanced Streamlit scaffold with small UX improvements:
- export filtered tables as CSV
- list available saved models and allow download

This file is additive — the original `streamlit_app.py` remains unchanged.
Run with:
    streamlit run app/streamlit_designed.py
"""
from pathlib import Path
import io
import sys
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    import pycountry
except ModuleNotFoundError:
    pycountry = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOLD_DIR, MODEL_DIR, ANTHROPIC_API_KEY


st.set_page_config(page_title="TED Procurement — Design", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        /* full app background: layered gradient (blue -> teal -> purple -> orange) */
        .stApp, .css-1d391kg, .css-ffhzg2 {
            background: linear-gradient(135deg, #012840 0%, #004E64 35%, #5A2D81 70%, #FF6A00 100%);
            background-attachment: fixed;
            color: #ffffff;
        }
        /* sidebar with subtle tint and translucency */
        section[data-testid='stSidebar'] {
            background: linear-gradient(180deg, rgba(0,42,86,0.92) 0%, rgba(0,31,66,0.88) 100%) !important;
            color: #ffffff;
            border-right: 1px solid rgba(255,255,255,0.06);
            backdrop-filter: blur(6px);
        }
        section[data-testid='stSidebar'] .css-1d391kg, section[data-testid='stSidebar'] .css-ffhzg2 {
            background: transparent !important;
        }
        section[data-testid='stSidebar'] .stMarkdown, section[data-testid='stSidebar'] .stText, section[data-testid='stSidebar'] .stTitle, section[data-testid='stSidebar'] .stCaption {
            color: #D6E4FF;
        }
        .stSidebar .stButton>button, .stButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg,#0078D4 0%, #06D6A0 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 999px !important;
            padding: 0.75rem 1.2rem !important;
            box-shadow: 0 12px 32px rgba(0,0,0,0.35) !important;
        }
        .stSidebar .stButton>button:hover, .stButton>button:hover, .stDownloadButton>button:hover {
            filter: brightness(0.93) !important;
        }
        .stSidebar .stSlider, .stSidebar .stDateInput, .stSidebar .stNumberInput, .stSidebar .stSelectbox {
            color: #ffffff;
        }
        .dashboard-hero {display:flex; flex-wrap:wrap; gap:24px; justify-content:space-between; align-items:flex-start; margin-bottom:32px;}
        .hero-panel {background:linear-gradient(135deg,#003E72 0%, #06D6A0 40%, #5A2D81 75%); border-radius:28px; padding:32px; box-shadow:0 28px 68px rgba(0, 0, 0, 0.36); flex:1 1 540px; min-width:320px;}
        .hero-title {font-size:3.5rem; line-height:1.05; margin:0 0 12px; color:#ffffff;}
        .hero-subtitle {margin:0 0 20px; color:#D6E4FF; font-size:1.05rem; max-width:680px;}
        .hero-pill {display:inline-flex; align-items:center; gap:8px; background:rgba(0,120,212,0.18); color:#D6E4FF; padding:10px 16px; border-radius:999px; font-size:0.80rem; letter-spacing:0.08em; text-transform:uppercase;}
        .hero-columns {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:20px;}
        .hero-card {background:linear-gradient(180deg,#00365D 0%, #002F4F 100%); border:1px solid rgba(255,255,255,0.04); border-radius:24px; padding:28px; min-height:170px;}
        .hero-card .label {color:#A8D1FF; font-size:0.90rem; margin-bottom:14px;}
        .hero-card .value {font-size:2.4rem; font-weight:800; color:#ffffff; margin-bottom:10px;}
        .hero-card .detail {color:#C7D9FF; font-size:0.95rem;}
        .feature-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:20px; margin-bottom:32px;}
        .feature-card {background:linear-gradient(180deg,#003A64 0%, #002A56 100%); border:1px solid rgba(255,255,255,0.04); border-radius:24px; padding:24px;}
        .chart-panel {background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.06)); border:1px solid rgba(255,255,255,0.03); border-radius:16px; padding:12px;}
        .feature-card h3 {margin:0 0 12px; color:#ffffff; font-size:1.05rem;}
        .feature-card p {margin:0; color:#D6E4FF; line-height:1.6;}
        .chart-panel {background:#002F56; border:1px solid rgba(255,255,255,0.08); border-radius:24px; padding:18px;}
        .section-heading {margin-top:0; color:#ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Visual theme palettes (can be selected in the sidebar)
PALETTE_MAP = {
    "Microsoft": {
        "continuous": ["#00338D", "#0078D4", "#00BCF2", "#06D6A0", "#FF6A00"],
        "discrete": ["#0078D4", "#06D6A0", "#FF6A00", "#9B5DE5", "#FFD166"],
    },
    "Warm": {
        "continuous": ["#7F2704", "#FF6A00", "#FF9E3B", "#FFD166"],
        "discrete": ["#FF6A00", "#FF9E3B", "#FFD166", "#FFB4A2", "#E85D04"],
    },
    "Teal-Purple": {
        "continuous": ["#005F73", "#0A9396", "#94D2BD", "#5A2D81"],
        "discrete": ["#06D6A0", "#118AB2", "#5A2D81", "#9B5DE5", "#B0E3E6"],
    },
    "High contrast": {
        "continuous": ["#001219", "#005F73", "#FB8500", "#F28482"],
        "discrete": ["#001219", "#005F73", "#FB8500", "#FFD166", "#EF476F"],
    },
}

def build_sparkline(values, color="#00BCF2", height=80):
    fig = px.area(x=list(range(len(values))), y=values, height=height)
    fig.update_traces(fillcolor=color, line_color=color, opacity=0.85)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

@st.cache_data
def create_demo_data() -> dict[str, pd.DataFrame]:
    opp = pd.DataFrame([
        {
            "notice_id": "CN-2026-001",
            "pub_date": "2026-01-05",
            "buyer_name": "City of Amsterdam",
            "buyer_country": "Netherlands",
            "estimated": 2250000,
            "num_lots": 3,
            "notice_type": "Contract Notice",
            "project_title": "Municipal IT services",
            "proc_type": "Services",
            "cpv_division_name": "IT services",
            "submission_deadline": "2026-02-01",
            "notice_date": "2026-01-04",
        },
        {
            "notice_id": "CN-2026-002",
            "pub_date": "2026-01-10",
            "buyer_name": "Ministry of Transport",
            "buyer_country": "Germany",
            "estimated": 5700000,
            "num_lots": 5,
            "notice_type": "Contract Notice",
            "project_title": "Road maintenance contract",
            "proc_type": "Works",
            "cpv_division_name": "Construction works",
            "submission_deadline": "2026-02-20",
            "notice_date": "2026-01-09",
        },
    ])

    awards = pd.DataFrame([
        {
            "notice_id": "CAN-2026-051",
            "award_date": "2026-01-18",
            "buyer_name": "City of Paris",
            "buyer_country": "France",
            "awarded_eur": 3200000,
            "savings_pct": 8.5,
            "avg_tenders_per_lot": 4.2,
            "sme_winner": 1,
            "project_title": "Public lighting upgrade",
            "notice_type": "Award Notice",
            "publication_date": "2026-01-15",
            "notice_date": "2026-01-14",
        },
        {
            "notice_id": "CAN-2026-052",
            "award_date": "2026-01-22",
            "buyer_name": "Transport Authority",
            "buyer_country": "Spain",
            "awarded_eur": 1450000,
            "savings_pct": 6.1,
            "avg_tenders_per_lot": 3.5,
            "sme_winner": 0,
            "project_title": "Railway signal systems",
            "notice_type": "Award Notice",
            "publication_date": "2026-01-20",
            "notice_date": "2026-01-19",
        },
    ])

    market = pd.DataFrame([
        {"dimension": "country", "dimension_value": "France", "total_awarded": 3200000, "avg_savings_pct": 8.5},
        {"dimension": "country", "dimension_value": "Spain", "total_awarded": 1450000, "avg_savings_pct": 6.1},
    ])

    cpv = pd.DataFrame([
        {"cpv_division_name": "IT services", "avg_competition": 6.8, "sme_wins": 42},
        {"cpv_division_name": "Construction works", "avg_competition": 5.1, "sme_wins": 37},
    ])

    return {"opp": opp, "awards": awards, "market": market, "cpv": cpv}


@st.cache_data
def load_data():
    def _load(name):
        p = GOLD_DIR / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    data = {
        "opp": _load("gold_opportunities"),
        "awards": _load("gold_awards"),
        "market": _load("gold_market_summary"),
        "cpv": _load("gold_cpv_analysis"),
    }
    return data if has_gold_data(data) else create_demo_data()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buf.getvalue()


def get_unique_country_count(*dfs: pd.DataFrame) -> int:
    unique_countries = set()
    for df in dfs:
        if "buyer_country" in df.columns:
            unique_countries |= set(df["buyer_country"].dropna().unique())
    return len(unique_countries)


def create_demo_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib_models = list(MODEL_DIR.glob("*.joblib"))
    if joblib_models:
        return

    num_pipe = Pipeline([
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, ["estimated", "num_lots"]),
        ("cat", cat_pipe, ["buyer_country", "proc_type"]),
    ], remainder="drop")

    dummy_clf = Pipeline([
        ("preproc", preprocessor),
        ("model", DummyClassifier(strategy="prior")),
    ])
    dummy_reg = Pipeline([
        ("preproc", preprocessor),
        ("model", DummyRegressor(strategy="mean")),
    ])

    joblib.dump({"model": dummy_clf, "preprocessor": None}, MODEL_DIR / "demo_win_probability.joblib")
    joblib.dump({"model": dummy_reg, "preprocessor": None}, MODEL_DIR / "demo_competition_intensity.joblib")
    joblib.dump({"model": dummy_reg, "preprocessor": None}, MODEL_DIR / "demo_bid_estimation.joblib")

    for name in ["demo_win_probability", "demo_competition_intensity", "demo_bid_estimation"]:
        with open(MODEL_DIR / f"{name}_features.json", "w") as f:
            json.dump({"demo_feature": 1.0}, f)


def has_gold_data(data: dict[str, pd.DataFrame]) -> bool:
    return any(len(df) > 0 for df in data.values())


data = load_data()
opp = data["opp"]
awards = data["awards"]
market = data["market"]
cpv = data["cpv"]

create_demo_models()

opp_count = len(opp)
award_count = len(awards)
country_count = get_unique_country_count(opp, awards)
total_estimated = opp["estimated"].sum() if "estimated" in opp.columns else 0
total_awarded = awards["awarded_eur"].sum() if "awarded_eur" in awards.columns else 0
sme_pct_value = (awards["sme_winner"].sum() / len(awards) * 100) if ("sme_winner" in awards.columns and len(awards) > 0) else 0

st.markdown(
    f"""
    <div class="dashboard-hero">
      <div class="hero-panel">
        <span class="hero-pill">Executive Dashboard</span>
        <h1 class="hero-title">TED Procurement Intelligence</h1>
        <p class="hero-subtitle">A high-impact analytics dashboard for European tender opportunities, awards performance, and model forecasting.</p>
        <div class="hero-columns">
          <div class="hero-card">
            <div class="label">Open opportunities</div>
            <div class="value">{opp_count:,}</div>
            <div class="detail">Live scan of current procurement notices</div>
          </div>
          <div class="hero-card">
            <div class="label">Award records</div>
            <div class="value">{award_count:,}</div>
            <div class="detail">Monitored award notices and outcomes</div>
          </div>
          <div class="hero-card">
            <div class="label">Global coverage</div>
            <div class="value">{country_count:,}</div>
            <div class="detail">Distinct buyer countries in the dataset</div>
          </div>
        </div>
      </div>
      <div style="min-width:320px; display:grid; gap:20px;">
        <div class="hero-card">
          <div class="label">Revenue impact</div>
          <div class="value">€{total_estimated/1e6:.1f}M</div>
          <div class="detail">Total estimated contract volume</div>
        </div>
        <div class="hero-card">
          <div class="label">Awarded value</div>
          <div class="value">€{total_awarded/1e6:.1f}M</div>
          <div class="detail">Sum of awarded contracts</div>
        </div>
        <div class="hero-card">
          <div class="label">SME share</div>
          <div class="value">{sme_pct_value:.1f}%</div>
          <div class="detail">SME winner percentage of awards</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not has_gold_data(data):
    st.warning(
        "No Gold data files found in `data/gold/`. Showing demo data for the dashboard UI. "
        "Run `python -m pipeline.run_pipeline` to load real TED procurement tables."
    )

st.markdown("## What you are seeing")
st.markdown(
    "This dashboard is built to show procurement intelligence from TED notices and awards in a Microsoft-style analytics layout. "
    "The hero section surfaces portfolio-level metrics, and the tabs let you explore opportunities, awards, and model artifacts."
)
st.info(
    "The top cards summarize the volume of notices, awarded value, country reach, and SME participation. "
    "The charts below show the geographic footprint, category mix, and award efficiency so you can spot high-potential markets quickly."
)


def normalize_country_name(name: str) -> str:
    if pycountry is None:
        return name

    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_3
    except Exception:
        return name


def style_plotly_figure(fig):
    fig.update_layout(
        paper_bgcolor="#001E3C",
        plot_bgcolor="#001E3C",
        font=dict(color="#ffffff"),
        legend=dict(font=dict(color="#ffffff")),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#ffffff"))
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color="#ffffff"))
    return fig


def build_top_country_bar(df, count_col="buyer_country"):
    if count_col not in df.columns:
        return None
    top = df[count_col].value_counts().head(8).reset_index()
    top.columns = [count_col, "count"]
    theme = globals().get("vis_theme", "Microsoft")
    pal = PALETTE_MAP.get(theme, PALETTE_MAP["Microsoft"]) if "PALETTE_MAP" in globals() else None
    cont_scale = pal["continuous"] if pal else ["#00338D", "#0078D4", "#00BCF2"]

    fig = px.bar(
        top,
        x="count",
        y=count_col,
        orientation="h",
        color="count",
        color_continuous_scale=cont_scale,
        labels={count_col: "Country", "count": "Notices"},
        text="count",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside", marker=dict(line=dict(color="#ffffff", width=1)))
    return style_plotly_figure(fig)


def build_proc_type_pie(df):
    if "proc_type" not in df.columns or df["proc_type"].dropna().empty:
        return None
    proc = df["proc_type"].value_counts().reset_index()
    proc.columns = ["proc_type", "count"]
    theme = globals().get("vis_theme", "Microsoft")
    pal = PALETTE_MAP.get(theme, PALETTE_MAP["Microsoft"]) if "PALETTE_MAP" in globals() else None
    disc = pal["discrete"] if pal else ["#0078D4", "#06D6A0", "#FF6A00", "#9B5DE5", "#FFD166"]

    fig = px.pie(
        proc,
        names="proc_type",
        values="count",
        hole=0.45,
        color_discrete_sequence=disc,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=12)
    fig.update_layout(legend_title_text="Procedure type")
    return style_plotly_figure(fig)


def build_award_savings_bar(df):
    if "savings_pct" not in df.columns:
        return None
    savings = df.groupby("buyer_country")["savings_pct"].mean().nlargest(8).reset_index()
    theme = globals().get("vis_theme", "Microsoft")
    pal = PALETTE_MAP.get(theme, PALETTE_MAP["Microsoft"]) if "PALETTE_MAP" in globals() else None
    cont_scale = pal["continuous"] if pal else ["#06D6A0", "#00BCF2", "#0078D4", "#5A2D81"]

    fig = px.bar(
        savings,
        x="savings_pct",
        y="buyer_country",
        orientation="h",
        color="savings_pct",
        color_continuous_scale=cont_scale,
        labels={"buyer_country": "Country", "savings_pct": "Avg savings %"},
        text="savings_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker=dict(line=dict(color="#ffffff", width=1)))
    return style_plotly_figure(fig)


def build_awarded_value_bar(df):
    if "buyer_country" not in df.columns or "awarded_eur" not in df.columns:
        return None
    awarded = df.groupby("buyer_country")["awarded_eur"].sum().nlargest(8).reset_index()
    theme = globals().get("vis_theme", "Microsoft")
    pal = PALETTE_MAP.get(theme, PALETTE_MAP["Microsoft"]) if "PALETTE_MAP" in globals() else None
    cont_scale = pal["continuous"] if pal else ["#00BCF2", "#0078D4", "#00338D"]
    fig = px.bar(
        awarded,
        x="awarded_eur",
        y="buyer_country",
        orientation="h",
        color="awarded_eur",
        color_continuous_scale=cont_scale,
        labels={"buyer_country": "Country", "awarded_eur": "Awarded €"},
        text="awarded_eur",
    )
    fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside", marker=dict(line=dict(color="#ffffff", width=1)))
    return style_plotly_figure(fig)


with st.sidebar:
    st.image("https://ted.europa.eu/o/ted2-theme/images/eu/condensed/logo-eu--en.svg", width=72)
    st.title("TED Procurement\nDesign UI")
    vis_theme = st.selectbox("Visual theme", list(PALETTE_MAP.keys()), index=0)
    show_sparklines = st.checkbox("Show hero sparklines", value=True)
    rows = st.number_input("Preview rows", min_value=5, max_value=200, value=50)
    st.markdown("---")
    st.subheader("Date filter")
    start_date = st.date_input("Start date", value=None)
    end_date = st.date_input("End date", value=None)
    st.markdown("---")
    st.caption("Small UX: export filtered tables, inspect saved models")


tab1, tab2, tab3 = st.tabs(["Opportunities", "Awards", "Models"])

# Global KPI summary before the tabs
st.markdown("## Global Procurement KPIs")
summary_cols = st.columns(6)
summary_cols[0].metric("Open opportunities", f"{len(opp):,}")
summary_cols[1].metric("Award records", f"{len(awards):,}")
summary_cols[2].metric("Countries", f"{get_unique_country_count(opp, awards):,}")
summary_cols[3].metric(
    "Total estimated",
    f"€{opp['estimated'].sum()/1e6:.1f}M" if 'estimated' in opp.columns else "—"
)
summary_cols[4].metric(
    "Total awarded",
    f"€{awards['awarded_eur'].sum()/1e6:.1f}M" if 'awarded_eur' in awards.columns else "—"
)
sme_pct = awards['sme_winner'].sum() / len(awards) * 100 if ('sme_winner' in awards.columns and len(awards) > 0) else 0
summary_cols[5].metric("SME awards", f"{sme_pct:.1f}%")

# Small hero sparklines (monthly mini-trends)
if globals().get("show_sparklines", True):
    try:
        def make_month_series(df, date_cols, value_col=None, periods=8, agg="count"):
            for col in date_cols:
                if col in df.columns:
                    try:
                        s = pd.to_datetime(df[col], errors="coerce").dropna()
                        if s.empty:
                            break
                        idx = s.dt.to_period("M")
                        if agg == "count":
                            counts = s.groupby(idx).size()
                        else:
                            counts = df.assign(_dt=s).groupby(idx)[value_col].sum()
                        last = counts.tail(periods)
                        vals = list(last.values)
                        if len(vals) < periods:
                            vals = [0] * (periods - len(vals)) + vals
                        return vals
                    except Exception:
                        break
            base = (np.linspace(5, 20, periods) + np.random.randint(0, 3, periods)).round(0).tolist()
            return base

        theme = globals().get("vis_theme", "Microsoft")
        pal = PALETTE_MAP.get(theme, PALETTE_MAP["Microsoft"]) if "PALETTE_MAP" in globals() else None
        disc = pal["discrete"] if pal else ["#0078D4", "#06D6A0", "#FF6A00"]

        opp_series = make_month_series(opp, ["pub_date", "publication_date", "notice_date"]) if not opp.empty else make_month_series(pd.DataFrame(), [])
        award_series = make_month_series(awards, ["award_date", "publication_date"]) if not awards.empty else make_month_series(pd.DataFrame(), [])
        if ("sme_winner" in awards.columns) and (not awards.empty):
            # monthly SME win rate
            try:
                d = awards.copy()
                d["_dt"] = pd.to_datetime(d[[c for c in ("award_date", "publication_date", "notice_date") if c in d.columns][0]], errors="coerce").dt.to_period("M")
                sme_month = d.dropna(subset=["_dt"]).groupby("_dt")["sme_winner"].mean().tail(8) * 100
                sme_vals = list(sme_month.values)
                if len(sme_vals) < 8:
                    sme_vals = [0] * (8 - len(sme_vals)) + sme_vals
            except Exception:
                sme_vals = make_month_series(pd.DataFrame(), [])
        else:
            sme_vals = make_month_series(pd.DataFrame(), [])

        if globals().get("show_sparklines", True):
            spark_cols = st.columns(3)
            spark_cols[0].plotly_chart(build_sparkline(opp_series, color=disc[0]), use_container_width=True)
            spark_cols[0].caption("Opportunities — recent months")
            spark_cols[1].plotly_chart(build_sparkline(award_series, color=disc[1] if len(disc) > 1 else disc[0]), use_container_width=True)
            spark_cols[1].caption("Awards — recent months")
            spark_cols[2].plotly_chart(build_sparkline(sme_vals, color=disc[2] if len(disc) > 2 else disc[0]), use_container_width=True)
            spark_cols[2].caption("SME win % — recent months")
    except Exception:
        pass

# ---------- ETL health panel ----------
def get_parquet_info(folder: Path) -> list[dict]:
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            rows = len(df)
        except Exception:
            rows = None
        out.append({
            "file": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "rows": rows,
            "modified": p.stat().st_mtime,
        })
    return out


st.markdown("## ETL health")
cols = st.columns([1, 1])
with cols[0]:
    st.subheader("Gold files")
    gold_info = get_parquet_info(GOLD_DIR)
    if not gold_info:
        st.info("No Gold parquet files found in data/gold/")
    else:
        dfg = pd.DataFrame(gold_info)[["file", "rows", "size_bytes"]]
        dfg["size_bytes"] = dfg["size_bytes"].map(lambda x: f"{x:,}")
        st.dataframe(dfg, use_container_width=True)

with cols[1]:
    st.subheader("Silver / Bronze / Raw")
    silver_info = get_parquet_info(Path("data/silver"))
    bronze_info = get_parquet_info(Path("data/bronze"))
    raw_files = list(Path("data/raw").glob("*"))
    splot = pd.DataFrame(silver_info)[["file", "rows"]] if silver_info else pd.DataFrame()
    bplot = pd.DataFrame(bronze_info)[["file", "rows"]] if bronze_info else pd.DataFrame()
    st.markdown("**Silver:**")
    if splot.empty:
        st.write("No Silver parquet files found in data/silver/")
    else:
        st.dataframe(splot, use_container_width=True)
    st.markdown("**Bronze:**")
    if bplot.empty:
        st.write("No Bronze parquet files found in data/bronze/")
    else:
        st.dataframe(bplot, use_container_width=True)
    st.markdown("**Raw:**")
    st.write(f"{len(raw_files)} files in data/raw/")


with tab1:
    st.header("Opportunities — Preview & Export")
    if opp.empty:
        st.warning("No opportunity data found in Gold folder.")
    else:
        df_display = opp.copy()
        if start_date and end_date:
            for col in ("publication_date", "submission_deadline", "notice_date"):
                if col in df_display.columns:
                    try:
                        df_display[col] = pd.to_datetime(df_display[col], errors="coerce")
                        df_display = df_display[(df_display[col] >= pd.to_datetime(start_date)) & (df_display[col] <= pd.to_datetime(end_date))]
                        break
                    except Exception:
                        continue

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opportunities", f"{len(df_display):,}")
        c2.metric("Countries", f"{df_display['buyer_country'].nunique() if 'buyer_country' in df_display.columns else 0}")
        c3.metric(
            "Total estimated",
            f"€{df_display['estimated'].sum()/1e6:.1f}M" if 'estimated' in df_display.columns else "—"
        )
        c4.metric(
            "Avg lots",
            f"{df_display['num_lots'].mean():.1f}" if 'num_lots' in df_display.columns else "—"
        )

        st.markdown("### Opportunity insights")
        chart_cols = st.columns([2, 1])
        fig_country = build_top_country_bar(df_display)
        if fig_country is not None:
            chart_cols[0].plotly_chart(fig_country, use_container_width=True)
        else:
            chart_cols[0].info("No country-level opportunity chart available.")

        fig_proc = build_proc_type_pie(df_display)
        if fig_proc is not None:
            chart_cols[1].plotly_chart(fig_proc, use_container_width=True)
        else:
            chart_cols[1].info("No procurement type breakdown available.")

        st.markdown("---")
        st.dataframe(df_display.head(rows), use_container_width=True)
        csv = df_to_csv_bytes(df_display)
        st.download_button("Download opportunities (CSV)", csv, file_name="opportunities.csv")
        try:
            xlsx = df_to_excel_bytes(df_display)
            st.download_button("Download opportunities (Excel)", xlsx, file_name="opportunities.xlsx")
        except Exception:
            pass
        st.markdown("---")
        st.subheader("Geography — Buyers by country")
        # Map controls
        map_metric = st.selectbox("Metric", ["count", "total_estimated"], index=0,
                                  help="Count = number of notices; total_estimated = sum of `estimated`")
        map_dataset = st.selectbox("Dataset", ["Opportunities", "Awards"], index=0)

        if map_dataset == "Opportunities":
            df_map = opp.copy()
            value_col = "estimated"
        else:
            df_map = awards.copy()
            value_col = "awarded_eur" if "awarded_eur" in awards.columns else "estimated"

        if df_map.empty or "buyer_country" not in df_map.columns:
            st.info("No country data available for the selected dataset.")
        else:
            grp = df_map.dropna(subset=["buyer_country"])
            if map_metric == "count":
                agg = grp.groupby("buyer_country")[value_col].count().reset_index(name="value")
            else:
                # sum of monetary column (may be NaN)
                agg = grp.groupby("buyer_country").agg(value=(value_col, "sum")).reset_index()

            if agg["value"].sum() == 0 or agg.empty:
                st.info("No numeric values to show on the map.")
            else:
                agg = agg.copy()
                agg["iso3"] = agg["buyer_country"].apply(normalize_country_name)
                num_iso = agg["iso3"].apply(lambda x: isinstance(x, str) and len(x) == 3).sum()
                theme = globals().get("vis_theme", "Microsoft")
                pal = PALETTE_MAP.get(theme, PALETTE_MAP["Microsoft"]) if "PALETTE_MAP" in globals() else None
                ms_color_scale = pal["continuous"] if pal else ["#00338D", "#0078D4", "#00BCF2", "#06D6A0", "#FF6A00", "#9B5DE5"]
                if num_iso >= max(5, len(agg) // 3):
                    fig = px.choropleth(
                        agg,
                        locations="iso3",
                        color="value",
                        hover_name="buyer_country",
                        color_continuous_scale=ms_color_scale,
                        labels={"value": map_metric},
                        locationmode="ISO-3",
                    )
                else:
                    fig = px.choropleth(
                        agg,
                        locations="buyer_country",
                        locationmode="country names",
                        color="value",
                        hover_name="buyer_country",
                        color_continuous_scale=ms_color_scale,
                        labels={"value": map_metric},
                    )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="#001E3C",
                    plot_bgcolor="#001E3C",
                    geo_bgcolor="#001E3C",
                    coloraxis_colorbar=dict(
                        title=map_metric,
                        tickfont=dict(color="#ffffff"),
                        titlefont=dict(color="#ffffff"),
                        bgcolor="#001E3C",
                        outlinecolor="#ffffff",
                    ),
                    font=dict(color="#ffffff"),
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Awards — Preview & Export")
    if awards.empty:
        st.warning("No awards data found in Gold folder.")
    else:
        df_display = awards.copy()
        if start_date and end_date:
            for col in ("award_date", "notice_date", "publication_date"):
                if col in df_display.columns:
                    try:
                        df_display[col] = pd.to_datetime(df_display[col], errors="coerce")
                        df_display = df_display[(df_display[col] >= pd.to_datetime(start_date)) & (df_display[col] <= pd.to_datetime(end_date))]
                        break
                    except Exception:
                        continue

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Awards", f"{len(df_display):,}")
        c2.metric("Countries", f"{df_display['buyer_country'].nunique() if 'buyer_country' in df_display.columns else 0}")
        c3.metric(
            "Total awarded",
            f"€{df_display['awarded_eur'].sum()/1e6:.1f}M" if 'awarded_eur' in df_display.columns else "—"
        )
        c4.metric(
            "Avg savings",
            f"{df_display['savings_pct'].mean():.1f}%" if 'savings_pct' in df_display.columns else "—"
        )
        c5.metric(
            "SME wins",
            f"{(df_display['sme_winner'].sum() / len(df_display) * 100):.1f}%" if 'sme_winner' in df_display.columns else "—"
        )

        st.markdown("### Award insights")
        award_charts = st.columns([2, 1])
        fig_awarded_value = build_awarded_value_bar(df_display)
        if fig_awarded_value is not None:
            award_charts[0].plotly_chart(fig_awarded_value, use_container_width=True)
        else:
            award_charts[0].info("No awarded value breakdown available.")

        fig_savings = build_award_savings_bar(df_display)
        if fig_savings is not None:
            award_charts[1].plotly_chart(fig_savings, use_container_width=True)
        else:
            award_charts[1].info("No savings percentage chart available.")

        st.markdown("---")
        st.dataframe(df_display.head(rows), use_container_width=True)
        csv = df_to_csv_bytes(df_display)
        st.download_button("Download awards (CSV)", csv, file_name="awards.csv")
        try:
            xlsx = df_to_excel_bytes(df_display)
            st.download_button("Download awards (Excel)", xlsx, file_name="awards.xlsx")
        except Exception:
            pass

with tab3:
    st.header("Saved models")
    st.markdown("This lists files under the configured `MODEL_DIR`. You can download model artifacts here.")
    md = MODEL_DIR
    if not md.exists():
        st.info("Model directory not found or empty.")
    else:
        files = sorted([p for p in md.iterdir() if p.is_file()])
        if not files:
            st.info("No saved models found in MODEL_DIR.")
        else:
            for p in files:
                with st.expander(p.name):
                    st.write(f"Size: {p.stat().st_size:,} bytes")
                    with open(p, "rb") as fh:
                        data_bytes = fh.read()
                    st.download_button(f"Download {p.name}", data_bytes, file_name=p.name)

    # ---------- Model prediction playground ----------
    st.markdown("---")
    st.subheader("Model prediction playground")
    model_files = sorted([p for p in md.iterdir() if p.suffix == ".joblib"])
    if not model_files:
        st.info("No .joblib models available to run predictions.")
    else:
        sel = st.selectbox("Choose model", [p.name for p in model_files])
        model_path = md / sel
        if st.button("Load model"):
            try:
                mdl_obj = joblib.load(model_path)
                st.success(f"Loaded {sel}")
                st.session_state["loaded_model"] = {"path": str(model_path), "obj": mdl_obj}
            except Exception as e:
                st.error(f"Failed to load model: {e}")

        if "loaded_model" in st.session_state and st.session_state["loaded_model"]["path"] == str(md / sel):
            mdl_obj = st.session_state["loaded_model"]["obj"]
            if isinstance(mdl_obj, dict):
                st.write("Model keys:", list(mdl_obj.keys()))
            else:
                st.write("Model type:", type(mdl_obj))

            # show feature importance file if present
            stem = Path(sel).stem
            feat_file = md / f"{stem}_features.json"
            if feat_file.exists():
                try:
                    with open(feat_file, "r") as f:
                        feats = json.load(f)
                    st.write("Top features:")
                    st.dataframe(pd.DataFrame(list(feats.items()), columns=["feature", "importance"]).head(20))
                except Exception:
                    pass

            # Sample selection from awards (if available)
            if not awards.empty:
                samples = awards.reset_index().head(200)
                labels = samples.apply(lambda r: f"{r['index']} — {str(r.get('project_title',''))[:80]}", axis=1).tolist()
                choice = st.selectbox("Pick a sample row (awards)", labels)
                idx = int(choice.split(" — ")[0])
                row = awards.loc[idx:idx]

                if st.button("Run prediction on sample"):
                    try:
                        model = mdl_obj
                        preproc = None
                        if isinstance(mdl_obj, dict):
                            model = mdl_obj.get("model", mdl_obj)
                            preproc = mdl_obj.get("preprocessor")

                        X = row.copy()
                        if preproc is not None:
                            cols = []
                            try:
                                for t in preproc.transformers_:
                                    cols += list(t[2])
                            except Exception:
                                cols = []
                            if cols:
                                X = row[cols]
                            X_t = preproc.transform(X)
                            pred = model.predict(X_t)
                        else:
                            # If model is a pipeline, it will handle DataFrame input
                            pred = model.predict(X)

                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X if preproc is None else X_t)[:, 1]
                            st.success(f"Probability (positive): {proba[0]:.3f}")
                        else:
                            if "bid_estimation" in sel or "bid" in sel:
                                val = np.expm1(pred)[0]
                                st.success(f"Predicted awarded EUR: €{val:,.0f}")
                            else:
                                st.success(f"Prediction: {pred[0]}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.caption("Next steps: wire model prediction UI, add maps, add country selectors.")
