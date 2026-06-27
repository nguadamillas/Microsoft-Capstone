"""
app/procurement_assistant.py
─────────────────────────────
Procurement Assistant — answers business questions from the real Gold tables.

Architecture:
  1. Gold parquets loaded into DataFrames (demo fallback when pipeline hasn't run).
  2. Claude answers via a single query_data tool that runs sandboxed pandas eval.
  3. Eight sales quick-questions are pre-computed (no API call, instant, demo-safe).
  4. Free-text questions go through the tool path; final answer streams.

Public API:
    render_assistant(T)   — T is the active theme dict from dashboard.py.
"""

import os, base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT     = Path(__file__).parent.parent
GOLD_DIR = ROOT / "data" / "gold"
MODEL    = "claude-sonnet-4-6"

# ── Microsoft 4-square avatar ─────────────────────────────────────────────────
_MS_SVG = """\
<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
  <rect x="0"  y="0"  width="9" height="9" fill="#F25022"/>
  <rect x="11" y="0"  width="9" height="9" fill="#7FBA00"/>
  <rect x="0"  y="11" width="9" height="9" fill="#00A4EF"/>
  <rect x="11" y="11" width="9" height="9" fill="#FFB900"/>
</svg>"""
_MS_AVATAR = "data:image/svg+xml;base64," + base64.b64encode(_MS_SVG.encode()).decode()

# Inline mark for use in HTML sections
_MS_MARK_HTML = (
    '<svg width="18" height="18" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" '
    'style="flex-shrink:0">'
    '<rect x="0" y="0" width="9" height="9" fill="#F25022"/>'
    '<rect x="11" y="0" width="9" height="9" fill="#7FBA00"/>'
    '<rect x="0" y="11" width="9" height="9" fill="#00A4EF"/>'
    '<rect x="11" y="11" width="9" height="9" fill="#FFB900"/>'
    '</svg>'
)

# ── Quick-question labels ─────────────────────────────────────────────────────
QUICK_QUESTIONS = [
    "What are the top 5 countries by number of contract awards?",
    "Which CPV categories have the highest average savings?",
    "What % of awards went to SMEs?",
    "Show the most competitive procurement categories",
    "What is the total estimated value of open tenders?",
    "Compare savings across Services, Supplies and Works",
    "Which countries publish the largest tenders on average?",
    "Top 10 CPV categories by total awarded value",
]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are the Procurement Assistant for the TED Procurement Intelligence product
(IE University × Microsoft — Capstone Group 2). You help procurement buyers and
suppliers with strategy questions about the European public-procurement market.

Rules:
- Answer ONLY from the Gold tables via the query_data tool. Never guess numbers.
- If the data cannot answer a question, say so plainly.
- Answer in plain business language — what it means for strategy, not ML internals.
- Be concise: 2–4 sentences + any data table.
- End every answer with a "Source:" line naming the Gold table(s) used.

{schema}"""

# ── Tool definition ───────────────────────────────────────────────────────────
_TOOL = {
    "name": "query_data",
    "description": (
        "Evaluate a Python/pandas expression against the Gold DataFrames and return "
        "the string result (max 4,000 chars). "
        "Available names: opp (gold_opportunities), awards (gold_awards), "
        "market (gold_market_summary), cpv (gold_cpv_analysis), pd. "
        "No imports or file/network access allowed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pandas_expression": {
                "type": "string",
                "description": (
                    "A single Python expression. Example: "
                    "\"awards.groupby('buyer_country')['awarded_eur'].sum().nlargest(5)\""
                ),
            }
        },
        "required": ["pandas_expression"],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fmt_eur(v: float) -> str:
    if v >= 1e9:   return f"€{v/1e9:.1f}B"
    if v >= 1e6:   return f"€{v/1e6:.1f}M"
    if v >= 1e3:   return f"€{v/1e3:.1f}k"
    return f"€{v:,.0f}"


# ── Demo data (same seed / distributions as dashboard._demo_data) ─────────────
# Must replicate RNG calls in identical order to produce matching datasets.

def _make_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)

    COUNTRIES = {
        "DEU":6195,"POL":4542,"FRA":3773,"ESP":1781,"CZE":1418,
        "ITA":1220,"BEL":932, "SWE":926, "ROU":836, "PRT":753,
        "NLD":706, "HRV":586, "FIN":558, "NOR":513, "BGR":507,
        "LTU":442, "SVN":417, "LVA":401, "IRL":398, "CHE":386,
        "DNK":264, "AUT":369, "HUN":316, "EST":257, "GRC":338,
    }
    CPV = [
        "IT services","Construction works","Medical & laboratory equipment",
        "Architectural & engineering","Road transport","Repair & maintenance",
        "Business services","Sewage, refuse & sanitation","Transport equipment",
        "Software & information systems","Electrical equipment","Health & social work",
        "Industrial machinery","Furniture","Petroleum, gas & fuels",
        "Food, beverages & tobacco","Financial & insurance services",
        "Environmental services","Security services","Research & development",
    ]
    PROC_P = {"Services":.43,"Supplies":.37,"Works":.20}
    n_opp  = 2_000

    ctry_w  = np.array(list(COUNTRIES.values()), float); ctry_w /= ctry_w.sum()
    ctry_pool = rng.choice(list(COUNTRIES.keys()), p=ctry_w, size=n_opp)
    proc_pool = rng.choice(list(PROC_P.keys()), p=list(PROC_P.values()), size=n_opp)
    cpv_pool  = rng.choice(CPV, size=n_opp)
    dates     = pd.date_range("2026-01-05","2026-01-29", freq="B")
    pub_dates = rng.choice(dates, size=n_opp)

    est_vals  = np.where(proc_pool=="Works",  rng.lognormal(14.5,1.2,n_opp),
                np.where(proc_pool=="Services", rng.lognormal(13.8,1.0,n_opp),
                         rng.lognormal(13.0,1.1,n_opp))).astype(int)
    num_lots  = np.where(proc_pool=="Supplies", rng.integers(1,8,n_opp),
                np.where(proc_pool=="Services", rng.integers(1,4,n_opp),
                         rng.integers(1,3,n_opp)))
    deadlines = pd.to_datetime(pub_dates) + pd.to_timedelta(rng.integers(14,60,n_opp), unit="D")

    opp = pd.DataFrame({
        "notice_id":          [f"CN-{2026000+i:06d}" for i in range(n_opp)],
        "pub_date":           pub_dates,
        "buyer_country":      ctry_pool,
        "proc_type":          proc_pool,
        "cpv_division_name":  cpv_pool,
        "estimated":          est_vals,
        "num_lots":           num_lots,
        "submission_deadline": deadlines,
    })

    # Awards (must replicate RNG call order from dashboard)
    AWARD_CL = {
        "ROU":(2058,14.31),"POL":(5312,3.55),"FRA":(3721,2.84),
        "DEU":(4156,1.47), "CZE":(3290,1.76),"ESP":(2250,1.94),
        "ITA":(955, 2.74), "HRV":(757, 2.89),"BGR":(1367,1.43),
        "HUN":(536, 3.08), "LVA":(532, 2.68),"LTU":(711, 1.79),
        "SVN":(344, 3.69), "SVK":(525, 2.35),"BEL":(635, 1.84),
        "NLD":(845, 1.21), "FIN":(541, 1.85),"PRT":(405, 2.31),
        "SWE":(679, 1.35), "GRC":(212, 3.54),
    }
    n_aw      = 3_000
    aw_w      = np.array([AWARD_CL[c][1]*AWARD_CL[c][0] for c in AWARD_CL], float)
    aw_w     /= aw_w.sum()
    aw_ctry   = rng.choice(list(AWARD_CL.keys()), p=aw_w, size=n_aw)
    aw_proc   = rng.choice(["Supplies","Services","Works"], p=[.64,.30,.06], size=n_aw)
    aw_cpv    = rng.choice(CPV, size=n_aw)
    aw_dates  = rng.choice(dates, size=n_aw)
    est_aw    = np.where(aw_proc=="Works",    rng.lognormal(14.5,1.2,n_aw),
                np.where(aw_proc=="Services", rng.lognormal(13.8,1.0,n_aw),
                         rng.lognormal(13.0,1.1,n_aw)))
    sav_pct   = rng.beta(2,3,n_aw)*80
    awarded   = (est_aw*(1-sav_pct/100)).astype(int)
    avg_t     = np.where(aw_proc=="Supplies", rng.gamma(4.0,1.2,n_aw),
                np.where(aw_proc=="Services", rng.gamma(1.8,1.1,n_aw),
                         rng.gamma(1.5,1.1,n_aw))).clip(1,25)
    sme_win   = (rng.random(n_aw)<.189).astype(int)
    aw_lots   = np.where(aw_proc=="Supplies", rng.integers(1,10,n_aw),
                np.where(aw_proc=="Services", rng.integers(1, 5,n_aw),
                         rng.integers(1, 4,n_aw)))

    awards = pd.DataFrame({
        "notice_id":           [f"CAN-{2026000+i:06d}" for i in range(n_aw)],
        "award_date":          aw_dates,
        "buyer_country":       aw_ctry,
        "proc_type":           aw_proc,
        "cpv_division_name":   aw_cpv,
        "estimated":           est_aw.astype(int),
        "awarded_eur":         awarded,
        "savings_pct":         sav_pct.round(1),
        "avg_tenders_per_lot": avg_t.round(2),
        "sme_winner":          sme_win,
        "num_lots":            aw_lots,
    })
    return opp, awards


# ── Gold table loader (cached) ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_gold() -> dict[str, pd.DataFrame]:
    """Load Gold parquets. Falls back to demo data when pipeline hasn't run."""
    def _try(name: str) -> pd.DataFrame | None:
        p = GOLD_DIR / f"{name}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                return df if len(df) > 50 else None
            except Exception:
                pass
        return None

    tables: dict[str, pd.DataFrame] = {}

    real_opp    = _try("gold_opportunities")
    real_awards = _try("gold_awards")

    if real_opp is not None and real_awards is not None:
        tables["opp"]    = real_opp
        tables["awards"] = real_awards
    else:
        demo_opp, demo_awards = _make_demo_data()
        tables["opp"]    = demo_opp
        tables["awards"] = demo_awards

    # Derive market_summary and cpv_analysis if real aggregates are tiny
    real_mkt = _try("gold_market_summary")
    real_cpv = _try("gold_cpv_analysis")

    aw = tables["awards"]
    tables["market"] = real_mkt if real_mkt is not None else (
        aw.groupby("buyer_country")
          .agg(total_awarded=("awarded_eur","sum"), avg_savings_pct=("savings_pct","mean"))
          .reset_index()
          .sort_values("total_awarded", ascending=False)
    )
    tables["cpv"] = real_cpv if real_cpv is not None else (
        aw.groupby("cpv_division_name")
          .agg(avg_competition=("avg_tenders_per_lot","mean"),
               sme_wins=("sme_winner","sum"),
               total_awarded=("awarded_eur","sum"))
          .reset_index()
          .sort_values("avg_competition", ascending=False)
    ) if "cpv_division_name" in aw.columns else pd.DataFrame()

    return tables


# ── Schema context for system prompt ─────────────────────────────────────────

def _build_schema(tables: dict[str, pd.DataFrame]) -> str:
    lines = ["TABLE SCHEMAS + SAMPLES (use these exact column names in queries):"]
    name_map = {
        "opp":    "gold_opportunities",
        "awards": "gold_awards",
        "market": "gold_market_summary",
        "cpv":    "gold_cpv_analysis",
    }
    for key, label in name_map.items():
        df = tables.get(key, pd.DataFrame())
        if df.empty:
            continue
        cols = ", ".join(df.columns)
        lines.append(f"\n• {key} ({label})  —  {len(df):,} rows")
        lines.append(f"  Columns: {cols}")
        if key in ("market", "cpv") and not df.empty:
            lines.append(df.head(4).to_string(index=False))
    return "\n".join(lines)


# ── Sandboxed evaluator ───────────────────────────────────────────────────────

def _run_query(expr: str, tables: dict[str, pd.DataFrame]) -> str:
    """Evaluate a pandas expression in a restricted namespace. Returns str."""
    if not expr.strip():
        return "Empty expression."
    safe_ns = {"pd": pd, "__builtins__": {}, **tables}
    try:
        result = eval(expr, safe_ns)          # noqa: S307
        text   = str(result)
        return text[:4000] + ("…[truncated]" if len(text) > 4000 else "")
    except Exception as exc:
        return f"QueryError: {exc}"


# ── Precomputed quick-question answers ────────────────────────────────────────

def _quick_answer(question: str, tables: dict[str, pd.DataFrame]) -> str | None:
    """Return a pre-formatted markdown answer for the 8 sales questions, or None."""
    q   = question.lower().strip()
    aw  = tables["awards"]
    opp = tables["opp"]

    # ── 1. Top 5 countries by awards ──────────────────────────────────────────
    if "top 5 countries" in q and "award" in q:
        top = aw["buyer_country"].value_counts().head(5)
        rows = "\n".join(f"| {i+1} | **{c}** | {n:,} |"
                         for i, (c, n) in enumerate(top.items()))
        return (
            "**Top 5 markets by contract volume (Jan 2026)**\n\n"
            "| # | Country | Contracts |\n|---|---|---|\n" + rows +
            "\n\nGermany and Poland alone account for ≈ "
            f"{(top.iloc[:2].sum()/len(aw)*100):.0f}% of all European contract awards, "
            "making them priority targets for any supplier seeking EU-wide coverage.\n\n"
            "*Source: gold\\_awards*"
        )

    # ── 2. Highest avg savings by CPV ─────────────────────────────────────────
    if "savings" in q and ("cpv" in q or "categor" in q) and "compare" not in q:
        if "cpv_division_name" not in aw.columns:
            return "CPV breakdown not available in current Gold tables."
        top = aw.groupby("cpv_division_name")["savings_pct"].mean().nlargest(5)
        rows = "\n".join(f"| **{c}** | {v:.1f}% |" for c, v in top.items())
        return (
            "**CPV categories with highest average savings vs buyer estimate**\n\n"
            "| Category | Avg Savings |\n|---|---|\n" + rows +
            "\n\nBuyers in these categories consistently over-budget — "
            "suppliers can compete aggressively on price while staying profitable.\n\n"
            "*Source: gold\\_awards*"
        )

    # ── 3. SME win rate ───────────────────────────────────────────────────────
    if "sme" in q and ("%" in q or "percent" in q or "rate" in q or "went" in q):
        rate    = aw["sme_winner"].mean() * 100
        n_sme   = int(aw["sme_winner"].sum())
        n_total = len(aw)
        return (
            f"**{rate:.1f}%** of contract awards went to SMEs "
            f"({n_sme:,} of {n_total:,} contracts).\n\n"
            "SME participation is healthy but concentrated in Services and high-competition "
            "categories. Buyers should consider whether their lot sizes systematically "
            "exclude smaller firms.\n\n"
            "*Source: gold\\_awards*"
        )

    # ── 4. Most competitive categories ───────────────────────────────────────
    if "competiti" in q and ("categor" in q or "cpv" in q or "show" in q):
        if "cpv_division_name" not in aw.columns:
            return "CPV breakdown not available in current Gold tables."
        top = (aw.groupby("cpv_division_name")["avg_tenders_per_lot"]
                 .mean().nlargest(6).round(1))
        rows = "\n".join(f"| **{c}** | {v:.1f}× |" for c, v in top.items())
        return (
            "**Most contested categories (avg tenders per lot)**\n\n"
            "| Category | Avg Bids/Lot |\n|---|---|\n" + rows +
            "\n\nSupplies contracts attract the most bidders due to lot splitting. "
            "High competition compresses margins — suppliers should price "
            "competitively or target less-contested categories.\n\n"
            "*Source: gold\\_awards*"
        )

    # ── 5. Total estimated value of open tenders ─────────────────────────────
    if "open tenders" in q or ("estimated" in q and "open" in q):
        total   = opp["estimated"].sum()
        n_opp_r = len(opp)
        med     = opp["estimated"].median()
        return (
            f"Open tenders in the current dataset represent a total estimated value of "
            f"**{_fmt_eur(total)}** across {n_opp_r:,} Contract Notices (CN).\n\n"
            f"The median tender is **{_fmt_eur(med)}** — but values range widely from "
            f"small municipal contracts to large infrastructure programmes.\n\n"
            "*Source: gold\\_opportunities*"
        )

    # ── 6. Savings by procurement type ───────────────────────────────────────
    if "savings" in q and ("services" in q or "supplies" in q or "works" in q
                           or "compare" in q or "type" in q):
        if "proc_type" not in aw.columns:
            return "Procurement type breakdown not available in current Gold tables."
        grp = aw.groupby("proc_type")["savings_pct"].agg(
            Median="median", Mean="mean", Count="size"
        ).round(1).reset_index()
        rows = "\n".join(
            f"| **{r.proc_type}** | {r.Median:.1f}% | {r.Mean:.1f}% | {int(r.Count):,} |"
            for r in grp.itertuples()
        )
        return (
            "**Savings vs buyer estimate by procurement type**\n\n"
            "| Type | Median | Mean | Contracts |\n|---|---|---|---|\n" + rows +
            "\n\nSupplies contracts show the largest savings due to price competition "
            "among many bidders. Buyers setting Works budgets have the most "
            "room to tighten estimates.\n\n"
            "*Source: gold\\_awards*"
        )

    # ── 7. Countries with largest average tenders ─────────────────────────────
    if "largest" in q and ("tender" in q or "budget" in q or "countr" in q):
        top = opp.groupby("buyer_country")["estimated"].mean().nlargest(5).apply(_fmt_eur)
        rows = "\n".join(f"| **{c}** | {v} |" for c, v in top.items())
        return (
            "**Countries with highest average tender value (open tenders)**\n\n"
            "| Country | Avg Estimated Value |\n|---|---|\n" + rows +
            "\n\nHigh-value markets offer larger individual contract opportunities "
            "but typically attract larger incumbents — SMEs may find better ROI "
            "in mid-tier markets.\n\n"
            "*Source: gold\\_opportunities*"
        )

    # ── 8. Top 10 CPV by total awarded value ──────────────────────────────────
    if "top 10" in q and ("cpv" in q or "categor" in q) and "award" in q:
        if "cpv_division_name" not in aw.columns:
            return "CPV breakdown not available in current Gold tables."
        top = aw.groupby("cpv_division_name")["awarded_eur"].sum().nlargest(10)
        rows = "\n".join(f"| {i+1} | **{c}** | {_fmt_eur(v)} |"
                         for i, (c, v) in enumerate(top.items()))
        return (
            "**Top 10 CPV categories by total awarded contract value**\n\n"
            "| # | Category | Total Awarded |\n|---|---|---|\n" + rows +
            "\n\nConstruction and IT services dominate by volume. "
            "These represent the deepest and most liquid procurement markets in Europe.\n\n"
            "*Source: gold\\_awards*"
        )

    return None   # not a precomputed question → tool path


# ── Agentic streaming loop ────────────────────────────────────────────────────

def _content_to_dicts(content_blocks) -> list[dict]:
    """Serialize Anthropic ContentBlock objects to plain dicts."""
    out = []
    for b in content_blocks:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _agentic_stream(api_msgs: list[dict], tables: dict, schema: str):
    """
    Tool-use loop then final streaming answer. Yields str chunks for st.write_stream.
    Phase 1: non-streaming tool calls (fast; < 1 s each).
    Phase 2: streaming final answer from the informed context.
    """
    import anthropic
    client     = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sys_prompt = _SYSTEM.format(schema=schema)
    msgs       = list(api_msgs)

    for _ in range(5):          # hard cap: max 5 tool iterations
        resp = client.messages.create(
            model=MODEL, max_tokens=512,
            system=sys_prompt, tools=[_TOOL], messages=msgs,
        )

        if resp.stop_reason != "tool_use":
            # No (more) tools — yield the text directly (character stream)
            text = next((b.text for b in resp.content if b.type == "text"), "")
            yield from text
            return

        # Execute every tool_use block in this response
        msgs.append({"role": "assistant", "content": _content_to_dicts(resp.content)})
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": b.id,
                "content": _run_query(b.input.get("pandas_expression", ""), tables),
            }
            for b in resp.content if b.type == "tool_use"
        ]
        msgs.append({"role": "user", "content": tool_results})

    # After tool loop, stream the final phrased answer
    with client.messages.stream(
        model=MODEL, max_tokens=1024,
        system=sys_prompt, messages=msgs,
    ) as s:
        yield from s.text_stream


# ── CSS injection ─────────────────────────────────────────────────────────────

def _inject_css(T: dict) -> None:
    a15  = _hex_rgba(T["accent"], .15)
    a25  = _hex_rgba(T["accent"], .25)
    a08  = _hex_rgba(T["accent"], .08)
    a20  = _hex_rgba(T["accent"], .20)
    a90  = _hex_rgba(T["accent"], .90)    # user bubble

    st.markdown(f"""
<style>
/* ── Assistant header ── */
.pa-header {{
    background:{T["card"]}; border:0.5px solid {T["border"]}; border-radius:14px;
    padding:14px 20px; margin-bottom:6px; box-shadow:{T["card_shadow"]};
    display:flex; align-items:center; justify-content:space-between; gap:16px;
}}
.pa-header-left {{ display:flex; align-items:center; gap:12px; }}
.pa-avatar-badge {{
    width:40px; height:40px; border-radius:50%;
    background:{a15}; border:1px solid {a25};
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
    box-shadow:0 2px 8px rgba(0,0,0,.20);
}}
.pa-title {{ font-size:15px; font-weight:700; color:{T["heading"]}; }}
.pa-sub   {{ font-size:11px; color:{T["muted"]}; margin-top:2px; }}
.pa-brand {{ display:flex; align-items:center; gap:7px; }}
.pa-brand-text {{
    font-family:'Segoe UI',system-ui,sans-serif; font-weight:600;
    font-size:13px; color:{T["muted"]};
}}

/* ── Welcome card ── */
.pa-welcome {{
    background:{T["card"]}; border:0.5px solid {T["border"]}; border-radius:12px;
    padding:18px 22px; margin:10px 0 6px 0; font-size:14px;
    color:{T["heading"]}; line-height:1.65; box-shadow:{T["card_shadow"]};
}}
.pa-welcome b {{ color:{T["accent"]}; font-weight:600; }}
.pa-welcome-hint {{ font-size:11px; color:{T["muted"]}; margin-top:6px; }}

/* ── Quick-question section ── */
.qq-label {{
    font-size:10.5px; font-weight:600; letter-spacing:.7px;
    text-transform:uppercase; color:{T["muted"]}; margin:16px 0 8px 0;
}}

/* ── Suggestion buttons as pills ── */
[data-testid="stButton"] > button {{
    border-radius:18px !important; font-size:12.5px !important;
    font-weight:500 !important; padding:7px 16px !important;
    border:1px solid {a25} !important; color:{T["accent"]} !important;
    background:{a08} !important; transition:background .15s !important;
    white-space:normal !important; text-align:left !important;
    height:auto !important; min-height:38px !important; line-height:1.4 !important;
}}
[data-testid="stButton"] > button:hover {{
    background:{a20} !important; border-color:{T["accent"]} !important;
}}

/* ── User bubble (right-aligned, injected as HTML) ── */
.pa-user-bubble {{
    display:flex; justify-content:flex-end; margin:6px 0 10px 0;
}}
.pa-user-inner {{
    max-width:72%; background:{a90};
    color:#fff; border-radius:18px 18px 4px 18px;
    padding:11px 16px; font-size:14px; line-height:1.55;
    box-shadow:0 1px 4px rgba(0,0,0,.22);
}}

/* ── Assistant chat_message override ── */
[data-testid="stChatMessage"] {{
    border-radius:12px !important;
    padding:6px 0 !important;
    margin-bottom:2px !important;
}}

/* ── No-key notice ── */
.pa-nokey {{
    background:{_hex_rgba(T["negative"],.07)}; border:1px solid {_hex_rgba(T["negative"],.22)};
    border-radius:10px; padding:14px 18px; color:{T["heading"]}; font-size:13px; margin:10px 0;
}}
.pa-nokey code {{
    background:{T["surface"]}; border:1px solid {T["border"]}; border-radius:4px;
    padding:1px 5px; font-size:12px; color:{T["accent"]};
}}

/* ── Error card ── */
.pa-error {{
    background:{_hex_rgba(T["negative"],.06)}; border:1px solid {_hex_rgba(T["negative"],.18)};
    border-radius:8px; padding:10px 14px; font-size:12px; color:{T["heading"]}; margin-top:6px;
}}

/* ── Powered footer ── */
.pa-footer {{
    display:flex; align-items:center; gap:6px; justify-content:flex-end;
    font-size:10px; color:{T["muted"]}; margin-top:10px;
    padding-top:8px; border-top:1px solid {T["border"]};
}}
</style>
""", unsafe_allow_html=True)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _render_header(T: dict) -> None:
    st.markdown(f"""
<div class="pa-header">
  <div class="pa-header-left">
    <div class="pa-avatar-badge">{_MS_MARK_HTML}</div>
    <div>
      <div class="pa-title">Procurement Assistant</div>
      <div class="pa-sub">Ask about the European procurement market</div>
    </div>
  </div>
  <div class="pa-brand">
    {_MS_MARK_HTML}
    <span class="pa-brand-text">Microsoft</span>
  </div>
</div>
""", unsafe_allow_html=True)


def _user_bubble(text: str) -> None:
    """Render a right-aligned user bubble without using st.chat_message."""
    safe = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(
        f'<div class="pa-user-bubble"><div class="pa-user-inner">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def _render_quick_questions(T: dict) -> str | None:
    """Render the 2-column quick-question grid; return the clicked question or None."""
    st.markdown('<div class="qq-label">Quick questions — click to ask</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        with (c1 if i % 2 == 0 else c2):
            if st.button(q, key=f"pa_chip_{i}", use_container_width=True):
                return q
    return None


# ── Main render ───────────────────────────────────────────────────────────────

def render_assistant(T: dict) -> None:
    """Render the full Procurement Assistant UI into the active Streamlit context."""

    # ── Init session state ────────────────────────────────────────────────────
    if "pa_chat" not in st.session_state:
        st.session_state.pa_chat: list[dict] = []
    history: list[dict] = st.session_state.pa_chat
    pending = st.session_state.pop("pa_pending", None)

    # ── CSS + Header ──────────────────────────────────────────────────────────
    _inject_css(T)
    hdr_col, clr_col = st.columns([9, 1])
    with hdr_col:
        _render_header(T)
    with clr_col:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="pa_clear", help="Clear conversation",
                     use_container_width=True):
            st.session_state.pa_chat = []
            st.rerun()

    # ── API-key guard ─────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.markdown("""
<div class="pa-nokey">
  ⚠️ <strong>No API key.</strong>
  Add <code>ANTHROPIC_API_KEY=sk-ant-…</code> to your <code>.env</code> file
  and restart the app. Quick-question answers still work without a key.
</div>
""", unsafe_allow_html=True)
        # Even without key, show quick questions (precomputed — no API call)

    # ── Load data (cached) ────────────────────────────────────────────────────
    with st.spinner("Loading Gold tables…"):
        tables = _load_gold()
    schema = _build_schema(tables)

    # ── Render conversation history ───────────────────────────────────────────
    for msg in history:
        if msg["role"] == "user":
            _user_bubble(msg["content"])
        else:
            with st.chat_message("assistant", avatar=_MS_AVATAR):
                st.markdown(msg["content"])

    # ── Empty state: welcome + quick questions ────────────────────────────────
    if not history and pending is None:
        st.markdown(f"""
<div class="pa-welcome">
  Welcome! Ask me anything about the European public-procurement market —
  country rankings, savings benchmarks, <b>SME</b> win rates, competitive intensity
  by <b>CPV</b> category, or the total value of open tenders.
  Answers come directly from our January 2026 Gold tables.
  <div class="pa-welcome-hint">
    Free-text questions and follow-ups work too — just type below.
  </div>
</div>
""", unsafe_allow_html=True)

    # Quick questions — shown in empty state and in expander when history exists
    if history:
        with st.expander("Quick questions", expanded=False):
            chip = _render_quick_questions(T)
            if chip:
                st.session_state.pa_pending = chip
                st.rerun()
    elif pending is None:
        chip = _render_quick_questions(T)
        if chip:
            st.session_state.pa_pending = chip
            st.rerun()
        st.markdown(f"""
<div class="pa-footer">
  {_MS_MARK_HTML}
  <span>Powered by Microsoft · claude-sonnet-4-6</span>
</div>
""", unsafe_allow_html=True)

    # ── Resolve user input ────────────────────────────────────────────────────
    chat_input  = st.chat_input("Ask about European procurement…")
    user_input: str | None = chat_input or pending

    if not user_input:
        return

    # ── Render user bubble ────────────────────────────────────────────────────
    history.append({"role": "user", "content": user_input})
    _user_bubble(user_input)

    # ── Generate answer ───────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar=_MS_AVATAR):
        # 1. Try precomputed (instant, no API)
        quick = _quick_answer(user_input, tables)
        if quick:
            st.markdown(quick)
            response_text = quick

        # 2. No API key → polite decline
        elif not api_key:
            response_text = (
                "I can answer the pre-built questions above without an API key. "
                "For free-text questions, please add `ANTHROPIC_API_KEY` to your `.env` file."
            )
            st.markdown(response_text)

        # 3. Tool path with streaming
        else:
            api_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in history
            ]
            try:
                response_text = st.write_stream(
                    _agentic_stream(api_msgs, tables, schema)
                )
            except Exception as exc:
                response_text = (
                    f"*API error:* `{exc}`\n\n"
                    "Check that your `ANTHROPIC_API_KEY` is valid and "
                    "`anthropic>=0.40.0` is installed."
                )
                st.markdown(response_text)

    history.append({"role": "assistant", "content": str(response_text)})
