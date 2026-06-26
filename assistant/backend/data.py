"""
Data layer for the Procurement Assistant.

Loads the project's Gold parquet tables when present, and otherwise falls back to a
small, clearly-labelled SAMPLE dataset so the app runs end-to-end for design work.
Also exposes a sandboxed evaluator for the `query_data` tool.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Tables we expect from the Gold layer (see README).
GOLD_TABLES = [
    "gold_opportunities",
    "gold_awards",
    "gold_market_summary",
    "gold_cpv_analysis",
]

# Where the dashboard keeps its Gold parquet files. Override with GOLD_DIR if needed.
GOLD_DIR = Path(os.environ.get("GOLD_DIR", "data/gold"))


# --------------------------------------------------------------------------------------
# Sample fallback data (only used when real parquet files are not found)
# --------------------------------------------------------------------------------------
def _sample_tables() -> dict[str, pd.DataFrame]:
    """Synthetic, plausibly-shaped Gold tables so the UI works without the real data.

    These numbers are illustrative ONLY. When data/gold/*.parquet exists, it is used
    instead and these are never touched.
    """
    rng = np.random.default_rng(7)

    countries = ["DEU", "POL", "FRA", "ESP", "CZE", "ITA", "ROU", "BEL", "NLD", "PRT",
                 "NOR", "SWE", "FIN", "HRV", "BGR", "IRL", "LVA", "GRC", "LTU", "CHE"]
    country_weight = np.array([440, 295, 278, 115, 94, 88, 61, 58, 57, 55,
                               46, 46, 42, 36, 36, 34, 29, 28, 27, 26], dtype=float)

    cpvs = ["Transport equipment", "Furniture", "Road transport", "Business services",
            "Electrical equipment", "Software & information systems", "Environmental services",
            "Architectural & engineering", "Health & social work", "Petroleum, gas & fuels",
            "Repair & maintenance", "Food, beverages & tobacco", "IT services",
            "Construction work", "Medical equipment", "Education services",
            "Security services", "Laboratory equipment", "Postal services", "Energy"]
    ptypes = ["Services", "Supplies", "Works"]

    # ---- gold_opportunities (open Contract Notices) ----
    n_opp = 6000
    opp_country = rng.choice(countries, size=n_opp, p=country_weight / country_weight.sum())
    opp = pd.DataFrame({
        "notice_id": [f"CN-{i:06d}" for i in range(n_opp)],
        "buyer_country": opp_country,
        "cpv_division_name": rng.choice(cpvs, size=n_opp),
        "estimated": np.round(rng.lognormal(mean=12.6, sigma=1.1, size=n_opp), 0),
        "num_lots": rng.integers(1, 6, size=n_opp),
    })

    # ---- gold_awards (Contract Award Notices) ----
    n_awd = 8000
    awd_type = rng.choice(ptypes, size=n_awd, p=[0.44, 0.37, 0.19])
    lots_by_type = {"Services": 1.92, "Supplies": 4.69, "Works": 1.68}
    awd = pd.DataFrame({
        "notice_id": [f"CAN-{i:06d}" for i in range(n_awd)],
        "buyer_country": rng.choice(countries, size=n_awd, p=country_weight / country_weight.sum()),
        "cpv_division_name": rng.choice(cpvs, size=n_awd),
        "procurement_type": awd_type,
        "awarded_eur": np.round(rng.lognormal(mean=12.4, sigma=1.2, size=n_awd), 0),
        "savings_pct": np.clip(rng.normal(0.40, 0.18, size=n_awd), -0.2, 0.9),
        "avg_tenders_per_lot": np.round(
            [rng.normal(lots_by_type[t], 1.2) for t in awd_type], 2).clip(1, None),
        "sme_winner": rng.random(n_awd) < 0.189,
    })

    # ---- gold_market_summary (pre-aggregated by dimension) ----
    by_country = (awd.groupby("buyer_country")
                  .agg(total_awarded=("awarded_eur", "sum"),
                       avg_savings_pct=("savings_pct", "mean"))
                  .reset_index()
                  .rename(columns={"buyer_country": "dimension_value"}))
    by_country.insert(0, "dimension", "country")
    by_type = (awd.groupby("procurement_type")
               .agg(total_awarded=("awarded_eur", "sum"),
                    avg_savings_pct=("savings_pct", "mean"))
               .reset_index()
               .rename(columns={"procurement_type": "dimension_value"}))
    by_type.insert(0, "dimension", "procurement_type")
    market = pd.concat([by_country, by_type], ignore_index=True)

    # ---- gold_cpv_analysis ----
    cpv = (awd.groupby("cpv_division_name")
           .agg(avg_competition=("avg_tenders_per_lot", "mean"),
                sme_wins=("sme_winner", "sum"),
                total_awarded=("awarded_eur", "sum"),
                avg_savings_pct=("savings_pct", "mean"))
           .reset_index())

    return {
        "gold_opportunities": opp,
        "gold_awards": awd,
        "gold_market_summary": market,
        "gold_cpv_analysis": cpv,
    }


@lru_cache(maxsize=1)
def load_gold() -> tuple[dict[str, pd.DataFrame], bool]:
    """Return ({table_name: DataFrame}, using_sample_data).

    Reads real parquet from GOLD_DIR when all expected tables exist; otherwise falls
    back to sample data so the app still runs.
    """
    paths = {t: GOLD_DIR / f"{t}.parquet" for t in GOLD_TABLES}
    if all(p.exists() for p in paths.values()):
        return {t: pd.read_parquet(p) for t, p in paths.items()}, False
    return _sample_tables(), True


def schema_context(tables: dict[str, pd.DataFrame]) -> str:
    """Compact schema for the system prompt.

    Example rows exist solely to show column format. The model MUST call
    query_data to compute any real figure from the full table.
    """
    blocks = []
    for name, df in tables.items():
        cols = ", ".join(f"{c} ({str(dt)})" for c, dt in df.dtypes.items())
        sample = df.head(2).to_dict(orient="records")
        blocks.append(
            f"TABLE {name} — {len(df):,} rows total.\n"
            f"  ALWAYS call query_data on the FULL table; never read numbers from the example rows.\n"
            f"  columns: {cols}\n"
            f"  (example rows — column FORMAT ONLY, NOT the dataset): {sample}"
        )
    return "\n\n".join(blocks)


def run_query(expression: str, tables: dict[str, pd.DataFrame]) -> str:
    """Sandboxed evaluator for the query_data tool.

    Only the Gold DataFrames and `pd` are in scope. No builtins, imports, files, or
    network. Result is stringified and length-capped.
    """
    safe_builtins = {
        "round": round, "len": len, "sum": sum, "min": min, "max": max,
        "sorted": sorted, "abs": abs, "list": list, "dict": dict, "set": set,
        "float": float, "int": int, "str": str, "range": range, "zip": zip,
    }
    safe_globals = {"__builtins__": safe_builtins}
    safe_locals = {"pd": pd, **tables}
    try:
        result = eval(expression, safe_globals, safe_locals)  # noqa: S307 - sandboxed
    except Exception as exc:  # surfaced back to the model so it can correct itself
        return f"ERROR: {type(exc).__name__}: {exc}"
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_string()[:4000]
    return str(result)[:4000]
