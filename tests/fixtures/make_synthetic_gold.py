"""
tests/fixtures/make_synthetic_gold.py
──────────────────────────────────────
Generate synthetic Gold tables matching docs/gold_contract.md, so the chatbot
engine and dashboard can be developed and tested with ZERO real data.

The shapes mirror `_demo_data()` in app/dashboard.py but produce the full
contract column set (and are Streamlit-free, so they import cleanly in tests).

Usage:
    # in-memory dict (for tests)
    from tests.fixtures.make_synthetic_gold import make_synthetic_gold
    dfs = make_synthetic_gold()

    # write parquet to data/gold/ (to exercise the real load path / dashboard)
    python -m tests.fixtures.make_synthetic_gold            # -> data/gold/*.parquet
    python -m tests.fixtures.make_synthetic_gold /tmp/gold  # -> /tmp/gold/*.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CPV_DIVISIONS  # noqa: E402

COUNTRIES = [
    "Germany", "France", "Spain", "Italy", "Netherlands",
    "Poland", "Sweden", "Denmark", "Belgium", "Austria",
    "Portugal", "Finland", "Ireland", "Czech Republic", "Romania",
]
PROC_TYPES = ["Services", "Supplies", "Works"]
PROCEDURE_TYPES = ["open", "restricted", "negotiated", "competitive-dialogue"]


def _dates(start: str, n: int, freq: str = "6h") -> list[str]:
    return pd.date_range(start, periods=n, freq=freq).strftime("%Y-%m-%d").tolist()


def make_synthetic_gold(n_opp: int = 150, n_awards: int = 120, seed: int = 42
                        ) -> dict[str, pd.DataFrame]:
    """Return the four contract tables keyed by their chatbot variable names:
    'opportunities', 'awards', 'market_summary', 'cpv_analysis'."""
    rng = np.random.default_rng(seed)
    cpv_names = list(CPV_DIVISIONS.values())
    cpv_codes = list(CPV_DIVISIONS.keys())

    def pick_cpv(n):
        idx = rng.integers(0, len(cpv_codes), n)
        return ([cpv_codes[i] for i in idx], [cpv_names[i] for i in idx])

    # ── gold_opportunities (CN) ──────────────────────────────────────────────
    o_div, o_name = pick_cpv(n_opp)
    # ~43% of estimates populated, matching the documented coverage
    o_est = rng.lognormal(14, 1.8, n_opp).clip(50_000, 60_000_000)
    o_est[rng.random(n_opp) > 0.43] = np.nan
    opportunities = pd.DataFrame({
        "notice_id":           [f"CN-2026-{i:04d}" for i in range(n_opp)],
        "pub_date":            _dates("2026-01-02", n_opp, "5h"),
        "buyer_name":          [f"Authority {i % 60}" for i in range(n_opp)],
        "buyer_country":       rng.choice(COUNTRIES, n_opp),
        "buyer_org_id":        [f"ORG-{rng.integers(0, 400)}" for _ in range(n_opp)],
        "project_title":       [f"Open tender {i}" for i in range(n_opp)],
        "proc_type":           rng.choice(PROC_TYPES, n_opp),
        "procedure_type":      rng.choice(PROCEDURE_TYPES, n_opp),
        "cpv_main":            [f"{d}000000" for d in o_div],
        "cpv_division":        o_div,
        "cpv_division_name":   o_name,
        "estimated":           o_est,
        "num_lots":            rng.integers(1, 9, n_opp),
        "avg_lot_duration":    rng.uniform(3, 48, n_opp).round(1),
        "submission_deadline": _dates("2026-02-01", n_opp, "7h"),
        "framework":           rng.choice([True, False], n_opp, p=[0.2, 0.8]),
        "source_file":         [f"notice_{i:05d}.xml" for i in range(n_opp)],
    })

    # ── gold_awards (CAN) ────────────────────────────────────────────────────
    a_div, a_name = pick_cpv(n_awards)
    est = rng.lognormal(14, 1.8, n_awards).clip(50_000, 60_000_000)
    awarded = est * rng.uniform(0.70, 1.06, n_awards)
    savings = ((est - awarded) / est * 100).round(1)
    awards = pd.DataFrame({
        "notice_id":           [f"CAN-2026-{i:04d}" for i in range(n_awards)],
        "pub_date":            _dates("2026-01-05", n_awards, "8h"),
        "buyer_name":          [f"Authority {i % 60}" for i in range(n_awards)],
        "buyer_country":       rng.choice(COUNTRIES, n_awards),
        "buyer_org_id":        [f"ORG-{rng.integers(0, 400)}" for _ in range(n_awards)],
        "project_title":       [f"Awarded project {i}" for i in range(n_awards)],
        "proc_type":           rng.choice(PROC_TYPES, n_awards),
        "cpv_main":            [f"{d}000000" for d in a_div],
        "cpv_division":        a_div,
        "cpv_division_name":   a_name,
        "estimated":           est,
        "awarded_eur":         awarded,
        "savings_pct":         savings,
        "num_lots":            rng.integers(1, 7, n_awards),
        "num_awarded_lots":    rng.integers(1, 5, n_awards),
        "avg_tenders_per_lot": rng.exponential(4.5, n_awards).clip(1, 22).round(1),
        "total_tenders":       rng.integers(1, 40, n_awards),
        "winner_names":        [f"Supplier {chr(65 + i % 26)}" for i in range(n_awards)],
        "winner_countries":    rng.choice(COUNTRIES, n_awards),
        "sme_winner":          rng.choice([True, False], n_awards, p=[0.45, 0.55]),
        "avg_lot_duration":    rng.uniform(3, 48, n_awards).round(1),
        "source_file":         [f"award_{i:05d}.xml" for i in range(n_awards)],
    })

    # ── gold_market_summary (long format over 3 dimensions) ──────────────────
    market_summary = _build_market_summary(awards)

    # ── gold_cpv_analysis ────────────────────────────────────────────────────
    cpv_analysis = _build_cpv_analysis(awards, opportunities)

    return {
        "opportunities":   opportunities,
        "awards":          awards,
        "market_summary":  market_summary,
        "cpv_analysis":    cpv_analysis,
    }


def _build_market_summary(awards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in ["buyer_country", "cpv_division_name", "proc_type"]:
        agg = (
            awards.groupby(dim)
            .agg(
                notice_count=("notice_id", "count"),
                total_estimated=("estimated", "sum"),
                total_awarded=("awarded_eur", "sum"),
                avg_savings_pct=("savings_pct", "mean"),
                avg_tenders=("avg_tenders_per_lot", "mean"),
                sme_winner_count=("sme_winner", "sum"),
            )
            .reset_index()
            .rename(columns={dim: "dimension_value"})
        )
        agg["dimension"] = dim
        rows.append(agg)
    df = pd.concat(rows, ignore_index=True)
    df["avg_savings_pct"] = df["avg_savings_pct"].round(1)
    df["avg_tenders"] = df["avg_tenders"].round(1)
    cols = ["dimension", "dimension_value", "notice_count", "total_estimated",
            "total_awarded", "avg_savings_pct", "avg_tenders", "sme_winner_count"]
    return df[cols]


def _build_cpv_analysis(awards: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    aw = (
        awards.groupby(["cpv_division", "cpv_division_name"])
        .agg(
            awards_count=("notice_id", "count"),
            total_awarded=("awarded_eur", "sum"),
            avg_savings=("savings_pct", "mean"),
            avg_competition=("avg_tenders_per_lot", "mean"),
            sme_wins=("sme_winner", "sum"),
        )
        .reset_index()
    )
    op = (
        opportunities.groupby(["cpv_division", "cpv_division_name"])
        .agg(
            open_notices=("notice_id", "count"),
            total_open_est=("estimated", "sum"),
        )
        .reset_index()
    )
    df = aw.merge(op, on=["cpv_division", "cpv_division_name"], how="outer")
    df["avg_savings"] = df["avg_savings"].round(1)
    df["avg_competition"] = df["avg_competition"].round(1)
    return df.sort_values("total_awarded", ascending=False, na_position="last").reset_index(drop=True)


def write_parquet(out_dir: Path, **kwargs) -> dict[str, Path]:
    """Write the synthetic tables as gold_*.parquet into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "opportunities":  "gold_opportunities",
        "awards":         "gold_awards",
        "market_summary": "gold_market_summary",
        "cpv_analysis":   "gold_cpv_analysis",
    }
    dfs = make_synthetic_gold(**kwargs)
    written = {}
    for key, fname in file_map.items():
        path = out_dir / f"{fname}.parquet"
        dfs[key].to_parquet(path, index=False)
        written[fname] = path
    return written


if __name__ == "__main__":
    from config import GOLD_DIR
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLD_DIR
    written = write_parquet(target)
    print(f"Wrote synthetic Gold tables to {target}:")
    for name, path in written.items():
        df = pd.read_parquet(path)
        print(f"  ✓ {path.name:<28} ({len(df):,} rows, {len(df.columns)} cols)")
