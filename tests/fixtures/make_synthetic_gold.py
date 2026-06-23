"""
tests/fixtures/make_synthetic_gold.py
──────────────────────────────────────
Generate synthetic Gold tables that mirror the **real** schema delivered in the
capstone Drive (verified 2026-06-23), so the chatbot engine and tests run with
ZERO real data and still reflect reality.

Mirrors these real quirks on purpose:
  - `gold_awards` is **lot-grained** and uses `awarded_amount`/`total_awarded`
    (NOT `awarded_eur`), `tenders_count` (NOT `avg_tenders_per_lot`); it has NO
    `savings_pct`.
  - `is_sme`, `winner_name`, `winner_country` are **100% empty** (as in real data).
  - Includes the two ML-output tables `gold_notice_enrichment` and
    `gold_bid_win_probability`.

Keys returned match the names the chatbot engine expects:
  opportunities, awards, market_summary, cpv_analysis, notice_enrichment,
  bid_win_probability

Usage:
    from tests.fixtures.make_synthetic_gold import make_synthetic_gold
    dfs = make_synthetic_gold()
    python -m tests.fixtures.make_synthetic_gold /tmp/gold   # write parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CPV_DIVISIONS  # noqa: E402

COUNTRIES = ["DEU", "FRA", "ESP", "ITA", "NLD", "POL", "SWE", "DNK", "BEL",
             "AUT", "PRT", "FIN", "IRL", "CZE", "ROU"]
PROC_TYPES = ["Services", "Supplies", "Works"]
PROCEDURE_TYPES = ["open", "restricted", "negotiated", "competitive-dialogue"]


def _dates(start: str, n: int, freq: str = "6h") -> list[str]:
    return pd.date_range(start, periods=n, freq=freq).strftime("%Y-%m-%d").tolist()


def make_synthetic_gold(n_opp: int = 150, n_awards: int = 200, n_notices: int = 300,
                        n_bids: int = 400, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cpv_codes = [int(k) for k in CPV_DIVISIONS.keys()]
    cpv_names = list(CPV_DIVISIONS.values())

    def pick_cpv(n):
        idx = rng.integers(0, len(cpv_codes), n)
        return ([cpv_codes[i] for i in idx], [cpv_names[i] for i in idx])

    # ── gold_opportunities (CN) — matches real schema ────────────────────────
    o_div, o_name = pick_cpv(n_opp)
    o_est = rng.lognormal(14, 1.8, n_opp).clip(50_000, 60_000_000)
    o_est[rng.random(n_opp) > 0.43] = np.nan
    opportunities = pd.DataFrame({
        "notice_id":           [f"CN-{i:05d}" for i in range(n_opp)],
        "pub_date":            _dates("2026-01-02", n_opp, "5h"),
        "buyer_name":          [f"Authority {i % 60}" for i in range(n_opp)],
        "buyer_country":       rng.choice(COUNTRIES, n_opp),
        "buyer_org_id":        [f"ORG-{rng.integers(0, 400)}" for _ in range(n_opp)],
        "project_title":       [f"Open tender {i}" for i in range(n_opp)],
        "proc_type":           rng.choice(PROC_TYPES, n_opp),
        "procedure_type":      rng.choice(PROCEDURE_TYPES, n_opp),
        "cpv_main":            [f"{d:02d}000000" for d in o_div],
        "cpv_division":        o_div,
        "cpv_division_name":   o_name,
        "estimated":           o_est,
        "num_lots":            rng.integers(1, 9, n_opp),
        "avg_lot_duration":    rng.uniform(3, 48, n_opp).round(1),
        "submission_deadline": _dates("2026-02-01", n_opp, "7h"),
        "framework":           rng.choice([True, False], n_opp, p=[0.2, 0.8]),
        "source_file":         [f"notice_{i:05d}.xml" for i in range(n_opp)],
    })

    # ── gold_awards (CAN) — LOT-GRAINED, real column names ───────────────────
    a_div, a_name = pick_cpv(n_awards)
    est = rng.lognormal(14, 1.8, n_awards).clip(50_000, 60_000_000)
    est[rng.random(n_awards) > 0.43] = np.nan
    awarded = rng.lognormal(13.5, 1.7, n_awards).clip(1_000, 50_000_000)
    awards = pd.DataFrame({
        "notice_id":         [f"CAN-{i // 2:05d}" for i in range(n_awards)],
        "lot_id":            [f"LOT-{i % 5:04d}" for i in range(n_awards)],
        "lot_result_id":     [f"LR-{i:05d}" for i in range(n_awards)],
        "notice_type":       "CAN",
        "pub_date":          _dates("2026-01-05", n_awards, "4h"),
        "buyer_name":        [f"Authority {i % 60}" for i in range(n_awards)],
        "buyer_country":     rng.choice(COUNTRIES, n_awards),
        "project_title":     [f"Awarded project {i}" for i in range(n_awards)],
        "lot_title":         [f"Lot {i}" for i in range(n_awards)],
        "proc_type":         rng.choice(PROC_TYPES, n_awards),
        "procedure_type":    rng.choice(PROCEDURE_TYPES, n_awards),
        "cpv_code":          [f"{d:02d}000000" for d in a_div],
        "cpv_division":      a_div,
        "cpv_division_name": a_name,
        "result_code":       "selec-w",
        "is_awarded":        True,
        "winner_org_id":     [f"TPA-{rng.integers(1, 5000):04d}" for _ in range(n_awards)],
        # These three are 100% empty in the real data — mirror that here.
        "winner_name":       pd.Series([None] * n_awards, dtype="object"),
        "winner_country":    pd.Series([None] * n_awards, dtype="object"),
        "is_sme":            pd.Series([None] * n_awards, dtype="object"),
        "tenders_count":     pd.array(rng.integers(1, 25, n_awards), dtype="Int64"),
        "awarded_amount":    awarded,
        "contract_value":    awarded * rng.uniform(0.95, 1.05, n_awards),
        "tender_value":      awarded * rng.uniform(0.9, 1.1, n_awards),
        "estimated":         est,
        "total_awarded":     awarded,
    })

    # ── gold_market_summary (long format over 3 dimensions) ──────────────────
    market_summary = _market_summary(awards)

    # ── gold_cpv_analysis ────────────────────────────────────────────────────
    cpv_analysis = _cpv_analysis(awards, opportunities, rng)

    # ── gold_notice_enrichment (ML outputs, per notice) ──────────────────────
    e_div, e_name = pick_cpv(n_notices)
    p_div, p_name = pick_cpv(n_notices)
    actual = rng.lognormal(13.5, 1.7, n_notices).clip(1_000, 50_000_000)
    actual[rng.random(n_notices) > 0.43] = np.nan
    n_est = rng.lognormal(14, 1.8, n_notices).clip(50_000, 60_000_000)
    n_est[rng.random(n_notices) > 0.43] = np.nan
    notice_enrichment = pd.DataFrame({
        "notice_id":                  [f"N-{i:05d}" for i in range(n_notices)],
        "notice_type":                rng.choice(["CN", "CAN", "PIN"], n_notices, p=[0.41, 0.56, 0.03]),
        "buyer_country":              rng.choice(COUNTRIES, n_notices),
        "proc_type":                  rng.choice(PROC_TYPES, n_notices),
        "cpv_division_original":      [float(d) for d in e_div],
        "cpv_division_name_original": e_name,
        "cpv_pred":                   [float(d) for d in p_div],
        "cpv_confidence":             rng.uniform(0.5, 1.0, n_notices).round(4),
        "cpv_pred_name":              p_name,
        "cpv_review_flag":            rng.choice([True, False], n_notices, p=[0.02, 0.98]),
        "predicted_award_value":      rng.lognormal(13.5, 1.5, n_notices).clip(1_000, 40_000_000).round(0),
        "total_awarded_actual":       actual,
        "estimated_value":            n_est,
    })

    # ── gold_bid_win_probability (ML outputs, per bid, competitive lots) ─────
    bid_win_probability = pd.DataFrame({
        "notice_id":          [f"N-{rng.integers(0, n_notices):05d}" for _ in range(n_bids)],
        "lot_id":             [f"LOT-{rng.integers(0, 5):04d}" for _ in range(n_bids)],
        "tender_id":          [f"TEN-{i:05d}" for i in range(n_bids)],
        "tendering_party_id": [f"TPA-{rng.integers(1, 5000):04d}" for _ in range(n_bids)],
        "tender_value":       rng.lognormal(13, 1.6, n_bids).clip(500, 40_000_000).round(0),
        "n_bids":             rng.integers(2, 20, n_bids),
        "win_probability":    rng.uniform(0, 1, n_bids).round(4),
        "is_winner_actual":   rng.choice([0, 1], n_bids, p=[0.7, 0.3]),
    })

    return {
        "opportunities":       opportunities,
        "awards":              awards,
        "market_summary":      market_summary,
        "cpv_analysis":        cpv_analysis,
        "notice_enrichment":   notice_enrichment,
        "bid_win_probability": bid_win_probability,
    }


def _savings(estimated: pd.Series, awarded: pd.Series) -> pd.Series:
    return np.where(estimated > 0, (estimated - awarded) / estimated * 100, np.nan)


def _market_summary(awards: pd.DataFrame) -> pd.DataFrame:
    a = awards.copy()
    a["_sav"] = _savings(a["estimated"], a["awarded_amount"])
    rows = []
    for dim in ["buyer_country", "cpv_division_name", "proc_type"]:
        agg = (a.groupby(dim).agg(
            notice_count=("notice_id", "nunique"),
            total_estimated=("estimated", "sum"),
            total_awarded=("awarded_amount", "sum"),
            avg_savings_pct=("_sav", "mean"),
            avg_tenders=("tenders_count", "mean"),
        ).reset_index().rename(columns={dim: "dimension_value"}))
        agg["sme_winner_count"] = 0      # is_sme is empty in real data
        agg["dimension"] = dim
        rows.append(agg)
    df = pd.concat(rows, ignore_index=True)
    df["avg_savings_pct"] = df["avg_savings_pct"].round(1)
    df["avg_tenders"] = df["avg_tenders"].astype(float).round(1)
    return df[["dimension_value", "notice_count", "total_estimated", "total_awarded",
               "avg_savings_pct", "avg_tenders", "sme_winner_count", "dimension"]]


def _cpv_analysis(awards: pd.DataFrame, opportunities: pd.DataFrame, rng) -> pd.DataFrame:
    a = awards.copy()
    a["_sav"] = _savings(a["estimated"], a["awarded_amount"])
    aw = (a.groupby(["cpv_division", "cpv_division_name"]).agg(
        awards_count=("notice_id", "count"),
        total_awarded=("awarded_amount", "sum"),
        avg_savings=("_sav", "mean"),
        avg_competition=("tenders_count", "mean"),
    ).reset_index())
    aw["sme_wins"] = 0
    op = (opportunities.groupby(["cpv_division", "cpv_division_name"]).agg(
        open_notices=("notice_id", "count"),
        total_open_est=("estimated", "sum"),
    ).reset_index())
    df = aw.merge(op, on=["cpv_division", "cpv_division_name"], how="outer")
    df["avg_savings"] = df["avg_savings"].round(1)
    df["avg_competition"] = df["avg_competition"].astype(float).round(1)
    return df.sort_values("total_awarded", ascending=False, na_position="last").reset_index(drop=True)


def write_parquet(out_dir: Path, **kwargs) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "opportunities":       "gold_opportunities",
        "awards":              "gold_awards",
        "market_summary":      "gold_market_summary",
        "cpv_analysis":        "gold_cpv_analysis",
        "notice_enrichment":   "gold_notice_enrichment",
        "bid_win_probability": "gold_bid_win_probability",
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
        print(f"  ✓ {path.name:<32} ({len(df):,} rows, {len(df.columns)} cols)")
