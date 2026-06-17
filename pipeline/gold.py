"""
pipeline/gold.py
─────────────────
Silver → Gold.

Produces 4 business-ready analytical tables:

  gold_opportunities   — open Contract Notices (CN) enriched with buyer & CPV info
  gold_awards          — Contract Award Notices (CAN) with winner, savings %, competition
  gold_market_summary  — aggregated KPIs by country / CPV / proc_type
  gold_cpv_analysis    — CPV-level competition and value stats

All monetary values in EUR. Joins use notice_id as the spine.

Usage:
    python -m pipeline.gold
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SILVER_DIR, GOLD_DIR


SILVER_TABLES = [
    "notices",
    "lots",
    "organisations",
    "cpv_codes",
    "lot_results",
    "contracts",
    "tendering_parties",
    "tenders",
]

DOCUMENTED_MISSING_SOURCE_FIELDS = {
    "notices": ["estimated_value"],
    "tendering_parties": ["tenderer_id"],
    "tenders": ["bid_amount"],
}

SMOKE_REPORT_PATH = GOLD_DIR / "gold_smoke_test_report.md"


def _load(table: str) -> pd.DataFrame:
    p = SILVER_DIR / f"silver_{table}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}. Run silver.py first.")
    return pd.read_parquet(p)


def _load_all() -> dict[str, pd.DataFrame]:
    return {table: _load(table) for table in SILVER_TABLES}


def _non_negative(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    return values.mask(values < 0)


def _clean_date_string(s: pd.Series) -> pd.Series:
    values = s.astype("string").str.slice(0, 10)
    return values.mask(values.isin(["", "None", "nan", "NaT", "<NA>"]))


def _sum_non_negative(s: pd.Series) -> float:
    return _non_negative(s).sum(min_count=1)


def _mean_non_negative(s: pd.Series) -> float:
    return _non_negative(s).mean()


def _unique_join(s: pd.Series) -> str | None:
    values = [str(value) for value in pd.unique(s.dropna()) if str(value)]
    return ", ".join(values) if values else None


def _min_date(s: pd.Series) -> str | None:
    values = s.dropna().astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        return None
    return values.min()


def _null_empty_sum(df: pd.DataFrame, value_col: str, count_col: str) -> pd.DataFrame:
    df.loc[df[count_col].eq(0), value_col] = np.nan
    return df.drop(columns=[count_col])


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


# ── Gold opportunities (CN) ────────────────────────────────────────────────────

def build_opportunities() -> pd.DataFrame:
    notices = _load("notices")
    lots    = _load("lots")

    cn = notices[notices["notice_type"] == "CN"].copy()

    # Aggregate lot-level info per notice
    lot_agg = (
        lots.groupby("notice_id")
        .agg(
            num_lots          = ("lot_id",  "count"),
            total_est_lots    = ("lot_est", "sum"),
            avg_lot_duration  = ("duration_val", "mean"),
        )
        .reset_index()
    )

    df = cn.merge(lot_agg, on="notice_id", how="left")

    # Use estimated amount from notice if lot-level is missing
    df["estimated"] = df["estimated"].fillna(df["total_est_lots"])

    keep = ["notice_id", "pub_date", "buyer_name", "buyer_country", "buyer_org_id",
            "project_title", "proc_type", "procedure_type", "cpv_main",
            "cpv_division", "cpv_division_name", "estimated", "num_lots",
            "avg_lot_duration", "submission_deadline", "framework", "source_file"]
    return df[[c for c in keep if c in df.columns]]


# ── Gold awards (CAN) ──────────────────────────────────────────────────────────

def build_awards() -> pd.DataFrame:
    notices    = _load("notices")
    lots       = _load("lots")
    lot_results = _load("lot_results")
    orgs       = _load("organisations")

    can = notices[notices["notice_type"] == "CAN"].copy()

    # Aggregate lot results per notice
    lr_agg = (
        lot_results.groupby("notice_id")
        .agg(
            num_lots              = ("lot_id",        "count"),
            total_awarded         = ("awarded_amount", "sum"),
            avg_tenders_per_lot   = ("tenders_count",  "mean"),
            total_tenders         = ("tenders_count",  "sum"),
            num_awarded_lots      = ("is_awarded",     "sum"),
        )
        .reset_index()
    )

    # Winner org names
    awarded = lot_results[lot_results["is_awarded"]].copy()
    winner_orgs = (
        awarded.merge(
            orgs[["notice_id", "org_id", "org_name", "org_country", "is_sme"]],
            left_on=["notice_id", "winner_org_id"],
            right_on=["notice_id", "org_id"],
            how="left"
        )
        .groupby("notice_id")
        .agg(
            winner_names     = ("org_name",    lambda x: ", ".join(x.dropna().unique())),
            winner_countries = ("org_country", lambda x: ", ".join(x.dropna().unique())),
            sme_winner       = ("is_sme",      "any"),
        )
        .reset_index()
    )

    # Lot summary
    lot_agg = (
        lots.groupby("notice_id")
        .agg(avg_lot_duration=("duration_val", "mean"))
        .reset_index()
    )

    df = (
        can
        .merge(lr_agg,     on="notice_id", how="left")
        .merge(winner_orgs, on="notice_id", how="left")
        .merge(lot_agg,    on="notice_id", how="left")
    )

    # Prefer lot-level totals over notice-level
    df["estimated"]     = df["estimated"].fillna(df["total_est_lots"] if "total_est_lots" in df else np.nan)
    df["awarded_eur"]   = df["total_awarded_x"] if "total_awarded_x" in df.columns else df["total_awarded"]
    if "total_awarded_x" in df.columns:
        df["awarded_eur"] = df["total_awarded_x"].fillna(df["total_awarded_y"])

    # savings_pct = (estimated - awarded) / estimated × 100
    df["savings_pct"] = np.where(
        df["estimated"] > 0,
        (df["estimated"] - df["awarded_eur"]) / df["estimated"] * 100,
        np.nan,
    )
    df["savings_pct"] = df["savings_pct"].round(1)

    keep = ["notice_id", "pub_date", "buyer_name", "buyer_country", "buyer_org_id",
            "project_title", "proc_type", "cpv_main", "cpv_division", "cpv_division_name",
            "estimated", "awarded_eur", "savings_pct",
            "num_lots", "num_awarded_lots", "avg_tenders_per_lot", "total_tenders",
            "winner_names", "winner_countries", "sme_winner",
            "avg_lot_duration", "source_file"]
    return df[[c for c in keep if c in df.columns]]


# ── Market summary ─────────────────────────────────────────────────────────────

def build_market_summary(awards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ["buyer_country", "cpv_division_name", "proc_type"]:
        agg = (
            awards.groupby(group_col)
            .agg(
                notice_count      = ("notice_id",         "count"),
                total_estimated   = ("estimated",         "sum"),
                total_awarded     = ("awarded_eur",        "sum"),
                avg_savings_pct   = ("savings_pct",       "mean"),
                avg_tenders       = ("avg_tenders_per_lot","mean"),
                sme_winner_count  = ("sme_winner",        "sum"),
            )
            .reset_index()
            .rename(columns={group_col: "dimension_value"})
        )
        agg["dimension"] = group_col
        rows.append(agg)

    df = pd.concat(rows, ignore_index=True)
    df["avg_savings_pct"] = df["avg_savings_pct"].round(1)
    df["avg_tenders"]     = df["avg_tenders"].round(1)
    return df


# ── CPV analysis ───────────────────────────────────────────────────────────────

def build_cpv_analysis(awards: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    aw = (
        awards.groupby(["cpv_division", "cpv_division_name"])
        .agg(
            awards_count    = ("notice_id",          "count"),
            total_awarded   = ("awarded_eur",         "sum"),
            avg_savings     = ("savings_pct",         "mean"),
            avg_competition = ("avg_tenders_per_lot", "mean"),
            sme_wins        = ("sme_winner",          "sum"),
        )
        .reset_index()
    )
    op = (
        opportunities.groupby(["cpv_division", "cpv_division_name"])
        .agg(
            open_notices  = ("notice_id", "count"),
            total_open_est = ("estimated", "sum"),
        )
        .reset_index()
    )
    df = aw.merge(op, on=["cpv_division", "cpv_division_name"], how="outer")
    df["avg_savings"]     = df["avg_savings"].round(1)
    df["avg_competition"] = df["avg_competition"].round(1)
    return df.sort_values("total_awarded", ascending=False)


# ── Starter / smoke-test Gold tables ───────────────────────────────────────────

def build_gold_notices(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    notices = tables["notices"].copy()
    lots = tables["lots"].copy()
    cpv_codes = tables["cpv_codes"]
    lot_results = tables["lot_results"].copy()
    contracts = tables["contracts"].copy()
    tendering_parties = tables["tendering_parties"]
    tenders = tables["tenders"].copy()

    notices["cpv_code"] = notices["cpv_main"]
    lots["lot_est_clean"] = _non_negative(lots["lot_est"])
    lots["duration_val_clean"] = _non_negative(lots["duration_val"])
    lot_results["awarded_amount_clean"] = _non_negative(lot_results["awarded_amount"])
    lot_results["tenders_count_clean"] = _non_negative(lot_results["tenders_count"])
    contracts["contract_value_clean"] = _non_negative(contracts["contract_value"])
    contracts["contract_date_clean"] = _clean_date_string(contracts["contract_date"])
    tenders["tender_value_clean"] = _non_negative(tenders["tender_value"])

    cpv_agg = (
        cpv_codes.groupby("notice_id")
        .agg(
            cpv_codes=("cpv_code", _unique_join),
            num_cpv_codes=("cpv_code", "nunique"),
        )
        .reset_index()
    )
    lot_agg = (
        lots.groupby("notice_id")
        .agg(
            lot_count=("lot_id", "nunique"),
            lot_estimated_total=("lot_est_clean", "sum"),
            lot_estimated_count=("lot_est_clean", "count"),
            avg_lot_duration=("duration_val_clean", "mean"),
        )
        .reset_index()
    )
    lot_agg = _null_empty_sum(lot_agg, "lot_estimated_total", "lot_estimated_count")
    result_agg = (
        lot_results.groupby("notice_id")
        .agg(
            result_count=("lot_result_id", "nunique"),
            awarded_lot_count=("is_awarded", "sum"),
            awarded_amount=("awarded_amount_clean", "sum"),
            awarded_amount_count=("awarded_amount_clean", "count"),
            avg_tenders_count=("tenders_count_clean", "mean"),
        )
        .reset_index()
    )
    result_agg = _null_empty_sum(result_agg, "awarded_amount", "awarded_amount_count")
    contract_agg = (
        contracts.groupby("notice_id")
        .agg(
            contract_count=("contract_id", "nunique"),
            contract_value=("contract_value_clean", "sum"),
            contract_value_count=("contract_value_clean", "count"),
            first_contract_date=("contract_date_clean", "min"),
        )
        .reset_index()
    )
    contract_agg = _null_empty_sum(contract_agg, "contract_value", "contract_value_count")
    party_agg = (
        tendering_parties.groupby("notice_id")
        .agg(tendering_party_count=("party_id", "nunique"))
        .reset_index()
    )
    tender_agg = (
        tenders.groupby("notice_id")
        .agg(
            tender_count=("tender_id", "nunique"),
            tender_value=("tender_value_clean", "sum"),
            tender_value_count=("tender_value_clean", "count"),
        )
        .reset_index()
    )
    tender_agg = _null_empty_sum(tender_agg, "tender_value", "tender_value_count")

    df = (
        notices
        .merge(cpv_agg, on="notice_id", how="left")
        .merge(lot_agg, on="notice_id", how="left")
        .merge(result_agg, on="notice_id", how="left")
        .merge(contract_agg, on="notice_id", how="left")
        .merge(party_agg, on="notice_id", how="left")
        .merge(tender_agg, on="notice_id", how="left")
    )

    keep = [
        "notice_id", "notice_type", "pub_date", "submission_deadline",
        "buyer_org_id", "buyer_name", "buyer_country", "project_title",
        "proc_type", "procedure_type", "cpv_code", "cpv_main", "cpv_codes",
        "cpv_division", "cpv_division_name", "estimated", "total_awarded",
        "lot_count", "lot_estimated_total", "avg_lot_duration",
        "result_count", "awarded_lot_count", "awarded_amount",
        "contract_count", "contract_value", "first_contract_date",
        "tender_count", "tendering_party_count", "avg_tenders_count",
        "tender_value", "framework", "source_file",
    ]
    return df[[col for col in keep if col in df.columns]]


def build_gold_lots(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    notices = tables["notices"].copy()
    lots = tables["lots"]
    lot_results = tables["lot_results"].copy()
    contracts = tables["contracts"].copy()
    tenders = tables["tenders"].copy()

    notices["cpv_code"] = notices["cpv_main"]
    lot_results["awarded_amount_clean"] = _non_negative(lot_results["awarded_amount"])
    lot_results["tenders_count_clean"] = _non_negative(lot_results["tenders_count"])
    contracts["contract_value_clean"] = _non_negative(contracts["contract_value"])
    contracts["contract_date_clean"] = _clean_date_string(contracts["contract_date"])
    tenders["tender_value_clean"] = _non_negative(tenders["tender_value"])

    notice_cols = [
        "notice_id", "notice_type", "pub_date", "buyer_name", "buyer_country",
        "project_title", "proc_type", "procedure_type", "cpv_code",
        "cpv_division", "cpv_division_name", "estimated",
    ]
    result_agg = (
        lot_results.groupby(["notice_id", "lot_id"])
        .agg(
            result_code=("result_code", "first"),
            is_awarded=("is_awarded", "any"),
            winner_org_id=("winner_org_id", "first"),
            tenders_count=("tenders_count_clean", "mean"),
            awarded_amount=("awarded_amount_clean", "sum"),
            awarded_amount_count=("awarded_amount_clean", "count"),
            lot_result_count=("lot_result_id", "nunique"),
        )
        .reset_index()
    )
    result_agg = _null_empty_sum(result_agg, "awarded_amount", "awarded_amount_count")

    contract_lots = contracts.merge(
        lot_results[["notice_id", "lot_result_id", "lot_id"]].drop_duplicates(),
        on=["notice_id", "lot_result_id"],
        how="left",
    )
    contract_agg = (
        contract_lots.dropna(subset=["lot_id"])
        .groupby(["notice_id", "lot_id"])
        .agg(
            contract_value=("contract_value_clean", "sum"),
            contract_value_count=("contract_value_clean", "count"),
            contract_date=("contract_date_clean", "min"),
            contract_count=("contract_id", "nunique"),
        )
        .reset_index()
    )
    contract_agg = _null_empty_sum(contract_agg, "contract_value", "contract_value_count")

    tender_agg = (
        tenders.groupby(["notice_id", "lot_id"])
        .agg(
            tender_count=("tender_id", "nunique"),
            tender_value=("tender_value_clean", "sum"),
            tender_value_count=("tender_value_clean", "count"),
            avg_tender_value=("tender_value_clean", "mean"),
        )
        .reset_index()
    )
    tender_agg = _null_empty_sum(tender_agg, "tender_value", "tender_value_count")

    df = (
        lots
        .merge(notices[notice_cols], on="notice_id", how="left")
        .merge(result_agg, on=["notice_id", "lot_id"], how="left")
        .merge(contract_agg, on=["notice_id", "lot_id"], how="left")
        .merge(tender_agg, on=["notice_id", "lot_id"], how="left")
    )

    keep = [
        "notice_id", "lot_id", "notice_type", "pub_date", "buyer_name",
        "buyer_country", "project_title", "lot_title", "lot_desc",
        "proc_type", "procedure_type", "cpv_code", "cpv_division",
        "cpv_division_name", "estimated", "lot_est", "lot_currency",
        "duration_val", "duration_unit", "award_criteria", "result_code",
        "is_awarded", "winner_org_id", "tenders_count", "awarded_amount",
        "contract_value", "contract_date", "contract_count", "tender_count",
        "tender_value", "avg_tender_value", "lot_result_count",
    ]
    return df[[col for col in keep if col in df.columns]]


def build_gold_awards(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    notices = tables["notices"].copy()
    lots = tables["lots"]
    lot_results = tables["lot_results"].copy()
    contracts = tables["contracts"].copy()
    organisations = tables["organisations"]
    tenders = tables["tenders"].copy()

    notices["cpv_code"] = notices["cpv_main"]
    lot_results["awarded_amount"] = _non_negative(lot_results["awarded_amount"])
    tenders["tender_value"] = _non_negative(tenders["tender_value"])
    contracts["contract_value_clean"] = _non_negative(contracts["contract_value"])
    contracts["contract_date_clean"] = _clean_date_string(contracts["contract_date"])

    awards = lot_results[lot_results["is_awarded"]].copy()
    notice_cols = [
        "notice_id", "notice_type", "pub_date", "buyer_name", "buyer_country",
        "project_title", "proc_type", "procedure_type", "cpv_code",
        "cpv_division", "cpv_division_name", "estimated", "total_awarded",
    ]
    lot_cols = ["notice_id", "lot_id", "lot_title", "lot_desc", "lot_est", "lot_currency"]
    winner_cols = ["notice_id", "org_id", "org_name", "org_country", "is_sme"]

    contract_agg = (
        contracts.groupby(["notice_id", "lot_result_id"])
        .agg(
            contract_value=("contract_value_clean", "sum"),
            contract_value_count=("contract_value_clean", "count"),
            contract_date=("contract_date_clean", "min"),
            contract_count=("contract_id", "nunique"),
        )
        .reset_index()
    )
    contract_agg = _null_empty_sum(contract_agg, "contract_value", "contract_value_count")
    winner_tenders = tenders[["notice_id", "tender_id", "tender_value", "tender_currency"]]

    df = (
        awards
        .merge(notices[notice_cols], on="notice_id", how="left")
        .merge(lots[lot_cols], on=["notice_id", "lot_id"], how="left")
        .merge(
            organisations[winner_cols],
            left_on=["notice_id", "winner_org_id"],
            right_on=["notice_id", "org_id"],
            how="left",
        )
        .merge(contract_agg, on=["notice_id", "lot_result_id"], how="left")
        .merge(
            winner_tenders,
            left_on=["notice_id", "winner_tender_id"],
            right_on=["notice_id", "tender_id"],
            how="left",
        )
    )
    df = df.rename(columns={"org_name": "winner_name", "org_country": "winner_country"})

    keep = [
        "notice_id", "lot_id", "lot_result_id", "notice_type", "pub_date",
        "buyer_name", "buyer_country", "project_title", "lot_title",
        "proc_type", "procedure_type", "cpv_code", "cpv_division",
        "cpv_division_name", "result_code", "is_awarded", "winner_org_id",
        "winner_name", "winner_country", "is_sme", "winner_tender_id",
        "tenders_count", "sme_tenders", "awarded_amount", "award_currency",
        "contract_value", "contract_date", "contract_count", "tender_value",
        "tender_currency", "estimated", "total_awarded", "lot_est",
        "lot_currency",
    ]
    return df[[col for col in keep if col in df.columns]]


def build_gold_country_kpis(gold_notices: pd.DataFrame, gold_awards: pd.DataFrame) -> pd.DataFrame:
    notices = gold_notices.copy()
    notices["buyer_country"] = notices["buyer_country"].astype("string").fillna("Unknown")

    kpis = (
        notices.groupby("buyer_country")
        .agg(
            notice_count=("notice_id", "nunique"),
            contract_notice_count=("notice_type", lambda s: int((s == "CN").sum())),
            award_notice_count=("notice_type", lambda s: int((s == "CAN").sum())),
            lot_count=("lot_count", "sum"),
            result_count=("result_count", "sum"),
            awarded_lot_count=("awarded_lot_count", "sum"),
            contract_count=("contract_count", "sum"),
            tender_count=("tender_count", "sum"),
            total_estimated=("estimated", _sum_non_negative),
            total_lot_estimated=("lot_estimated_total", _sum_non_negative),
            total_awarded_amount=("awarded_amount", _sum_non_negative),
            total_contract_value=("contract_value", _sum_non_negative),
            total_tender_value=("tender_value", _sum_non_negative),
            avg_tenders_count=("avg_tenders_count", "mean"),
        )
        .reset_index()
    )

    awards = gold_awards.copy()
    awards["buyer_country"] = awards["buyer_country"].astype("string").fillna("Unknown")
    award_kpis = (
        awards.groupby("buyer_country")
        .agg(
            award_rows=("lot_result_id", "count"),
            winner_count=("winner_org_id", "nunique"),
        )
        .reset_index()
    )

    return (
        kpis.merge(award_kpis, on="buyer_country", how="left")
        .fillna({"award_rows": 0, "winner_count": 0})
        .sort_values(["total_contract_value", "notice_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_gold_cpv_kpis(gold_notices: pd.DataFrame, gold_awards: pd.DataFrame) -> pd.DataFrame:
    notices = gold_notices.copy()
    notices["cpv_division"] = notices["cpv_division"].astype("string").fillna("Unknown")
    notices["cpv_division_name"] = notices["cpv_division_name"].astype("string").fillna("Unknown")

    kpis = (
        notices.groupby(["cpv_division", "cpv_division_name"])
        .agg(
            notice_count=("notice_id", "nunique"),
            lot_count=("lot_count", "sum"),
            result_count=("result_count", "sum"),
            awarded_lot_count=("awarded_lot_count", "sum"),
            contract_count=("contract_count", "sum"),
            tender_count=("tender_count", "sum"),
            buyer_country_count=("buyer_country", "nunique"),
            total_estimated=("estimated", _sum_non_negative),
            total_lot_estimated=("lot_estimated_total", _sum_non_negative),
            total_awarded_amount=("awarded_amount", _sum_non_negative),
            total_contract_value=("contract_value", _sum_non_negative),
            total_tender_value=("tender_value", _sum_non_negative),
            avg_tenders_count=("avg_tenders_count", "mean"),
        )
        .reset_index()
    )

    awards = gold_awards.copy()
    awards["cpv_division"] = awards["cpv_division"].astype("string").fillna("Unknown")
    awards["cpv_division_name"] = awards["cpv_division_name"].astype("string").fillna("Unknown")
    award_kpis = (
        awards.groupby(["cpv_division", "cpv_division_name"])
        .agg(
            award_rows=("lot_result_id", "count"),
            winner_count=("winner_org_id", "nunique"),
        )
        .reset_index()
    )

    return (
        kpis.merge(award_kpis, on=["cpv_division", "cpv_division_name"], how="left")
        .fillna({"award_rows": 0, "winner_count": 0})
        .sort_values(["total_contract_value", "notice_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _missing_source_fields(tables: dict[str, pd.DataFrame]) -> list[list[str]]:
    rows: list[list[str]] = []
    for table, fields in DOCUMENTED_MISSING_SOURCE_FIELDS.items():
        missing = [field for field in fields if field not in tables[table].columns]
        if missing:
            rows.append([table, ", ".join(missing), "Documented missing from validated Silver schema"])
    return rows


def _smoke_checks(outputs: dict[str, pd.DataFrame]) -> list[list[object]]:
    checks: list[list[object]] = []

    for name, df in outputs.items():
        checks.append(["PASS" if len(df) else "WARN", name, "row count", f"{len(df):,} rows"])

    key_checks = {
        "gold_notices": ["notice_id"],
        "gold_lots": ["notice_id", "lot_id"],
        "gold_awards": ["notice_id", "lot_result_id"],
        "gold_country_kpis": ["buyer_country"],
        "gold_cpv_kpis": ["cpv_division", "cpv_division_name"],
    }
    for name, key in key_checks.items():
        df = outputs[name]
        duplicate_rows = int(df.duplicated(subset=key, keep=False).sum())
        checks.append([
            "PASS" if duplicate_rows == 0 else "FAIL",
            name,
            "natural key uniqueness",
            f"{key}; duplicate rows={duplicate_rows:,}",
        ])

    awards = outputs["gold_awards"]
    non_awarded = int((awards["is_awarded"] != True).sum()) if "is_awarded" in awards.columns else len(awards)
    checks.append([
        "PASS" if non_awarded == 0 else "FAIL",
        "gold_awards",
        "awarded rows only",
        f"non-awarded rows={non_awarded:,}",
    ])

    for name, col in [
        ("gold_awards", "awarded_amount"),
        ("gold_awards", "contract_value"),
        ("gold_awards", "tender_value"),
        ("gold_lots", "contract_value"),
        ("gold_lots", "tender_value"),
    ]:
        df = outputs[name]
        if col not in df.columns:
            checks.append(["WARN", name, f"{col} non-negative", "column not present"])
            continue
        negative = int((_non_negative(df[col]).isna() & pd.to_numeric(df[col], errors="coerce").lt(0)).sum())
        checks.append([
            "PASS" if negative == 0 else "WARN",
            name,
            f"{col} non-negative",
            f"negative rows={negative:,}",
        ])

    return checks


def build_smoke_report(
    tables: dict[str, pd.DataFrame],
    outputs: dict[str, pd.DataFrame],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_rows = [
        [f"silver_{table}.parquet", f"{len(df):,}", ", ".join(df.columns)]
        for table, df in tables.items()
    ]
    output_rows = [
        [f"{name}.parquet", f"{len(df):,}", ", ".join(df.columns)]
        for name, df in outputs.items()
    ]
    missing_rows = _missing_source_fields(tables)
    if not missing_rows:
        missing_rows = [["none", "none", "All documented fields are present"]]

    sections = [
        "# Gold Smoke Test Report",
        f"Generated at: {generated_at}",
        "This is a starter/smoke-test Gold layer for dashboard readiness. It is intentionally simple and not a replacement for final teammate-owned Gold logic.",
        "Only validated Silver parquet files from `data/silver/` are used as inputs.",
        "## Silver Inputs",
        _md_table(["file", "rows", "columns"], input_rows),
        "## Gold Outputs",
        _md_table(["file", "rows", "columns"], output_rows),
        "## Missing Source Fields",
        _md_table(["table", "missing_fields", "note"], missing_rows),
        "## Smoke Checks",
        _md_table(["status", "table", "check", "detail"], _smoke_checks(outputs)),
    ]
    return "\n\n".join(sections) + "\n"


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run() -> None:
    print(f"\n{'─'*55}")
    print(f"  Gold layer starter / smoke test")
    print(f"{'─'*55}")

    tables = _load_all()

    gold_notices = build_gold_notices(tables)
    gold_lots = build_gold_lots(tables)
    gold_awards = build_gold_awards(tables)
    gold_country_kpis = build_gold_country_kpis(gold_notices, gold_awards)
    gold_cpv_kpis = build_gold_cpv_kpis(gold_notices, gold_awards)

    outputs = {
        "gold_notices": gold_notices,
        "gold_lots": gold_lots,
        "gold_awards": gold_awards,
        "gold_country_kpis": gold_country_kpis,
        "gold_cpv_kpis": gold_cpv_kpis,
    }

    for name, df in outputs.items():
        out = GOLD_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"  ✓ {out.name:<28} ({len(df):,} rows)")

    report = build_smoke_report(tables, outputs)
    SMOKE_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  ✓ {SMOKE_REPORT_PATH.name:<28} ({len(report):,} chars)")

    missing_fields = _missing_source_fields(tables)
    if missing_fields:
        print("\n  Missing source fields documented")
        for table, fields, _note in missing_fields:
            print(f"  - {table}: {fields}")

    print(f"\n✓ Starter Gold smoke test complete → {GOLD_DIR}")
    print("  Note: this is a starter/smoke-test Gold layer, not final teammate-owned Gold logic.")


if __name__ == "__main__":
    run()
