"""
pipeline/silver.py
──────────────────
Bronze → Silver.

Transformations applied:
  - Strip timezone suffixes from dates (+01:00, Z)
  - Cast amounts to float, counts to int
  - Filter by DATE_FILTER if set in config
  - Deduplicate buyers on org_id
  - Add cpv_division (first 2 digits) and cpv_division_name
  - Standardise procedure_type and proc_type codes

Usage:
    python -m pipeline.silver
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRONZE_DIR, SILVER_DIR, DATE_FILTER, CPV_DIVISIONS


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_parquet_safe(path: Path) -> pd.DataFrame:
    """Read Parquet with fallbacks for files pyarrow cannot decode."""
    print(f"  [read] {path}")

    errors = []

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        errors.append(("pandas default", exc))
        print(f"    pandas default failed: {exc}")

    try:
        return pd.read_parquet(path, engine="fastparquet")
    except ImportError as exc:
        errors.append(("pandas fastparquet", exc))
        print("    fastparquet is not installed.")
        print("    Install with: python -m pip install fastparquet")
    except Exception as exc:
        errors.append(("pandas fastparquet", exc))
        print(f"    pandas fastparquet failed: {exc}")

    try:
        import duckdb
    except ImportError as exc:
        errors.append(("duckdb", exc))
        print("    duckdb is not installed.")
        print("    Install with: python -m pip install duckdb")
    else:
        try:
            return duckdb.read_parquet(str(path)).df()
        except Exception as exc:
            errors.append(("duckdb", exc))
            print(f"    duckdb failed: {exc}")

    details = "\n".join(f"  - {name}: {exc}" for name, exc in errors)
    raise OSError(f"Could not read Parquet file: {path}\n{details}")


def _clean_date(s: pd.Series) -> pd.Series:
    """Remove timezone suffixes and parse to date."""
    return (
        s.str.replace(r"[\+\-]\d{2}:\d{2}$", "", regex=True)
         .str.replace("Z$", "", regex=True)
         .str[:10]  # keep YYYY-MM-DD
    )


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _cpv_division(s: pd.Series) -> pd.Series:
    """First 2 digits of CPV code."""
    return s.str[:2].where(s.notna())


def _fill_null_from(df: pd.DataFrame, target: str, source: str) -> pd.Series:
    if target not in df.columns:
        df[target] = pd.NA
    return df[target].where(df[target].notna(), df[source])


NATURAL_KEYS = {
    "notices": ["notice_id"],
    "lots": ["notice_id", "lot_id"],
    "organisations": ["notice_id", "org_id"],
    "cpv_codes": ["notice_id", "cpv_code"],
    "lot_results": ["notice_id", "lot_result_id"],
    "contracts": ["notice_id", "lot_result_id", "contract_id"],
    "tendering_parties": ["notice_id", "party_id"],
    "tenders": ["notice_id", "tender_id"],
}

QUALITY_VALUE_COLUMNS = [
    "estimated",
    "total_awarded",
    "lot_est",
    "duration_val",
    "awarded_amount",
    "contract_value",
    "tender_value",
    "subcontracting",
]


def _dedupe_on_key(df: pd.DataFrame, table: str) -> pd.DataFrame:
    key = NATURAL_KEYS.get(table)
    if not key:
        return df

    missing = [col for col in key if col not in df.columns]
    if missing:
        print(f"  [warn] {table}: cannot deduplicate; missing key columns {missing}")
        return df

    duplicate_rows = int(df.duplicated(subset=key, keep=False).sum())
    if duplicate_rows == 0:
        return df

    work = df.copy()
    work["_silver_source_order"] = range(len(work))

    quality = work.notna().sum(axis=1).astype("int64")
    for col in QUALITY_VALUE_COLUMNS:
        if col not in work.columns:
            continue

        numeric = pd.to_numeric(work[col], errors="coerce")
        quality += (numeric.notna() & numeric.ge(0)).astype("int64") * 3
        quality -= (numeric.notna() & numeric.lt(0)).astype("int64") * 3

    work["_silver_quality_score"] = quality

    temp_cols = ["_silver_source_order", "_silver_quality_score"]
    sort_cols = key + ["_silver_quality_score"]
    for col in ["pub_date", "contract_date", "submission_deadline"]:
        if col in work.columns:
            sort_col = f"_silver_sort_{col}"
            work[sort_col] = pd.to_datetime(work[col], errors="coerce")
            temp_cols.append(sort_col)
            sort_cols.append(sort_col)

    sort_cols.append("_silver_source_order")
    before = len(work)
    work = (
        work.sort_values(sort_cols, kind="mergesort")
            .drop_duplicates(subset=key, keep="last")
            .sort_values("_silver_source_order", kind="mergesort")
            .drop(columns=temp_cols)
            .reset_index(drop=True)
    )
    print(f"  [dedupe] {table}: removed {before - len(work):,} rows on {key}")
    return work


# ── Per-table transforms ───────────────────────────────────────────────────────

def transform_notices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pub_date"]     = _clean_date(df["publication_date"])
    df["estimated"]    = _to_float(df["est_amount"])
    if "estimated_value" in df.columns:
        df["estimated_value"] = _to_float(df["estimated_value"])
    df["estimated_value"] = _fill_null_from(df, "estimated_value", "estimated")
    df["total_awarded"] = _to_float(df["total_awarded"])
    df["cpv_division"] = _cpv_division(df["cpv_main"])
    df["cpv_division_name"] = df["cpv_division"].map(CPV_DIVISIONS)

    if DATE_FILTER:
        df = df[df["pub_date"].str.startswith(DATE_FILTER, na=False)]

    # Standardise procurement type labels
    type_map = {
        "services": "Services", "supplies": "Supplies",
        "works": "Works", "service": "Services", "supply": "Supplies",
    }
    df["proc_type"] = df["proc_type"].str.lower().map(type_map).fillna(df["proc_type"])

    keep = ["notice_id", "notice_type", "pub_date", "buyer_org_id", "buyer_name",
            "buyer_country", "project_title", "proc_type", "procedure_type",
            "cpv_main", "cpv_division", "cpv_division_name", "estimated", "estimated_value", "total_awarded",
            "submission_deadline", "framework", "source_file"]
    return df[[c for c in keep if c in df.columns]]


def transform_lots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lot_est"]      = _to_float(df["lot_est"])
    df["duration_val"] = _to_float(df["duration_val"])
    return df


def transform_organisations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_sme"] = df["sme"].str.lower().isin(["sme", "small", "medium", "true", "yes"])
    return _dedupe_on_key(df, "organisations")


def transform_lot_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["awarded_amount"] = _to_float(df["awarded_amount"])
    df["tenders_count"]  = _to_int(df["tenders_count"])
    df["sme_tenders"]    = _to_int(df["sme_tenders"])
    result_code = df["result_code"].astype("string").str.lower()
    df["is_awarded"]     = result_code.eq("selec-w").fillna(False)
    return df


def transform_tenders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tender_value"]    = _to_float(df["tender_value"])
    if "bid_amount" in df.columns:
        df["bid_amount"] = _to_float(df["bid_amount"])
    df["bid_amount"] = _fill_null_from(df, "bid_amount", "tender_value")
    df["subcontracting"]  = _to_float(df["subcontracting"])
    return df


def transform_contracts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    contract_value = _to_float(df["contract_value"])
    negative_value = contract_value < 0
    df["contract_value_raw"] = contract_value
    df["contract_value_was_negative_source"] = negative_value.fillna(False)
    df["contract_value"] = contract_value.mask(negative_value)
    df["contract_date"]  = _clean_date(df["contract_date"])
    return df


def transform_cpv_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpv_division"] = _cpv_division(df["cpv_code"].astype("string"))
    df["cpv_division_name"] = df["cpv_division"].map(CPV_DIVISIONS)
    return df


def transform_tendering_parties(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenderer_id"] = _fill_null_from(df, "tenderer_id", "party_id")
    return df


def enrich_notice_buyers(notices: pd.DataFrame, organisations: pd.DataFrame) -> pd.DataFrame:
    required_notice_cols = ["notice_id", "buyer_org_id"]
    required_org_cols = ["notice_id", "org_id", "org_name", "org_country"]

    missing_notice = [col for col in required_notice_cols if col not in notices.columns]
    missing_org = [col for col in required_org_cols if col not in organisations.columns]
    if missing_notice or missing_org:
        print(
            "  [warn] notices: buyer enrichment skipped; "
            f"missing notice columns={missing_notice or 'none'}, "
            f"missing organisation columns={missing_org or 'none'}"
        )
        return notices

    orgs = organisations[required_org_cols].copy()
    orgs = _dedupe_on_key(orgs, "organisations")
    orgs = orgs.rename(
        columns={
            "org_id": "buyer_org_id",
            "org_name": "_buyer_org_name",
            "org_country": "_buyer_org_country",
        }
    )

    enriched = notices.merge(orgs, on=["notice_id", "buyer_org_id"], how="left")

    fill_map = {
        "buyer_name": "_buyer_org_name",
        "buyer_country": "_buyer_org_country",
    }
    details = []
    for target, source in fill_map.items():
        if target not in enriched.columns:
            enriched[target] = pd.NA

        before = int(enriched[target].notna().sum())
        enriched[target] = enriched[target].astype("object").where(enriched[target].notna(), enriched[source])
        after = int(enriched[target].notna().sum())
        details.append(f"{target}: +{after - before:,}")

    enriched = enriched.drop(columns=["_buyer_org_name", "_buyer_org_country"])
    print(f"  [enrich] notices buyers from organisations ({'; '.join(details)})")
    return enriched


def load_organisations_for_notice_enrichment() -> pd.DataFrame | None:
    src = BRONZE_DIR / "bronze_organisations.parquet"
    if not src.exists():
        print("  [warn] notices: buyer enrichment skipped; bronze_organisations.parquet not found")
        return None

    organisations = read_parquet_safe(src)
    organisations = transform_organisations(organisations)
    organisations = _dedupe_on_key(organisations, "organisations")
    return organisations


# ── Orchestrator ───────────────────────────────────────────────────────────────

TRANSFORMS = {
    "notices":         transform_notices,
    "lots":            transform_lots,
    "organisations":   transform_organisations,
    "lot_results":     transform_lot_results,
    "tenders":         transform_tenders,
    "contracts":       transform_contracts,
    "cpv_codes":       transform_cpv_codes,
    "tendering_parties": transform_tendering_parties,
}


def run() -> None:
    print(f"\n{'─'*55}")
    print(f"  Silver layer")
    print(f"{'─'*55}")

    for table, fn in TRANSFORMS.items():
        src = BRONZE_DIR / f"bronze_{table}.parquet"
        if not src.exists():
            print(f"  [skip] bronze_{table}.parquet not found")
            continue

        df = read_parquet_safe(src)
        df = fn(df)
        df = _dedupe_on_key(df, table)
        if table == "notices":
            organisations = load_organisations_for_notice_enrichment()
            if organisations is not None:
                df = enrich_notice_buyers(df, organisations)
        out = SILVER_DIR / f"silver_{table}.parquet"
        df.to_parquet(out, index=False)
        print(f"  ✓ silver_{table}.parquet  ({len(df):,} rows)")

    print(f"\n✓ Silver complete → {SILVER_DIR}")


if __name__ == "__main__":
    run()
