"""
pipeline/validate_silver.py
---------------------------
Validate the Silver layer and write a data quality report.

Scope is intentionally limited to:
    - data/silver/
    - Silver Parquet tables created by pipeline/silver.py

Usage:
    python -m pipeline.validate_silver
"""
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SILVER_DIR


REPORT_PATH = SILVER_DIR / "silver_data_quality_report.md"

EXPECTED_SCHEMAS = {
    "notices": [
        "notice_id",
        "notice_type",
        "pub_date",
        "buyer_org_id",
        "buyer_name",
        "buyer_country",
        "project_title",
        "proc_type",
        "procedure_type",
        "cpv_main",
        "cpv_division",
        "cpv_division_name",
        "estimated",
        "estimated_value",
        "est_currency",
        "total_awarded",
        "submission_deadline",
        "framework",
        "source_file",
    ],
    "lots": [
        "notice_id",
        "lot_id",
        "lot_title",
        "lot_desc",
        "lot_est",
        "lot_currency",
        "duration_val",
        "duration_unit",
        "duration_months",
        "award_criteria",
    ],
    "organisations": [
        "notice_id",
        "org_id",
        "org_name",
        "org_country",
        "org_role",
        "sme",
        "is_sme",
        "nuts_code",
    ],
    "cpv_codes": [
        "notice_id",
        "cpv_code",
        "cpv_division",
        "cpv_division_name",
    ],
    "lot_results": [
        "notice_id",
        "lot_result_id",
        "lot_id",
        "result_code",
        "awarded_amount",
        "award_currency",
        "tenders_count",
        "sme_tenders",
        "winner_tender_id",
        "winner_org_id",
        "is_awarded",
        "result_status",
    ],
    "contracts": [
        "notice_id",
        "lot_result_id",
        "contract_id",
        "contract_date",
        "contract_value",
        "contract_currency",
    ],
    "tendering_parties": [
        "notice_id",
        "party_id",
        "tender_ids",
        "num_tenders",
    ],
    "tenders": [
        "notice_id",
        "tender_id",
        "lot_id",
        "tendering_party_id",
        "tender_value",
        "tender_currency",
        "rank",
        "is_ranked",
        "subcontracting",
    ],
}

OPTIONAL_COLUMNS = {
    "tendering_parties": ["tenderer_id", "tenderer_ids", "is_lead", "num_tenderers"],
    "tenders": ["bid_amount"],
}

PRIMARY_KEYS = {
    "notices": ["notice_id"],
    "lots": ["notice_id", "lot_id"],
    "organisations": ["notice_id", "org_id"],
    "cpv_codes": ["notice_id", "cpv_code"],
    "lot_results": ["notice_id", "lot_result_id"],
    "contracts": ["notice_id", "lot_result_id", "contract_id"],
    "tendering_parties": ["notice_id", "party_id"],
    "tenders": ["notice_id", "tender_id"],
}

CRITICAL_COLUMNS = {
    "notices": ["notice_id", "notice_type", "pub_date", "project_title", "source_file"],
    "lots": ["notice_id", "lot_id"],
    "organisations": ["notice_id", "org_id", "org_name", "org_country"],
    "cpv_codes": ["notice_id", "cpv_code"],
    "lot_results": ["notice_id", "lot_result_id", "lot_id"],
    "contracts": ["notice_id", "lot_result_id", "contract_id"],
    "tendering_parties": ["notice_id", "party_id"],
    "tenders": ["notice_id", "tender_id", "lot_id", "tendering_party_id"],
}

IMPORTANT_ML_FIELDS = [
    ("winner_org_id", "lot_results", "winner_org_id"),
    ("tenders_count", "lot_results", "tenders_count"),
    ("tenderer_id", "tendering_parties", "tenderer_id"),
    ("awarded_amount", "lot_results", "awarded_amount"),
    ("estimated_value", "notices", "estimated_value"),
    ("contract_value", "contracts", "contract_value"),
    ("contract_date", "contracts", "contract_date"),
    ("bid_amount", "tenders", "bid_amount"),
    ("tender_value", "tenders", "tender_value"),
]

EXPECTED_RESULT_STATUSES = {
    "selec-w": "Awarded",
    "clos-nw": "Closed without winner",
    "open-nw": "Open without winner",
    "unpublished": "Unpublished",
}

RESULT_STATUS_COLUMNS = [
    "result_status",
    "result",
    "result_code",
    "lot_result",
    "status",
    "award_status",
    "result_status_code",
]

RESULT_STATUS_CODE_COLUMNS = {"result_code", "result_status_code"}


@dataclass
class Check:
    status: str
    table: str
    check: str
    detail: str


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    return "WARN" if warn else "FAIL"


def _missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def _first_existing_column(df: pd.DataFrame, columns: list[str]) -> str | None:
    return next((col for col in columns if col in df.columns), None)


def _columns_available(
    checks: list[Check],
    table: str,
    df: pd.DataFrame,
    columns: list[str],
    check_name: str,
) -> bool:
    missing = _missing_columns(df, columns)
    if not missing:
        return True

    checks.append(Check("WARN", table, check_name, f"skipped; missing columns={missing}"))
    return False


def _tables_have_columns(
    checks: list[Check],
    tables: dict[str, pd.DataFrame],
    requirements: dict[str, list[str]],
    report_table: str,
    check_name: str,
) -> bool:
    missing: list[str] = []
    for table, columns in requirements.items():
        if table not in tables:
            missing.append(f"{table}=MISSING_TABLE")
            continue

        missing.extend(f"{table}.{col}" for col in _missing_columns(tables[table], columns))

    if not missing:
        return True

    checks.append(Check("WARN", report_table, check_name, f"skipped; missing columns={missing}"))
    return False


def _invalid_numeric_below(series: pd.Series, minimum: int | float) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    non_numeric = series.notna() & numeric.isna()
    below_minimum = numeric < minimum
    return int((non_numeric | below_minimum).sum())


def _fmt_int(value: int | float) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_tables() -> tuple[dict[str, pd.DataFrame], list[Check]]:
    tables: dict[str, pd.DataFrame] = {}
    checks: list[Check] = []

    for table in EXPECTED_SCHEMAS:
        path = SILVER_DIR / f"silver_{table}.parquet"
        if not path.exists():
            checks.append(Check("FAIL", table, "file exists", f"Missing {path.name}"))
            continue

        try:
            tables[table] = pd.read_parquet(path)
            checks.append(Check("PASS", table, "file readable", f"{path.name} loaded"))
        except Exception as exc:
            checks.append(Check("FAIL", table, "file readable", f"{type(exc).__name__}: {exc}"))

    return tables, checks


def _check_schemas(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []

    for table, expected in EXPECTED_SCHEMAS.items():
        if table not in tables:
            continue

        actual = list(tables[table].columns)
        optional = OPTIONAL_COLUMNS.get(table, [])
        allowed = expected + optional
        missing = [col for col in expected if col not in actual]
        extra = [col for col in actual if col not in allowed]
        optional_present = [col for col in optional if col in actual]
        if not missing and not extra:
            detail = f"{len(actual)} columns"
            if optional_present:
                detail += f"; optional present={optional_present}"
            checks.append(Check("PASS", table, "schema", detail))
        elif missing:
            detail = f"missing={missing}; unexpected={extra or 'none'}"
            checks.append(Check("WARN", table, "schema", detail))
        else:
            detail = f"missing=none; unexpected={extra}"
            checks.append(Check("WARN", table, "schema", detail))

    return checks


def _check_primary_keys(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []

    for table, keys in PRIMARY_KEYS.items():
        if table not in tables:
            continue

        df = tables[table]
        missing_keys = [key for key in keys if key not in df.columns]
        if missing_keys:
            checks.append(Check("FAIL", table, "primary key columns", f"Missing {missing_keys}"))
            continue

        duplicate_count = int(df.duplicated(subset=keys, keep=False).sum())
        null_key_rows = int(df[keys].isna().any(axis=1).sum())
        if duplicate_count == 0 and null_key_rows == 0:
            detail = f"unique key: {', '.join(keys)}"
            checks.append(Check("PASS", table, "primary key", detail))
        else:
            detail = f"duplicate rows={duplicate_count:,}; null-key rows={null_key_rows:,}"
            checks.append(Check("FAIL", table, "primary key", detail))

    return checks


def _check_critical_completeness(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []

    for table, columns in CRITICAL_COLUMNS.items():
        if table not in tables:
            continue

        df = tables[table]
        row_count = len(df)
        for col in columns:
            if col not in df.columns:
                checks.append(Check("FAIL", table, f"critical completeness: {col}", "missing column"))
                continue
            null_count = int(df[col].isna().sum())
            null_pct = null_count / row_count if row_count else 0
            status = _status(null_count == 0, warn=null_pct < 0.01)
            detail = f"nulls={null_count:,} ({null_pct:.1%})"
            checks.append(Check(status, table, f"critical completeness: {col}", detail))

    return checks


def _check_references(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []
    if "notices" not in tables:
        return checks

    if not _columns_available(checks, "notices", tables["notices"], ["notice_id"], "notice_id reference"):
        return checks

    notice_ids = set(tables["notices"]["notice_id"].dropna().astype(str))
    for table, df in tables.items():
        if table == "notices":
            continue
        if not _columns_available(checks, table, df, ["notice_id"], "notice_id reference"):
            continue
        missing = int((~df["notice_id"].astype(str).isin(notice_ids)).sum())
        status = _status(missing == 0)
        checks.append(Check(status, table, "notice_id reference", f"unmatched rows={missing:,}"))

    if {"notices", "organisations"}.issubset(tables) and _tables_have_columns(
        checks,
        tables,
        {
            "notices": ["notice_id", "buyer_org_id"],
            "organisations": ["notice_id", "org_id"],
        },
        "notices",
        "buyer organisation reference",
    ):
        notices = tables["notices"].dropna(subset=["notice_id", "buyer_org_id"])
        org_keys = set(
            tables["organisations"][["notice_id", "org_id"]]
            .dropna()
            .astype(str)
            .agg("|".join, axis=1)
        )
        buyer_keys = notices[["notice_id", "buyer_org_id"]].astype(str).agg("|".join, axis=1)
        missing = int((~buyer_keys.isin(org_keys)).sum())
        status = _status(missing == 0, warn=missing <= 25)
        checks.append(Check(status, "notices", "buyer organisation reference", f"unmatched rows={missing:,}"))

    if {"lots", "lot_results"}.issubset(tables) and _tables_have_columns(
        checks,
        tables,
        {
            "lots": ["notice_id", "lot_id"],
            "lot_results": ["notice_id", "lot_id"],
        },
        "lot_results",
        "lot reference",
    ):
        lot_keys = set(
            tables["lots"][["notice_id", "lot_id"]]
            .dropna()
            .astype(str)
            .agg("|".join, axis=1)
        )
        result_lots = tables["lot_results"].dropna(subset=["notice_id", "lot_id"])
        result_keys = result_lots[["notice_id", "lot_id"]].astype(str).agg("|".join, axis=1)
        missing = int((~result_keys.isin(lot_keys)).sum())
        status = _status(missing == 0)
        checks.append(Check(status, "lot_results", "lot reference", f"unmatched rows={missing:,}"))

    if {"lots", "tenders"}.issubset(tables) and _tables_have_columns(
        checks,
        tables,
        {
            "lots": ["notice_id", "lot_id"],
            "tenders": ["notice_id", "lot_id"],
        },
        "tenders",
        "lot reference",
    ):
        lot_keys = set(
            tables["lots"][["notice_id", "lot_id"]]
            .dropna()
            .astype(str)
            .agg("|".join, axis=1)
        )
        tenders = tables["tenders"].dropna(subset=["notice_id", "lot_id"])
        tender_lot_keys = tenders[["notice_id", "lot_id"]].astype(str).agg("|".join, axis=1)
        missing = int((~tender_lot_keys.isin(lot_keys)).sum())
        status = _status(missing == 0)
        checks.append(Check(status, "tenders", "lot reference", f"unmatched rows={missing:,}"))

    if {"tendering_parties", "tenders"}.issubset(tables) and _tables_have_columns(
        checks,
        tables,
        {
            "tendering_parties": ["notice_id", "party_id"],
            "tenders": ["notice_id", "tendering_party_id"],
        },
        "tenders",
        "tendering party reference",
    ):
        party_keys = set(
            tables["tendering_parties"][["notice_id", "party_id"]]
            .dropna()
            .astype(str)
            .agg("|".join, axis=1)
        )
        tenders = tables["tenders"].dropna(subset=["notice_id", "tendering_party_id"])
        tender_party_keys = (
            tenders[["notice_id", "tendering_party_id"]]
            .astype(str)
            .agg("|".join, axis=1)
        )
        missing = int((~tender_party_keys.isin(party_keys)).sum())
        status = _status(missing == 0)
        checks.append(Check(status, "tenders", "tendering party reference", f"unmatched rows={missing:,}"))

    if {"lot_results", "tenders"}.issubset(tables) and "winner_tender_id" in tables["lot_results"].columns:
        if _tables_have_columns(
            checks,
            tables,
            {
                "lot_results": ["notice_id", "winner_tender_id"],
                "tenders": ["notice_id", "tender_id"],
            },
            "lot_results",
            "winner tender reference",
        ):
            tender_keys = set(
                tables["tenders"][["notice_id", "tender_id"]]
                .dropna()
                .astype(str)
                .agg("|".join, axis=1)
            )
            winners = tables["lot_results"].dropna(subset=["notice_id", "winner_tender_id"])
            winner_keys = winners[["notice_id", "winner_tender_id"]].astype(str).agg("|".join, axis=1)
            missing = int((~winner_keys.isin(tender_keys)).sum())
            status = _status(missing == 0)
            checks.append(Check(status, "lot_results", "winner tender reference", f"unmatched rows={missing:,}"))

    if {"tendering_parties", "tenders"}.issubset(tables) and "tender_ids" in tables["tendering_parties"].columns:
        if _tables_have_columns(
            checks,
            tables,
            {
                "tendering_parties": ["notice_id", "tender_ids"],
                "tenders": ["notice_id", "tender_id"],
            },
            "tendering_parties",
            "tender_ids reference",
        ):
            tender_keys = set(
                tables["tenders"][["notice_id", "tender_id"]]
                .dropna()
                .astype(str)
                .agg("|".join, axis=1)
            )
            refs = []
            party_rows = tables["tendering_parties"].dropna(subset=["notice_id", "tender_ids"])
            for notice_id, tender_ids in party_rows[["notice_id", "tender_ids"]].astype(str).itertuples(index=False):
                refs.extend(
                    f"{notice_id}|{tender_id.strip()}"
                    for tender_id in tender_ids.split(",")
                    if tender_id.strip()
                )
            missing = sum(1 for ref in refs if ref not in tender_keys)
            status = _status(missing == 0)
            checks.append(Check(status, "tendering_parties", "tender_ids reference", f"unmatched ids={missing:,}"))

    if {"lot_results", "contracts"}.issubset(tables) and _tables_have_columns(
        checks,
        tables,
        {
            "lot_results": ["notice_id", "lot_result_id"],
            "contracts": ["notice_id", "lot_result_id"],
        },
        "contracts",
        "lot_result reference",
    ):
        result_keys = set(
            tables["lot_results"][["notice_id", "lot_result_id"]]
            .dropna()
            .astype(str)
            .agg("|".join, axis=1)
        )
        contracts = tables["contracts"].dropna(subset=["notice_id", "lot_result_id"])
        contract_keys = contracts[["notice_id", "lot_result_id"]].astype(str).agg("|".join, axis=1)
        missing = int((~contract_keys.isin(result_keys)).sum())
        status = _status(missing == 0)
        checks.append(Check(status, "contracts", "lot_result reference", f"unmatched rows={missing:,}"))

    return checks


def _check_domains(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []

    if "notices" in tables:
        notices = tables["notices"]
        if _columns_available(checks, "notices", notices, ["cpv_main"], "cpv_main format"):
            invalid_cpv = int(
                notices["cpv_main"].dropna().astype(str).str.match(r"^\d{8}$").eq(False).sum()
            )
            checks.append(Check(_status(invalid_cpv == 0), "notices", "cpv_main format", f"invalid rows={invalid_cpv:,}"))

        if _columns_available(checks, "notices", notices, ["cpv_main", "cpv_division"], "cpv_division derivation"):
            invalid_div = int(
                notices.dropna(subset=["cpv_main", "cpv_division"])
                .apply(lambda row: str(row["cpv_main"])[:2] != str(row["cpv_division"]), axis=1)
                .sum()
            )
            checks.append(Check(_status(invalid_div == 0), "notices", "cpv_division derivation", f"invalid rows={invalid_div:,}"))

        if _columns_available(checks, "notices", notices, ["pub_date"], "pub_date parsed"):
            bad_dates = int((notices["pub_date"].isna()).sum())
            checks.append(Check(_status(bad_dates == 0), "notices", "pub_date parsed", f"null dates={bad_dates:,}"))

    if "cpv_codes" in tables:
        cpv = tables["cpv_codes"]
        if _columns_available(checks, "cpv_codes", cpv, ["cpv_code"], "cpv_code format"):
            invalid = int(cpv["cpv_code"].dropna().astype(str).str.match(r"^\d{8}$").eq(False).sum())
            checks.append(Check(_status(invalid == 0), "cpv_codes", "cpv_code format", f"invalid rows={invalid:,}"))

    if "lot_results" in tables:
        lr = tables["lot_results"]
        if _columns_available(checks, "lot_results", lr, ["is_awarded", "result_code"], "is_awarded mapping"):
            mismatch_awarded = int((lr["is_awarded"] != lr["result_code"].eq("selec-w")).sum())
            checks.append(Check(_status(mismatch_awarded == 0), "lot_results", "is_awarded mapping", f"invalid rows={mismatch_awarded:,}"))

        if _columns_available(checks, "lot_results", lr, ["result_code"], "result_code domain"):
            known = lr["result_code"].isin(EXPECTED_RESULT_STATUSES) | lr["result_code"].isna()
            unknown = int((~known).sum())
            checks.append(Check(_status(unknown == 0, warn=True), "lot_results", "result_code domain", f"unknown rows={unknown:,}"))

        status_col = _first_existing_column(lr, RESULT_STATUS_COLUMNS)
        if status_col is None:
            checks.append(
                Check(
                    "WARN",
                    "lot_results",
                    "result_status domain",
                    "skipped because no result status column exists",
                )
            )
        else:
            values = lr[status_col].dropna().astype(str)
            if status_col in RESULT_STATUS_CODE_COLUMNS:
                valid_values = set(EXPECTED_RESULT_STATUSES)
            else:
                valid_values = set(EXPECTED_RESULT_STATUSES) | set(EXPECTED_RESULT_STATUSES.values()) | {"Unknown"}
            unknown = int((~values.isin(valid_values)).sum())
            detail = f"using {status_col}; unknown rows={unknown:,}"
            checks.append(Check(_status(unknown == 0, warn=True), "lot_results", "result_status domain", detail))

            if status_col == "result_status" and "result_code" in lr.columns:
                expected_status = lr["result_code"].map(EXPECTED_RESULT_STATUSES).fillna("Unknown")
                bad_status = int((lr["result_status"].astype(str) != expected_status.astype(str)).sum())
                checks.append(Check(_status(bad_status == 0), "lot_results", "result_status mapping", f"invalid rows={bad_status:,}"))

    if "organisations" in tables:
        orgs = tables["organisations"]
        if _columns_available(checks, "organisations", orgs, ["org_country"], "org_country format"):
            invalid_country = int(
                orgs["org_country"].dropna().astype(str).str.match(r"^[A-Z]{3}$").eq(False).sum()
            )
            checks.append(Check(_status(invalid_country == 0, warn=True), "organisations", "org_country format", f"invalid rows={invalid_country:,}"))

    if "contracts" in tables:
        contracts = tables["contracts"]
        if _columns_available(checks, "contracts", contracts, ["contract_value"], "contract_value non-negative"):
            invalid_value = _invalid_numeric_below(contracts["contract_value"], 0)
            checks.append(Check(_status(invalid_value == 0), "contracts", "contract_value non-negative", f"invalid rows={invalid_value:,}"))

    if "tendering_parties" in tables:
        parties = tables["tendering_parties"]
        if _columns_available(checks, "tendering_parties", parties, ["num_tenders"], "num_tenders non-negative"):
            invalid_num_tenders = _invalid_numeric_below(parties["num_tenders"], 0)
            checks.append(Check(_status(invalid_num_tenders == 0), "tendering_parties", "num_tenders non-negative", f"invalid rows={invalid_num_tenders:,}"))

        if _columns_available(checks, "tendering_parties", parties, ["tender_ids", "num_tenders"], "tender_ids count"):
            counts = parties["tender_ids"].dropna().astype(str).str.split(",").map(
                lambda values: len([value.strip() for value in values if value.strip()])
            )
            expected = pd.to_numeric(parties.loc[counts.index, "num_tenders"], errors="coerce")
            mismatched_counts = int((expected.notna() & counts.ne(expected)).sum())
            checks.append(Check(_status(mismatched_counts == 0), "tendering_parties", "tender_ids count", f"invalid rows={mismatched_counts:,}"))

    if "tenders" in tables:
        tenders = tables["tenders"]
        if _columns_available(checks, "tenders", tenders, ["tender_value"], "tender_value non-negative"):
            invalid_value = _invalid_numeric_below(tenders["tender_value"], 0)
            checks.append(Check(_status(invalid_value == 0, warn=True), "tenders", "tender_value non-negative", f"invalid rows={invalid_value:,}"))

        if _columns_available(checks, "tenders", tenders, ["rank"], "rank domain"):
            invalid_rank = _invalid_numeric_below(tenders["rank"], 1)
            checks.append(Check(_status(invalid_rank == 0, warn=True), "tenders", "rank domain", f"invalid rows={invalid_rank:,}"))

    return checks


def _table_summary_rows(tables: dict[str, pd.DataFrame]) -> list[list[object]]:
    rows: list[list[object]] = []
    for table in EXPECTED_SCHEMAS:
        if table not in tables:
            rows.append([table, "MISSING", "", "", ""])
            continue
        df = tables[table]
        rows.append(
            [
                table,
                _fmt_int(len(df)),
                _fmt_int(len(df.columns)),
                _fmt_int(df.memory_usage(deep=True).sum()),
                ", ".join(df.columns),
            ]
        )
    return rows


def _completeness_rows(tables: dict[str, pd.DataFrame]) -> list[list[object]]:
    rows: list[list[object]] = []
    for table in EXPECTED_SCHEMAS:
        if table not in tables:
            continue
        df = tables[table]
        row_count = len(df)
        for col in df.columns:
            nulls = int(df[col].isna().sum())
            rows.append([table, col, str(df[col].dtype), _fmt_int(nulls), _fmt_pct(nulls / row_count if row_count else 0)])
    return rows


def _ml_field_rows(tables: dict[str, pd.DataFrame]) -> list[list[object]]:
    rows: list[list[object]] = []
    for field, table, col in IMPORTANT_ML_FIELDS:
        if table not in tables:
            rows.append([field, table, col, "MISSING_TABLE", "", ""])
            continue

        df = tables[table]
        if col not in df.columns:
            rows.append([field, table, col, "MISSING_COLUMN", "", ""])
            continue

        row_count = len(df)
        non_null = int(df[col].notna().sum())
        rows.append(
            [
                field,
                table,
                col,
                _fmt_int(non_null),
                _fmt_pct(non_null / row_count if row_count else 0),
                str(df[col].dtype),
            ]
        )
    return rows


def _profile_rows(tables: dict[str, pd.DataFrame]) -> list[list[object]]:
    rows: list[list[object]] = []

    profiles = [
        ("notices", "notice_type"),
        ("notices", "proc_type"),
        ("notices", "buyer_country"),
        ("notices", "cpv_division_name"),
        ("lot_results", "result_status"),
        ("organisations", "sme"),
        ("tenders", "is_ranked"),
        ("tenders", "subcontracting"),
    ]
    for table, col in profiles:
        if table not in tables or col not in tables[table].columns:
            continue
        counts = tables[table][col].value_counts(dropna=False).head(10)
        for value, count in counts.items():
            rows.append([table, col, value if pd.notna(value) else "<null>", _fmt_int(count)])

    return rows


def _date_range_rows(tables: dict[str, pd.DataFrame]) -> list[list[object]]:
    rows: list[list[object]] = []
    date_cols = [
        ("notices", "pub_date"),
        ("notices", "submission_deadline"),
        ("contracts", "contract_date"),
    ]
    for table, col in date_cols:
        if table not in tables or col not in tables[table].columns:
            continue
        s = pd.to_datetime(tables[table][col], errors="coerce")
        rows.append(
            [
                table,
                col,
                _fmt_int(s.notna().sum()),
                s.min().date().isoformat() if s.notna().any() else "",
                s.max().date().isoformat() if s.notna().any() else "",
            ]
        )
    return rows


def _build_report(tables: dict[str, pd.DataFrame], checks: list[Check]) -> str:
    status_counts = pd.Series([check.status for check in checks]).value_counts()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_rows = [[status, _fmt_int(status_counts.get(status, 0))] for status in ["PASS", "WARN", "FAIL"]]
    check_rows = [[c.status, c.table, c.check, c.detail] for c in checks]

    sections = [
        "# Silver Data Quality Report",
        f"Generated at: {generated_at}",
        "Scope: Silver layer only. Source files checked are the eight `data/silver/silver_*.parquet` tables.",
        "No Gold, model, app, or pipeline orchestration files are read or executed by this validator.",
        "## Status Summary",
        _md_table(["status", "checks"], status_rows),
        "## Table Summary",
        _md_table(["table", "rows", "columns", "memory_bytes", "column_names"], _table_summary_rows(tables)),
        "## Validation Checks",
        _md_table(["status", "table", "check", "detail"], check_rows),
        "## Important ML Field Coverage",
        _md_table(["field", "table", "column", "non_null", "non_null_pct", "dtype"], _ml_field_rows(tables)),
        "## Date Ranges",
        _md_table(["table", "column", "non_null", "min", "max"], _date_range_rows(tables)),
        "## Column Completeness",
        _md_table(["table", "column", "dtype", "nulls", "null_pct"], _completeness_rows(tables)),
        "## Value Profiles",
        _md_table(["table", "column", "value", "count"], _profile_rows(tables)),
    ]
    return "\n\n".join(sections) + "\n"


def run() -> None:
    print(f"\n{'-'*55}")
    print("  Silver data quality validation")
    print(f"{'-'*55}")

    tables, checks = _load_tables()
    checks.extend(_check_schemas(tables))
    checks.extend(_check_primary_keys(tables))
    checks.extend(_check_critical_completeness(tables))
    checks.extend(_check_references(tables))
    checks.extend(_check_domains(tables))

    report = _build_report(tables, checks)
    REPORT_PATH.write_text(report, encoding="utf-8")

    counts = pd.Series([check.status for check in checks]).value_counts()
    for status in ["PASS", "WARN", "FAIL"]:
        print(f"  {status:<4} {int(counts.get(status, 0)):>3}")

    print("\n  Important ML field coverage")
    for field, table, col, non_null, pct, dtype in _ml_field_rows(tables):
        location = f"{table}.{col}"
        if non_null in {"MISSING_TABLE", "MISSING_COLUMN"}:
            print(f"  - {field:<16} {location:<35} {non_null}")
        else:
            print(f"  - {field:<16} {location:<35} {pct:>6} non-null ({non_null} rows, {dtype})")

    print(f"\n✓ Silver report written -> {REPORT_PATH}")

    if counts.get("FAIL", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
