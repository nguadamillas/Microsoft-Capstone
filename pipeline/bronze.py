"""
pipeline/bronze.py
──────────────────
Raw XML → Bronze (Parquet) — FIXED for actual eForms UBL 2.3 structure.

Key structural facts learned from real XML inspection:
  - efac:NoticeResult is inside ext:UBLExtensions/.../efext:EformsExtension
  - tenders_count is in efac:ReceivedSubmissionsStatistics, not cbc:ReceivedTendersQuantity
  - efac:LotTender (with values) lives at NoticeResult level, NOT inside LotResult
  - LotResult only references tender IDs via efac:LotTender/cbc:ID
  - winner = LotTender with RankCode=1 linked to the LotResult
  - SettledContract details (value/date) are in a separate efac:SettledContract block
  - TenderingParty is referenced by ID from LotTender
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, BRONZE_DIR, MAX_WORKERS, CHUNK_SIZE

NS = {
    "cbc":   "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
    "cac":   "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
    "efac":  "http://data.europa.eu/p27/eforms-ubl-extension-aggregate-components/1",
    "efbc":  "http://data.europa.eu/p27/eforms-ubl-extension-basic-components/1",
    "efext": "http://data.europa.eu/p27/eforms-ubl-extensions/1",
    "ext":   "urn:oasis:names:specification:ubl:schema:xsd:CommonExtensionComponents-2",
}


def _t(el, xpath):
    nodes = el.xpath(xpath, namespaces=NS)
    if not nodes:
        return None
    node = nodes[0]
    return node.text.strip() if hasattr(node, "text") and node.text else str(node).strip() or None


def _attr(el, xpath, attr):
    nodes = el.xpath(xpath, namespaces=NS)
    return nodes[0].get(attr) if nodes else None


def parse_xml(xml_path):
    rows = {k: [] for k in [
        "notices", "lots", "organisations", "cpv_codes",
        "lot_results", "tenders", "contracts", "tendering_parties",
    ]}

    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
    except etree.XMLSyntaxError:
        return rows

    tag = etree.QName(root.tag).localname
    notice_type = {
        "ContractNotice": "CN",
        "ContractAwardNotice": "CAN",
        "PriorInformationNotice": "PIN",
    }.get(tag, tag)

    notice_id = _t(root, "cbc:ID") or xml_path.stem

    # ── NOTICE ────────────────────────────────────────────────────────────────
    # estimated amount: try notice-level first, then lot-level aggregation later
    est_amount = _t(root, ".//cac:ProcurementProject/cbc:EstimatedOverallContractAmount")
    est_currency = _attr(root, ".//cac:ProcurementProject/cbc:EstimatedOverallContractAmount", "currencyID")

    # total awarded is inside EformsExtension/NoticeResult
    notice_result = root.xpath(
        ".//efext:EformsExtension/efac:NoticeResult", namespaces=NS
    )
    total_awarded = None
    if notice_result:
        total_awarded = _t(notice_result[0], "cbc:TotalAmount")

    rows["notices"].append({
        "notice_id":          notice_id,
        "notice_type":        notice_type,
        "publication_date":   _t(root, "cbc:IssueDate"),
        "buyer_org_id":       _t(root, ".//cac:ContractingParty/cac:Party/cac:PartyIdentification/cbc:ID"),
        "buyer_name":         _t(root, ".//cac:ContractingParty/cac:Party/cac:PartyName/cbc:Name"),
        "buyer_country":      _t(root, ".//cac:ContractingParty/cac:Party/cac:PostalAddress/cac:Country/cbc:IdentificationCode"),
        "project_title":      _t(root, ".//cac:ProcurementProject/cbc:Name"),
        "proc_type":          _t(root, ".//cac:ProcurementProject/cbc:ProcurementTypeCode"),
        "procedure_type":     _t(root, ".//cac:TenderingProcess/cbc:ProcedureCode"),
        "cpv_main":           _t(root, ".//cac:ProcurementProject/cac:MainCommodityClassification/cbc:ItemClassificationCode"),
        "est_amount":         _t(root, ".//cac:RequestedTenderTotal/cbc:EstimatedOverallContractAmount"),
        "est_currency":       _attr(root, ".//cac:RequestedTenderTotal/cbc:EstimatedOverallContractAmount", "currencyID"),
        "total_awarded":      total_awarded,
        "submission_deadline": _t(root, ".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate"),
        "framework":          _t(root, ".//efbc:ContractFrameworkIndicator"),
        "source_file":        xml_path.name,
    })

    # ── CPV CODES ─────────────────────────────────────────────────────────────
    for cpv_el in root.xpath(".//cac:AdditionalCommodityClassification/cbc:ItemClassificationCode", namespaces=NS):
        rows["cpv_codes"].append({
            "notice_id": notice_id,
            "cpv_code":  cpv_el.text.strip() if cpv_el.text else None,
        })

    # ── LOTS ──────────────────────────────────────────────────────────────────
    for lot in root.xpath(".//cac:ProcurementProjectLot", namespaces=NS):
        lot_id = _t(lot, "cbc:ID")
        rows["lots"].append({
            "notice_id":     notice_id,
            "lot_id":        lot_id,
            "lot_title":     _t(lot, "cac:ProcurementProject/cbc:Name"),
            "lot_desc":      _t(lot, "cac:ProcurementProject/cbc:Description"),
            "lot_est":       _t(lot, ".//cac:RequestedTenderTotal/cbc:EstimatedOverallContractAmount"),
            "lot_currency":  _attr(lot, ".//cac:RequestedTenderTotal/cbc:EstimatedOverallContractAmount", "currencyID"),
            "duration_val":  _t(lot, ".//cac:PlannedPeriod/cbc:DurationMeasure"),
            "duration_unit": _attr(lot, ".//cac:PlannedPeriod/cbc:DurationMeasure", "unitCode"),
            "award_criteria": _t(lot, ".//cac:AwardingTerms/cac:AwardingCriterion/cac:SubordinateAwardingCriterion/cbc:AwardingCriterionTypeCode"),
        })

    # ── ORGANISATIONS ─────────────────────────────────────────────────────────
    for org in root.xpath(".//efac:Organization", namespaces=NS):
        rows["organisations"].append({
            "notice_id":   notice_id,
            "org_id":      _t(org, "efac:Company/cac:PartyIdentification/cbc:ID"),
            "org_name":    _t(org, "efac:Company/cac:PartyName/cbc:Name"),
            "org_country": _t(org, "efac:Company/cac:PostalAddress/cac:Country/cbc:IdentificationCode"),
            "sme":         _t(org, "efac:Company/efbc:CompanySizeCode"),
            "nuts_code":   _t(org, "efac:Company/cac:PostalAddress/cbc:CountrySubentityCode"),
        })

    # ── EformsExtension block — contains LotResult, LotTender, SettledContract
    eforms = root.xpath(".//efext:EformsExtension", namespaces=NS)
    if not eforms:
        return rows
    eforms = eforms[0]

    # Build a lookup: tender_id → LotTender element (for value, rank, party)
    lot_tender_map = {}
    for lt in eforms.xpath("efac:NoticeResult/efac:LotTender", namespaces=NS):
        tid = _t(lt, "cbc:ID")
        if tid:
            lot_tender_map[tid] = lt

    # Build a lookup: contract_id → SettledContract element (for value, date)
    # SettledContract with full details is at NoticeResult level
    contract_detail_map = {}
    for sc in eforms.xpath("efac:NoticeResult/efac:SettledContract", namespaces=NS):
        cid = _t(sc, "cbc:ID")
        if cid:
            contract_detail_map[cid] = sc

    # TenderingParty is only a reference ID inside LotTender
    tp_map = {}
    for lt in eforms.xpath("efac:NoticeResult/efac:LotTender", namespaces=NS):
        tid = _t(lt, "cbc:ID")
        tp_ref = _t(lt, "efac:TenderingParty/cbc:ID")
        if tp_ref:
            tp_map.setdefault(tp_ref, []).append(tid or "")

    # ── LOT RESULTS ───────────────────────────────────────────────────────────
    for lr in eforms.xpath("efac:NoticeResult/efac:LotResult", namespaces=NS):
        lot_result_id = _t(lr, "cbc:ID")
        ref_lot       = _t(lr, "efac:TenderLot/cbc:ID")
        result_code   = _t(lr, "cbc:TenderResultCode")

        # tenders_count from ReceivedSubmissionsStatistics where code = "tenders"
        tenders_count = None
        sme_tenders   = None
        for stat in lr.xpath("efac:ReceivedSubmissionsStatistics", namespaces=NS):
            code = _t(stat, "efbc:StatisticsCode")
            val  = _t(stat, "efbc:StatisticsNumeric")
            if code == "tenders":
                tenders_count = val
            elif code == "t-sme":
                sme_tenders = val

        # Winner: find LotTender refs in this LotResult, get rank=1 tender
        winner_tender_id  = None
        winner_org_id     = None
        awarded_amount    = None
        award_currency    = None

        tender_ids_in_result = [
            _t(lt_ref, "cbc:ID")
            for lt_ref in lr.xpath("efac:LotTender", namespaces=NS)
        ]

        for tid in tender_ids_in_result:
            lt_el = lot_tender_map.get(tid)
            if lt_el is None:
                continue
            rank = _t(lt_el, "cbc:RankCode")
            ranked = _t(lt_el, "efbc:TenderRankedIndicator")
            # Winner = rank 1 or TenderRankedIndicator=true
            if rank == "1" or (ranked and ranked.lower() == "true"):
                winner_tender_id = tid
                awarded_amount   = _t(lt_el, "cac:LegalMonetaryTotal/cbc:PayableAmount")
                award_currency   = _attr(lt_el, "cac:LegalMonetaryTotal/cbc:PayableAmount", "currencyID")
                # Get org via TenderingParty reference (tp_map stores tender_id lists)
                tp_ref_id = _t(lt_el, "efac:TenderingParty/cbc:ID")
                # winner_org_id will be resolved via organisations table join in Silver
                winner_org_id = tp_ref_id  # store party ID for now
                break  # first ranked winner is enough

        rows["lot_results"].append({
            "notice_id":      notice_id,
            "lot_result_id":  lot_result_id,
            "lot_id":         ref_lot,
            "result_code":    result_code,
            "tenders_count":  tenders_count,
            "sme_tenders":    sme_tenders,
            "winner_tender_id": winner_tender_id,
            "winner_org_id":  winner_org_id,
            "awarded_amount": awarded_amount,
            "award_currency": award_currency,
        })

        # ── CONTRACTS (from LotResult SettledContract refs → detail map) ──────
        for sc_ref in lr.xpath("efac:SettledContract", namespaces=NS):
            cid    = _t(sc_ref, "cbc:ID")
            sc_el  = contract_detail_map.get(cid)
            c_value = None
            c_currency = None
            c_date  = None
            if sc_el is not None:
                c_date     = _t(sc_el, "cbc:IssueDate")
                c_framework = _t(sc_el, "efbc:ContractFrameworkIndicator")
                # contract value: get from linked LotTender
                linked_tender_id = _t(sc_el, "efac:LotTender/cbc:ID")
                linked_lt = lot_tender_map.get(linked_tender_id)
                if linked_lt is not None:
                    c_value    = _t(linked_lt, "cac:LegalMonetaryTotal/cbc:PayableAmount")
                    c_currency = _attr(linked_lt, "cac:LegalMonetaryTotal/cbc:PayableAmount", "currencyID")
                else:
                    c_value    = None
                    c_currency = None
            rows["contracts"].append({
                "notice_id":      notice_id,
                "lot_result_id":  lot_result_id,
                "contract_id":    cid,
                "contract_value": c_value,
                "contract_currency": c_currency,
                "contract_date":  c_date,
            })

    # ── TENDERS (all LotTender elements with full details) ────────────────────
    for lt in eforms.xpath("efac:NoticeResult/efac:LotTender", namespaces=NS):
        tender_id    = _t(lt, "cbc:ID")
        tp_ref_id    = _t(lt, "efac:TenderingParty/cbc:ID")
        lot_ref      = _t(lt, "efac:TenderLot/cbc:ID")
        tender_value = _t(lt, "cac:LegalMonetaryTotal/cbc:PayableAmount")
        t_currency   = _attr(lt, "cac:LegalMonetaryTotal/cbc:PayableAmount", "currencyID")
        rank         = _t(lt, "cbc:RankCode")
        ranked       = _t(lt, "efbc:TenderRankedIndicator")
        subcon_code  = _t(lt, "efac:SubcontractingTerm/efbc:TermCode")

        rows["tenders"].append({
            "notice_id":        notice_id,
            "tender_id":        tender_id,
            "lot_id":           lot_ref,
            "tendering_party_id": tp_ref_id,
            "tender_value":     tender_value,
            "tender_currency":  t_currency,
            "rank":             rank,
            "is_ranked":        ranked,
            "subcontracting":   subcon_code,
        })

    # ── TENDERING PARTIES ─────────────────────────────────────────────────────
    for tp_id, tender_ids in tp_map.items():
        rows["tendering_parties"].append({
            "notice_id":   notice_id,
            "party_id":    tp_id,
            "tender_ids":  ", ".join(tender_ids),
            "num_tenders": len(tender_ids),
        })

    return rows


def _collect_xml_files():
    return sorted(RAW_DIR.rglob("*.xml"))


def run():
    xml_files = _collect_xml_files()
    if not xml_files:
        print("No XML files found in data/raw/. Run pipeline/ingest.py first.")
        return

    print(f"\n{'─'*55}")
    print(f"  Bronze layer (FIXED) — {len(xml_files):,} XML files")
    print(f"{'─'*55}")

    all_rows = {k: [] for k in [
        "notices", "lots", "organisations", "cpv_codes",
        "lot_results", "tenders", "contracts", "tendering_parties",
    ]}

    chunk_idx = 0
    buffer = list(xml_files)

    while buffer:
        chunk = buffer[:CHUNK_SIZE]
        buffer = buffer[CHUNK_SIZE:]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(parse_xml, f): f for f in chunk}
            for fut in tqdm(as_completed(futures), total=len(chunk),
                            desc=f"  chunk {chunk_idx+1}", unit="file"):
                result = fut.result()
                for table, rows in result.items():
                    all_rows[table].extend(rows)

        chunk_idx += 1

    for table, rows in all_rows.items():
        if not rows:
            print(f"  [empty] {table} — 0 rows, skipping")
            continue
        df = pd.DataFrame(rows)
        out = BRONZE_DIR / f"bronze_{table}.parquet"
        df.to_parquet(out, index=False)
        print(f"  ✓ bronze_{table}.parquet  ({len(df):,} rows)")

    print(f"\n✓ Bronze complete → {BRONZE_DIR}")


if __name__ == "__main__":
    run()
