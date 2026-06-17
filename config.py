"""
config.py — Central configuration for the TED Procurement Intelligence project.
Edit this file to set date ranges, paths, and API keys.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR   = DATA_DIR / "gold"
MODEL_DIR  = BASE_DIR / "models" / "saved"

for d in [RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── TED data source ───────────────────────────────────────────────────────────
# Daily packages follow: https://ted.europa.eu/packages/daily/{YYYYNNNNN}
# NNNNN = OJ issue number (sequential per year, not day-of-year).
# TED publishes Mon–Fri. Use the release calendar to map dates → issue numbers:
#   https://ted.europa.eu/release-calendar
#
# January 2026: issues 002–022 (approx 20 publication days)
# ~30,000 notices · ~1.2 GB raw XML · ~15 MB Gold output

TED_BASE_URL = "https://ted.europa.eu/packages/daily"

# Full January 2026 — covers ~1 month of procurement activity
TED_PACKAGES = [
    "202600002", "202600003", "202600004", "202600005", "202600006",
    "202600007", "202600008", "202600009", "202600010", "202600011",
    "202600012", "202600013", "202600014", "202600015", "202600016",
    "202600017", "202600018", "202600019", "202600020", "202600021",
    "202600022",
]

# ── Pipeline settings ─────────────────────────────────────────────────────────
DATE_FILTER  = None   # "2026-01" to keep only January; None = keep everything
CHUNK_SIZE   = 500    # notices processed per chunk (keeps RAM under ~2 GB)
MAX_WORKERS  = 4      # parallel XML parsing threads

# ── API keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── CPV division lookup (first 2 digits → human-readable label) ───────────────
CPV_DIVISIONS = {
    "03": "Agriculture, forestry & fishing",
    "09": "Petroleum, gas & fuels",
    "14": "Mining & basic metals",
    "15": "Food, beverages & tobacco",
    "16": "Agricultural machinery",
    "18": "Clothing & textiles",
    "22": "Printed matter",
    "24": "Chemical products",
    "30": "Office & computing machinery",
    "31": "Electrical equipment",
    "32": "Communications equipment",
    "33": "Medical & laboratory equipment",
    "34": "Transport equipment",
    "35": "Security & safety equipment",
    "38": "Laboratory & optical equipment",
    "39": "Furniture",
    "42": "Industrial machinery",
    "44": "Construction materials",
    "45": "Construction works",
    "48": "Software & information systems",
    "50": "Repair & maintenance",
    "55": "Hotel & restaurant services",
    "60": "Road transport",
    "63": "Auxiliary transport services",
    "64": "Postal & telecoms",
    "65": "Public utilities",
    "66": "Financial & insurance services",
    "70": "Real estate services",
    "71": "Architectural & engineering",
    "72": "IT services",
    "73": "Research & development",
    "75": "Public administration & defence",
    "79": "Business services",
    "80": "Education & training",
    "85": "Health & social work",
    "90": "Sewage, refuse & sanitation",
    "92": "Recreational & cultural services",
    "98": "Other community services",
}
