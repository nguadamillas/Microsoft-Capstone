# %% [markdown]
# # Silver EDA + Baselines — ML/DL Track (Procurement Intelligence Challenge)
#
# Reads the Silver parquet layer, profiles ML label readiness, runs a CPV-division
# classification baseline, profiles the award-value target, and prints a data-gap
# report (columns that are 100% null and must be requested from the pipeline team).
#
# Runs as a plain script (`python eda_baseline.py`) or as a notebook
# (VS Code / Jupyter recognises the `# %%` cell markers).

# %%
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

# Locate the Silver folder (handles a few common repo layouts)
try:
    HERE = Path(__file__).resolve().parent
except NameError:               # interactive / notebook
    HERE = Path.cwd()
CANDIDATES = [
    HERE.parent / "data-2" / "silver",
    HERE.parent / "data" / "silver",
    HERE / "data" / "silver",
    Path("data/silver"),
]
SILVER = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])
print("Silver folder:", SILVER)


def load(name, cols=None):
    """Load a silver table. Uses whatever parquet engine is installed."""
    return pd.read_parquet(SILVER / f"silver_{name}.parquet", columns=cols)


# %%
# ---------------------------------------------------------------- load tables
notices = load("notices")
lots = load("lots")
lot_results = load("lot_results")
cpv_codes = load("cpv_codes")

for nm, t in [("notices", notices), ("lots", lots),
              ("lot_results", lot_results), ("cpv_codes", cpv_codes)]:
    print(f"{nm:12s} rows={len(t):>7,}  cols={t.shape[1]}")
print("date range:", notices.pub_date.min(), "->", notices.pub_date.max())
print(notices.notice_type.value_counts().to_string())

# %% [markdown]
# ## 1. CPV label readiness (classification target)

# %%
print("labeled notices (title & division non-null):",
      notices.dropna(subset=["project_title", "cpv_division"]).shape[0])
print("distinct cpv_main (8-digit):", notices.cpv_main.nunique())
print("distinct cpv_division (2-digit):", notices.cpv_division.nunique())
print("distinct cpv 3-digit group:",
      notices.cpv_main.dropna().astype(str).str[:3].nunique())
print("\nTop CPV divisions:")
print(notices.cpv_division_name.value_counts().head(10).to_string())

# %% [markdown]
# ## 2. CPV classification baseline
# Text = project_title + concatenated lot descriptions  ->  CPV division (45 classes).
# Time-based split (train on older notices, test on newer) to mimic deployment.

# %%
# aggregate lot text per notice
lot_text = (lots.assign(t=lots.lot_title.fillna("") + " " + lots.lot_desc.fillna(""))
                .groupby("notice_id")["t"]
                .apply(lambda s: " ".join(s)[:3000])
                .rename("lot_text"))

df = (notices[["notice_id", "project_title", "cpv_division", "pub_date"]]
      .merge(lot_text, on="notice_id", how="left")
      .dropna(subset=["project_title", "cpv_division"]))
df["text"] = (df.project_title.fillna("") + " " + df.lot_text.fillna("")).str.strip()
df = df.sort_values("pub_date")

cut = int(0.8 * len(df))
train, test = df.iloc[:cut], df.iloc[cut:]
majority = test.cpv_division.value_counts(normalize=True).iloc[0]
print(f"train={len(train):,}  test={len(test):,}  majority-class floor={majority:.1%}")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000,
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(train.text)
    Xte = vec.transform(test.text)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, C=4.0)
    clf.fit(Xtr, train.cpv_division)

    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    acc = accuracy_score(test.cpv_division, pred)
    f1 = f1_score(test.cpv_division, pred, average="macro")
    top3 = top_k_accuracy_score(test.cpv_division, proba, k=3,
                                labels=clf.classes_)
    print(f"\nTF-IDF + LogisticRegression  ->  CPV division")
    print(f"  accuracy : {acc:.1%}")
    print(f"  macro-F1 : {f1:.1%}")
    print(f"  top-3 acc: {top3:.1%}")
    print(f"  (majority-class floor was {majority:.1%})")
except ImportError:
    print("\n[scikit-learn not installed -> skipping the linear baseline]")
    print("Reference: a pure-NB title-only baseline scored 62.1% acc / 44.3% macro-F1.")
    print("Install with:  pip install scikit-learn")

# %% [markdown]
# ### Next step for this model
# Fine-tune a multilingual transformer (XLM-R or paraphrase-multilingual-MiniLM)
# on the same text->division task; expect ~80%+ accuracy. Then add CPV 3-digit
# group (308 classes) as a stretch target.

# %% [markdown]
# ## 3. Award-value target profile (regression)

# %%
ta = notices.total_awarded
print("total_awarded non-null:", int(ta.notna().sum()),
      f"({ta.notna().mean():.1%} of all notices)")
can = notices[notices.notice_type == "CAN"]
print("among CAN:", int(can.total_awarded.notna().sum()), "of", len(can),
      f"({can.total_awarded.notna().mean():.1%})")
clean = ta[(ta.notna()) & (ta > 0)]
print("usable (>0):", len(clean))
print(np.log1p(clean).describe().round(2).to_string())
print("\nNOTE: lot-level money (awarded_amount, contract_value, lot_est, estimated) "
      "is 100% null -> notice-level regression only for now.")

# %% [markdown]
# ## 4. Data-gap report — request these from the pipeline team

# %%
EXPECTED = [
    ("lot_results", "tenders_count", lot_results),
    ("lot_results", "sme_tenders", lot_results),
    ("lot_results", "winner_org_id", lot_results),
    ("lot_results", "awarded_amount", lot_results),
    ("notices", "submission_deadline", notices),
    ("notices", "estimated", notices),
]
print("column                              non-null %   status")
print("-" * 60)
for tbl, col, frame in EXPECTED:
    pct = frame[col].notna().mean() * 100
    flag = "EMPTY - REQUEST" if pct == 0 else ("LOW" if pct < 50 else "ok")
    print(f"{tbl}.{col:<24} {pct:9.1f}%   {flag}")
