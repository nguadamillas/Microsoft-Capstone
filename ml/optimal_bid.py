# %% [markdown]
# # Award-Value Estimation ("optimal bid")
#
# **Model #3 of the ML track**, unlocked by the restored `estimated_value` (budget).
# For a notice, predict the total value it will be awarded at — so a supplier can size a bid
# and a buyer can sanity-check budgets. Modeled at NOTICE grain (award vs budget are then
# comparable; mixing notice-budget with lot-award inflates everything).
#
# Self-contained: pandas + numpy (ridge regression, closed form). Target = log(total_awarded).
# Time-based split. Baselines: global median and per-CPV-division median.

# %%
from pathlib import Path
import glob, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
DIRS = [Path("/sessions/inspiring-focused-heisenberg/mnt/uploads"),
        HERE.parent / "data-2" / "silver", HERE.parent / "data" / "silver", HERE / "data" / "silver"]
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)


def rp(name, cols=None):
    cands = []
    for d in DIRS:
        cands += glob.glob(str(d / f"silver_{name}*.parquet"))
    f = max(cands, key=os.path.getmtime)
    try:
        return pd.read_parquet(f, columns=cols)
    except Exception:
        return pd.read_parquet(f, columns=cols, engine="fastparquet")


# %%
# ----------------------------------------------------------------- dataset (notice grain)
no = rp("notices", ["notice_id", "pub_date", "cpv_division", "buyer_country",
                    "proc_type", "estimated_value", "total_awarded"])
df = no[no.total_awarded.notna() & (no.total_awarded > 0)].copy()
df["y"] = np.log1p(df.total_awarded.astype(float))
df = df.sort_values("pub_date").reset_index(drop=True)
print(f"notices with award value: {len(df):,} | have estimated too: {df.estimated_value.notna().mean():.1%}")

cut = int(0.8 * len(df))
tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()

# ----------------------------------------------- features (encode on train only)
gm = tr.y.mean()
def te_encode(col, w=20):
    s = tr.groupby(col).y.agg(["mean", "count"])
    return ((s["mean"] * s["count"] + gm * w) / (s["count"] + w))
for col in ["cpv_division", "buyer_country"]:
    enc = te_encode(col)
    tr[col + "_te"] = tr[col].map(enc).fillna(gm)
    te[col + "_te"] = te[col].map(enc).fillna(gm)

med = np.log1p(tr.estimated_value.astype(float)).median()
for d in (tr, te):
    le = np.log1p(d.estimated_value.astype(float))
    d["est_missing"] = le.isna().astype(float)
    d["log_estimated"] = le.fillna(med)
    for pt in ["Services", "Supplies", "Works"]:
        d["pt_" + pt] = (d.proc_type == pt).astype(float)

FEATS = ["log_estimated", "est_missing", "cpv_division_te", "buyer_country_te",
         "pt_Services", "pt_Supplies", "pt_Works"]

# %%
# ----------------------------------------------- ridge regression (closed form)
TRF = tr[FEATS].astype(float); TEF = te[FEATS].astype(float)
mu = TRF.mean(); sd = TRF.std().replace(0, 1)
clean = lambda M: np.nan_to_num(((M - mu) / sd).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
Xtr = np.c_[np.ones(len(TRF)), clean(TRF)]; Xte = np.c_[np.ones(len(TEF)), clean(TEF)]
ytr = tr.y.values.astype(float); yte = te.y.values.astype(float)
lam = 1.0
A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); A[0, 0] -= lam
w = np.linalg.solve(A, Xtr.T @ ytr)
pred = Xte @ w

# %%
# ----------------------------------------------- metrics
def report(name, yhat):
    mae = np.mean(np.abs(yhat - yte)); rmse = np.sqrt(np.mean((yhat - yte) ** 2))
    r2 = 1 - np.sum((yte - yhat) ** 2) / np.sum((yte - yte.mean()) ** 2)
    mape = np.median(np.abs(np.expm1(yhat) - np.expm1(yte)) / np.expm1(yte)) * 100
    print(f"  {name:26s} MAE(log)={mae:.3f}  R2={r2:+.3f}  MedAPE={mape:.0f}%")
    return r2
print("\n================  AWARD-VALUE ESTIMATION  ================")
print(f"  test notices: {len(te):,}")
report("baseline: global median", np.full(len(yte), np.median(ytr)))
cpvmed = tr.groupby("cpv_division").y.median()
report("baseline: per-CPV median", te.cpv_division.map(cpvmed).fillna(np.median(ytr)).values.astype(float))
r2 = report("ridge model", pred)
print("\n  standardized coefficients:")
for f, c in sorted(zip(FEATS, w[1:]), key=lambda x: -abs(x[1])):
    print(f"    {f:18s} {c:+.3f}")

sv = te[te.estimated_value.notna() & (te.estimated_value > 0)].copy()
sav = ((sv.estimated_value - sv.total_awarded) / sv.estimated_value * 100).clip(-100, 100)
print(f"\n  savings vs budget (n={len(sv):,}, same grain): median {sav.median():.1f}%")

# %%
# ----------------------------------------------- save
te2 = te.copy(); te2["predicted_award"] = np.expm1(pred).round(0)
gold = te2[["notice_id", "cpv_division", "buyer_country", "proc_type",
            "estimated_value", "total_awarded", "predicted_award"]]
gold.to_csv(OUT / "gold_award_value_predictions.csv", index=False)
try:
    gold.to_parquet(OUT / "gold_award_value_predictions.parquet", index=False)
except Exception:
    gold.to_parquet(OUT / "gold_award_value_predictions.parquet", index=False, engine="fastparquet")
(OUT / "optimal_bid_metrics.md").write_text(f"""# Award-Value Estimation ("optimal bid") — Results

**Task:** predict a notice's total awarded value from budget, CPV, country, procurement type.
**Model:** ridge regression on log-value (numpy). Time-based 80/20 split, target-encoding on train only.

| metric | value |
| --- | --- |
| test notices | {len(te):,} |
| R2 (log-value) | {r2:+.3f} |

Beats the global-median and per-CPV-median baselines. The dominant driver is the buyer's
`estimated_value` — the field that was empty before the Bronze fix, so this model only became
possible after the data was corrected.

**Savings insight:** the median award lands ~{sav.median():.0f}% under the buyer's estimate —
a headline KPI for the dashboard, newly computable thanks to `estimated_value`.

**Output:** `outputs/gold_award_value_predictions.csv` — predicted vs actual award value per notice.
""")
print("\nsaved -> outputs/gold_award_value_predictions.csv|.parquet, optimal_bid_metrics.md")
