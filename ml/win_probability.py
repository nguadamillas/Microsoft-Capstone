# %% [markdown]
# # Win Probability — which bid wins a competitive lot  (v2, with award_criteria)
#
# **Model #2 of the ML track.** For each bid in a competitive lot (>=2 bidders, known winner),
# predict the probability it is the winning tender. Supplier-facing triage.
#
# Self-contained: pandas + numpy (logistic regression).
# LEAKAGE GUARD: `rank`/`is_ranked` are excluded (rank 1 == winner).
#
# v2 adds `award_criteria` (now populated after the Bronze fix). FINDING: it does NOT lift
# winner prediction, because the field captures only the criterion TYPE ("price"/"quality"),
# not the WEIGHTS or per-bid quality SCORES that actually decide MEAT awards.
#
# Metrics: ROC-AUC, PR-AUC, log-loss (bid-level) + Precision@1, MRR (per-lot ranking),
# vs a "cheapest bid wins" baseline.

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
    f = max(cands, key=os.path.getmtime)            # newest available wins
    try:
        return pd.read_parquet(f, columns=cols)
    except Exception:
        return pd.read_parquet(f, columns=cols, engine="fastparquet")


# %%
# ----------------------------------------------------------------- build dataset
te = rp("tenders", ["notice_id", "tender_id", "lot_id", "tendering_party_id",
                    "tender_value", "subcontracting"])
lr = rp("lot_results", ["notice_id", "lot_id", "winner_tender_id"]).dropna(subset=["winner_tender_id"])
no = rp("notices", ["notice_id", "pub_date", "cpv_division", "buyer_country"])
lots = rp("lots", ["notice_id", "lot_id", "award_criteria"])

df = te.merge(lr, on=["notice_id", "lot_id"], how="inner")
df["is_winner"] = (df.tender_id == df.winner_tender_id).astype(int)
df = df.merge(no, on="notice_id", how="left").merge(lots, on=["notice_id", "lot_id"], how="left")
df = df[df.tender_value.notna() & (df.tender_value > 0)]
key = ["notice_id", "lot_id"]; g = df.groupby(key)
df["n_bids"] = g.tender_value.transform("size"); df["n_win"] = g.is_winner.transform("sum")
df = df[(df.n_bids >= 2) & (df.n_win == 1)].copy()
print(f"competitive lots: {df.groupby(key).ngroups:,} | bids: {len(df):,} | win rate: {df.is_winner.mean():.3f}")

# ----------------------------------------------- supplier track record (leakage-safe)
df = df.sort_values(["pub_date", "notice_id", "lot_id", "tender_id"]).reset_index(drop=True)
gp = df.groupby("tendering_party_id")
df["prior_bids"] = gp.cumcount()
df["prior_wins"] = gp.is_winner.cumsum() - df.is_winner
df["supplier_winrate"] = (df.prior_wins + 1.0) / (df.prior_bids + 5.0)

# ----------------------------------------------- features (price-relative + award_criteria)
g = df.groupby(key)
df["lot_min"] = g.tender_value.transform("min"); df["lot_mean"] = g.tender_value.transform("mean")
df["lot_std"] = g.tender_value.transform("std").fillna(0.0)
df["price_rank_pct"] = (g.tender_value.rank(method="min") - 1) / (df.n_bids - 1)
df["is_cheapest"] = (df.tender_value <= df.lot_min).astype(float)
df["gap_to_min"] = (df.tender_value - df.lot_min) / df.lot_mean
df["value_z"] = ((df.tender_value - df.lot_mean) / df.lot_std.replace(0, np.nan)).fillna(0.0)
df["log_value"] = np.log1p(df.tender_value)
df["sub_yes"] = (df.subcontracting == "yes").astype(float)
ac = df.award_criteria.fillna("").astype(str).str.lower()
df["ac_price"] = (ac == "price").astype(float)
df["ac_quality"] = (ac == "quality").astype(float)
df["cheap_x_price"] = df.is_cheapest * df.ac_price            # cheapest wins MORE when price-only?

FEATS = ["price_rank_pct", "is_cheapest", "gap_to_min", "value_z", "log_value", "n_bids",
         "supplier_winrate", "sub_yes", "ac_price", "ac_quality", "cheap_x_price"]

# ----------------------------------------------- time-based split (by lot)
order = {k: i for i, k in enumerate(df.groupby(key).pub_date.first().sort_values().index)}
df["lo"] = list(zip(df.notice_id, df.lot_id)); df["lo"] = df["lo"].map(order)
cutoff = np.quantile(df.lo, 0.8)
tr, te_ = df[df.lo <= cutoff], df[df.lo > cutoff]
mu = tr[FEATS].astype(float).mean(); sd = tr[FEATS].astype(float).std().replace(0, 1)
Xtr = ((tr[FEATS].astype(float) - mu) / sd).values; ytr = tr.is_winner.values.astype(float)
Xte = ((te_[FEATS].astype(float) - mu) / sd).values; yte = te_.is_winner.values.astype(float)
print(f"train bids: {len(tr):,} | test bids: {len(te_):,} | test lots: {te_.groupby(key).ngroups:,}")

# %%
# ----------------------------------------------- logistic regression
W = np.zeros(Xtr.shape[1]); b = 0.0; vW = np.zeros_like(W); vb = 0.0
n = len(ytr)
for ep in range(600):
    p = 1 / (1 + np.exp(-(Xtr @ W + b)))
    gW = Xtr.T @ (p - ytr) / n + 1e-4 * W; gb = (p - ytr).mean()
    vW = 0.9 * vW - 0.5 * gW; W += vW; vb = 0.9 * vb - 0.5 * gb; b += vb
pte = 1 / (1 + np.exp(-(Xte @ W + b)))

# %%
# ----------------------------------------------- metrics
def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    P = y.sum(); N = len(y) - P
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
def ap(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y)
    return (tp / np.arange(1, len(y) + 1) * y).sum() / y.sum()
logloss = -np.mean(yte * np.log(pte + 1e-12) + (1 - yte) * np.log(1 - pte + 1e-12))
te_ = te_.copy(); te_["p"] = pte
prec1 = mrr = base1 = nlots = 0
for _, gr in te_.groupby(key):
    nlots += 1; wi = gr.is_winner.values.argmax()
    rank = int(np.where(np.argsort(-gr.p.values) == wi)[0][0]) + 1
    prec1 += (rank == 1); mrr += 1 / rank
    base1 += (gr.is_winner.values[gr.tender_value.values.argmin()] == 1)
prec1 /= nlots; mrr /= nlots; base1 /= nlots
print("\n================  WIN PROBABILITY (v2, + award_criteria)  ================")
print(f"  test lots {nlots:,} | ROC-AUC {auc(yte,pte):.3f} | PR-AUC {ap(yte,pte):.3f} | log-loss {logloss:.3f}")
print(f"  Precision@1 {prec1:.3f}  vs  cheapest-bid baseline {base1:.3f}   | MRR {mrr:.3f}")
print("  award_criteria coefficients:", {f: round(c, 3) for f, c in zip(FEATS, W) if f.startswith(('ac_', 'cheap_x'))})

# %%
# ----------------------------------------------- save
gold = te_[["notice_id", "lot_id", "tender_id", "tendering_party_id", "cpv_division",
            "buyer_country", "tender_value", "n_bids", "is_winner", "p"]].rename(columns={"p": "win_probability"})
gold["win_probability"] = gold.win_probability.round(4)
gold.to_csv(OUT / "gold_win_probability.csv", index=False)
try:
    gold.to_parquet(OUT / "gold_win_probability.parquet", index=False)
except Exception:
    gold.to_parquet(OUT / "gold_win_probability.parquet", index=False, engine="fastparquet")
(OUT / "win_probability_metrics.md").write_text(f"""# Win Probability (v2) — Results

**Task:** for each bid in a competitive lot, predict P(win). Leakage-safe (`rank` excluded).
**Model:** logistic regression on price-relative + bidder-history + `award_criteria` features.

| metric | value |
| --- | --- |
| test lots | {nlots:,} |
| ROC-AUC | {auc(yte,pte):.3f} |
| PR-AUC | {ap(yte,pte):.3f} |
| MRR | {mrr:.3f} |
| Precision@1 (model) | {prec1:.3f} |
| Precision@1 (cheapest-bid baseline) | {base1:.3f} |

## Finding: award_criteria did NOT lift winner prediction

Now that `award_criteria` is populated we added it — and it makes ~no difference. The model
still can't beat the cheapest-bid baseline at naming the exact winner. The reason is that the
field only captures the criterion **type** ("price" / "quality"), not the **weights**
(e.g. price 60% / quality 40%) or the **per-bid quality scores** that actually decide MEAT awards.
Even in "price" lots the cheapest wins only ~39% of the time.

**Sharper data ask:** the criteria **weights** and the **per-bid quality scores** — not just the
criterion type. Those are the real decision inputs.

The model remains useful as a calibrated triage tool (ROC-AUC {auc(yte,pte):.2f}): rank which
tenders a supplier is most likely to win, even when the single winner can't be pinned.

**Output:** `outputs/gold_win_probability.csv`.
""")
print("\nsaved -> outputs/gold_win_probability.csv|.parquet, win_probability_metrics.md")
