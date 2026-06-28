# %% [markdown]
# # CPV Auto-Classification  +  Mis-coding Detector  +  Semantic Search
#
# **Model #1 of the ML track.** Self-contained (pandas + numpy only — no sklearn, no
# internet), so it trains and runs anywhere the Silver layer is readable.
#
# What it does:
#   1. Predicts a notice's CPV **division** from its title text.
#   2. Attaches a **confidence** (posterior probability) to each prediction.
#   3. Flags likely **mis-coded** notices (model strongly disagrees with the buyer's CPV)
#      -> a data-quality signal that is genuinely useful to Microsoft.
#   4. "**Find similar opportunities**" via TF-IDF cosine — a prototype for the demo's
#      company-contract-fit / semantic layer, built on data we already have.
#   5. Writes a **Gold-style output table** (predictions as columns) for the dashboard + chatbot.
#
# Algorithm: multinomial Naive Bayes (a fast, strong text baseline). Empirically,
# title-only unigrams beat title+lot-descriptions and bigrams for NB — the lot text is
# boilerplate-heavy and adds noise. A transformer can exploit that extra text; NB can't.
#
# UPGRADE PATH (do these in your own env, where sklearn/torch are installed):
#   * sklearn TF-IDF + LogisticRegression  -> ~75-80% (see eda_baseline.py)
#   * XLM-R / multilingual-MiniLM fine-tune -> ~80%+, the production model.

# %%
from pathlib import Path
from collections import Counter
import re, math, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
CAND = [HERE.parent / "data-2" / "silver", HERE.parent / "data" / "silver",
        HERE / "data" / "silver", Path("data/silver")]
SILVER = next((p for p in CAND if p.exists()), CAND[0])
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
print("Silver:", SILVER)


def rp(name, cols=None):
    try:
        return pd.read_parquet(SILVER / f"silver_{name}.parquet", columns=cols)
    except Exception:
        return pd.read_parquet(SILVER / f"silver_{name}.parquet", columns=cols,
                               engine="fastparquet")


# %%
# ----------------------------------------------------------------- load
df = rp("notices", ["notice_id", "project_title", "cpv_division",
                    "cpv_division_name", "pub_date", "notice_type", "buyer_country"])
df = df.dropna(subset=["project_title", "cpv_division"]).sort_values("pub_date").reset_index(drop=True)
df["text"] = df.project_title.astype(str)
print(f"usable notices: {len(df):,} | CPV divisions: {df.cpv_division.nunique()}")
DIVNAME = (df.dropna(subset=["cpv_division_name"]).drop_duplicates("cpv_division")
             .set_index("cpv_division")["cpv_division_name"].to_dict())

# %%
# ----------------------------------------------------------------- tokenizer + vocab
TOK = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokens(s):
    return [t for t in TOK.findall(str(s).lower()) if len(t) >= 2]


cut = int(0.8 * len(df))
tr, te = df.iloc[:cut], df.iloc[cut:]
classes = sorted(df.cpv_division.unique()); C = len(classes); c2i = {c: i for i, c in enumerate(classes)}

dfreq = Counter()
doc_tok = [tokens(s) for s in df.text]
for d in doc_tok[:cut]:
    dfreq.update(set(d))
V = 30000
vocab = [w for w, _ in dfreq.most_common(V)]; V = len(vocab); w2i = {w: i for i, w in enumerate(vocab)}
idf = np.array([math.log((1 + cut) / (1 + dfreq[vocab[i]])) + 1.0 for i in range(V)])
print(f"vocabulary: {V:,} terms")


def bow(tk):
    c = Counter(t for t in tk if t in w2i)
    if not c:
        return np.empty(0, int), np.empty(0)
    idx = np.fromiter((w2i[t] for t in c), int, len(c))
    return idx, np.fromiter(c.values(), float, len(c))


# %%
# ------------------------------------------------- train multinomial Naive Bayes
alpha = 0.1
cnt = np.zeros((C, V)); cls_cnt = np.zeros(C)
for d, c in zip(doc_tok[:cut], tr.cpv_division):
    ci = c2i[c]; cls_cnt[ci] += 1
    idx, x = bow(d); cnt[ci, idx] += x
flp = np.log((cnt + alpha) / (cnt.sum(1, keepdims=True) + alpha * V))
clp = np.log(cls_cnt / cls_cnt.sum())


def predict(idx, x):
    logj = clp + (flp[:, idx] @ x if len(idx) else 0.0)
    order = np.argsort(-logj)
    p = np.exp(logj - logj.max()); p = p / p.sum()
    return order, p


# %%
# ----------------------------------------------------------------- evaluate (held-out)
yt, yp, conf, p_assigned, top3 = [], [], [], [], 0
for d, c in zip(doc_tok[cut:], te.cpv_division):
    idx, x = bow(d); order, p = predict(idx, x)
    yt.append(c); yp.append(classes[order[0]]); conf.append(float(p[order[0]]))
    p_assigned.append(float(p[c2i[c]]))
    if c in (classes[order[0]], classes[order[1]], classes[order[2]]):
        top3 += 1
acc = np.mean([a == b for a, b in zip(yt, yp)]); top3 /= len(te)
tp, fp, fn = Counter(), Counter(), Counter()
for a, b in zip(yt, yp):
    if a == b:
        tp[a] += 1
    else:
        fp[b] += 1; fn[a] += 1
f1 = [2 * (tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0) * (tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0) /
      (((tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0) + (tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0)) or 1)
      for c in classes]
macro = float(np.mean(f1)); major = float(te.cpv_division.value_counts(normalize=True).iloc[0])
print("\n================  CPV DIVISION CLASSIFIER (Naive Bayes)  ================")
print(f"  test notices   : {len(te):,}")
print(f"  majority floor : {major:.1%}   (the number to beat)")
print(f"  accuracy       : {acc:.1%}")
print(f"  macro-F1       : {macro:.1%}")
print(f"  top-3 accuracy : {top3:.1%}")

# %%
# ------------------------------------------------ mis-coding detector (added value)
# Conservative: the model must be near-certain of a DIFFERENT division AND consider the
# buyer's assigned division almost impossible. NB is overconfident, so this is a
# screening list of "review candidates", not ground truth.
res = te[["notice_id", "notice_type", "buyer_country", "cpv_division", "project_title"]].copy()
res["cpv_division_name"] = res.cpv_division.map(DIVNAME)
res["cpv_pred"] = yp
res["cpv_pred_name"] = res.cpv_pred.map(DIVNAME)
res["confidence"] = np.round(conf, 3)
res["p_assigned"] = np.round(p_assigned, 4)
res["review_flag"] = ((res.cpv_division != res.cpv_pred) &
                      (res.confidence >= 0.99) & (res.p_assigned <= 0.01))
flagged = res[res.review_flag].sort_values("confidence", ascending=False)
print("\n================  MIS-CODING REVIEW CANDIDATES  ================")
print(f"  flagged for review: {len(flagged):,} ({len(flagged)/len(res):.1%} of test)")
for _, r in flagged.head(6).iterrows():
    print(f"   buyer=[{r.cpv_division_name}] -> model=[{r.cpv_pred_name}] :: {str(r.project_title)[:60]!r}")

# %%
# --------------------------------- find-similar-opportunities (semantic prototype)
cn_pos = np.where(df.notice_type.values == "CN")[0][:6000]


def tfidf(tk):
    idx, x = bow(tk)
    if not len(idx):
        return idx, x
    val = (1 + np.log(x)) * idf[idx]; nrm = np.linalg.norm(val)
    return idx, (val / nrm if nrm else val)


cn_vec = {int(p): tfidf(doc_tok[p]) for p in cn_pos}


def similar(qpos, k=5):
    qi, qv = cn_vec[qpos]; qm = dict(zip(qi.tolist(), qv.tolist()))
    out = []
    for p, (idx, val) in cn_vec.items():
        if p == qpos:
            continue
        s = sum(qm.get(int(i), 0.0) * v for i, v in zip(idx.tolist(), val.tolist()))
        out.append((s, p))
    out.sort(reverse=True)
    return out[:k]


print("\n================  SEMANTIC SEARCH (company-contract fit prototype)  ====")
if len(cn_pos):
    q = int(cn_pos[0])
    print(f"  query: {df.project_title.iloc[q][:80]!r}")
    for s, p in similar(q, 5):
        print(f"   sim {s:.2f} [{df.cpv_division_name.iloc[p]}] {df.project_title.iloc[p][:60]!r}")

# %%
# ----------------------------------------------------------------- save outputs
gold = res.drop(columns=["project_title"])
gold.to_csv(OUT / "gold_cpv_predictions.csv", index=False)
try:
    gold.to_parquet(OUT / "gold_cpv_predictions.parquet", index=False)
except Exception:
    gold.to_parquet(OUT / "gold_cpv_predictions.parquet", index=False, engine="fastparquet")

metrics = {"model": "multinomial Naive Bayes (title-only unigrams)",
           "test_notices": int(len(te)), "majority_floor": round(major, 4),
           "accuracy": round(float(acc), 4), "macro_f1": round(macro, 4),
           "top3_accuracy": round(float(top3), 4), "review_candidates": int(len(flagged))}
(OUT / "cpv_model_metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "cpv_model_metrics.md").write_text(f"""# CPV Classifier — Results (real Silver data)

**Task:** predict CPV division (45 classes) from notice title text.
**Model:** multinomial Naive Bayes, title-only unigrams. Time-based 80/20 split.

| metric | value |
| --- | --- |
| test notices | {len(te):,} |
| majority-class floor | {major:.1%} |
| **accuracy** | **{acc:.1%}** |
| macro-F1 | {macro:.1%} |
| top-3 accuracy | {top3:.1%} |
| mis-coding review candidates | {len(flagged):,} |

This is the self-contained baseline (numpy only). The stronger models — sklearn
TF-IDF + LogisticRegression (~75-80%, see `eda_baseline.py`) and an XLM-R fine-tune
(~80%+, production) — run in an environment with those libraries installed.

**Outputs** (`ml_track/outputs/`): `gold_cpv_predictions.csv` / `.parquet` — per-notice
prediction, confidence, p(assigned), and review flag, ready to join into Gold.

**Added value beyond the demo:** confidence scoring, a mis-coding review detector
(data-quality intelligence), and a TF-IDF semantic search seeding company-contract fit.

**Next:** add the supplier models (competition intensity, optimal bid, win probability)
once the pipeline team restores `tenders_count`, `estimated`, and `winner_org_id`.
""")
print("\nsaved -> outputs/gold_cpv_predictions.csv|.parquet, cpv_model_metrics.md|.json")
