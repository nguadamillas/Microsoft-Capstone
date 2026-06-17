"""
models/train_models.py
───────────────────────
Trains three ML models on Gold-layer procurement data:

  1. Win Probability      — binary classifier: will an SME win this contract?
  2. Competition Intensity — regressor: how many tenders will a lot receive?
  3. Bid Estimation       — regressor: what will the awarded amount be?

Models are saved to models/saved/ as .joblib files.
A feature importance report is printed for each model.

Usage:
    python -m models.train_models
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, mean_absolute_error,
    mean_absolute_percentage_error, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOLD_DIR, MODEL_DIR


# ── Load data ──────────────────────────────────────────────────────────────────

def load_awards() -> pd.DataFrame:
    p = GOLD_DIR / "gold_awards.parquet"
    if not p.exists():
        raise FileNotFoundError("gold_awards.parquet not found. Run the full pipeline first.")
    return pd.read_parquet(p)


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features useful for all three models."""
    df = df.copy()

    # Log-scale monetary amounts (handles heavy right skew in procurement data)
    df["log_estimated"]  = np.log1p(df["estimated"].fillna(0))
    df["log_awarded"]    = np.log1p(df["awarded_eur"].fillna(0))

    # CPV division as integer (first 2 digits)
    df["cpv_div_int"] = pd.to_numeric(df["cpv_division"], errors="coerce")

    # Publication month and weekday (procurement timing patterns)
    df["pub_month"]   = pd.to_datetime(df["pub_date"], errors="coerce").dt.month
    df["pub_weekday"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.dayofweek

    # Has framework agreement
    df["is_framework"] = df.get("framework", pd.Series(dtype=str)).notna().astype(int)

    # Num lots (competition proxy)
    df["num_lots"]    = df["num_lots"].fillna(1).astype(float)

    # SME winner flag (binary target for model 1)
    df["sme_winner_int"] = df["sme_winner"].fillna(False).astype(int)

    return df


NUMERIC_FEATURES = [
    "log_estimated", "cpv_div_int", "pub_month", "pub_weekday",
    "num_lots", "is_framework",
]
CATEGORICAL_FEATURES = [
    "buyer_country", "proc_type", "cpv_division_name",
]


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def feature_importance_report(model, preprocessor, label: str) -> dict:
    """Extract and print top-10 feature importances from LightGBM."""
    ohe_features = list(
        preprocessor.named_transformers_["cat"]["ohe"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    all_features = NUMERIC_FEATURES + ohe_features

    importances = model.feature_importances_
    fi = pd.Series(importances, index=all_features).sort_values(ascending=False)

    print(f"\n  Top-10 features ({label}):")
    for feat, imp in fi.head(10).items():
        bar = "█" * int(imp / fi.max() * 20)
        print(f"    {feat:<40} {bar} {imp:.0f}")

    return fi.head(20).to_dict()


# ── Model 1: Win Probability (SME wins) ────────────────────────────────────────

def train_win_probability(df: pd.DataFrame):
    """
    Binary classifier: predict whether an SME will win the contract.
    Target: sme_winner_int (1 = SME winner, 0 = large company winner)
    """
    print(f"\n{'─'*55}")
    print("  Model 1: Win Probability (SME vs large company)")
    print(f"{'─'*55}")

    df = df.dropna(subset=["sme_winner_int", "estimated"])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["sme_winner_int"]

    print(f"  Dataset: {len(df):,} samples | Class balance: {y.mean():.1%} SME wins")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05,
        num_leaves=31, class_weight="balanced",
        random_state=42, verbose=-1,
    )
    model.fit(X_train_t, y_train,
              eval_set=[(X_test_t, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred  = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  Test AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Large", "SME"]))

    fi = feature_importance_report(model, preprocessor, "Win Probability")

    # Save
    joblib.dump({"model": model, "preprocessor": preprocessor}, MODEL_DIR / "win_probability.joblib")
    with open(MODEL_DIR / "win_probability_features.json", "w") as f:
        json.dump(fi, f, indent=2)
    print(f"\n  ✓ Saved to models/saved/win_probability.joblib")
    return {"auc": auc}


# ── Model 2: Competition Intensity ────────────────────────────────────────────

def train_competition_intensity(df: pd.DataFrame):
    """
    Regressor: predict avg_tenders_per_lot.
    High competition = more bidders = lower expected margin for suppliers.
    Useful for Microsoft's customers scoping bids.
    """
    print(f"\n{'─'*55}")
    print("  Model 2: Competition Intensity (avg tenders per lot)")
    print(f"{'─'*55}")

    df = df.dropna(subset=["avg_tenders_per_lot", "estimated"])
    df = df[df["avg_tenders_per_lot"] > 0]
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["avg_tenders_per_lot"]

    print(f"  Dataset: {len(df):,} samples | Mean tenders/lot: {y.mean():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1,
    )
    model.fit(X_train_t, y_train,
              eval_set=[(X_test_t, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred = model.predict(X_test_t)
    mae    = mean_absolute_error(y_test, y_pred)
    mape   = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n  MAE:  {mae:.2f} tenders   |   MAPE: {mape:.1%}")

    fi = feature_importance_report(model, preprocessor, "Competition Intensity")

    joblib.dump({"model": model, "preprocessor": preprocessor}, MODEL_DIR / "competition_intensity.joblib")
    with open(MODEL_DIR / "competition_intensity_features.json", "w") as f:
        json.dump(fi, f, indent=2)
    print(f"\n  ✓ Saved to models/saved/competition_intensity.joblib")
    return {"mae": mae, "mape": mape}


# ── Model 3: Bid Estimation ────────────────────────────────────────────────────

def train_bid_estimation(df: pd.DataFrame):
    """
    Regressor: predict awarded_eur from estimated + context features.
    savings_pct = (estimated - awarded) / estimated.
    Useful for suppliers estimating realistic bid prices.
    """
    print(f"\n{'─'*55}")
    print("  Model 3: Bid Estimation (predicted awarded amount)")
    print(f"{'─'*55}")

    df = df.dropna(subset=["awarded_eur", "estimated"])
    df = df[(df["awarded_eur"] > 0) & (df["estimated"] > 0)]

    # Add competition as a feature here (it's available at award time)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    extra = ["avg_tenders_per_lot", "num_awarded_lots"]
    for col in extra:
        if col in df.columns:
            features = [col] + features   # prepend

    X = df[features].copy()
    y = np.log1p(df["awarded_eur"])  # log-transform target

    print(f"  Dataset: {len(df):,} samples | Median awarded: €{df['awarded_eur'].median():,.0f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Dynamic preprocessor to handle extra features
    num_feats = [f for f in features if f not in CATEGORICAL_FEATURES]
    cat_feats  = [f for f in features if f in CATEGORICAL_FEATURES]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)])

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04,
        num_leaves=63, min_child_samples=10,
        random_state=42, verbose=-1,
    )
    model.fit(X_train_t, y_train,
              eval_set=[(X_test_t, y_test)],
              callbacks=[lgb.early_stopping(60, verbose=False)])

    y_pred_log = model.predict(X_test_t)
    y_pred_eur = np.expm1(y_pred_log)
    y_test_eur = np.expm1(y_test)

    mae  = mean_absolute_error(y_test_eur, y_pred_eur)
    mape = mean_absolute_percentage_error(y_test_eur, y_pred_eur)
    print(f"\n  MAE:  €{mae:,.0f}   |   MAPE: {mape:.1%}")

    # Feature importance (adapt to dynamic feature list)
    ohe_features = list(preprocessor.named_transformers_["cat"]["ohe"]
                        .get_feature_names_out(cat_feats))
    all_features = num_feats + ohe_features
    fi_series = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
    print(f"\n  Top-10 features (Bid Estimation):")
    for feat, imp in fi_series.head(10).items():
        bar = "█" * int(imp / fi_series.max() * 20)
        print(f"    {feat:<40} {bar} {imp:.0f}")

    joblib.dump({"model": model, "preprocessor": preprocessor,
                 "num_feats": num_feats, "cat_feats": cat_feats},
                MODEL_DIR / "bid_estimation.joblib")
    with open(MODEL_DIR / "bid_estimation_features.json", "w") as f:
        json.dump(fi_series.head(20).to_dict(), f, indent=2)
    print(f"\n  ✓ Saved to models/saved/bid_estimation.joblib")
    return {"mae": mae, "mape": mape}


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run():
    print(f"\n{'═'*55}")
    print(f"  ML Training — TED Procurement")
    print(f"{'═'*55}")

    df_raw  = load_awards()
    df      = engineer_features(df_raw)

    metrics = {}
    metrics["win_probability"]       = train_win_probability(df)
    metrics["competition_intensity"] = train_competition_intensity(df)
    metrics["bid_estimation"]        = train_bid_estimation(df)

    # Save metrics summary
    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"  ✓ All models trained and saved to models/saved/")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    run()
