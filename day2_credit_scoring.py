# =============================================================
#   DAY 2 — AI Banking Risk Intelligence Platform
#   Credit Scoring Model: XGBoost + SMOTE + Optuna Tuning
#   Run: python day2_credit_scoring.py
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/data",   exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("=" * 60)
print("  DAY 2 — Credit Scoring Model  |  XGBoost + SMOTE + Optuna")
print("=" * 60)



# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def save_chart(filename):
    plt.tight_layout()
    path = f"outputs/charts/{filename}"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅  Chart saved → {path}")


# =============================================================
# SECTION 1 — LOAD CLEAN PARQUETS FROM DAY 1
# =============================================================

print("\n── Loading clean datasets from Day 1 ──────────────────")

def load_parquet(path, name):
    try:
        df = pd.read_parquet(path)
        print(f"   ✅  {name} loaded → {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df, True
    except FileNotFoundError:
        print(f"   ⚠️   {path} not found — skipping {name}")
        return None, False

lc, LC_OK = load_parquet("outputs/data/lendingclub_clean.parquet",  "LendingClub")
gc, GC_OK = load_parquet("outputs/data/german_credit_clean.parquet","German Credit")
hc, HC_OK = load_parquet("outputs/data/home_credit_clean.parquet",  "Home Credit")


# =============================================================
# SECTION 2 — FEATURE PREP HELPER
# =============================================================

def prepare_features(df, target_col="target", sample_n=None):
    """
    Encode categoricals, drop leakage-y columns, return X, y arrays.
    """
    df = df.copy()
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)

    # Drop columns that are obvious leakage or non-informative
    drop_cols = [target_col, "target_str", "loan_status",
                 "total_pymnt", "out_prncp",  # post-outcome leakage
                 "dti_band"]                  # redundant with dti
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)

    # Label-encode any remaining categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.astype("float32")
    return X, y


# =============================================================
# SECTION 3 — OPTUNA OBJECTIVE
# =============================================================

def optuna_objective(trial, X_tr, y_tr, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "use_label_encoder": False,
        "eval_metric":       "auc",
        "random_state":      42,
        "n_jobs":            -1,
        "tree_method":       "hist",
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# =============================================================
# SECTION 4 — TRAIN ONE DATASET
# =============================================================

def train_dataset(df, name, target_col="target",
                  sample_n=50_000, n_trials=30, cv_folds=5):
    """
    Full pipeline: prep → split → SMOTE → Optuna → eval → charts.
    Returns best model and results dict.
    """
    print(f"\n{'─'*55}")
    print(f"  Training on: {name}")
    print(f"{'─'*55}")

    X, y = prepare_features(df, target_col=target_col, sample_n=sample_n)
    print(f"   Dataset shape  : {X.shape}  |  Default rate: {y.mean()*100:.1f}%")

    # ── Train / test split ────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── SMOTE on training set only ────────────────────────────
    print(f"   Before SMOTE   : {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    print(f"   After  SMOTE   : {dict(zip(*np.unique(y_tr_sm, return_counts=True)))}")

    # ── Optuna hyperparameter search ──────────────────────────
    print(f"\n   🔍 Optuna tuning ({n_trials} trials × {cv_folds}-fold CV) ...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: optuna_objective(trial, X_tr_sm, y_tr_sm, cv),
        n_trials=n_trials,
        show_progress_bar=False
    )
    best_params = study.best_params
    best_cv_auc = study.best_value
    print(f"   ✅  Best CV AUC  : {best_cv_auc:.4f}")
    print(f"   Best params    : {best_params}")

    # ── Train final model on full SMOTE-d training set ───────
    final_model = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    final_model.fit(X_tr_sm, y_tr_sm)

    # ── Evaluate ──────────────────────────────────────────────
    y_prob = final_model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_auc = roc_auc_score(y_te, y_prob)
    ap       = average_precision_score(y_te, y_prob)

    print(f"\n   📊 Test  AUC    : {test_auc:.4f}")
    print(f"   📊 Avg Precision: {ap:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['No Default','Default'])}")

    results = {
        "name":       name,
        "model":      final_model,
        "X_tr":       X_tr_sm,
        "y_tr":       y_tr_sm,
        "X_te":       X_te,
        "y_te":       y_te,
        "y_prob":     y_prob,
        "y_pred":     y_pred,
        "test_auc":   test_auc,
        "avg_prec":   ap,
        "best_cv":    best_cv_auc,
        "features":   list(X.columns),
        "study":      study,
    }
    return results


# =============================================================
# SECTION 5 — RUN TRAINING ON EACH DATASET
# =============================================================

all_results = {}

if LC_OK:
    all_results["LendingClub"] = train_dataset(
        lc, "LendingClub", target_col="target",
        sample_n=60_000, n_trials=30, cv_folds=5
    )

if GC_OK:
    all_results["German Credit"] = train_dataset(
        gc, "German Credit", target_col="target",
        sample_n=None, n_trials=25, cv_folds=5
    )

if HC_OK:
    all_results["Home Credit"] = train_dataset(
        hc, "Home Credit", target_col="target",
        sample_n=50_000, n_trials=30, cv_folds=5
    )


# =============================================================
# SECTION 6 — VISUALIZATIONS
# =============================================================

print("\n── Generating charts ───────────────────────────────────")

# ── Chart 10: ROC curves — all datasets ──────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"LendingClub": "#378ADD", "German Credit": "#1D9E75", "Home Credit": "#E24B4A"}
for name, r in all_results.items():
    RocCurveDisplay.from_predictions(
        r["y_te"], r["y_prob"],
        name=f"{name}  (AUC={r['test_auc']:.3f})",
        ax=ax, color=colors.get(name, "gray")
    )
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC Curves — All Datasets", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
save_chart("10_roc_curves.png")


# ── Chart 11: Precision-Recall curves ────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, r in all_results.items():
    prec, rec, _ = precision_recall_curve(r["y_te"], r["y_prob"])
    ax.plot(rec, prec, label=f"{name}  (AP={r['avg_prec']:.3f})",
            color=colors.get(name, "gray"), lw=2)
ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
save_chart("11_precision_recall.png")


# ── Chart 12: Confusion matrices ─────────────────────────────
n = len(all_results)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
if n == 1: axes = [axes]
for ax, (name, r) in zip(axes, all_results.items()):
    cm = confusion_matrix(r["y_te"], r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred OK","Pred Def"],
                yticklabels=["Actual OK","Actual Def"],
                cbar=False)
    ax.set_title(f"{name}\nAUC={r['test_auc']:.3f}", fontsize=11, fontweight="bold")
plt.suptitle("Confusion Matrices (threshold = 0.50)", fontsize=13, fontweight="bold")
save_chart("12_confusion_matrices.png")


# ── Chart 13: Feature importance — top 15 per dataset ────────
for name, r in all_results.items():
    model = r["model"]
    feats = r["features"]
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feats).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 5))
    fi[::-1].plot(kind="barh", ax=ax,
                  color=colors.get(name, "#378ADD"), edgecolor="white")
    ax.set_title(f"Top 15 Feature Importances — {name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("XGBoost gain importance")
    safe_name = name.lower().replace(" ", "_")
    save_chart(f"13_feature_importance_{safe_name}.png")


# ── Chart 14: Optuna optimisation history ────────────────────
fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 4))
if len(all_results) == 1: axes = [axes]
for ax, (name, r) in zip(axes, all_results.items()):
    trials = r["study"].trials
    vals   = [t.value for t in trials if t.value is not None]
    running_best = pd.Series(vals).cummax()
    ax.plot(vals, "o", alpha=0.4, color=colors.get(name, "gray"),
            markersize=4, label="Trial AUC")
    ax.plot(running_best, lw=2, color="#E24B4A", label="Best so far")
    ax.set_title(f"Optuna History — {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV AUC")
    ax.legend(fontsize=9)
plt.suptitle("Hyperparameter Tuning Progress", fontsize=13, fontweight="bold")
save_chart("14_optuna_history.png")


# ── Chart 15: Score distribution (predicted prob by true label) ──
fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4))
if len(all_results) == 1: axes = [axes]
for ax, (name, r) in zip(axes, all_results.items()):
    prob = pd.Series(r["y_prob"])
    label = pd.Series(r["y_te"].values)
    ax.hist(prob[label == 0], bins=50, alpha=0.6, color="#378ADD",
            label="No Default", density=True)
    ax.hist(prob[label == 1], bins=50, alpha=0.6, color="#E24B4A",
            label="Default",    density=True)
    ax.axvline(0.5, color="black", lw=1.2, linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("Predicted probability")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
plt.suptitle("Credit Score Distribution by True Label",
             fontsize=13, fontweight="bold")
save_chart("15_score_distributions.png")


# =============================================================
# SECTION 7 — SAVE MODELS + PREDICTIONS
# =============================================================

print("\n── Saving models and predictions ───────────────────────")

for name, r in all_results.items():
    safe_name = name.lower().replace(" ", "_")

    # Save model
    model_path = f"outputs/models/xgb_{safe_name}.pkl"
    joblib.dump(r["model"], model_path)
    print(f"   ✅  Model  → {model_path}")

    # Save test predictions
    pred_df = pd.DataFrame({
        "y_true": r["y_te"].values,
        "y_pred": r["y_pred"],
        "y_prob": r["y_prob"],
    })
    pred_path = f"outputs/data/predictions_{safe_name}.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"   ✅  Preds  → {pred_path}")

    # Save feature importances
    fi = pd.Series(r["model"].feature_importances_,
                   index=r["features"]).sort_values(ascending=False)
    fi_path = f"outputs/data/feature_importance_{safe_name}.csv"
    fi.to_csv(fi_path, header=["importance"])
    print(f"   ✅  FI     → {fi_path}")

    # Save best Optuna params
    params_path = f"outputs/data/best_params_{safe_name}.csv"
    pd.Series(r["study"].best_params).to_csv(params_path, header=["value"])
    print(f"   ✅  Params → {params_path}")


# =============================================================
# SECTION 8 — FINAL SUMMARY TABLE
# =============================================================

print("\n" + "=" * 60)
print("  DAY 2 — MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"\n  {'Dataset':<18} {'Test AUC':>10} {'Avg Prec':>10} {'Best CV AUC':>12}")
print(f"  {'-'*52}")
for name, r in all_results.items():
    print(f"  {name:<18} {r['test_auc']:>10.4f} {r['avg_prec']:>10.4f} {r['best_cv']:>12.4f}")

print("\n" + "=" * 60)
print("  DAY 2 COMPLETE!")
print("=" * 60)
print("""
  outputs/charts/  → Charts 10-15 (ROC, PR, CM, FI, Optuna, Score dist)
  outputs/models/  → Trained XGBoost models (.pkl)
  outputs/data/    → Predictions + Feature importances + Best params

  Tomorrow Day 3 — Model Explainability (SHAP)
  Load with:
    import joblib
    model = joblib.load('outputs/models/xgb_lendingclub.pkl')
    preds = pd.read_parquet('outputs/data/predictions_lendingclub.parquet')
""")
