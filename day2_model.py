# =============================================================
#   DAY 2 — XGBoost Credit Scoring Model
#   Run: python day2_model.py
#   Install: pip install xgboost optuna imbalanced-learn shap
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
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("=" * 60)
print("  DAY 2 — XGBoost Credit Scoring Model")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(f"outputs/charts/{filename}", bbox_inches="tight")
    plt.close()
    print(f"   ✅ Chart saved → outputs/charts/{filename}")


# =============================================================
# SECTION 1 — LOAD CLEAN DATA
# =============================================================

print("\n── Loading clean datasets ──────────────────────────────")

lc = pd.read_parquet("outputs/data/lendingclub_clean.parquet")
gc = pd.read_parquet("outputs/data/german_credit_clean.parquet")
hc = pd.read_parquet("outputs/data/home_credit_clean.parquet")

print(f"   LendingClub  → {lc.shape[0]:,} rows × {lc.shape[1]} cols")
print(f"   German Credit→ {gc.shape[0]:,} rows × {gc.shape[1]} cols")
print(f"   Home Credit  → {hc.shape[0]:,} rows × {hc.shape[1]} cols")


# =============================================================
# SECTION 2 — PREPARE LENDINGLUB (PRIMARY DATASET)
# =============================================================

print("\n── Preparing features ──────────────────────────────────")

# Drop non-feature columns
drop_cols = ["target_str", "dti_band", "income_band"]
lc_model = lc.drop(columns=[c for c in drop_cols if c in lc.columns])

# Encode any remaining object columns
le = LabelEncoder()
for col in lc_model.select_dtypes(include="object").columns:
    lc_model[col] = le.fit_transform(lc_model[col].astype(str))

X = lc_model.drop("target", axis=1)
y = lc_model["target"]

print(f"   Features     : {X.shape[1]}")
print(f"   Samples      : {X.shape[0]:,}")
print(f"   Default rate : {y.mean()*100:.1f}%")
print(f"   Class ratio  : {int((y==0).sum()/(y==1).sum())}:1 (imbalanced)")

# Train / test split — stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train size   : {X_train.shape[0]:,}")
print(f"   Test size    : {X_test.shape[0]:,}")


# =============================================================
# SECTION 3 — SMOTE (Fix class imbalance on train only)
# =============================================================

print("\n── Applying SMOTE to training set ──────────────────────")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"   Before SMOTE : {y_train.value_counts().to_dict()}")
print(f"   After SMOTE  : {pd.Series(y_train_sm).value_counts().to_dict()}")


# =============================================================
# SECTION 4 — BASELINE MODELS
# =============================================================

print("\n── Training baseline models ────────────────────────────")

imbalance_ratio = int((y == 0).sum() / (y == 1).sum())

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sm, y_train_sm)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print(f"   Logistic Regression AUC : {lr_auc:.4f}")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight="balanced", n_jobs=-1, random_state=42
)
rf.fit(X_train_sm, y_train_sm)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"   Random Forest AUC       : {rf_auc:.4f}")


# =============================================================
# SECTION 5 — XGBOOST WITH OPTUNA TUNING
# =============================================================

print("\n── XGBoost + Optuna hyperparameter tuning (50 trials) ──")

def objective(trial):
    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 100, 600),
        "max_depth"         : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight"  : trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight"  : imbalance_ratio,
        "eval_metric"       : "auc",
        "random_state"      : 42,
        "n_jobs"            : -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_sm, y_train_sm,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"   Best trial AUC (CV)  : {study.best_value:.4f}")
print(f"   Best params          : {study.best_params}")

# Train final XGBoost with best params
best_params = study.best_params
best_params.update({"scale_pos_weight": imbalance_ratio,
                    "eval_metric": "auc", "random_state": 42, "n_jobs": -1})

xgb = XGBClassifier(**best_params)
xgb.fit(X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)], verbose=False)

xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"\n   Final XGBoost Test AUC : {xgb_auc:.4f}")


# =============================================================
# SECTION 6 — FULL EVALUATION
# =============================================================

print("\n── Model evaluation ────────────────────────────────────")

models = {
    "Logistic Regression" : lr,
    "Random Forest"       : rf,
    "XGBoost (Tuned)"     : xgb,
}

results = {}
for name, model in models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = {"auc": auc, "proba": y_proba, "pred": y_pred}
    print(f"\n  {name}  |  AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Fully Paid", "Charged Off"]))

# 5-fold CV on XGBoost
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X, y, cv=cv5, scoring="roc_auc", n_jobs=-1)
print(f"\n  XGBoost 5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("  (Use this number in your report!)")


# =============================================================
# SECTION 7 — CHARTS
# =============================================================

print("\n── Saving evaluation charts ────────────────────────────")

# Chart 1 — ROC curves (all 3 models)
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#534AB7", "#1D9E75", "#E24B4A"]
for (name, res), color in zip(results.items(), colors):
    RocCurveDisplay.from_predictions(
        y_test, res["proba"], name=f"{name} (AUC={res['auc']:.3f})",
        ax=ax, color=color)
ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
save_chart("10_roc_curves.png")

# Chart 2 — Confusion matrix (XGBoost)
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, results["XGBoost (Tuned)"]["pred"])
ConfusionMatrixDisplay(cm, display_labels=["Fully Paid","Charged Off"]).plot(
    cmap="Blues", ax=ax)
ax.set_title("XGBoost — Confusion Matrix", fontsize=13, fontweight="bold")
save_chart("11_confusion_matrix.png")

# Chart 3 — Feature importance (XGBoost)
imp_df = pd.DataFrame({
    "feature"   : X.columns,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(imp_df["feature"], imp_df["importance"], color="#378ADD")
ax.set_title("XGBoost — Top 15 Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance score")
save_chart("12_feature_importance.png")

# Chart 4 — Model comparison bar
fig, ax = plt.subplots(figsize=(8, 4))
names = list(results.keys())
aucs  = [results[n]["auc"] for n in names]
bars  = ax.bar(names, aucs, color=["#534AB7","#1D9E75","#E24B4A"], width=0.4)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(min(aucs)*0.95, 1.0)
ax.set_title("Model AUC Comparison", fontsize=13, fontweight="bold")
ax.set_ylabel("ROC-AUC Score")
save_chart("13_model_comparison.png")

# Chart 5 — Precision-Recall curve (XGBoost)
fig, ax = plt.subplots(figsize=(8, 5))
prec, rec, _ = precision_recall_curve(y_test, results["XGBoost (Tuned)"]["proba"])
ax.plot(rec, prec, color="#E24B4A", lw=2)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("XGBoost — Precision-Recall Curve", fontsize=13, fontweight="bold")
ax.fill_between(rec, prec, alpha=0.1, color="#E24B4A")
save_chart("14_precision_recall.png")


# =============================================================
# SECTION 8 — SHAP EXPLAINABILITY
# =============================================================

print("\n── SHAP Explainability ─────────────────────────────────")

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# SHAP Summary plot (beeswarm)
fig = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP — Feature Impact on Predictions", fontsize=13, fontweight="bold")
save_chart("15_shap_summary.png")

# SHAP Bar plot (global importance)
fig = plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP — Mean Absolute Feature Importance", fontsize=13, fontweight="bold")
save_chart("16_shap_importance_bar.png")

# SHAP Waterfall for 1 customer (for your viva demo)
sample_idx = 0
fig = plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values         = shap_values[sample_idx],
        base_values    = explainer.expected_value,
        data           = X_test.iloc[sample_idx],
        feature_names  = X_test.columns.tolist()
    ), show=False
)
plt.title("SHAP Waterfall — Single Customer Explanation", fontsize=12, fontweight="bold")
save_chart("17_shap_waterfall_customer.png")

print("   SHAP explanation function saved (use in Streamlit Day 5)")


# =============================================================
# SECTION 9 — SAVE MODELS
# =============================================================

print("\n── Saving models ───────────────────────────────────────")

joblib.dump(xgb, "outputs/models/xgboost_credit_model.pkl")
joblib.dump(rf,  "outputs/models/random_forest_model.pkl")
joblib.dump(lr,  "outputs/models/logistic_regression_model.pkl")
joblib.dump(list(X.columns), "outputs/models/feature_columns.pkl")

print("   ✅ outputs/models/xgboost_credit_model.pkl")
print("   ✅ outputs/models/random_forest_model.pkl")
print("   ✅ outputs/models/logistic_regression_model.pkl")
print("   ✅ outputs/models/feature_columns.pkl")


# =============================================================
# DONE
# =============================================================

print("\n" + "=" * 60)
print("  DAY 2 COMPLETE!")
print("=" * 60)
print(f"""
  Models saved  → outputs/models/
  Charts saved  → outputs/charts/ (charts 10-17)

  Key results to put in your report:
  ───────────────────────────────────────────────
  Logistic Regression AUC : {lr_auc:.4f}
  Random Forest AUC       : {rf_auc:.4f}
  XGBoost (Tuned) AUC     : {xgb_auc:.4f}
  5-Fold CV AUC           : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  ───────────────────────────────────────────────
  Tomorrow Day 3 — Fraud Detection (Isolation Forest)
""")
