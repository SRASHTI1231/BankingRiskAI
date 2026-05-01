# =============================================================
#   DAY 3 — Fraud Detection Model
#   Isolation Forest + Unified Risk Score
#   Run: python day3_fraud.py
#   Install: pip install tensorflow scikit-learn xgboost joblib
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

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("=" * 60)
print("  DAY 3 — Fraud Detection + Unified Risk Score")
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
# SECTION 1 — LOAD DATA
# =============================================================

print("\n── Loading datasets ────────────────────────────────────")

lc = pd.read_parquet("outputs/data/lendingclub_clean.parquet")
hc = pd.read_parquet("outputs/data/home_credit_clean.parquet")

print(f"   LendingClub  → {lc.shape[0]:,} rows")
print(f"   Home Credit  → {hc.shape[0]:,} rows")


# =============================================================
# SECTION 2 — PREPARE FRAUD FEATURES (LendingClub)
# =============================================================

print("\n── Preparing fraud detection features ──────────────────")

# Drop non-numeric / non-feature columns
drop_cols = ["target_str", "dti_band", "income_band"]
lc_fraud = lc.drop(columns=[c for c in drop_cols if c in lc.columns]).copy()

# Encode object columns
le = LabelEncoder()
for col in lc_fraud.select_dtypes(include="object").columns:
    lc_fraud[col] = le.fit_transform(lc_fraud[col].astype(str))

# Separate features and true labels
X_fraud = lc_fraud.drop("target", axis=1)
y_true  = lc_fraud["target"]  # we'll use this only for evaluation

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fraud)

print(f"   Features     : {X_fraud.shape[1]}")
print(f"   Samples      : {X_fraud.shape[0]:,}")
print(f"   Actual fraud : {y_true.mean()*100:.1f}% (ground truth)")


# =============================================================
# SECTION 3 — ISOLATION FOREST
# =============================================================

print("\n── Training Isolation Forest ───────────────────────────")

# contamination = expected % of anomalies (match actual default rate)
contamination = round(float(y_true.mean()), 3)

iso_forest = IsolationForest(
    n_estimators  = 300,
    contamination = contamination,
    max_samples   = "auto",
    random_state  = 42,
    n_jobs        = -1
)
iso_forest.fit(X_scaled)

# Predictions: -1 = anomaly (fraud), 1 = normal
iso_pred   = iso_forest.predict(X_scaled)
iso_labels = np.where(iso_pred == -1, 1, 0)  # convert to 0/1

# Anomaly scores (lower = more anomalous)
iso_scores = iso_forest.decision_function(X_scaled)
# Normalise to 0-1 fraud probability (higher = more suspicious)
iso_fraud_score = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

# Evaluate against true labels
iso_auc = roc_auc_score(y_true, iso_fraud_score)
print(f"   Isolation Forest AUC     : {iso_auc:.4f}")
print(f"   Flagged as anomaly       : {iso_labels.sum():,} ({iso_labels.mean()*100:.1f}%)")
print(f"\n   Classification report vs true labels:")
print(classification_report(y_true, iso_labels,
      target_names=["Normal", "Fraud/Default"]))


# =============================================================
# SECTION 4 — AUTOENCODER (Neural Network Anomaly Detection)
# =============================================================

print("\n── Training AutoEncoder ────────────────────────────────")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")

    # Train AutoEncoder ONLY on normal transactions
    X_normal = X_scaled[y_true == 0]

    n_features = X_scaled.shape[1]

    # Encoder
    inp      = Input(shape=(n_features,))
    encoded  = Dense(32, activation="relu")(inp)
    encoded  = Dropout(0.2)(encoded)
    encoded  = Dense(16, activation="relu")(encoded)
    bottleneck = Dense(8, activation="relu")(encoded)

    # Decoder
    decoded  = Dense(16, activation="relu")(bottleneck)
    decoded  = Dropout(0.2)(decoded)
    decoded  = Dense(32, activation="relu")(decoded)
    output   = Dense(n_features, activation="linear")(decoded)

    autoencoder = Model(inp, output)
    autoencoder.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    autoencoder.fit(
        X_normal, X_normal,
        epochs          = 50,
        batch_size      = 256,
        validation_split= 0.1,
        callbacks       = [early_stop],
        verbose         = 0
    )

    # Reconstruction error = fraud score
    X_reconstructed   = autoencoder.predict(X_scaled, verbose=0)
    reconstruction_err = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

    # Normalise to 0-1
    ae_fraud_score = (reconstruction_err - reconstruction_err.min()) / \
                     (reconstruction_err.max() - reconstruction_err.min())

    ae_auc = roc_auc_score(y_true, ae_fraud_score)
    print(f"   AutoEncoder AUC          : {ae_auc:.4f}")

    # Threshold: flag top contamination% as fraud
    ae_threshold  = np.percentile(reconstruction_err, (1 - contamination) * 100)
    ae_labels     = (reconstruction_err > ae_threshold).astype(int)
    AE_OK         = True

    joblib.dump(scaler, "outputs/models/scaler.pkl")
    autoencoder.save("outputs/models/autoencoder.keras")
    print("   ✅ AutoEncoder saved")

except ImportError:
    print("   ⚠️  TensorFlow not installed — skipping AutoEncoder")
    print("   Install with: pip install tensorflow")
    ae_fraud_score = iso_fraud_score  # fallback
    ae_auc         = iso_auc
    ae_labels      = iso_labels
    AE_OK          = False


# =============================================================
# SECTION 5 — UNIFIED RISK SCORE
# =============================================================

print("\n── Creating Unified Risk Score ─────────────────────────")

# Load credit model predictions
xgb_model    = joblib.load("outputs/models/xgboost_credit_model.pkl")
feat_cols     = joblib.load("outputs/models/feature_columns.pkl")

# Align features to what the credit model expects
X_credit = X_fraud.reindex(columns=feat_cols, fill_value=0)
credit_score = xgb_model.predict_proba(X_credit)[:, 1]  # default probability

# Unified Risk Score = weighted average of both signals
# 60% credit default risk + 40% fraud/anomaly risk
CREDIT_WEIGHT = 0.60
FRAUD_WEIGHT  = 0.40

unified_score = (CREDIT_WEIGHT * credit_score) + (FRAUD_WEIGHT * ae_fraud_score)

# Risk tier classification
def get_risk_tier(score):
    if score < 0.30:   return "LOW"
    elif score < 0.60: return "MEDIUM"
    else:              return "HIGH"

risk_tiers = pd.Series(unified_score).apply(get_risk_tier)

print(f"   Credit model weight  : {CREDIT_WEIGHT*100:.0f}%")
print(f"   Fraud model weight   : {FRAUD_WEIGHT*100:.0f}%")
print(f"\n   Risk tier distribution:")
print(risk_tiers.value_counts().to_string())

# Save unified scores
results_df = X_fraud.copy()
results_df["true_label"]     = y_true.values
results_df["credit_score"]   = credit_score
results_df["fraud_score"]    = ae_fraud_score
results_df["unified_score"]  = unified_score
results_df["risk_tier"]      = risk_tiers.values

results_df.to_parquet("outputs/data/unified_risk_scores.parquet", index=False)
results_df.to_csv("outputs/data/unified_risk_scores.csv", index=False)
print("\n   ✅ Unified scores saved → outputs/data/unified_risk_scores.parquet")
print("   ✅ CSV saved            → outputs/data/unified_risk_scores.csv")
print("   (Import this CSV into Power BI on Day 8!)")


# =============================================================
# SECTION 6 — CHARTS
# =============================================================

print("\n── Saving fraud detection charts ───────────────────────")

# Chart 1 — Isolation Forest anomaly score distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(iso_fraud_score[y_true==0], bins=50, alpha=0.6,
             color="#378ADD", label="Normal", density=True)
axes[0].hist(iso_fraud_score[y_true==1], bins=50, alpha=0.6,
             color="#E24B4A", label="Default/Fraud", density=True)
axes[0].set_title("Isolation Forest — Fraud Score Distribution")
axes[0].set_xlabel("Fraud score (0=normal, 1=anomaly)")
axes[0].legend()

# Risk tier pie
tier_counts = risk_tiers.value_counts()
colors_pie  = {"LOW":"#1D9E75","MEDIUM":"#EF9F27","HIGH":"#E24B4A"}
axes[1].pie(tier_counts.values,
            labels=tier_counts.index,
            colors=[colors_pie[t] for t in tier_counts.index],
            autopct="%1.1f%%", startangle=90)
axes[1].set_title("Unified Risk Tier Distribution")
save_chart("18_fraud_score_distribution.png")

# Chart 2 — Unified score vs true label
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(unified_score[y_true==0], bins=50, alpha=0.6,
        color="#378ADD", label="No Default", density=True)
ax.hist(unified_score[y_true==1], bins=50, alpha=0.6,
        color="#E24B4A", label="Default", density=True)
ax.axvline(0.30, color="#1D9E75", linestyle="--", label="Low/Med threshold")
ax.axvline(0.60, color="#EF9F27", linestyle="--", label="Med/High threshold")
ax.set_title("Unified Risk Score — Distribution by Outcome",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Unified risk score")
ax.legend()
save_chart("19_unified_score_distribution.png")

# Chart 3 — Risk tier vs default rate
fig, ax = plt.subplots(figsize=(8, 5))
tier_default = results_df.groupby("risk_tier")["true_label"].mean()
tier_order   = ["LOW", "MEDIUM", "HIGH"]
tier_default = tier_default.reindex([t for t in tier_order if t in tier_default.index])
colors_bar   = ["#1D9E75","#EF9F27","#E24B4A"][:len(tier_default)]
bars = ax.bar(tier_default.index, tier_default.values * 100, color=colors_bar, width=0.4)
for bar, val in zip(bars, tier_default.values * 100):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Actual Default Rate by Risk Tier", fontsize=13, fontweight="bold")
ax.set_ylabel("Default rate (%)")
ax.set_ylim(0, tier_default.max() * 100 * 1.3)
save_chart("20_risk_tier_default_rate.png")

# Chart 4 — Credit score vs Fraud score scatter
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    credit_score, ae_fraud_score,
    c=y_true, cmap="RdBu_r", alpha=0.3, s=5
)
ax.set_xlabel("Credit default score")
ax.set_ylabel("Fraud / anomaly score")
ax.set_title("Credit Score vs Fraud Score (coloured by true label)",
             fontsize=12, fontweight="bold")
plt.colorbar(scatter, label="0=Normal, 1=Default")
save_chart("21_credit_vs_fraud_scatter.png")

# Chart 5 — Confusion matrix Isolation Forest
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_true, iso_labels)
ConfusionMatrixDisplay(cm, display_labels=["Normal","Anomaly"]).plot(
    cmap="Oranges", ax=ax)
ax.set_title("Isolation Forest — Confusion Matrix", fontsize=12, fontweight="bold")
save_chart("22_isolation_forest_confusion.png")


# =============================================================
# SECTION 7 — SAVE FRAUD MODELS
# =============================================================

print("\n── Saving fraud models ─────────────────────────────────")

joblib.dump(iso_forest, "outputs/models/isolation_forest.pkl")
joblib.dump(scaler,     "outputs/models/scaler.pkl")
joblib.dump({
    "credit_weight"    : CREDIT_WEIGHT,
    "fraud_weight"     : FRAUD_WEIGHT,
    "contamination"    : contamination,
    "iso_score_min"    : iso_scores.min(),
    "iso_score_max"    : iso_scores.max(),
}, "outputs/models/risk_config.pkl")

print("   ✅ outputs/models/isolation_forest.pkl")
print("   ✅ outputs/models/scaler.pkl")
print("   ✅ outputs/models/risk_config.pkl")


# =============================================================
# DONE
# =============================================================

print("\n" + "=" * 60)
print("  DAY 3 COMPLETE!")
print("=" * 60)
print(f"""
  Models saved  → outputs/models/
  Charts saved  → outputs/charts/ (charts 18-22)
  Data saved    → outputs/data/unified_risk_scores.csv

  Key results for your report:
  ──────────────────────────────────────────────────
  Isolation Forest AUC : {iso_auc:.4f}
  AutoEncoder AUC      : {ae_auc:.4f}  {"✅" if AE_OK else "(fallback)"}
  Risk tiers created   : LOW / MEDIUM / HIGH
  Unified score        : 60% credit + 40% fraud
  ──────────────────────────────────────────────────

  Power BI (Day 8): import unified_risk_scores.csv
  Streamlit (Day 5): load all models from outputs/models/

  Tomorrow Day 4 — SHAP Deep Dive + Report Figures
""")
