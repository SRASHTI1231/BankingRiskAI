# =============================================================
#   DAY 4 — SHAP Deep Dive + All Report Figures
#   Run: python day4_shap.py
#   Requires: Day 2 and Day 3 completed first
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

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/shap",   exist_ok=True)

print("=" * 60)
print("  DAY 4 — SHAP Explainability + Report Figures")
print("=" * 60)


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → {path}")


# =============================================================
# SECTION 1 — LOAD DATA + MODELS
# =============================================================

print("\n── Loading data and models ─────────────────────────────")

lc        = pd.read_parquet("outputs/data/lendingclub_clean.parquet")
feat_cols  = joblib.load("outputs/models/feature_columns.pkl")
xgb       = joblib.load("outputs/models/xgboost_credit_model.pkl")
rf        = joblib.load("outputs/models/random_forest_model.pkl")

drop_cols  = ["target_str", "dti_band", "income_band"]
lc_model   = lc.drop(columns=[c for c in drop_cols if c in lc.columns])

le = LabelEncoder()
for col in lc_model.select_dtypes(include="object").columns:
    lc_model[col] = le.fit_transform(lc_model[col].astype(str))

X = lc_model.drop("target", axis=1).reindex(columns=feat_cols, fill_value=0)
y = lc_model["target"]

print(f"   Features : {X.shape[1]}  |  Samples : {X.shape[0]:,}")


# =============================================================
# SECTION 2 — GLOBAL SHAP (full dataset)
# =============================================================

print("\n── Computing SHAP values ───────────────────────────────")

explainer   = shap.TreeExplainer(xgb)
# Use a sample of 2000 for speed
sample      = X.sample(min(2000, len(X)), random_state=42)
shap_values = explainer.shap_values(sample)

print(f"   SHAP computed on {len(sample):,} samples")

# Chart 1 — Beeswarm (best chart for report)
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_values, sample, show=False, max_display=15)
plt.title("SHAP — Feature Impact (Beeswarm)", fontsize=13, fontweight="bold")
save_chart("outputs/charts/23_shap_beeswarm.png")

# Chart 2 — Bar (mean absolute SHAP)
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=15)
plt.title("SHAP — Mean Absolute Feature Importance", fontsize=13, fontweight="bold")
save_chart("outputs/charts/24_shap_bar.png")

# Chart 3 — Dependence plot for top feature
top_feature = pd.DataFrame({
    "feature": sample.columns,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False).iloc[0]["feature"]

plt.figure(figsize=(9, 5))
shap.dependence_plot(top_feature, shap_values, sample,
                     show=False, interaction_index=None)
plt.title(f"SHAP Dependence — {top_feature}", fontsize=13, fontweight="bold")
save_chart(f"outputs/charts/25_shap_dependence_{top_feature}.png")


# =============================================================
# SECTION 3 — PER CUSTOMER EXPLANATION FUNCTION
# =============================================================

print("\n── Building explain_customer() function ────────────────")

def explain_customer(customer_row_df, save_path=None):
    """
    Takes a single row DataFrame (same columns as X),
    returns SHAP waterfall chart and risk verdict.
    Use this function inside Streamlit on Day 5.
    """
    row_aligned = customer_row_df.reindex(columns=feat_cols, fill_value=0)
    shap_vals   = explainer.shap_values(row_aligned)
    credit_prob = xgb.predict_proba(row_aligned)[0][1]

    if credit_prob < 0.30:
        tier, color = "LOW RISK ✅", "#1D9E75"
    elif credit_prob < 0.60:
        tier, color = "MEDIUM RISK ⚠️", "#EF9F27"
    else:
        tier, color = "HIGH RISK 🔴", "#E24B4A"

    fig = plt.figure(figsize=(11, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_vals[0],
            base_values   = explainer.expected_value,
            data          = row_aligned.iloc[0],
            feature_names = feat_cols
        ), show=False
    )
    plt.title(f"Customer Explanation | Default Probability: {credit_prob:.1%} | {tier}",
              fontsize=11, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return credit_prob, tier, fig


# Explain 3 sample customers and save charts
for i, idx in enumerate([0, 10, 50]):
    row = X.iloc[[idx]]
    prob, tier, _ = explain_customer(
        row,
        save_path=f"outputs/shap/customer_{i+1}_explanation.png"
    )
    print(f"   Customer {i+1}: prob={prob:.1%}  tier={tier}")

print("   ✅ 3 customer explanation charts saved → outputs/shap/")


# =============================================================
# SECTION 4 — LIME (backup explainability)
# =============================================================

print("\n── LIME explainability (backup) ────────────────────────")

try:
    import lime
    import lime.lime_tabular

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X.values,
        feature_names  = X.columns.tolist(),
        class_names    = ["Fully Paid", "Charged Off"],
        mode           = "classification",
        random_state   = 42
    )

    lime_exp = lime_explainer.explain_instance(
        X.iloc[0].values,
        xgb.predict_proba,
        num_features=10
    )

    fig = lime_exp.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    plt.title("LIME — Customer 1 Explanation", fontsize=12, fontweight="bold")
    save_chart("outputs/charts/26_lime_explanation.png")
    print("   ✅ LIME chart saved")

except ImportError:
    print("   ⚠️  LIME not installed — run: pip install lime")


# =============================================================
# SECTION 5 — PORTFOLIO RISK DASHBOARD FIGURES
# =============================================================

print("\n── Generating portfolio figures for report ─────────────")

unified = pd.read_csv("outputs/data/unified_risk_scores.csv")

# Chart — Risk tier distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

tier_counts = unified["risk_tier"].value_counts()
colors_map  = {"LOW":"#1D9E75","MEDIUM":"#EF9F27","HIGH":"#E24B4A"}
tier_order  = [t for t in ["LOW","MEDIUM","HIGH"] if t in tier_counts.index]
bar_colors  = [colors_map[t] for t in tier_order]

axes[0].bar(tier_order,
            [tier_counts[t] for t in tier_order],
            color=bar_colors, width=0.4)
axes[0].set_title("Portfolio — Customer Risk Tier Counts")
axes[0].set_ylabel("Number of customers")

# Avg unified score by tier
avg_score = unified.groupby("risk_tier")["unified_score"].mean().reindex(tier_order)
axes[1].bar(tier_order, avg_score.values, color=bar_colors, width=0.4)
axes[1].set_title("Average Unified Score by Tier")
axes[1].set_ylabel("Avg unified risk score")
save_chart("outputs/charts/27_portfolio_risk_tiers.png")

# Chart — Credit score distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(unified["credit_score"], bins=50, color="#378ADD",
        edgecolor="white", alpha=0.8)
ax.axvline(0.30, color="#1D9E75", linestyle="--", lw=2, label="Low/Med (0.30)")
ax.axvline(0.60, color="#E24B4A", linestyle="--", lw=2, label="Med/High (0.60)")
ax.set_title("Credit Default Score Distribution — Full Portfolio",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Credit default probability")
ax.set_ylabel("Count")
ax.legend()
save_chart("outputs/charts/28_credit_score_distribution.png")

print("\n" + "=" * 60)
print("  DAY 4 COMPLETE!")
print("=" * 60)
print("""
  Charts saved  → outputs/charts/ (charts 23-28)
  SHAP charts   → outputs/shap/   (3 customer explanations)

  Key outputs for your viva:
  ──────────────────────────────────────────────────
  23_shap_beeswarm.png      ← best chart for report
  24_shap_bar.png           ← feature importance
  outputs/shap/customer_*.png ← live demo in viva

  Tomorrow Day 5 — Streamlit App
  Run: pip install streamlit
  Then: streamlit run day5_app.py
""")
