# =============================================================
#   DAY 8 — Power BI Data Preparation
#   Run this BEFORE opening Power BI
#   python day8_powerbi_prep.py
# =============================================================
#
#   This script creates perfectly formatted CSV files
#   ready to import directly into Power BI — no extra cleaning needed
# =============================================================

import os
import pandas as pd
import numpy as np

os.makedirs("outputs/powerbi", exist_ok=True)

print("=" * 60)
print("  DAY 8 — Power BI Data Preparation")
print("=" * 60)


# =============================================================
# SECTION 1 — LOAD UNIFIED RISK SCORES
# =============================================================

print("\n── Loading unified risk data ───────────────────────────")

unified = pd.read_csv("outputs/data/unified_risk_scores.csv")
lc      = pd.read_parquet("outputs/data/lendingclub_clean.parquet")

print(f"   Unified scores : {len(unified):,} rows")
print(f"   LendingClub    : {len(lc):,} rows")


# =============================================================
# SECTION 2 — TABLE 1: CUSTOMER RISK SUMMARY
#   (Main table for Power BI — one row per customer)
# =============================================================

print("\n── Building Table 1: Customer Risk Summary ─────────────")

drop_cols = ["target_str", "dti_band", "income_band"]
lc_clean  = lc.drop(columns=[c for c in drop_cols if c in lc.columns])

# Add unified scores to lending club
customer_df = lc_clean.copy().reset_index(drop=True)

# Safely add score columns
n = min(len(customer_df), len(unified))
customer_df = customer_df.iloc[:n].copy()
unified_sub = unified.iloc[:n].copy()

customer_df["credit_score"]  = unified_sub["credit_score"].values
customer_df["fraud_score"]   = unified_sub["fraud_score"].values
customer_df["unified_score"] = unified_sub["unified_score"].values
customer_df["risk_tier"]     = unified_sub["risk_tier"].values

# Decode risk tier to numeric for Power BI sorting
tier_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
customer_df["risk_tier_num"] = customer_df["risk_tier"].map(tier_map)

# Add decision column
customer_df["decision"] = np.where(customer_df["credit_score"] >= 0.5,
                                    "REJECT", "APPROVE")

# Add risk score as 0-100 for gauges
customer_df["risk_score_100"] = (customer_df["unified_score"] * 100).round(1)

# Select key columns for Power BI
pbi_cols = [c for c in [
    "loan_amnt", "int_rate", "term", "grade", "purpose",
    "home_ownership", "annual_inc", "dti", "revol_util",
    "emp_length", "total_acc", "open_acc",
    "target", "credit_score", "fraud_score",
    "unified_score", "risk_score_100", "risk_tier",
    "risk_tier_num", "decision"
] if c in customer_df.columns]

customer_df = customer_df[pbi_cols]

# Round floats for cleaner Power BI display
for col in customer_df.select_dtypes(include="float").columns:
    customer_df[col] = customer_df[col].round(4)

customer_df.to_csv("outputs/powerbi/01_customer_risk_summary.csv", index=False)
print(f"   ✅ 01_customer_risk_summary.csv  ({len(customer_df):,} rows)")


# =============================================================
# SECTION 3 — TABLE 2: RISK TIER KPIs
#   (For KPI cards and donut charts)
# =============================================================

print("\n── Building Table 2: Risk Tier KPIs ───────────────────")

kpi_df = customer_df.groupby("risk_tier").agg(
    customer_count   = ("risk_tier",      "count"),
    avg_credit_score = ("credit_score",   "mean"),
    avg_fraud_score  = ("fraud_score",    "mean"),
    avg_unified_score= ("unified_score",  "mean"),
    default_rate     = ("target",         "mean"),
    approve_count    = ("decision",       lambda x: (x=="APPROVE").sum()),
    reject_count     = ("decision",       lambda x: (x=="REJECT").sum()),
).reset_index()

kpi_df["avg_credit_score"]  = kpi_df["avg_credit_score"].round(4)
kpi_df["avg_fraud_score"]   = kpi_df["avg_fraud_score"].round(4)
kpi_df["avg_unified_score"] = kpi_df["avg_unified_score"].round(4)
kpi_df["default_rate_pct"]  = (kpi_df["default_rate"] * 100).round(2)
kpi_df["approval_rate_pct"] = (kpi_df["approve_count"] / kpi_df["customer_count"] * 100).round(2)
kpi_df["risk_tier_num"]     = kpi_df["risk_tier"].map(tier_map)
kpi_df = kpi_df.sort_values("risk_tier_num")

kpi_df.to_csv("outputs/powerbi/02_risk_tier_kpis.csv", index=False)
print(f"   ✅ 02_risk_tier_kpis.csv  ({len(kpi_df)} rows)")
print(kpi_df[["risk_tier","customer_count","default_rate_pct","avg_credit_score"]].to_string(index=False))


# =============================================================
# SECTION 4 — TABLE 3: LOAN PURPOSE ANALYSIS
#   (For bar charts by purpose)
# =============================================================

print("\n── Building Table 3: Loan Purpose Analysis ─────────────")

if "purpose" in customer_df.columns:
    purpose_df = customer_df.groupby("purpose").agg(
        count         = ("purpose",       "count"),
        default_rate  = ("target",        "mean"),
        avg_loan      = ("loan_amnt",     "mean"),
        avg_risk      = ("unified_score", "mean"),
        high_risk_pct = ("risk_tier",     lambda x: (x=="HIGH").mean() * 100),
    ).reset_index()

    purpose_df["default_rate_pct"] = (purpose_df["default_rate"] * 100).round(2)
    purpose_df["avg_loan"]         = purpose_df["avg_loan"].round(0)
    purpose_df["avg_risk"]         = purpose_df["avg_risk"].round(4)
    purpose_df["high_risk_pct"]    = purpose_df["high_risk_pct"].round(2)
    purpose_df = purpose_df.sort_values("default_rate_pct", ascending=False)

    purpose_df.to_csv("outputs/powerbi/03_purpose_analysis.csv", index=False)
    print(f"   ✅ 03_purpose_analysis.csv  ({len(purpose_df)} rows)")


# =============================================================
# SECTION 5 — TABLE 4: GRADE ANALYSIS
#   (For loan grade heatmap)
# =============================================================

print("\n── Building Table 4: Grade Analysis ───────────────────")

if "grade" in customer_df.columns:
    grade_df = customer_df.groupby("grade").agg(
        count        = ("grade",        "count"),
        default_rate = ("target",       "mean"),
        avg_int_rate = ("int_rate",     "mean"),
        avg_risk     = ("unified_score","mean"),
    ).reset_index()

    grade_df["default_rate_pct"] = (grade_df["default_rate"] * 100).round(2)
    grade_df["avg_int_rate"]     = grade_df["avg_int_rate"].round(2)
    grade_df["avg_risk"]         = grade_df["avg_risk"].round(4)

    grade_df.to_csv("outputs/powerbi/04_grade_analysis.csv", index=False)
    print(f"   ✅ 04_grade_analysis.csv  ({len(grade_df)} rows)")


# =============================================================
# SECTION 6 — TABLE 5: MODEL PERFORMANCE SUMMARY
#   (For your model comparison page in Power BI)
# =============================================================

print("\n── Building Table 5: Model Performance ─────────────────")

model_perf = pd.DataFrame({
    "Model"       : ["Logistic Regression", "Random Forest", "XGBoost (Tuned)", "Isolation Forest"],
    "Type"        : ["Supervised", "Supervised", "Supervised", "Unsupervised"],
    "Purpose"     : ["Credit Baseline", "Credit Baseline", "Credit Scoring", "Fraud Detection"],
    "AUC_Score"   : ["See day2 output", "See day2 output", "See day2 output", "See day3 output"],
    "Tuning"      : ["None", "None", "Optuna 50 trials", "Contamination tuned"],
    "Imbalance"   : ["class_weight", "class_weight", "scale_pos_weight + SMOTE", "contamination rate"],
})
model_perf.to_csv("outputs/powerbi/05_model_performance.csv", index=False)
print(f"   ✅ 05_model_performance.csv")


# =============================================================
# SECTION 7 — PRINT POWER BI STEP BY STEP GUIDE
# =============================================================

print("\n" + "=" * 60)
print("  POWER BI COMPLETE SETUP GUIDE")
print("=" * 60)

print("""
  STEP 1 — Download Power BI Desktop (FREE)
  ─────────────────────────────────────────────
  Go to: https://powerbi.microsoft.com/desktop
  Click "Download free" → Install → Open

  STEP 2 — Import all 5 CSV files
  ─────────────────────────────────────────────
  Home → Get Data → Text/CSV → import each file:
    outputs/powerbi/01_customer_risk_summary.csv
    outputs/powerbi/02_risk_tier_kpis.csv
    outputs/powerbi/03_purpose_analysis.csv
    outputs/powerbi/04_grade_analysis.csv
    outputs/powerbi/05_model_performance.csv
  Click "Load" for each one.

  STEP 3 — Build Page 1: EXECUTIVE OVERVIEW
  ─────────────────────────────────────────────
  Visuals to add:
  [1] Card — Total Customers
      Field: COUNT of customer_risk_summary[loan_amnt]

  [2] Card — Overall Default Rate
      Field: AVERAGE of customer_risk_summary[target]
      Format as percentage

  [3] Card — High Risk Customers
      Field: from risk_tier_kpis → filter risk_tier = HIGH → customer_count

  [4] Donut Chart — Risk Tier Distribution
      Legend: risk_tier_kpis[risk_tier]
      Values: risk_tier_kpis[customer_count]
      Colors: LOW=#1D9E75, MEDIUM=#EF9F27, HIGH=#E24B4A

  [5] Bar Chart — Default Rate by Risk Tier
      X-axis: risk_tier_kpis[risk_tier]
      Y-axis: risk_tier_kpis[default_rate_pct]
      Sort by: risk_tier_num ascending

  [6] Slicer — Filter by risk_tier
      Field: customer_risk_summary[risk_tier]

  STEP 4 — Build Page 2: CREDIT RISK ANALYSIS
  ─────────────────────────────────────────────
  [1] Bar Chart — Default Rate by Loan Purpose
      X-axis: purpose_analysis[purpose]
      Y-axis: purpose_analysis[default_rate_pct]
      Sort descending

  [2] Bar Chart — Default Rate by Loan Grade
      X-axis: grade_analysis[grade]
      Y-axis: grade_analysis[default_rate_pct]
      Sort by grade A→G

  [3] Scatter Chart — Loan Amount vs Credit Score
      X-axis: customer_risk_summary[loan_amnt]
      Y-axis: customer_risk_summary[credit_score]
      Legend: customer_risk_summary[risk_tier]
      Colors: LOW=#1D9E75, MEDIUM=#EF9F27, HIGH=#E24B4A

  [4] Histogram (use Column chart with bins) — Credit Score Distribution
      X-axis: customer_risk_summary[credit_score] (binned)
      Y-axis: Count

  [5] Table — Top 20 Highest Risk Customers
      Columns: loan_amnt, int_rate, dti, credit_score,
               unified_score, risk_tier, decision
      Sort by: unified_score descending
      Top N filter: 20

  STEP 5 — Build Page 3: FRAUD DETECTION
  ─────────────────────────────────────────────
  [1] Card — Avg Fraud Score
      Field: AVERAGE of customer_risk_summary[fraud_score]

  [2] Card — Flagged High Risk
      Field: COUNTIF risk_tier = HIGH

  [3] Scatter Chart — Credit vs Fraud Score
      X-axis: credit_score
      Y-axis: fraud_score
      Legend: risk_tier
      Colors: LOW=#1D9E75, MEDIUM=#EF9F27, HIGH=#E24B4A

  [4] Bar Chart — Avg Fraud Score by Loan Purpose
      X-axis: purpose
      Y-axis: AVG fraud_score
      Sort descending

  [5] Gauge — Overall Portfolio Risk
      Value: AVERAGE unified_score * 100
      Min: 0, Max: 100
      Target: 30 (low risk threshold)

  STEP 6 — Build Page 4: MODEL PERFORMANCE
  ─────────────────────────────────────────────
  [1] Table — Model Comparison
      Source: model_performance table
      All columns

  [2] Text box — add your AUC scores manually from day2 output

  [3] Image — insert your charts from outputs/charts/:
      - 10_roc_curves.png
      - 11_confusion_matrix.png
      - 13_model_comparison.png
      - 23_shap_beeswarm.png
      (Insert → Image → select file)

  STEP 7 — STYLING TIPS
  ─────────────────────────────────────────────
  - Page background: #f8f9fa (light grey)
  - Title font: Segoe UI Bold, size 18
  - Card background: white with shadow
  - Use consistent colors: LOW=#1D9E75 MED=#EF9F27 HIGH=#E24B4A
  - Add your college name + project title as text box on each page

  STEP 8 — PUBLISH (get shareable link)
  ─────────────────────────────────────────────
  1. Sign up FREE at https://app.powerbi.com
  2. In Power BI Desktop: File → Publish → Publish to Power BI
  3. Select "My workspace"
  4. Go to app.powerbi.com → find your report
  5. Share → Get link → Copy link
  6. Paste this link in your project report!

  DAX MEASURES (add in Power BI):
  ─────────────────────────────────────────────
  High Risk % =
  DIVIDE(
      COUNTROWS(FILTER('01_customer_risk_summary', '01_customer_risk_summary'[risk_tier] = "HIGH")),
      COUNTROWS('01_customer_risk_summary')
  ) * 100

  Approval Rate =
  DIVIDE(
      COUNTROWS(FILTER('01_customer_risk_summary', '01_customer_risk_summary'[decision] = "APPROVE")),
      COUNTROWS('01_customer_risk_summary')
  ) * 100

  Avg Risk Score =
  AVERAGE('01_customer_risk_summary'[risk_score_100])
""")

print("=" * 60)
print("  Power BI files saved to outputs/powerbi/")
print("  DAY 8 COMPLETE!")
print("=" * 60)
