# =============================================================
#   DAY 6 — Streamlit Polish + Deploy
#   DAY 7 — Power BI + Final Submission Prep
#   Run this file to verify everything is ready
#   python day6_7_verify.py
# =============================================================

import os
import sys

print("=" * 60)
print("  DAY 6-7 VERIFICATION + DEPLOYMENT GUIDE")
print("=" * 60)


# =============================================================
# SECTION 1 — CHECK ALL FILES EXIST
# =============================================================

print("\n── Checking all required files ─────────────────────────")

required_files = {
    "Day 1 - Clean data" : [
        "outputs/data/lendingclub_clean.parquet",
        "outputs/data/german_credit_clean.parquet",
        "outputs/data/home_credit_clean.parquet",
    ],
    "Day 2 - Models" : [
        "outputs/models/xgboost_credit_model.pkl",
        "outputs/models/random_forest_model.pkl",
        "outputs/models/logistic_regression_model.pkl",
        "outputs/models/feature_columns.pkl",
    ],
    "Day 3 - Fraud models" : [
        "outputs/models/isolation_forest.pkl",
        "outputs/models/scaler.pkl",
        "outputs/models/risk_config.pkl",
        "outputs/data/unified_risk_scores.parquet",
        "outputs/data/unified_risk_scores.csv",
    ],
    "Day 5 - App" : [
        "day5_app.py",
    ]
}

all_ok = True
for section, files in required_files.items():
    print(f"\n  {section}:")
    for f in files:
        exists = os.path.exists(f)
        status = "✅" if exists else "❌ MISSING"
        print(f"    {status}  {f}")
        if not exists:
            all_ok = False

if all_ok:
    print("\n✅ ALL FILES PRESENT — Ready to deploy!")
else:
    print("\n❌ Some files missing — run the missing day scripts first")
    sys.exit(1)


# =============================================================
# SECTION 2 — GENERATE requirements.txt
# =============================================================

print("\n── Generating requirements.txt ─────────────────────────")

requirements = """pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.1.1
imbalanced-learn==0.12.3
optuna==3.6.1
shap==0.45.1
streamlit==1.36.0
plotly==5.22.0
matplotlib==3.9.1
seaborn==0.13.2
joblib==1.4.2
pyarrow==16.1.0
fastparquet==2024.5.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

print("   ✅ requirements.txt created")


# =============================================================
# SECTION 3 — GENERATE README.md
# =============================================================

print("\n── Generating README.md ────────────────────────────────")

readme = """# 🏦 AI Banking Risk Intelligence Platform
### Final Year Project — Data Science

## Overview
An AI-powered banking risk platform that combines **credit default prediction** 
and **fraud anomaly detection** with full **explainable AI (SHAP)** — 
deployed as an interactive Streamlit web application.

## Datasets Used
| Dataset | Rows | Purpose |
|---|---|---|
| LendingClub | 20,349 | Primary credit scoring |
| German Credit | 1,000 | Secondary risk benchmark |
| Home Credit | 307,511 | Large-scale banking data |

## Models
| Model | Type | Purpose | AUC |
|---|---|---|---|
| XGBoost | Supervised | Credit default prediction | ~0.90+ |
| Random Forest | Supervised | Baseline comparison | ~0.85+ |
| Isolation Forest | Unsupervised | Fraud anomaly detection | - |

## Key Features
- ✅ Real-time credit risk scoring with risk tier (LOW / MEDIUM / HIGH)
- ✅ SHAP waterfall charts explaining every prediction
- ✅ Fraud anomaly detection using Isolation Forest
- ✅ Unified Risk Score (60% credit + 40% fraud)
- ✅ Portfolio dashboard with interactive Plotly charts
- ✅ Batch scoring — upload CSV, download scored results for Power BI
- ✅ Class imbalance handled with SMOTE
- ✅ Hyperparameter tuning with Optuna (50 trials)

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run pipeline in order
```bash
python day1_final.py    # Data cleaning + EDA
python day2_model.py    # XGBoost credit model
python day3_fraud.py    # Fraud detection
python day4_shap.py     # SHAP explainability
```

### 3. Launch Streamlit app
```bash
streamlit run day5_app.py
```

## Project Structure
```
BankingRiskAI/
├── data/                          # Raw datasets
├── outputs/
│   ├── charts/                    # 28 EDA + model charts
│   ├── data/                      # Clean parquet files
│   ├── models/                    # Trained ML models
│   └── shap/                      # SHAP explanations
├── day1_final.py                  # Data pipeline
├── day2_model.py                  # Credit scoring model
├── day3_fraud.py                  # Fraud detection
├── day4_shap.py                   # Explainability
├── day5_app.py                    # Streamlit application
├── requirements.txt
└── README.md
```

## Technology Stack
- **Python 3.11**
- **Machine Learning**: XGBoost, Scikit-learn, Imbalanced-learn
- **Explainability**: SHAP, LIME
- **Hyperparameter Tuning**: Optuna
- **Web App**: Streamlit + Plotly
- **Business Intelligence**: Power BI
- **Data Processing**: Pandas, NumPy
"""

with open("README.md", "w") as f:
    f.write(readme)

print("   ✅ README.md created")


# =============================================================
# SECTION 4 — STREAMLIT DEPLOYMENT GUIDE
# =============================================================

print("\n── Day 6: Streamlit Deployment Steps ───────────────────")
print("""
  STEP 1: Test locally
  ─────────────────────────────────────────────
  streamlit run day5_app.py
  Open: http://localhost:8501

  STEP 2: Push to GitHub
  ─────────────────────────────────────────────
  git init
  git add .
  git commit -m "AI Banking Risk Platform - Final Year Project"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/BankingRiskAI.git
  git push -u origin main

  STEP 3: Deploy to Streamlit Cloud (FREE)
  ─────────────────────────────────────────────
  1. Go to https://share.streamlit.io
  2. Sign in with GitHub
  3. Click "New app"
  4. Select your repo → branch: main → file: day5_app.py
  5. Click Deploy
  6. Get your public URL → paste in report!

  NOTE: Add this to your GitHub repo as .streamlit/config.toml:
  [theme]
  primaryColor = "#0f3460"
  backgroundColor = "#ffffff"
  secondaryBackgroundColor = "#f8f9fa"
  textColor = "#1a1a2e"
""")


# =============================================================
# SECTION 5 — POWER BI GUIDE
# =============================================================

print("\n── Day 7: Power BI Setup Guide ─────────────────────────")
print("""
  FILE TO IMPORT INTO POWER BI:
  outputs/data/unified_risk_scores.csv

  POWER BI STEPS:
  ─────────────────────────────────────────────
  1. Open Power BI Desktop (free download from Microsoft)
  2. Home → Get Data → Text/CSV
  3. Select unified_risk_scores.csv → Load

  PAGE 1 — Credit Risk Overview:
  ─────────────────────────────────────────────
  - Donut chart: risk_tier distribution
  - KPI cards: total customers, avg credit_score, avg unified_score
  - Bar chart: count of HIGH / MEDIUM / LOW risk customers
  - Slicer: filter by risk_tier

  PAGE 2 — Fraud Analysis:
  ─────────────────────────────────────────────
  - Histogram: fraud_score distribution
  - Scatter: credit_score vs fraud_score (color = risk_tier)
  - Table: top 20 highest unified_score customers

  PAGE 3 — Portfolio KPIs:
  ─────────────────────────────────────────────
  - Card: % customers in HIGH risk tier
  - Card: avg credit default probability
  - Bar chart: actual default rate by risk tier
  - Line chart: unified_score distribution

  PUBLISH:
  ─────────────────────────────────────────────
  - File → Publish to Power BI service (free account)
  - Get shareable link → paste in report
""")


# =============================================================
# SECTION 6 — VIVA PREP
# =============================================================

print("\n── Viva Preparation ────────────────────────────────────")
print("""
  TOP 10 QUESTIONS EXAMINERS WILL ASK:
  ─────────────────────────────────────────────
  Q1: Why XGBoost over other models?
  A:  XGBoost handles tabular data best, supports class imbalance
      via scale_pos_weight, and Optuna found AUC of X.XX

  Q2: How did you handle class imbalance?
  A:  SMOTE on training set only (never test set) + scale_pos_weight
      in XGBoost + evaluated with ROC-AUC not accuracy

  Q3: What is SHAP and why did you use it?
  A:  SHapley Additive exPlanations — game theory approach that
      shows each feature's contribution. Banks need explainability
      for regulatory compliance (GDPR, model transparency)

  Q4: What is Isolation Forest?
  A:  Unsupervised anomaly detection — isolates outliers by random
      feature splits. More anomalous points need fewer splits.
      No labels needed — perfect for fraud where labels are rare.

  Q5: What is the Unified Risk Score?
  A:  60% credit default probability + 40% fraud anomaly score.
      Weighted average giving a single 0-1 risk index per customer.

  Q6: Why SMOTE?
  A:  Dataset was imbalanced (9.7% default). SMOTE creates synthetic
      minority samples by interpolating between existing ones.

  Q7: What is your model's AUC and what does it mean?
  A:  AUC = X.XX. Probability that model ranks a random defaulter
      higher than a random non-defaulter. 0.5 = random, 1.0 = perfect.

  Q8: Why Optuna for tuning?
  A:  Bayesian optimisation — smarter than grid search. Learns from
      previous trials to focus on promising parameter regions. 50
      trials finds near-optimal params much faster than grid search.

  Q9: How is your project different from basic loan prediction?
  A:  1) Dual model (credit + fraud) 2) SHAP explainability
      3) Unified risk score 4) Live Streamlit demo 5) Power BI
      business dashboard — end-to-end production-ready system

  Q10: What are your project's limitations?
  A:   German Credit target was rule-engineered (no ground truth),
       LendingClub subset is only 20K rows, AutoEncoder needs
       TensorFlow. Future work: real-time API, bank integration.
""")


# =============================================================
# SECTION 7 — FINAL CHECKLIST
# =============================================================

print("\n── Final Submission Checklist ──────────────────────────")
print("""
  REPORT (Day 6-7):
  □ Abstract (200 words)
  □ Introduction — why banking risk matters
  □ Literature review — 8-10 papers on XGBoost, SHAP, fraud detection
  □ Methodology — datasets, models, SMOTE, Optuna, SHAP
  □ Results — all AUC scores, charts 10-28, confusion matrix
  □ Discussion — what worked, limitations, future work
  □ Conclusion
  □ References

  SUBMISSION PACKAGE:
  □ Project report (PDF)
  □ GitHub repo with clean README
  □ Streamlit public URL (paste in report)
  □ Power BI shareable link (paste in report)
  □ Demo video (2 min walkthrough of Streamlit app)
  □ Code zip file

  VIVA DEMO ORDER:
  1. Open Streamlit app in browser
  2. Go to Credit Scorer → enter a customer → show SHAP waterfall
  3. Go to Fraud Detector → show anomaly gauge
  4. Go to Portfolio Dashboard → show risk distribution charts
  5. Go to Batch Scoring → upload CSV → download results
  6. Open Power BI → show executive dashboard
""")

print("\n" + "=" * 60)
print("  YOU ARE READY! GO SUBMIT. 🎓🔥")
print("=" * 60)
