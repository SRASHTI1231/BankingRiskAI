# =============================================================
#   DAY 1 — AI Banking Risk Intelligence Platform
#   FIXED VERSION 4 — boxplot palette fixed
#   Run: python day1_final.py
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/data",   exist_ok=True)

print("=" * 60)
print("  DAY 1 — Banking Risk Platform  |  VS Code Edition")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def save_chart(filename):
    plt.tight_layout()
    path = f"outputs/charts/{filename}"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Chart saved → {path}")


# =============================================================
# SECTION 1 — LOAD DATASETS
# =============================================================

print("\n── Loading datasets ────────────────────────────────────")

# ── LendingClub ───────────────────────────────────────────────
LC_COLS_WANTED = [
    "loan_amnt", "term", "int_rate", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "revol_util", "total_acc",
    "open_acc", "pub_rec", "mort_acc", "inq_last_6mths",
    "revol_bal", "total_pymnt", "out_prncp",
    "target", "loan_to_income", "dti_band"
]
try:
    lc_raw = pd.read_csv("data/loan.csv", nrows=150_000, low_memory=False)
    LC_COLS = [c for c in LC_COLS_WANTED if c in lc_raw.columns]
    lc = lc_raw[LC_COLS].copy()
    del lc_raw
    print(f"   [1/3] LendingClub loaded     → {len(lc):,} rows × {lc.shape[1]} cols")
    print(f"         Default rate           → {lc['target'].mean()*100:.1f}%")
    LC_OK = True
except FileNotFoundError:
    print("   [1/3] ⚠️  loan.csv not found in data/")
    LC_OK = False

# ── German Credit ─────────────────────────────────────────────
try:
    gc = pd.read_csv("data/german_credit_data.csv", index_col=0)
    gc.columns = [c.strip().lower().replace(" ", "_") for c in gc.columns]

    risk_score = pd.Series(0, index=gc.index)
    if "credit_amount" in gc.columns:
        risk_score += (gc["credit_amount"] > 7500).astype(int)
    if "duration" in gc.columns:
        risk_score += (gc["duration"] > 36).astype(int)
    if "checking_account" in gc.columns:
        risk_score += gc["checking_account"].isin(["little"]).astype(int)
        risk_score += gc["checking_account"].isna().astype(int)
    if "saving_accounts" in gc.columns:
        risk_score += gc["saving_accounts"].isna().astype(int)
        risk_score += gc["saving_accounts"].isin(["little"]).astype(int)

    gc["target"] = (risk_score >= 2).astype(int)
    print(f"   [2/3] German Credit loaded   → {len(gc):,} rows × {gc.shape[1]} cols")
    print(f"         Target auto-created    → {gc['target'].mean()*100:.1f}% flagged high risk")
    GC_OK = True
except FileNotFoundError:
    print("   [2/3] ⚠️  german_credit_data.csv not found in data/")
    GC_OK = False

# ── Home Credit ───────────────────────────────────────────────
HC_COLS = [
    "TARGET", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"
]
try:
    hc = pd.read_csv("data/application_train.csv", usecols=HC_COLS)
    hc.columns = [c.lower() for c in hc.columns]
    print(f"   [3/3] Home Credit loaded     → {len(hc):,} rows × {hc.shape[1]} cols")
    print(f"         Default rate           → {hc['target'].mean()*100:.1f}%")
    HC_OK = True
except FileNotFoundError:
    print("   [3/3] ⚠️  application_train.csv not found in data/")
    HC_OK = False


# =============================================================
# SECTION 2 — CLEAN LENDINGLUB
# =============================================================

if LC_OK:
    print("\n── Cleaning LendingClub ────────────────────────────────")

    if "term" in lc.columns:
        lc["term"] = pd.to_numeric(
            lc["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    for col in ["int_rate", "revol_util"]:
        if col in lc.columns:
            lc[col] = pd.to_numeric(
                lc[col].astype(str).str.strip("%"), errors="coerce")
    if "emp_length" in lc.columns:
        emp_map = {"< 1 year": "0.5", "10+ years": "10"}
        lc["emp_length"] = pd.to_numeric(
            lc["emp_length"].replace(emp_map).astype(str)
                            .str.extract(r"(\d+\.?\d*)")[0],
            errors="coerce")

    num_cols = lc.select_dtypes(include="number").columns.drop("target", errors="ignore")
    lc[num_cols] = lc[num_cols].fillna(lc[num_cols].median())
    for col in lc.select_dtypes(include="object").columns:
        lc[col] = lc[col].fillna(lc[col].mode()[0])

    if "loan_to_income" not in lc.columns:
        if "loan_amnt" in lc.columns and "annual_inc" in lc.columns:
            lc["loan_to_income"] = lc["loan_amnt"] / (lc["annual_inc"] + 1)
    if "loan_amnt" in lc.columns and "int_rate" in lc.columns and "term" in lc.columns:
        lc["monthly_payment_est"] = (lc["loan_amnt"] * (lc["int_rate"] / 100)) / (lc["term"] + 1)
    if "dti" in lc.columns:
        lc["high_dti_flag"] = (lc["dti"] > 30).astype(np.int8)
    if "int_rate" in lc.columns:
        lc["high_int_flag"] = (lc["int_rate"] > 18).astype(np.int8)
    if "annual_inc" in lc.columns:
        lc["annual_inc_log"] = np.log1p(lc["annual_inc"])
    if "loan_amnt" in lc.columns:
        lc["loan_amnt_log"] = np.log1p(lc["loan_amnt"])

    num_cols = lc.select_dtypes(include="number").columns.drop("target", errors="ignore")
    lc[num_cols] = lc[num_cols].astype("float32")

    # ── FIX: make target string so seaborn palette works ─────
    lc["target_str"] = lc["target"].map({0: "Fully Paid", 1: "Charged Off"})

    print(f"   Rows × Cols  : {lc.shape[0]:,} × {lc.shape[1]}")
    print(f"   Default rate : {lc['target'].mean()*100:.1f}%")
    print(f"   Missing      : {lc.isnull().mean().mean()*100:.2f}%")


# =============================================================
# SECTION 3 — CLEAN GERMAN CREDIT
# =============================================================

if GC_OK:
    print("\n── Cleaning German Credit ──────────────────────────────")

    gc_num = gc.select_dtypes(include="number").columns.drop("target", errors="ignore")
    gc[gc_num] = gc[gc_num].fillna(gc[gc_num].median())
    for col in gc.select_dtypes(include="object").columns:
        gc[col] = gc[col].fillna(gc[col].mode()[0])
    if "credit_amount" in gc.columns and "duration" in gc.columns:
        gc["monthly_rate"] = gc["credit_amount"] / (gc["duration"] + 1)
    if "age" in gc.columns:
        gc["young_borrower_flag"] = (gc["age"] < 25).astype(int)

    le = LabelEncoder()
    for col in gc.select_dtypes(include="object").columns:
        gc[col] = le.fit_transform(gc[col].astype(str))

    print(f"   Rows × Cols  : {gc.shape[0]:,} × {gc.shape[1]}")
    print(f"   High risk %  : {gc['target'].mean()*100:.1f}%")


# =============================================================
# SECTION 4 — CLEAN HOME CREDIT
# =============================================================

if HC_OK:
    print("\n── Cleaning Home Credit ────────────────────────────────")

    hc["age_years"]      = (-hc["days_birth"] / 365).round(1)
    hc["days_employed"]  = hc["days_employed"].replace(365243, np.nan)
    hc["employed_years"] = (-hc["days_employed"] / 365).clip(lower=0)
    hc.drop(["days_birth", "days_employed"], axis=1, inplace=True)

    hc["credit_to_income"]  = hc["amt_credit"]  / (hc["amt_income_total"] + 1)
    hc["annuity_to_income"] = hc["amt_annuity"] / (hc["amt_income_total"] + 1)
    hc["ext_source_mean"]   = hc[["ext_source_1","ext_source_2","ext_source_3"]].mean(axis=1)
    hc["ext_source_min"]    = hc[["ext_source_1","ext_source_2","ext_source_3"]].min(axis=1)

    hc_num = hc.select_dtypes(include="number").columns.drop("target", errors="ignore")
    hc[hc_num] = hc[hc_num].fillna(hc[hc_num].median())
    for col in hc.select_dtypes(include="object").columns:
        hc[col] = hc[col].fillna(hc[col].mode()[0])

    le = LabelEncoder()
    for col in hc.select_dtypes(include="object").columns:
        hc[col] = le.fit_transform(hc[col].astype(str))

    hc[hc_num] = hc[hc_num].astype("float32")

    print(f"   Rows × Cols  : {hc.shape[0]:,} × {hc.shape[1]}")
    print(f"   Default rate : {hc['target'].mean()*100:.1f}%")


# =============================================================
# SECTION 5 — EDA CHARTS
# =============================================================

print("\n── Generating EDA charts ───────────────────────────────")

if LC_OK:

    # Chart 1 — Class balance + default by purpose
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    counts = lc["target"].value_counts()
    bars = axes[0].bar(["Fully Paid","Charged Off"],
                       counts.values, color=["#378ADD","#E24B4A"], width=0.45)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 50,
                     f"{val:,}\n({val/len(lc)*100:.1f}%)",
                     ha="center", fontsize=10)
    axes[0].set_title("Class balance — LendingClub")
    axes[0].set_ylabel("Count")
    if "purpose" in lc.columns:
        purpose_def = lc.groupby("purpose", observed=True)["target"].mean().sort_values()
        axes[1].barh(purpose_def.index.astype(str), purpose_def.values, color="#534AB7")
        axes[1].set_title("Default rate by loan purpose")
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    save_chart("01_class_balance_purpose.png")

    # Chart 2 — Feature distributions
    plot_features = [(c, l) for c, l in [
        ("int_rate",       "Interest rate (%)"),
        ("dti",            "Debt-to-income ratio"),
        ("revol_util",     "Revolving utilisation (%)"),
        ("emp_length",     "Employment length (yrs)"),
        ("annual_inc_log", "Annual income (log)"),
        ("loan_amnt_log",  "Loan amount (log)"),
    ] if c in lc.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (col, label) in enumerate(plot_features):
        lc[lc["target"]==0][col].hist(ax=axes[i], bins=40, alpha=0.6,
                                       color="#378ADD", label="Paid", density=True)
        lc[lc["target"]==1][col].hist(ax=axes[i], bins=40, alpha=0.6,
                                       color="#E24B4A", label="Default", density=True)
        axes[i].set_title(label)
        axes[i].legend(fontsize=9)
    plt.suptitle("Feature distributions — Paid vs Default",
                 fontsize=13, fontweight="bold", y=1.01)
    save_chart("02_feature_distributions.png")

    # Chart 3 — Correlation heatmap
    lc_num_df = lc.select_dtypes(include="number")
    corr = lc_num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.4,
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation matrix", fontsize=13, fontweight="bold")
    save_chart("03_correlation_heatmap.png")

    # Chart 4 — Boxplots (FIX: use target_str with string palette)
    box_cols = [(c, l) for c, l in [
        ("int_rate","Interest rate"),("dti","DTI ratio"),("revol_util","Revolving util")
    ] if c in lc.columns]
    fig, axes = plt.subplots(1, len(box_cols), figsize=(5*len(box_cols), 4))
    if len(box_cols) == 1: axes = [axes]
    for ax, (col, label) in zip(axes, box_cols):
        sns.boxplot(data=lc, x="target_str", y=col, ax=ax,
                    order=["Fully Paid","Charged Off"],
                    palette={"Fully Paid":"#378ADD","Charged Off":"#E24B4A"})
        ax.set_xlabel("")
        ax.set_title(label)
    plt.suptitle("Risk indicators by outcome", fontsize=13, fontweight="bold")
    save_chart("04_boxplots.png")

    # Chart 5 — Home ownership + grade
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    if "home_ownership" in lc.columns:
        ho = lc.groupby("home_ownership", observed=True)["target"].mean().sort_values(ascending=False)
        axes[0].bar(ho.index.astype(str), ho.values, color="#1D9E75", width=0.5)
        axes[0].set_title("Default rate by home ownership")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    if "grade" in lc.columns:
        grade = lc.groupby("grade", observed=True)["target"].mean().sort_index()
        axes[1].bar(grade.index.astype(str), grade.values, color="#EF9F27", width=0.5)
        axes[1].set_title("Default rate by loan grade")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    save_chart("05_categorical_rates.png")

    # Chart 6 — Engineered features
    eng_cols = [(c, l) for c, l in [
        ("loan_to_income",      "Loan-to-income ratio"),
        ("monthly_payment_est", "Est. monthly payment"),
        ("high_dti_flag",       "High DTI flag"),
    ] if c in lc.columns]
    if eng_cols:
        fig, axes = plt.subplots(1, min(3, len(eng_cols)), figsize=(15, 4))
        if len(eng_cols) == 1: axes = [axes]
        for ax, (col, label) in zip(axes, eng_cols[:3]):
            if lc[col].nunique() <= 2:
                fd = lc.groupby(col)["target"].mean()
                ax.bar([str(x) for x in fd.index], fd.values,
                       color=["#378ADD","#E24B4A"], width=0.4)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            else:
                lo, hi = lc[col].quantile(.01), lc[col].quantile(.99)
                lc[lc["target"]==0][col].clip(lo, hi).hist(
                    ax=ax, bins=40, alpha=0.6, color="#378ADD", label="Paid", density=True)
                lc[lc["target"]==1][col].clip(lo, hi).hist(
                    ax=ax, bins=40, alpha=0.6, color="#E24B4A", label="Default", density=True)
                ax.legend(fontsize=9)
            ax.set_title(label)
        plt.suptitle("Engineered features", fontsize=13, fontweight="bold")
        save_chart("06_engineered_features.png")

if GC_OK:
    gc_plot = [c for c in ["credit_amount","duration","age"] if c in gc.columns]
    if gc_plot:
        fig, axes = plt.subplots(1, len(gc_plot), figsize=(5*len(gc_plot), 4))
        if len(gc_plot) == 1: axes = [axes]
        for ax, col in zip(axes, gc_plot):
            gc[gc["target"]==0][col].hist(ax=ax, bins=20, alpha=0.6,
                                           color="#378ADD", label="Low risk", density=True)
            gc[gc["target"]==1][col].hist(ax=ax, bins=20, alpha=0.6,
                                           color="#E24B4A", label="High risk", density=True)
            ax.set_title(col.replace("_"," ").title())
            ax.legend(fontsize=9)
        plt.suptitle("German Credit — Feature distributions", fontsize=13, fontweight="bold")
        save_chart("07_german_credit_eda.png")

if HC_OK:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, label in zip(axes,
        ["age_years","credit_to_income","ext_source_mean"],
        ["Age (years)","Credit-to-income","Ext credit score"]):
        v0 = hc[hc["target"]==0][col].dropna()
        v1 = hc[hc["target"]==1][col].dropna()
        ax.hist(v0.clip(v0.quantile(.01), v0.quantile(.99)),
                bins=40, alpha=0.6, color="#378ADD", label="No default", density=True)
        ax.hist(v1.clip(v1.quantile(.01), v1.quantile(.99)),
                bins=40, alpha=0.6, color="#E24B4A", label="Default", density=True)
        ax.set_title(label)
        ax.legend(fontsize=9)
    plt.suptitle("Home Credit — Key features", fontsize=13, fontweight="bold")
    save_chart("08_home_credit_eda.png")

# Chart 9 — Cross dataset comparison
fig, ax = plt.subplots(figsize=(8, 4))
ds = {}
if LC_OK: ds["LendingClub\n(20K rows)"]   = lc["target"].mean() * 100
if GC_OK: ds["German Credit\n(1K rows)"]  = gc["target"].mean() * 100
if HC_OK: ds["Home Credit\n(307K rows)"]  = hc["target"].mean() * 100
colors = ["#378ADD","#1D9E75","#534AB7"]
bars = ax.bar(list(ds.keys()), list(ds.values()), color=colors[:len(ds)], width=0.4)
for bar, val in zip(bars, ds.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Default/risk rate across all 3 datasets", fontsize=13, fontweight="bold")
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, max(ds.values()) * 1.3)
save_chart("09_cross_dataset_comparison.png")


# =============================================================
# SECTION 6 — SAVE PARQUET FILES
# =============================================================

print("\n── Saving clean datasets ───────────────────────────────")

if LC_OK:
    lc_save = lc.drop(columns=["target_str"], errors="ignore").copy()
    for col in lc_save.select_dtypes(include="category").columns:
        lc_save[col] = lc_save[col].astype(str)
    lc_save.to_parquet("outputs/data/lendingclub_clean.parquet", index=False)
    print("   ✅ outputs/data/lendingclub_clean.parquet")

if GC_OK:
    gc.to_parquet("outputs/data/german_credit_clean.parquet", index=False)
    print("   ✅ outputs/data/german_credit_clean.parquet")

if HC_OK:
    hc.to_parquet("outputs/data/home_credit_clean.parquet", index=False)
    print("   ✅ outputs/data/home_credit_clean.parquet")


# =============================================================
# SECTION 7 — FINAL SUMMARY
# =============================================================

print("\n" + "=" * 60)
print("  DATASET SUMMARY")
print("=" * 60)

all_ds = {}
if LC_OK: all_ds["LendingClub"]   = lc
if GC_OK: all_ds["German Credit"] = gc
if HC_OK: all_ds["Home Credit"]   = hc

for name, df in all_ds.items():
    num_df = df.select_dtypes(include="number")
    top4 = (num_df.corr()["target"]
                  .drop("target", errors="ignore")
                  .abs()
                  .sort_values(ascending=False)
                  .head(4))
    print(f"\n  {name}")
    print(f"  Rows: {df.shape[0]:,}  |  Cols: {df.shape[1]}  |  Risk: {df['target'].mean()*100:.1f}%")
    print(f"  Top predictors:")
    for feat, val in top4.items():
        print(f"    {feat:<35} corr = {val:.4f}")

print("\n" + "=" * 60)
print("  DAY 1 COMPLETE!")
print("=" * 60)
print("""
  outputs/charts/  -> 9 EDA charts  (paste into your report!)
  outputs/data/    -> 3 parquet files (load directly in Day 2)

  Tomorrow Day 2 — XGBoost Credit Scoring Model
  Load with:
    lc = pd.read_parquet('outputs/data/lendingclub_clean.parquet')
    gc = pd.read_parquet('outputs/data/german_credit_clean.parquet')
    hc = pd.read_parquet('outputs/data/home_credit_clean.parquet')
""")
