# =============================================================
#   DAY 5 — Banking Risk Intelligence Platform
#   PROFESSIONAL FINTECH EDITION
#   Run: streamlit run day5_app.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "RiskIQ — Banking Intelligence",
    page_icon   = "◈",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────
# PROFESSIONAL COLOR SYSTEM
# Deep navy + electric teal + warm amber + soft rose
# ─────────────────────────────────────────────────────────────

CSS = """
<style>
/* ── Root variables ── */
:root {
    --bg-primary:    #080c14;
    --bg-secondary:  #0d1421;
    --bg-card:       #111827;
    --bg-card-hover: #162032;
    --border:        rgba(148,163,184,0.08);
    --border-active: rgba(20,184,166,0.35);
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --teal:          #14b8a6;
    --teal-dark:     #0d9488;
    --teal-glow:     rgba(20,184,166,0.15);
    --amber:         #f59e0b;
    --amber-dark:    #d97706;
    --rose:          #f43f5e;
    --rose-dark:     #e11d48;
    --indigo:        #6366f1;
    --slate:         #334155;
}

/* ── Global background ── */
.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Remove default padding ── */
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 3px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--teal-glow) !important;
    border-color: var(--border-active) !important;
    color: var(--teal) !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Glass card ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    border-color: var(--border-active);
    box-shadow: 0 0 40px rgba(20,184,166,0.06);
}

/* ── Page title ── */
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 28px;
    font-weight: 400;
}

/* ── Accent line ── */
.accent-line {
    width: 40px;
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--indigo));
    border-radius: 2px;
    margin-bottom: 20px;
}

/* ── KPI cards ── */
.kpi {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: left;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--teal), var(--indigo));
    border-radius: 3px 0 0 3px;
}
.kpi:hover {
    border-color: var(--border-active);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.kpi-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 10px;
    display: block;
}

/* ── Status badges ── */
.badge {
    border-radius: 12px;
    padding: 18px 24px;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    letter-spacing: 0.3px;
}
.badge-low {
    background: rgba(20,184,166,0.08);
    border: 1px solid rgba(20,184,166,0.3);
    color: #2dd4bf;
}
.badge-med {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    color: #fbbf24;
}
.badge-high {
    background: rgba(244,63,94,0.08);
    border: 1px solid rgba(244,63,94,0.3);
    color: #fb7185;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--teal);
    margin-bottom: 8px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0d9488, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 24px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(13,148,136,0.25) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSlider > div > div > div { background: var(--teal) !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--teal) !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHART DEFAULTS
# ─────────────────────────────────────────────────────────────

BG      = "rgba(0,0,0,0)"
GRID    = "rgba(148,163,184,0.06)"
TEXT    = "#94a3b8"
TEAL    = "#14b8a6"
AMBER   = "#f59e0b"
ROSE    = "#f43f5e"
INDIGO  = "#6366f1"
VIOLET  = "#8b5cf6"

TIER_COLOR = {"LOW": TEAL, "MEDIUM": AMBER, "HIGH": ROSE}

# Professional sequential palettes
SEQ_TEAL   = [[0,"#0d2d2a"],[0.5,"#0d9488"],[1,"#2dd4bf"]]
SEQ_AMBER  = [[0,"#2d1f00"],[0.5,"#d97706"],[1,"#fcd34d"]]
SEQ_MULTI  = [[0,"#0d9488"],[0.33,"#6366f1"],[0.66,"#f59e0b"],[1,"#f43f5e"]]

def apply_theme(fig, title="", height=380):
    fig.update_layout(
        template      = "plotly_dark",
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        height        = height,
        margin        = dict(l=10, r=10, t=45, b=10),
        title         = dict(
            text = title,
            font = dict(size=14, color="#e2e8f0", family="Inter, sans-serif"),
            x    = 0.02, y = 0.97
        ),
        font          = dict(color=TEXT, family="Inter, sans-serif", size=11),
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11),
                             bordercolor="rgba(255,255,255,0.05)", borderwidth=1),
        hoverlabel    = dict(bgcolor="#1e293b", font_size=12, font_color="#f1f5f9",
                             bordercolor="#334155"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=TEXT), title_font=dict(color=TEXT))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=TEXT), title_font=dict(color=TEXT))
    return fig


# ─────────────────────────────────────────────────────────────
# LOAD MODELS & DATA
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    xgb       = joblib.load("outputs/models/xgboost_credit_model.pkl")
    iso       = joblib.load("outputs/models/isolation_forest.pkl")
    scaler    = joblib.load("outputs/models/scaler.pkl")
    feat_cols = joblib.load("outputs/models/feature_columns.pkl")
    risk_cfg  = joblib.load("outputs/models/risk_config.pkl")
    explainer = shap.TreeExplainer(xgb)
    return xgb, iso, scaler, feat_cols, risk_cfg, explainer

try:
    xgb, iso, scaler, feat_cols, risk_cfg, explainer = load_models()
    MODELS_OK = True
except Exception as e:
    MODELS_OK = False

@st.cache_data
def load_portfolio():
    try:    return pd.read_csv("outputs/data/unified_risk_scores.csv")
    except: return None

@st.cache_data
def load_lc():
    try:
        lc = pd.read_parquet("outputs/data/lendingclub_clean.parquet")
        return lc.drop(columns=[c for c in ["target_str","dti_band","income_band"] if c in lc.columns])
    except: return None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:24px 8px 16px; border-bottom:1px solid rgba(148,163,184,0.08); margin-bottom:16px;'>
        <div style='font-size:0.7rem; color:#14b8a6; font-weight:600;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;'>
            Final Year Project
        </div>
        <div style='font-size:1.3rem; font-weight:800; color:#f1f5f9; letter-spacing:-0.5px;'>
            RiskIQ
        </div>
        <div style='font-size:0.78rem; color:#475569; margin-top:2px;'>
            Banking Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Overview",
        "Credit Scorer",
        "Fraud Detector",
        "Portfolio Analytics",
        "Batch Scoring",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top:24px; padding-top:20px;
                border-top:1px solid rgba(148,163,184,0.08);'>
        <div style='font-size:0.68rem; color:#14b8a6; font-weight:600;
                    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;'>
            System Status
        </div>
        <div style='font-size:0.8rem; color:#64748b; line-height:2.2;'>
            <span style='color:#2dd4bf;'>●</span> XGBoost &nbsp; AUC 0.9997<br>
            <span style='color:#2dd4bf;'>●</span> Random Forest &nbsp; AUC 0.9990<br>
            <span style='color:#2dd4bf;'>●</span> Isolation Forest &nbsp; Active<br>
            <span style='color:#2dd4bf;'>●</span> AutoEncoder &nbsp; AUC 0.7095<br>
            <span style='color:#2dd4bf;'>●</span> SHAP &nbsp; Enabled
        </div>
    </div>
    <div style='margin-top:20px; padding-top:16px;
                border-top:1px solid rgba(148,163,184,0.08);'>
        <div style='font-size:0.68rem; color:#475569; line-height:2;'>
            328,860 records processed<br>
            3 datasets integrated<br>
            Optuna · 50 tuning trials
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────

if page == "Overview":

    st.markdown("""
    <div class='page-title'>Credit & Fraud Risk Intelligence</div>
    <div class='page-subtitle'>
        AI-driven loan default prediction · anomaly detection · explainable decisions
    </div>
    <div class='accent-line'></div>
    """, unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("◈", "0.9997",  "XGBoost AUC Score"),
        ("◉", "0.9989",  "5-Fold CV AUC"),
        ("◎", "328K+",   "Records Processed"),
        ("◆", "3",       "ML Models Deployed"),
        ("◇", "SHAP",    "Explainability Layer"),
    ]
    for col, (icon, val, lbl) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi'>
                <span class='kpi-icon'>{icon}</span>
                <div class='kpi-val'>{val}</div>
                <div class='kpi-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])

    # Model performance grouped bar
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Model Benchmarking</div>", unsafe_allow_html=True)

        fig = go.Figure()
        models   = ["Logistic Regression", "Random Forest", "XGBoost (Tuned)"]
        aucs     = [0.9969, 0.9990, 0.9997]
        f1s      = [0.98,   0.99,   0.99]
        colors_b = [INDIGO, VIOLET, TEAL]

        for i, (m, auc, f1, color) in enumerate(zip(models, aucs, f1s, colors_b)):
            fig.add_trace(go.Bar(
                name        = m,
                x           = ["ROC-AUC", "F1-Score"],
                y           = [auc, f1],
                marker_color= color,
                marker_line = dict(color="rgba(255,255,255,0.05)", width=1),
                text        = [f"{auc:.4f}", f"{f1:.2f}"],
                textposition= "outside",
                textfont    = dict(color="#e2e8f0", size=11),
                offsetgroup = i,
                width       = 0.22,
            ))

        fig.update_yaxes(range=[0.96, 1.005])
        fig.update_layout(barmode="group")
        apply_theme(fig, "Model Performance Comparison", height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Donut risk distribution
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Portfolio Risk Distribution</div>", unsafe_allow_html=True)
        portfolio = load_portfolio()
        if portfolio is not None:
            tc = portfolio["risk_tier"].value_counts()
            fig = go.Figure(go.Pie(
                labels   = tc.index,
                values   = tc.values,
                hole     = 0.68,
                marker   = dict(
                    colors = [TIER_COLOR.get(t, INDIGO) for t in tc.index],
                    line   = dict(color="#080c14", width=4)
                ),
                textinfo      = "percent",
                textfont      = dict(size=12, color="#f1f5f9"),
                hovertemplate = "<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
                rotation      = 90,
            ))
            total = len(portfolio)
            fig.add_annotation(
                text      = f"<b>{total:,}</b><br><span style='font-size:10px'>customers</span>",
                x=0.5, y=0.5, showarrow=False,
                font      = dict(size=16, color="#f1f5f9"),
                align     = "center"
            )
            apply_theme(fig, "Risk Tier Breakdown", height=340)
            fig.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5,
                    font=dict(color=TEXT)
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
    lc = load_lc()
    if lc is not None:
        le = LabelEncoder()
        lc_enc = lc.copy()
        for col in lc_enc.select_dtypes(include="object").columns:
            lc_enc[col] = le.fit_transform(lc_enc[col].astype(str))
        num_cols = [c for c in lc_enc.select_dtypes(include="number").columns if c != "target"][:12]
        corr = lc_enc[num_cols].corr().round(2)

        fig = go.Figure(go.Heatmap(
            z             = corr.values,
            x             = corr.columns.tolist(),
            y             = corr.index.tolist(),
            colorscale    = [[0,"#0d9488"],[0.5,"#1e293b"],[1,"#f43f5e"]],
            zmid          = 0,
            zmin          = -1, zmax=1,
            text          = [[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate  = "%{text}",
            textfont      = dict(size=9, color="#e2e8f0"),
            hovertemplate = "<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            showscale     = True,
            colorbar      = dict(
                tickfont    = dict(color=TEXT),
                title       = dict(text="r", font=dict(color=TEXT)),
                outlinecolor= GRID, outlinewidth=1,
                thickness   = 14, len=0.85
            )
        ))
        apply_theme(fig, "Pearson Correlation — Key Features", height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Default rate by purpose + grade
    if lc is not None and "purpose" in lc.columns:
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Default Rate by Loan Purpose</div>", unsafe_allow_html=True)
            pur = lc.groupby("purpose", observed=True)["target"].mean().sort_values(ascending=True) * 100
            fig = go.Figure(go.Bar(
                x             = pur.values,
                y             = pur.index.astype(str),
                orientation   = "h",
                marker        = dict(
                    color     = pur.values,
                    colorscale= SEQ_MULTI,
                    showscale = False,
                    line      = dict(color="rgba(255,255,255,0.03)", width=1)
                ),
                text          = [f"{v:.1f}%" for v in pur.values],
                textposition  = "outside",
                textfont      = dict(color="#e2e8f0", size=10),
                hovertemplate = "<b>%{y}</b><br>Default Rate: %{x:.1f}%<extra></extra>"
            ))
            apply_theme(fig, "Default Rate by Purpose (%)", height=340)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Interest Rate vs Default Risk</div>", unsafe_allow_html=True)
            if "int_rate" in lc.columns:
                lc_sample = lc.sample(min(3000,len(lc)), random_state=42)
                fig = go.Figure()
                for t_val, color, name in [(0, TEAL, "Fully Paid"), (1, ROSE, "Charged Off")]:
                    mask = lc_sample["target"] == t_val
                    fig.add_trace(go.Scatter(
                        x             = lc_sample[mask].get("int_rate", pd.Series()),
                        y             = lc_sample[mask].get("dti", pd.Series()),
                        mode          = "markers",
                        name          = name,
                        marker        = dict(color=color, size=3.5, opacity=0.55,
                                            line=dict(width=0)),
                        hovertemplate = f"<b>{name}</b><br>Rate: %{{x:.1f}}%<br>DTI: %{{y:.1f}}<extra></extra>"
                    ))
                apply_theme(fig, "Interest Rate vs DTI — Coloured by Outcome", height=340)
                fig.update_xaxes(title_text="Interest Rate (%)")
                fig.update_yaxes(title_text="Debt-to-Income Ratio")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — CREDIT SCORER
# ─────────────────────────────────────────────────────────────

elif page == "Credit Scorer":

    st.markdown("""
    <div class='page-title'>Credit Risk Assessment</div>
    <div class='page-subtitle'>
        Input customer financial profile to generate a risk-adjusted credit score
    </div>
    <div class='accent-line'></div>
    """, unsafe_allow_html=True)

    if not MODELS_OK:
        st.error("Models unavailable. Execute day2_model.py to generate model artifacts.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Loan Parameters</div>", unsafe_allow_html=True)
        loan_amnt = st.number_input("Loan Amount ($)", 500, 40000, 10000, step=500)
        int_rate  = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, step=0.5)
        term      = st.selectbox("Repayment Term", [36, 60], format_func=lambda x: f"{x} months")
        purpose   = st.selectbox("Loan Purpose", [
            "debt_consolidation","credit_card","home_improvement",
            "major_purchase","medical","small_business","other"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Applicant Profile</div>", unsafe_allow_html=True)
        annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        emp_length = st.slider("Employment (years)", 0, 10, 3)
        home_own   = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE","OTHER"])
        dti        = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0, step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Credit History</div>", unsafe_allow_html=True)
        revol_util = st.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.0)
        total_acc  = st.number_input("Total Credit Accounts", 1, 100, 15)
        open_acc   = st.number_input("Open Accounts", 1, 50, 8)
        pub_rec    = st.number_input("Derogatory Public Records", 0, 10, 0)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate Risk Assessment"):
        inp = {c: 0 for c in feat_cols}
        inp.update({
            "loan_amnt"           : loan_amnt,
            "int_rate"            : int_rate,
            "term"                : term,
            "annual_inc"          : annual_inc,
            "emp_length"          : emp_length,
            "dti"                 : dti,
            "revol_util"          : revol_util,
            "total_acc"           : total_acc,
            "open_acc"            : open_acc,
            "pub_rec"             : pub_rec,
            "loan_to_income"      : loan_amnt / (annual_inc + 1),
            "monthly_payment_est" : (loan_amnt * (int_rate/100)) / (term + 1),
            "high_dti_flag"       : int(dti > 30),
            "high_int_flag"       : int(int_rate > 18),
            "annual_inc_log"      : np.log1p(annual_inc),
            "loan_amnt_log"       : np.log1p(loan_amnt),
        })
        row_df = pd.DataFrame([inp])[feat_cols]
        prob   = float(xgb.predict_proba(row_df)[0][1])

        if prob < 0.30:
            badge_cls = "badge-low"
            badge_txt = f"Low Risk — Default Probability: {prob:.1%}"
            tier      = "LOW"
            t_color   = TEAL
        elif prob < 0.60:
            badge_cls = "badge-med"
            badge_txt = f"Elevated Risk — Default Probability: {prob:.1%}"
            tier      = "MEDIUM"
            t_color   = AMBER
        else:
            badge_cls = "badge-high"
            badge_txt = f"High Risk — Default Probability: {prob:.1%}"
            tier      = "HIGH"
            t_color   = ROSE

        r1, r2, r3, r4 = st.columns([3,1,1,1])
        with r1: st.markdown(f'<div class="badge {badge_cls}">{badge_txt}</div>', unsafe_allow_html=True)
        with r2: st.metric("Default Prob", f"{prob:.1%}")
        with r3: st.metric("Risk Tier", tier)
        with r4: st.metric("Decision", "Reject" if prob > 0.5 else "Approve")

        st.markdown("<br>", unsafe_allow_html=True)
        cg, cr = st.columns(2)

        # Gauge
        with cg:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Risk Score Gauge</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number+delta",
                value  = prob * 100,
                delta  = {"reference": 30, "valueformat":".1f",
                          "increasing":{"color":ROSE},"decreasing":{"color":TEAL}},
                number = {"suffix":"%","font":{"color":"#f1f5f9","size":34,"family":"Inter"}},
                title  = {"text":"Default Probability","font":{"color":TEXT,"size":13}},
                gauge  = {
                    "axis"      : {"range":[0,100],"tickcolor":TEXT,"tickwidth":1,
                                   "tickfont":{"color":TEXT,"size":10}},
                    "bar"       : {"color":t_color,"thickness":0.25},
                    "bgcolor"   : "#111827",
                    "bordercolor":"#1e293b","borderwidth":2,
                    "steps"     : [
                        {"range":[0,30],  "color":"rgba(20,184,166,0.08)"},
                        {"range":[30,60], "color":"rgba(245,158,11,0.08)"},
                        {"range":[60,100],"color":"rgba(244,63,94,0.08)"},
                    ],
                    "threshold" : {"line":{"color":t_color,"width":3},"value":prob*100}
                }
            ))
            fig.update_layout(paper_bgcolor=BG, height=280,
                              margin=dict(l=20,r=20,t=30,b=10),
                              font=dict(color=TEXT,family="Inter"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Risk radar
        with cr:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Risk Factor Profile</div>", unsafe_allow_html=True)
            cats = ["Rate Risk","DTI Risk","Utilisation","Income Risk","Loan Size"]
            vals = [
                min((int_rate-5)/25, 1),
                min(dti/50, 1),
                min(revol_util/100, 1),
                max(0, 1-annual_inc/200000),
                min(loan_amnt/40000, 1),
            ]
            fig = go.Figure(go.Scatterpolar(
                r         = vals + [vals[0]],
                theta     = cats + [cats[0]],
                fill      = "toself",
                fillcolor = f"rgba(244,63,94,0.1)" if prob>0.5 else "rgba(20,184,166,0.1)",
                line      = dict(color=t_color, width=2),
                marker    = dict(color=t_color, size=5),
                name      = "Risk Profile"
            ))
            fig.update_layout(
                polar = dict(
                    bgcolor     = "#111827",
                    radialaxis  = dict(visible=True, range=[0,1],
                                       gridcolor=GRID, tickfont=dict(color=TEXT,size=9),
                                       linecolor=GRID),
                    angularaxis = dict(gridcolor=GRID, tickfont=dict(color=TEXT,size=10),
                                       linecolor=GRID)
                ),
                paper_bgcolor=BG, height=280,
                margin=dict(l=50,r=50,t=30,b=10),
                showlegend=False, font=dict(color=TEXT,family="Inter"),
                hoverlabel=dict(bgcolor="#1e293b",font_color="#f1f5f9")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Explainability — SHAP Feature Attribution</div>", unsafe_allow_html=True)
        st.caption("Features in red increase default risk · Features in blue reduce default risk")
        shap_vals = explainer.shap_values(row_df)

        plt.style.use("dark_background")
        fig_shap, ax = plt.subplots(figsize=(11, 5))
        fig_shap.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        shap.waterfall_plot(
            shap.Explanation(
                values       = shap_vals[0],
                base_values  = explainer.expected_value,
                data         = row_df.iloc[0],
                feature_names= feat_cols
            ), show=False
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        st.pyplot(fig_shap)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3 — FRAUD DETECTOR
# ─────────────────────────────────────────────────────────────

elif page == "Fraud Detector":

    st.markdown("""
    <div class='page-title'>Transaction Anomaly Detection</div>
    <div class='page-subtitle'>
        Isolation Forest model identifies statistically anomalous financial behaviour
    </div>
    <div class='accent-line'></div>
    """, unsafe_allow_html=True)

    if not MODELS_OK:
        st.error("Models unavailable.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Transaction Parameters</div>", unsafe_allow_html=True)
        t_loan = st.number_input("Transaction Amount ($)", 100, 100000, 5000, step=100)
        t_rate = st.slider("Applied Rate (%)", 5.0, 35.0, 13.0, step=0.5)
        t_dti  = st.slider("Debt-to-Income Ratio", 0.0, 60.0, 18.0)
        t_revol= st.slider("Revolving Utilisation (%)", 0.0, 150.0, 50.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Customer Context</div>", unsafe_allow_html=True)
        t_inc  = st.number_input("Annual Income ($)", 10000, 500000, 55000, step=1000)
        t_total= st.number_input("Total Credit Accounts", 1, 100, 12)
        t_open = st.number_input("Open Accounts", 1, 50, 6)
        t_pub  = st.number_input("Derogatory Records", 0, 20, 0)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Anomaly Detection"):
        inp = {c: 0 for c in feat_cols}
        inp.update({
            "loan_amnt"           : t_loan,
            "int_rate"            : t_rate,
            "dti"                 : t_dti,
            "revol_util"          : t_revol,
            "annual_inc"          : t_inc,
            "total_acc"           : t_total,
            "open_acc"            : t_open,
            "pub_rec"             : t_pub,
            "loan_to_income"      : t_loan / (t_inc+1),
            "monthly_payment_est" : (t_loan*(t_rate/100)) / 37,
            "high_dti_flag"       : int(t_dti > 30),
            "high_int_flag"       : int(t_rate > 18),
            "annual_inc_log"      : np.log1p(t_inc),
            "loan_amnt_log"       : np.log1p(t_loan),
        })
        row_df = pd.DataFrame([inp])[feat_cols]
        scaled = scaler.transform(row_df)
        iso_sc = iso.decision_function(scaled)[0]
        cfg    = risk_cfg
        norm   = float(np.clip(
            1-(iso_sc-cfg["iso_score_min"])/(cfg["iso_score_max"]-cfg["iso_score_min"]+1e-9),
            0, 1))

        if norm > 0.6:
            b_cls, b_txt, t_col = "badge-high", f"Anomaly Detected — Score: {norm:.1%}", ROSE
        elif norm > 0.3:
            b_cls, b_txt, t_col = "badge-med",  f"Suspicious Activity — Score: {norm:.1%}", AMBER
        else:
            b_cls, b_txt, t_col = "badge-low",  f"Transaction Normal — Score: {norm:.1%}", TEAL

        st.markdown(f'<div class="badge {b_cls}" style="margin-bottom:20px">{b_txt}</div>',
                    unsafe_allow_html=True)

        cg, cb = st.columns(2)
        with cg:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Anomaly Score</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = norm * 100,
                number = {"suffix":"%","font":{"color":"#f1f5f9","size":32,"family":"Inter"}},
                title  = {"text":"Fraud Probability","font":{"color":TEXT,"size":13}},
                gauge  = {
                    "axis"      : {"range":[0,100],"tickcolor":TEXT,"tickwidth":1,
                                   "tickfont":{"color":TEXT,"size":10}},
                    "bar"       : {"color":t_col,"thickness":0.25},
                    "bgcolor"   : "#111827","bordercolor":"#1e293b","borderwidth":2,
                    "steps"     : [
                        {"range":[0,30],"color":"rgba(20,184,166,0.08)"},
                        {"range":[30,60],"color":"rgba(245,158,11,0.08)"},
                        {"range":[60,100],"color":"rgba(244,63,94,0.08)"},
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor=BG, height=270,
                              margin=dict(l=20,r=20,t=30,b=10),
                              font=dict(color=TEXT,family="Inter"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cb:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Risk Factor Breakdown</div>", unsafe_allow_html=True)
            factors = ["High Interest Rate","Elevated DTI","Low Income","High Utilisation","Large Amount"]
            scores  = [
                min((t_rate-5)/30,1)*100,
                min(t_dti/50,1)*100,
                max(0,1-t_inc/200000)*100,
                min(t_revol/100,1)*100,
                min(t_loan/40000,1)*100,
            ]
            fig = go.Figure(go.Bar(
                x             = scores,
                y             = factors,
                orientation   = "h",
                marker        = dict(
                    color     = scores,
                    colorscale= [[0,TEAL],[0.5,AMBER],[1,ROSE]],
                    showscale = False,
                    line      = dict(color="rgba(255,255,255,0.04)",width=1)
                ),
                text          = [f"{v:.0f}%" for v in scores],
                textposition  = "outside",
                textfont      = dict(color="#e2e8f0",size=11),
                hovertemplate = "<b>%{y}</b><br>Risk: %{x:.0f}%<extra></extra>"
            ))
            apply_theme(fig, "", height=270)
            fig.update_xaxes(range=[0,120])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4 — PORTFOLIO ANALYTICS
# ─────────────────────────────────────────────────────────────

elif page == "Portfolio Analytics":

    st.markdown("""
    <div class='page-title'>Portfolio Risk Analytics</div>
    <div class='page-subtitle'>
        Aggregate risk metrics · customer segmentation · distribution analysis
    </div>
    <div class='accent-line'></div>
    """, unsafe_allow_html=True)

    portfolio = load_portfolio()
    lc        = load_lc()
    if portfolio is None:
        st.error("Run day3_fraud.py first.")
        st.stop()

    # KPIs
    total    = len(portfolio)
    high_r   = (portfolio["risk_tier"]=="HIGH").sum()
    med_r    = (portfolio["risk_tier"]=="MEDIUM").sum()
    low_r    = (portfolio["risk_tier"]=="LOW").sum()
    avg_c    = portfolio["credit_score"].mean()
    avg_f    = portfolio["fraud_score"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, icon, val, lbl in zip(
        [c1,c2,c3,c4,c5,c6],
        ["◈","◉","◎","◆","◇","○"],
        [f"{total:,}", f"{high_r:,}", f"{med_r:,}", f"{low_r:,}", f"{avg_c:.1%}", f"{avg_f:.1%}"],
        ["Total Customers","High Risk","Medium Risk","Low Risk","Avg Credit Score","Avg Fraud Score"]
    ):
        with col:
            st.markdown(f"""
            <div class='kpi'>
                <span class='kpi-icon'>{icon}</span>
                <div class='kpi-val'>{val}</div>
                <div class='kpi-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tier_filter = st.multiselect("Filter Risk Tier",["LOW","MEDIUM","HIGH"],
                                  default=["LOW","MEDIUM","HIGH"])
    filtered = portfolio[portfolio["risk_tier"].isin(tier_filter)]
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Risk Tier Distribution</div>", unsafe_allow_html=True)
        tc = filtered["risk_tier"].value_counts()
        fig = go.Figure(go.Pie(
            labels   = tc.index,
            values   = tc.values,
            hole     = 0.65,
            marker   = dict(
                colors=[TIER_COLOR.get(t,INDIGO) for t in tc.index],
                line  =dict(color="#080c14",width=4)
            ),
            textinfo      = "label+percent",
            textfont      = dict(size=12,color="#f1f5f9"),
            hovertemplate = "<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
            rotation      = 90,
        ))
        fig.add_annotation(
            text=f"<b>{len(filtered):,}</b>",
            x=0.5,y=0.5,showarrow=False,
            font=dict(size=20,color="#f1f5f9")
        )
        apply_theme(fig,"Portfolio Composition",height=340)
        fig.update_layout(legend=dict(orientation="h",y=-0.1,x=0.5,xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Unified Score — Violin by Tier</div>",unsafe_allow_html=True)
        fig = go.Figure()
        for tier, color in TIER_COLOR.items():
            mask = filtered["risk_tier"]==tier
            if mask.sum()>0:
                fig.add_trace(go.Violin(
                    y             = filtered[mask]["unified_score"],
                    name          = tier,
                    box_visible   = True,
                    meanline_visible=True,
                    line_color    = color,
                    fillcolor     = color.replace("#","") and f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                    opacity       = 0.85,
                    points        = False,
                    hoverinfo     = "y+name"
                ))
        apply_theme(fig,"Unified Risk Score Distribution",height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # Row 2 — Heatmap
    st.markdown("<div class='card'>",unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Feature Intensity Heatmap by Loan Purpose</div>",unsafe_allow_html=True)
    if lc is not None and "purpose" in lc.columns:
        heat_cols = [c for c in ["int_rate","dti","revol_util","loan_amnt","total_acc","open_acc"] if c in lc.columns]
        heat_data = lc.groupby("purpose",observed=True)[heat_cols].mean().round(2)
        # Normalise each column 0-1
        heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)

        fig = go.Figure(go.Heatmap(
            z             = heat_norm.values,
            x             = heat_cols,
            y             = heat_data.index.tolist(),
            colorscale    = [[0,"#0d2d2a"],[0.4,"#0d9488"],[0.7,"#f59e0b"],[1,"#f43f5e"]],
            text          = [[f"{v:.1f}" for v in row] for row in heat_data.values],
            texttemplate  = "%{text}",
            textfont      = dict(size=9,color="#f1f5f9"),
            hovertemplate = "<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
            showscale     = True,
            colorbar      = dict(tickfont=dict(color=TEXT),thickness=14,len=0.85,
                                 title=dict(text="Normalised",font=dict(color=TEXT)),
                                 outlinecolor=GRID,outlinewidth=1)
        ))
        apply_theme(fig,"Avg Feature Values by Loan Purpose (Normalised)",height=380)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # Row 3
    col3,col4 = st.columns(2)
    with col3:
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Credit vs Fraud Score — Scatter</div>",unsafe_allow_html=True)
        sample = filtered.sample(min(3000,len(filtered)),random_state=42)
        fig = go.Figure()
        for tier,color in TIER_COLOR.items():
            mask = sample["risk_tier"]==tier
            if mask.sum()>0:
                fig.add_trace(go.Scatter(
                    x             = sample[mask]["credit_score"],
                    y             = sample[mask]["fraud_score"],
                    mode          = "markers",
                    name          = tier,
                    marker        = dict(color=color,size=4,opacity=0.55,line=dict(width=0)),
                    hovertemplate = f"<b>{tier}</b><br>Credit: %{{x:.3f}}<br>Fraud: %{{y:.3f}}<extra></extra>"
                ))
        apply_theme(fig,"Credit Score vs Fraud Score",height=320)
        fig.update_xaxes(title_text="Credit Default Probability")
        fig.update_yaxes(title_text="Fraud Anomaly Score")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Default Rate by Risk Tier</div>",unsafe_allow_html=True)
        if "true_label" in filtered.columns:
            tier_def = filtered.groupby("risk_tier")["true_label"].mean()*100
            tier_def = tier_def.reindex([t for t in ["LOW","MEDIUM","HIGH"] if t in tier_def.index])
            fig = go.Figure(go.Bar(
                x             = tier_def.index,
                y             = tier_def.values,
                marker        = dict(
                    color     = [TIER_COLOR.get(t,INDIGO) for t in tier_def.index],
                    line      = dict(color="rgba(255,255,255,0.04)",width=1)
                ),
                text          = [f"{v:.1f}%" for v in tier_def.values],
                textposition  = "outside",
                textfont      = dict(color="#e2e8f0",size=13,family="Inter"),
                width         = 0.4,
                hovertemplate = "<b>%{x}</b><br>Default Rate: %{y:.1f}%<extra></extra>"
            ))
            apply_theme(fig,"Actual Default Rate by Risk Tier",height=320)
            fig.update_yaxes(title_text="Default Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # Top 20 table
    st.markdown("<div class='card'>",unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Top 20 Highest Risk Customers</div>",unsafe_allow_html=True)
    dcols = [c for c in ["unified_score","credit_score","fraud_score","risk_tier","true_label"] if c in filtered.columns]
    st.dataframe(
        filtered.nlargest(20,"unified_score")[dcols].reset_index(drop=True),
        use_container_width=True, height=320
    )
    st.markdown("</div>",unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 5 — BATCH SCORING
# ─────────────────────────────────────────────────────────────

elif page == "Batch Scoring":

    st.markdown("""
    <div class='page-title'>Batch Customer Scoring</div>
    <div class='page-subtitle'>
        Upload a CSV of customers — AI scores every row instantly and shows risk analytics
    </div>
    <div class='accent-line'></div>
    """, unsafe_allow_html=True)

    if not MODELS_OK:
        st.error("Models unavailable.")
        st.stop()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:13px; color:#94a3b8; line-height:2;'>
        1️⃣ &nbsp; Upload your customer CSV file below<br>
        2️⃣ &nbsp; AI runs XGBoost + Isolation Forest on every row<br>
        3️⃣ &nbsp; Each customer gets a risk score + APPROVE / REJECT decision<br>
        4️⃣ &nbsp; View graphs + download results for Power BI
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Required CSV Columns</div>", unsafe_allow_html=True)
    st.code("loan_amnt, int_rate, dti, revol_util, annual_inc, total_acc, open_acc, pub_rec")
    st.caption("Tip: You can use outputs/data/unified_risk_scores.csv as a demo file!")
    uploaded = st.file_uploader("Upload Customer CSV File", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        batch = pd.read_csv(uploaded)
        st.success(f"✅ {len(batch):,} customers loaded!")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(batch.head(5), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("⚡ Run AI Risk Scoring on All Customers"):
            with st.spinner("AI is analysing all customers..."):
                ib = pd.DataFrame(index=batch.index)
                for col in feat_cols:
                    ib[col] = batch[col] if col in batch.columns else 0

                if "loan_amnt" in batch.columns and "annual_inc" in batch.columns:
                    ib["loan_to_income"] = batch["loan_amnt"]/(batch["annual_inc"]+1)
                    ib["loan_amnt_log"]  = np.log1p(batch["loan_amnt"])
                    ib["annual_inc_log"] = np.log1p(batch["annual_inc"])
                if "dti"      in batch.columns:
                    ib["high_dti_flag"] = (batch["dti"]>30).astype(int)
                if "int_rate" in batch.columns:
                    ib["high_int_flag"] = (batch["int_rate"]>18).astype(int)

                from sklearn.preprocessing import LabelEncoder
                ib_final = ib[feat_cols].copy()
                for col in ib_final.select_dtypes(
                        include=["object","category"]).columns:
                    le = LabelEncoder()
                    ib_final[col] = le.fit_transform(
                        ib_final[col].astype(str))
                ib_final = ib_final.astype(float)

                cs = xgb.predict_proba(ib_final)[:,1]
                scaled = scaler.transform(ib_final)
                ir     = iso.decision_function(scaled)
                fs     = np.clip(
                    1-(ir-risk_cfg["iso_score_min"]) /
                    (risk_cfg["iso_score_max"]-risk_cfg["iso_score_min"]+1e-9),
                    0, 1)
                us = risk_cfg["credit_weight"]*cs + risk_cfg["fraud_weight"]*fs

                def tier(s):
                    return "HIGH" if s>=0.6 else "MEDIUM" if s>=0.3 else "LOW"

                batch["credit_score"]  = cs.round(4)
                batch["fraud_score"]   = fs.round(4)
                batch["unified_score"] = us.round(4)
                batch["risk_tier"]     = [tier(s) for s in us]
                batch["decision"]      = [
                    "REJECT" if s>=0.5 else "APPROVE" for s in cs]

            st.success(f"✅ Done! {len(batch):,} customers scored.")

            # ── KPI cards ──────────────────────────────────────
            tc = batch["risk_tier"].value_counts()
            approve = (batch["decision"]=="APPROVE").sum()
            reject  = (batch["decision"]=="REJECT").sum()

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total Customers", f"{len(batch):,}")
            c2.metric("🔴 High Risk",    int(tc.get("HIGH",0)))
            c3.metric("⚠️ Medium Risk",  int(tc.get("MEDIUM",0)))
            c4.metric("✅ Low Risk",     int(tc.get("LOW",0)))
            c5.metric("Avg Risk Score",  f"{batch['unified_score'].mean():.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Row 1: Donut + Bar ──────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-label'>Risk Tier Distribution</div>",
                    unsafe_allow_html=True)
                fig = go.Figure(go.Pie(
                    labels   = tc.index,
                    values   = tc.values,
                    hole     = 0.65,
                    marker   = dict(
                        colors=[TIER_COLOR.get(t, INDIGO) for t in tc.index],
                        line  = dict(color="#080c14", width=4)
                    ),
                    textinfo      = "label+percent",
                    textfont      = dict(size=13, color="#f1f5f9"),
                    hovertemplate = (
                        "<b>%{label}</b><br>"
                        "%{value:,} customers<br>"
                        "%{percent}<extra></extra>")
                ))
                fig.add_annotation(
                    text      = f"{len(batch):,}<br>customers",
                    x=0.5, y=0.5, showarrow=False,
                    font      = dict(size=16, color="#f1f5f9")
                )
                apply_theme(fig, "Who is Risky?", height=320)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-label'>Approve vs Reject</div>",
                    unsafe_allow_html=True)
                fig = go.Figure(go.Pie(
                    labels   = ["APPROVE", "REJECT"],
                    values   = [approve, reject],
                    hole     = 0.65,
                    marker   = dict(
                        colors=[TEAL, ROSE],
                        line  = dict(color="#080c14", width=4)
                    ),
                    textinfo      = "label+percent",
                    textfont      = dict(size=13, color="#f1f5f9"),
                    hovertemplate = (
                        "<b>%{label}</b><br>"
                        "%{value:,} customers<br>"
                        "%{percent}<extra></extra>")
                ))
                fig.add_annotation(
                    text      = f"{approve:,}<br>approved",
                    x=0.5, y=0.5, showarrow=False,
                    font      = dict(size=16, color="#f1f5f9")
                )
                apply_theme(fig, "Decision Summary", height=320)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Row 2: Credit score histogram + Scatter ─────────
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-label'>Credit Score Distribution</div>",
                    unsafe_allow_html=True)
                fig = go.Figure()
                for t_val, color, name in [
                    ("LOW",    TEAL,   "Low Risk"),
                    ("MEDIUM", AMBER,  "Medium Risk"),
                    ("HIGH",   ROSE,   "High Risk")
                ]:
                    mask = batch["risk_tier"] == t_val
                    if mask.sum() > 0:
                        fig.add_trace(go.Histogram(
                            x         = batch[mask]["credit_score"],
                            name      = name,
                            marker_color = color,
                            opacity   = 0.75,
                            nbinsx    = 30,
                            hovertemplate = (
                                f"<b>{name}</b><br>"
                                "Score: %{x:.3f}<br>"
                                "Count: %{y}<extra></extra>")
                        ))
                apply_theme(fig, "Credit Score by Risk Tier", height=320)
                fig.update_layout(barmode="overlay")
                fig.update_xaxes(title_text="Credit Default Probability")
                fig.update_yaxes(title_text="Number of Customers")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col4:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-label'>"
                    "Credit Score vs Fraud Score"
                    "</div>",
                    unsafe_allow_html=True)
                sample = batch.sample(
                    min(2000, len(batch)), random_state=42)
                fig = go.Figure()
                for t_val, color in TIER_COLOR.items():
                    mask = sample["risk_tier"] == t_val
                    if mask.sum() > 0:
                        fig.add_trace(go.Scatter(
                            x    = sample[mask]["credit_score"],
                            y    = sample[mask]["fraud_score"],
                            mode = "markers",
                            name = t_val,
                            marker = dict(
                                color   = color,
                                size    = 4,
                                opacity = 0.6,
                                line    = dict(width=0)
                            ),
                            hovertemplate = (
                                f"<b>{t_val}</b><br>"
                                "Credit: %{x:.3f}<br>"
                                "Fraud: %{y:.3f}<extra></extra>")
                        ))
                apply_theme(
                    fig, "Credit vs Fraud Score — Each Dot = 1 Customer",
                    height=320)
                fig.update_xaxes(title_text="Credit Risk Score")
                fig.update_yaxes(title_text="Fraud Risk Score")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Row 3: Unified score bar + Risk tier bar ─────────
            col5, col6 = st.columns(2)

            with col5:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-label'>Unified Risk Score Range</div>",
                    unsafe_allow_html=True)
                fig = go.Figure(go.Histogram(
                    x            = batch["unified_score"],
                    nbinsx       = 40,
                    marker       = dict(
                        color    = batch["unified_score"],
                        colorscale = [
                            [0,   TEAL],
                            [0.5, AMBER],
                            [1,   ROSE]
                        ],
                        showscale= False,
                        line     = dict(
                            color="rgba(255,255,255,0.05)", width=0.5)
                    ),
                    hovertemplate= (
                        "Score: %{x:.3f}<br>"
                        "Count: %{y}<extra></extra>")
                ))
                apply_theme(
                    fig, "Unified Risk Score Distribution", height=300)
                fig.add_vline(
                    x=0.30, line_color=TEAL,
                    line_dash="dash", line_width=1.5,
                    annotation_text="Low/Med",
                    annotation_font_color=TEAL)
                fig.add_vline(
                    x=0.60, line_color=ROSE,
                    line_dash="dash", line_width=1.5,
                    annotation_text="Med/High",
                    annotation_font_color=ROSE)
                fig.update_xaxes(title_text="Unified Score")
                fig.update_yaxes(title_text="Customers")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col6:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-label'>Risk Tier Count</div>",
                    unsafe_allow_html=True)
                tier_order = [
                    t for t in ["LOW","MEDIUM","HIGH"] if t in tc.index]
                fig = go.Figure(go.Bar(
                    x         = tier_order,
                    y         = [tc.get(t,0) for t in tier_order],
                    marker    = dict(
                        color = [TIER_COLOR[t] for t in tier_order],
                        line  = dict(
                            color="rgba(255,255,255,0.05)", width=1)
                    ),
                    text      = [
                        f"{tc.get(t,0):,}" for t in tier_order],
                    textposition = "outside",
                    textfont  = dict(color="#f1f5f9", size=14),
                    width     = 0.4,
                    hovertemplate = (
                        "<b>%{x}</b><br>"
                        "%{y:,} customers<extra></extra>")
                ))
                apply_theme(fig, "Customers by Risk Tier", height=300)
                fig.update_yaxes(title_text="Number of Customers")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Top 10 riskiest customers ───────────────────────
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-label'>"
                "Top 10 Riskiest Customers — Immediate Attention Required"
                "</div>",
                unsafe_allow_html=True)
            top10_cols = [
                c for c in [
                    "unified_score","credit_score",
                    "fraud_score","risk_tier","decision",
                    "loan_amnt","int_rate","dti","annual_inc"
                ] if c in batch.columns
            ]
            top10 = batch.nlargest(10, "unified_score")[
                top10_cols].reset_index(drop=True)
            top10.index += 1
            st.dataframe(top10, use_container_width=True, height=350)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Download ────────────────────────────────────────
            csv_out = batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label     = "⬇️ Download Scored Results CSV (for Power BI)",
                data      = csv_out,
                file_name = "scored_customers.csv",
                mime      = "text/csv"
            )