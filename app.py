import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 
# 1. BUILT-IN DATA CLEANING & FEATURE ENGINEERING
# 
FEATURE_COLUMNS = [
    "Income",
    "Age",
    "Experience",
    "Married/Single",
    "House_Ownership",
    "Car_Ownership",
    "Profession",
    "CITY",
    "STATE",
    "CURRENT_JOB_YRS",
    "CURRENT_HOUSE_YRS",
    "Income_per_Age",
    "Job_to_Exp_Ratio",
    "Exp_to_Age_Ratio",
    "Stability_Score",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw applicant records and engineer financial/stability ratios."""
    X = df.copy()
    for col in ("Id", "Risk_Flag"):
        if col in X.columns:
            X = X.drop(columns=[col])

    X["CITY"] = (
        X["CITY"]
        .astype(str)
        .str.replace(r"\[\d+\]", "", regex=True)
        .str.strip()
    )

    safe_age = np.maximum(X["Age"].astype(float), 1.0)
    X["Income_per_Age"] = X["Income"] / safe_age
    X["Job_to_Exp_Ratio"] = np.where(
        X["Experience"] > 0, X["CURRENT_JOB_YRS"] / X["Experience"], 0.0
    )
    X["Exp_to_Age_Ratio"] = X["Experience"] / safe_age
    X["Stability_Score"] = X["CURRENT_JOB_YRS"] + X["CURRENT_HOUSE_YRS"]

    return X[FEATURE_COLUMNS]



# 2. STREAMLIT PAGE SETUP & MODEL LOADING

st.set_page_config(
    page_title="Credit Risk Underwriting Decision Support",
    page_icon="🏦",
    layout="wide",
)

MODEL_PATH = "credit_risk_rf_artifact.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Could not find '{MODEL_PATH}'. Please run your model training cell first so the saved model file is created."
    )
    st.stop()


@st.cache_resource
def load_model_artifact(path: str):
    return joblib.load(path)


artifact = load_model_artifact(MODEL_PATH)
pipeline = artifact["pipeline"]
benchmarks = artifact["benchmarks"]

st.title("🏦Loan Default Risk Assessment — Underwriting Decision Support")
st.info(
    "**Human-in-the-Loop Governance Notice:** This tool provides an empirical risk estimate "
    "from a validated Random Forest model (Test ROC-AUC: 0.931, Recall: 76.9%) to prioritize "
    "and structure manual loan review. **It does not automatically approve or reject applicants.** "
    "Final credit decisions require verification of supporting documents by an authorized loan officer."
)

# --- Sidebar: Model Reference & Risk Tier Definitions ---
with st.sidebar:
    st.header("Model & Risk Tier Guide")
    st.markdown(
        """
        **Selected Model:** Random Forest  
        **Validation:** 3-Fold Stratified CV (`n = 201,600`)  
        **Untouched Test Metrics (`n = 50,400`):**
        - **Accuracy:** 89.30%
        - **Precision:** 54.62%
        - **Recall:** 76.85%
        - **F1-Score:** 0.6386
        ---
        **Review Routing Thresholds:**
        - **Low Risk (< 30%):** Standard verification track
        -  **Moderate Risk (30% – 55%):** Enhanced income & tenure verification
        -  **High Risk (≥ 55%):** Senior underwriter / committee review
        """
    )

# --- Main Form: Applicant Data Entry ---
st.subheader("1. Enter Loan Applicant Profile")

with st.form("applicant_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Financial & Career Profile**")
        income = st.number_input(
            "Annual Income",
            min_value=10_000,
            max_value=20_000_000,
            value=4_500_000,
            step=50_000,
        )
        profession = st.selectbox(
            "Profession",
            options=artifact["professions"],
            index=artifact["professions"].index("Software_Developer")
            if "Software_Developer" in artifact["professions"]
            else 0,
        )
        experience = st.slider("Total Work Experience (Years)", min_value=0, max_value=30, value=6)
        current_job_yrs = st.slider("Years in Current Job", min_value=0, max_value=20, value=3)

    with col2:
        st.markdown("**Demographics & Asset Ownership**")
        age = st.slider("Applicant Age (Years)", min_value=21, max_value=80, value=36)
        marital_status = st.selectbox("Marital Status", options=["single", "married"])
        house_ownership = st.selectbox(
            "House Ownership", options=["rented", "owned", "norent_noown"]
        )
        car_ownership = st.selectbox("Car Ownership", options=["no", "yes"])

    with col3:
        st.markdown("**Residential Stability & Location**")
        current_house_yrs = st.slider(
            "Years at Current Residence", min_value=10, max_value=14, value=12
        )
        state = st.selectbox(
            "State",
            options=artifact["states"],
            index=artifact["states"].index("Maharashtra")
            if "Maharashtra" in artifact["states"]
            else 0,
        )
        city = st.selectbox("City", options=artifact["cities"], index=0)

    submitted = st.form_submit_button("Generate Risk Assessment", type="primary")

if current_job_yrs > experience:
    st.warning(
        f"Note: Years in Current Job ({current_job_yrs}) exceeds Total Work Experience ({experience}). "
        "Please verify employment history dates during manual review."
    )

if submitted:
    raw_input = pd.DataFrame(
        [
            {
                "Income": income,
                "Age": age,
                "Experience": experience,
                "Married/Single": marital_status,
                "House_Ownership": house_ownership,
                "Car_Ownership": car_ownership,
                "Profession": profession,
                "CITY": city,
                "STATE": state,
                "CURRENT_JOB_YRS": current_job_yrs,
                "CURRENT_HOUSE_YRS": current_house_yrs,
            }
        ]
    )

    engineered_df = prepare_features(raw_input)
    default_prob = float(pipeline.predict_proba(engineered_df)[0, 1])
    model_flag = int(default_prob >= 0.50)

    st.divider()
    st.subheader("2. Model Risk Assessment & Key Factors")

    if default_prob < 0.30:
        tier_label = "LOW RISK — Standard Underwriting Review"
        tier_color = "green"
        guidance = (
            "Applicant profile aligns with low historical default incidence. "
            "Proceed with standard KYC, payslip, and bank-statement verification."
        )
    elif default_prob < 0.55:
        tier_label = "MODERATE / BORDERLINE RISK — Enhanced Verification Recommended"
        tier_color = "orange"
        guidance = (
            "Applicant sits near the decision boundary where false positives can occur "
            "(model precision at 0.50 cutoff is 54.6%). Verify debt-to-income obligations, "
            "employer stability, and liquidity reserves before deciding."
        )
    else:
        tier_label = "ELEVATED / HIGH RISK — Senior Underwriter Review Required"
        tier_color = "red"
        guidance = (
            "Applicant exhibits patterns strongly correlated with historical defaults. "
            "Escalate to a senior loan officer to evaluate collateral, co-signer options, "
            "or adjusted loan terms."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated Default Probability", f"{default_prob:.1%}")
    m2.metric("Portfolio Baseline Rate", f"{benchmarks['baseline_default_rate']:.1%}")
    m3.metric(
        "Income / Age Ratio",
        f"{engineered_df['Income_per_Age'].iloc[0]:,.0f}",
        delta=f"{engineered_df['Income_per_Age'].iloc[0] - benchmarks['median_income_per_age']:+,.0f} vs. median",
    )
    m4.metric(
        "Stability Score (Job + House Yrs)",
        f"{int(engineered_df['Stability_Score'].iloc[0])} yrs",
        delta=f"{engineered_df['Stability_Score'].iloc[0] - benchmarks['median_stability_score']:+.1f} vs. median",
    )

    st.progress(min(max(default_prob, 0.0), 1.0))
    if tier_color == "green":
        st.success(f"**{tier_label}** — {guidance}")
    elif tier_color == "orange":
        st.warning(f"**{tier_label}** — {guidance}")
    else:
        st.error(f"**{tier_label}** — {guidance}")

    st.markdown("**Applicant Profile Contextual Indicators:**")
    if income < benchmarks["median_income"]:
        st.markdown(
            f"- **Income Below Portfolio Median:** Reported income (`{income:,.0f}`) is below the "
            f"development-set median (`{benchmarks['median_income']:,.0f}`)."
        )
    else:
        st.markdown(
            f"- **Income Above Portfolio Median:** Reported income (`{income:,.0f}`) exceeds the "
            f"development-set median (`{benchmarks['median_income']:,.0f}`)."
        )

    if current_job_yrs < 3:
        st.markdown(
            f"- **Short Current Job Tenure ({current_job_yrs} yrs):** Early-tenure employment carries "
            "higher income-continuity risk; confirm probation completion."
        )
    else:
        st.markdown(
            f"- **Established Job Tenure ({current_job_yrs} yrs):** Supports steady repayment capacity."
        )

    if house_ownership == "rented":
        st.markdown(
            "- **Rented Housing:** Factor monthly rental outflow into manual debt-service ratio checks."
        )
    elif house_ownership == "owned":
        st.markdown(
            "- **Owned Housing:** Indicates stronger residential and asset stability."
        )

    # --- Section 3: Mandatory Human Loan Officer Review ---
    st.divider()
    st.subheader("3. Loan Officer Manual Review & Decision Log")
    c_left, c_right = st.columns(2)
    with c_left:
        st.checkbox("Verified government ID & KYC documentation")
        st.checkbox("Verified income statements / tax returns against declared Income")
        st.checkbox("Checked external credit bureau report & existing debt obligations")
    with c_right:
        officer_action = st.selectbox(
            "Underwriter Recommended Action (Human Decision)",
            options=[
                "Pending Additional Documentation",
                "Recommend Approval — Standard Terms",
                "Recommend Approval — Reduced Principal / Co-Signer",
                "Recommend Decline — Documented Credit Policy Reason",
            ],
        )
        reviewer_notes = st.text_area(
            "Underwriter Justification Notes (Required for Audit Trail)",
            placeholder="Record qualitative factors, verified cash flows, or mitigating circumstances...",
        )
        if st.button("Log Underwriter Assessment"):
            st.success(
                f"Review logged: **{officer_action}** (Model Risk Score: {default_prob:.1%}, "
                f"Binary Flag: {model_flag})."
            )
