import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from xgboost import XGBClassifier

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Credit Risk Engine",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Explainable Credit Risk Engine")
st.markdown("*Powered by XGBoost + SHAP — Built for transparent credit decision support*")

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("Borrower Information")

age = st.sidebar.slider("Age", 18, 80, 40)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 0, 50000, 5000, step=500)
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 1.0, 0.3, step=0.01)
revolving_util = st.sidebar.slider("Revolving Utilization of Unsecured Lines", 0.0, 1.0, 0.3, step=0.01)
num_open_credit = st.sidebar.slider("Number of Open Credit Lines and Loans", 0, 20, 5)
num_real_estate = st.sidebar.slider("Number of Real Estate Loans or Lines", 0, 10, 1)
num_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 1)
times_30_59 = st.sidebar.slider("Times 30-59 Days Past Due", 0, 10, 0)
times_60_89 = st.sidebar.slider("Times 60-89 Days Past Due", 0, 10, 0)
times_90_late = st.sidebar.slider("Times 90+ Days Late", 0, 10, 0)

# ── Build input dataframe ─────────────────────────────────────
input_data = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': revolving_util,
    'age': age,
    'NumberOfTime30-59DaysPastDueNotWorse': times_30_59,
    'DebtRatio': debt_ratio,
    'MonthlyIncome': monthly_income,
    'NumberOfOpenCreditLinesAndLoans': num_open_credit,
    'NumberOfTimes90DaysLate': times_90_late,
    'NumberRealEstateLoansOrLines': num_real_estate,
    'NumberOfTime60-89DaysPastDueNotWorse': times_60_89,
    'NumberOfDependents': num_dependents
}])

# ── Load data and train model ─────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('data/cs-training.csv', index_col=0)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    df = df[df['age'] > 18]
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='auc', verbosity=0)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer

with st.spinner("Loading model..."):
    model, explainer = load_model()

# ── Prediction ────────────────────────────────────────────────
prob = model.predict_proba(input_data)[0][1]
threshold = 0.24
decision = "🔴 High Risk — Loan Denied" if prob >= threshold else "🟢 Low Risk — Loan Approved"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Credit Decision")
    st.metric("Default Probability", f"{prob:.1%}")
    st.markdown(f"### {decision}")
    st.caption(f"Decision threshold: {threshold}")

with col2:
    st.subheader("Risk Gauge")
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.barh(0, prob, color='red' if prob >= threshold else 'green', height=0.4)
    ax.barh(0, 1 - prob, left=prob, color='lightgray', height=0.4)
    ax.axvline(x=threshold, color='orange', linestyle='--', linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Default Probability")
    st.pyplot(fig)

# ── SHAP Waterfall ────────────────────────────────────────────
st.subheader("Why this decision? — SHAP Explanation")

shap_values = explainer.shap_values(input_data)
fig2, ax2 = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=input_data.iloc[0],
    feature_names=input_data.columns.tolist()
), show=False)
st.pyplot(fig2)

st.markdown("---")
st.caption("Explainable Credit Risk Engine · Built with XGBoost + SHAP · For educational purposes only") 
