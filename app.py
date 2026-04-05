import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #F8FAFC; }
.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 24px;
    padding: 60px 48px;
    text-align: center;
    margin-bottom: 36px;
    box-shadow: 0 20px 60px rgba(102,126,234,0.3);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 20px;
    border-radius: 100px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.3);
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 16px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.85);
    max-width: 600px;
    margin: 0 auto 28px;
    line-height: 1.7;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 48px;
    margin-top: 32px;
    flex-wrap: wrap;
}
.hero-stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: white;
}
.hero-stat-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}
.card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06);
    border: 1px solid #E2E8F0;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #667eea;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #EEF2FF;
}
.risk-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    font-size: 1rem;
    color: #374151;
    font-weight: 500;
}
.badge-danger {
    background: #FEE2E2;
    color: #DC2626;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-safe {
    background: #DCFCE7;
    color: #16A34A;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
}
.result-danger {
    background: white;
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    border: 2px solid #FECACA;
    box-shadow: 0 8px 40px rgba(220,38,38,0.1);
}
.result-safe {
    background: white;
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    border: 2px solid #BBF7D0;
    box-shadow: 0 8px 40px rgba(22,163,74,0.1);
}
.result-icon { font-size: 4rem; margin-bottom: 16px; }
.result-heading-danger {
    font-size: 2rem;
    font-weight: 800;
    color: #DC2626;
    margin-bottom: 8px;
}
.result-heading-safe {
    font-size: 2rem;
    font-weight: 800;
    color: #16A34A;
    margin-bottom: 8px;
}
.result-sub { color: #718096; font-size: 1rem; margin-bottom: 28px; }
.prob-circle-danger {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: #FEE2E2;
    border: 5px solid #DC2626;
    font-size: 2rem;
    font-weight: 800;
    color: #DC2626;
    margin-bottom: 28px;
}
.prob-circle-safe {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: #DCFCE7;
    border: 5px solid #16A34A;
    font-size: 2rem;
    font-weight: 800;
    color: #16A34A;
    margin-bottom: 28px;
}
.rec-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: #F8FAFC;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.95rem;
    color: #374151;
    text-align: left;
    border: 1px solid #E2E8F0;
    font-weight: 500;
}
label { color: #374151 !important; font-size: 0.95rem !important; font-weight: 500 !important; }
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    background: #F8FAFC !important;
    font-size: 1rem !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 18px !important;
    letter-spacing: 0.5px !important;
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, 'models', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">🛡️ AI-Powered Churn Analytics</div>
    <div class="hero-title">Welcome to ChurnShield</div>
    <div class="hero-subtitle">
        Predict customer churn before it happens. Fill in the customer details
        below and let our Machine Learning model instantly assess their retention risk.
    </div>
    <div class="hero-stats">
        <div><div class="hero-stat-num">7,043</div><div class="hero-stat-label">Customers Trained</div></div>
        <div><div class="hero-stat-num">79%</div><div class="hero-stat-label">Model Accuracy</div></div>
        <div><div class="hero-stat-num">19</div><div class="hero-stat-label">Features Used</div></div>
        <div><div class="hero-stat-num">26.5%</div><div class="hero-stat-label">Avg Churn Rate</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:

    st.markdown('<div class="card"><div class="card-title">👤 Customer Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender           = st.selectbox("Gender", ["Female", "Male"])
        senior           = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner          = st.selectbox("Has Partner", ["No", "Yes"])
        dependents       = st.selectbox("Has Dependents", ["No", "Yes"])
        phone_service    = st.selectbox("Phone Service", ["No", "Yes"])
    with c2:
        multiple_lines   = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment          = st.selectbox("Payment Method", [
                               "Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🌐 Services Subscribed</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        online_security   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup     = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
with c4:
    tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">💰 Billing Information</div>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges   = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)
    st.markdown('</div>', unsafe_allow_html=True)

with right:

    st.markdown('<div class="card"><div class="card-title">📊 Live Risk Analysis</div>', unsafe_allow_html=True)

    def badge(condition, yes_label, no_label):
        if condition:
            return f'<span class="badge-danger">{yes_label}</span>'
        return f'<span class="badge-safe">{no_label}</span>'

    st.markdown(f"""
    <div class="risk-item">Contract {badge(contract=="Month-to-month","Risky","Safe")}</div>
    <div class="risk-item">Internet {badge(internet_service=="Fiber optic","High Churn","Low Churn")}</div>
    <div class="risk-item">Tenure {badge(tenure<12,"New Customer","Loyal Customer")}</div>
    <div class="risk-item">Monthly Charges {badge(monthly_charges>70,f"${monthly_charges:.0f} High",f"${monthly_charges:.0f} OK")}</div>
    <div class="risk-item">Online Security {badge(online_security=="No","Not Protected","Protected")}</div>
    <div class="risk-item">Tech Support {badge(tech_support=="No","No Support","Has Support")}</div>
    <div class="risk-item">Payment {badge(payment=="Electronic check","Risky","Stable")}</div>
    """, unsafe_allow_html=True)

    risk_score = sum([
        contract == "Month-to-month",
        internet_service == "Fiber optic",
        tenure < 12,
        monthly_charges > 70,
        online_security == "No",
        tech_support == "No",
        payment == "Electronic check"
    ])
    risk_pct   = int((risk_score / 7) * 100)
    bar_color  = "#DC2626" if risk_pct > 55 else "#D97706" if risk_pct > 30 else "#16A34A"
    risk_label = "High Risk" if risk_pct > 55 else "Medium Risk" if risk_pct > 30 else "Low Risk"

    st.markdown(f"""
    <div style="margin-top:16px;padding:18px;background:#F8FAFC;
                border-radius:14px;border:1px solid #E2E8F0">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px">
            <span style="font-size:1rem;font-weight:600;color:#374151">Overall Risk Score</span>
            <span style="font-size:1rem;font-weight:700;color:{bar_color}">{risk_label} — {risk_pct}%</span>
        </div>
        <div style="background:#E2E8F0;border-radius:100px;height:10px">
            <div style="width:{risk_pct}%;background:{bar_color};height:10px;border-radius:100px"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Insights
    st.markdown('<div class="card"><div class="card-title">💡 Key Insights</div>', unsafe_allow_html=True)
    ia, ib = st.columns(2)
    with ia:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#EEF2FF,#E0E7FF);
                    border-radius:14px;padding:16px;border:1px solid #C7D2FE;margin-bottom:12px">
            <div style="font-size:1.6rem;margin-bottom:6px">📋</div>
            <div style="font-size:0.75rem;font-weight:700;color:#3730A3;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Contract</div>
            <div style="font-size:0.88rem;color:#4338CA;line-height:1.5">
                Month-to-month churns <b>3x more</b> than yearly</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#F0FDF4,#DCFCE7);
                    border-radius:14px;padding:16px;border:1px solid #BBF7D0">
            <div style="font-size:1.6rem;margin-bottom:6px">🌐</div>
            <div style="font-size:0.75rem;font-weight:700;color:#15803D;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Internet</div>
            <div style="font-size:0.88rem;color:#16A34A;line-height:1.5">
                Fiber optic users churn <b>more</b> than DSL</div>
        </div>""", unsafe_allow_html=True)

    with ib:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#FFF7ED,#FFEDD5);
                    border-radius:14px;padding:16px;border:1px solid #FED7AA;margin-bottom:12px">
            <div style="font-size:1.6rem;margin-bottom:6px">⏱️</div>
            <div style="font-size:0.75rem;font-weight:700;color:#C2410C;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Tenure</div>
            <div style="font-size:0.88rem;color:#EA580C;line-height:1.5">
                Under <b>12 months</b> = highest risk</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#FFF1F2,#FFE4E6);
                    border-radius:14px;padding:16px;border:1px solid #FECDD3">
            <div style="font-size:1.6rem;margin-bottom:6px">💵</div>
            <div style="font-size:0.75rem;font-weight:700;color:#BE123C;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Charges</div>
            <div style="font-size:0.88rem;color:#E11D48;line-height:1.5">
                Above <b>$70/month</b> increases churn risk</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🔍  Predict Churn Risk Now", use_container_width=True)

# ── Prediction ──────────────────────────────────────────────
if predict:
    gender_e     = 0 if gender == "Female" else 1
    senior_e     = 1 if senior == "Yes" else 0
    partner_e    = 0 if partner == "No" else 1
    dependents_e = 0 if dependents == "No" else 1
    phone_e      = 0 if phone_service == "No" else 1
    paperless_e  = 0 if paperless == "No" else 1

    multi_map    = {"No": 0, "No phone service": 1, "Yes": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    ynmap        = {"No": 0, "No internet service": 1, "Yes": 2}
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    pay_map      = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    inp = pd.DataFrame([[
        gender_e, senior_e, partner_e, dependents_e,
        float(tenure), phone_e,
        multi_map[multiple_lines],
        internet_map[internet_service],
        ynmap[online_security],
        ynmap[online_backup],
        ynmap[device_protection],
        ynmap[tech_support],
        ynmap[streaming_tv],
        ynmap[streaming_movies],
        contract_map[contract],
        paperless_e,
        pay_map[payment],
        float(monthly_charges),
        float(total_charges)
    ]], columns=[
        'gender','SeniorCitizen','Partner','Dependents','tenure',
        'PhoneService','MultipleLines','InternetService','OnlineSecurity',
        'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
        'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
        'MonthlyCharges','TotalCharges'
    ])

    pred = model.predict(inp)[0]
    prob = model.predict_proba(inp)[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)
    _, res_col, _ = st.columns([0.3, 2, 0.3])

    with res_col:
        if pred == 1:
            st.markdown(f"""
            <div class="result-danger">
                <div class="result-icon">⚠️</div>
                <div class="result-heading-danger">High Churn Risk Detected</div>
                <div class="result-sub">This customer shows strong indicators of leaving soon</div>
                <div class="prob-circle-danger">{prob:.0f}%</div>
                <div style="font-size:0.9rem;color:#9CA3AF;margin-bottom:28px">Churn Probability</div>
                <div class="rec-item">💼 Offer a discounted annual or two-year contract immediately</div>
                <div class="rec-item">🎁 Provide a loyalty reward — free month or service upgrade</div>
                <div class="rec-item">📞 Assign a dedicated retention agent within 24 hours</div>
                <div class="rec-item">🔐 Bundle online security and tech support at no extra cost</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-heading-safe">Low Churn Risk</div>
                <div class="result-sub">This customer appears satisfied and is likely to stay</div>
                <div class="prob-circle-safe">{prob:.0f}%</div>
                <div style="font-size:0.9rem;color:#9CA3AF;margin-bottom:28px">Churn Probability</div>
                <div class="rec-item">🚀 Great opportunity to upsell premium services</div>
                <div class="rec-item">⭐ Send a satisfaction survey to maintain engagement</div>
                <div class="rec-item">🎯 Offer referral program — happy customers bring friends</div>
                <div class="rec-item">📊 Monitor monthly to catch any early warning signs</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<br><br>
<div style="text-align:center;color:#9CA3AF;font-size:0.88rem;
            padding:28px;border-top:1px solid #E2E8F0;">
    🛡️ ChurnShield &nbsp;|&nbsp; Built with Random Forest + Streamlit
    &nbsp;|&nbsp; AI &amp; Data Science Portfolio Project
</div>
""", unsafe_allow_html=True)