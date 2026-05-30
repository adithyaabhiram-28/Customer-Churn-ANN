import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-weight: 600;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    /* Selectbox and input styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #6366f1 !important;
    }

    .stSlider > div > div > div {
        background-color: #6366f1 !important;
    }

    /* Number input */
    .stNumberInput input {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
    }

    /* Headers */
    h1 { color: #f8fafc !important; font-weight: 700 !important; }
    h2 { color: #e2e8f0 !important; font-weight: 600 !important; }
    h3 { color: #cbd5e1 !important; font-weight: 600 !important; }

    /* Labels */
    label { color: #94a3b8 !important; font-weight: 500 !important; }

    /* Success/Error boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 12px !important;
        color: #4ade80 !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 12px !important;
        color: #f87171 !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #94a3b8 !important;
    }

    /* Divider */
    hr {
        border-color: rgba(148, 163, 184, 0.1) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler with error handling
@st.cache_resource
def load_resources():
    try:
        model = load_model("churn_model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_resources()

# Header Section
st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 8px; background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            📊 Customer Churn Prediction
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0;">
            AI-powered customer retention analytics dashboard
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with better organization
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding-bottom: 20px; border-bottom: 1px solid rgba(148,163,184,0.1); margin-bottom: 20px;">
            <h2 style="color: #f8fafc; font-size: 1.3rem; margin: 0;">📝 Customer Profile</h2>
            <p style="color: #64748b; font-size: 0.85rem; margin: 5px 0 0 0;">Configure customer details below</p>
        </div>
    """, unsafe_allow_html=True)

    # Personal Info
    st.markdown("<p style='color: #6366f1; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px;'>👤 PERSONAL INFO</p>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"], label_visibility="collapsed")
    st.caption("Gender")

    col1, col2 = st.columns(2)
    with col1:
        senior = st.selectbox("Senior Citizen", [0, 1], label_visibility="collapsed")
        st.caption("Senior")
    with col2:
        partner = st.selectbox("Partner", ["No", "Yes"], label_visibility="collapsed")
        st.caption("Partner")

    dependents = st.selectbox("Dependents", ["No", "Yes"], label_visibility="collapsed")
    st.caption("Dependents")

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(148,163,184,0.1);'>", unsafe_allow_html=True)

    # Services
    st.markdown("<p style='color: #6366f1; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px;'>📞 SERVICES</p>", unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"], label_visibility="collapsed")
    st.caption("Phone Service")

    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], label_visibility="collapsed")
    st.caption("Multiple Lines")

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], label_visibility="collapsed")
    st.caption("Internet Service")

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(148,163,184,0.1);'>", unsafe_allow_html=True)

    # Add-ons
    st.markdown("<p style='color: #6366f1; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px;'>🔒 ADD-ONS</p>", unsafe_allow_html=True)

    addon_cols = st.columns(2)
    with addon_cols[0]:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], label_visibility="collapsed")
        st.caption("Security")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], label_visibility="collapsed")
        st.caption("Backup")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], label_visibility="collapsed")
        st.caption("Support")
    with addon_cols[1]:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], label_visibility="collapsed")
        st.caption("Protection")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], label_visibility="collapsed")
        st.caption("TV")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], label_visibility="collapsed")
        st.caption("Movies")

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(148,163,184,0.1);'>", unsafe_allow_html=True)

    # Billing
    st.markdown("<p style='color: #6366f1; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px;'>💳 BILLING</p>", unsafe_allow_html=True)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], label_visibility="collapsed")
    st.caption("Contract Type")

    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], label_visibility="collapsed")
    st.caption("Paperless")

    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], label_visibility="collapsed")
    st.caption("Payment Method")

    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=50.0)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

# Main content area
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    # Customer Summary Card
    st.markdown("""
        <div class="metric-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #f8fafc; font-size: 1.2rem;">📋 Customer Summary</h3>
        </div>
    """, unsafe_allow_html=True)

    # Summary metrics in a grid
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("Tenure", f"{tenure} mo")
    with sum_col2:
        st.metric("Monthly", f"${monthly_charges:.0f}")
    with sum_col3:
        st.metric("Total", f"${total_charges:.0f}")
    with sum_col4:
        avg_monthly = total_charges / max(tenure, 1)
        st.metric("Avg/Month", f"${avg_monthly:.0f}")

    # Service overview
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    svc_col1, svc_col2, svc_col3 = st.columns(3)
    with svc_col1:
        phone_icon = "📞" if phone_service == "Yes" else "❌"
        st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 16px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{phone_icon}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Phone Service</div>
                <div style="color: #f8fafc; font-weight: 600;">{phone_service}</div>
            </div>
        """, unsafe_allow_html=True)
    with svc_col2:
        net_icon = "🌐" if internet_service != "No" else "❌"
        st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 16px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{net_icon}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Internet</div>
                <div style="color: #f8fafc; font-weight: 600;">{internet_service}</div>
            </div>
        """, unsafe_allow_html=True)
    with svc_col3:
        contract_colors = {"Month-to-month": "🟡", "One year": "🔵", "Two year": "🟢"}
        st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 16px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{contract_colors.get(contract, "📄")}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Contract</div>
                <div style="color: #f8fafc; font-weight: 600;">{contract}</div>
            </div>
        """, unsafe_allow_html=True)

with col_main2:
    # Risk indicators
    st.markdown("""
        <div class="metric-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #f8fafc; font-size: 1.2rem;">⚡ Risk Indicators</h3>
        </div>
    """, unsafe_allow_html=True)

    # Risk factors
    risk_score = 0
    risk_factors = []

    if contract == "Month-to-month":
        risk_score += 30
        risk_factors.append(("Month-to-month contract", "high"))
    if tenure < 12:
        risk_score += 25
        risk_factors.append(("Low tenure (< 12 mo)", "high"))
    if payment_method == "Electronic check":
        risk_score += 20
        risk_factors.append(("Electronic check payment", "medium"))
    if internet_service == "Fiber optic":
        risk_score += 15
        risk_factors.append(("Fiber optic service", "medium"))
    if senior == 1:
        risk_score += 10
        risk_factors.append(("Senior citizen", "low"))

    risk_score = min(risk_score, 100)

    # Risk gauge
    risk_color = "#22c55e" if risk_score < 40 else "#f59e0b" if risk_score < 70 else "#ef4444"

    st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; font-weight: 700; color: {risk_color};">{risk_score}%</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Base Risk Score</div>
            <div style="margin-top: 15px; background: rgba(255,255,255,0.05); border-radius: 8px; height: 8px; overflow: hidden;">
                <div style="width: {risk_score}%; height: 100%; background: linear-gradient(90deg, {risk_color}, {risk_color}88); border-radius: 8px; transition: width 0.5s ease;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    for factor, level in risk_factors[:3]:
        level_colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
        level_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 8px 12px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 6px;">
                <span style="margin-right: 8px;">{level_icons[level]}</span>
                <span style="color: #cbd5e1; font-size: 0.85rem;">{factor}</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Prediction Section
if predict_btn and model is not None and scaler is not None:
    # Encoding mappings
    mappings = {
        "gender": {"Female": 0, "Male": 1},
        "partner": {"No": 0, "Yes": 1},
        "dependents": {"No": 0, "Yes": 1},
        "phone_service": {"No": 0, "Yes": 1},
        "multiple_lines": {"No": 0, "No phone service": 1, "Yes": 2},
        "internet_service": {"DSL": 0, "Fiber optic": 1, "No": 2},
        "online_security": {"No": 0, "No internet service": 1, "Yes": 2},
        "online_backup": {"No": 0, "No internet service": 1, "Yes": 2},
        "device_protection": {"No": 0, "No internet service": 1, "Yes": 2},
        "tech_support": {"No": 0, "No internet service": 1, "Yes": 2},
        "streaming_tv": {"No": 0, "No internet service": 1, "Yes": 2},
        "streaming_movies": {"No": 0, "No internet service": 1, "Yes": 2},
        "contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "paperless_billing": {"No": 0, "Yes": 1},
        "payment_method": {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        }
    }

    input_data = pd.DataFrame([{
        "gender": mappings["gender"][gender],
        "SeniorCitizen": senior,
        "Partner": mappings["partner"][partner],
        "Dependents": mappings["dependents"][dependents],
        "tenure": tenure,
        "PhoneService": mappings["phone_service"][phone_service],
        "MultipleLines": mappings["multiple_lines"][multiple_lines],
        "InternetService": mappings["internet_service"][internet_service],
        "OnlineSecurity": mappings["online_security"][online_security],
        "OnlineBackup": mappings["online_backup"][online_backup],
        "DeviceProtection": mappings["device_protection"][device_protection],
        "TechSupport": mappings["tech_support"][tech_support],
        "StreamingTV": mappings["streaming_tv"][streaming_tv],
        "StreamingMovies": mappings["streaming_movies"][streaming_movies],
        "Contract": mappings["contract"][contract],
        "PaperlessBilling": mappings["paperless_billing"][paperless_billing],
        "PaymentMethod": mappings["payment_method"][payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    scaled_data = scaler.transform(input_data)
    probability = model.predict(scaled_data, verbose=0)[0][0]
    prediction = int(probability > 0.5)

    # Results display
    st.markdown("<hr style='border-color: rgba(148,163,184,0.1); margin: 30px 0;'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="font-size: 1.8rem; color: #f8fafc;">🔮 Prediction Result</h2>
        </div>
    """, unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

    with res_col2:
        if prediction == 1:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center; border: 2px solid rgba(239, 68, 68, 0.3);">
                    <div style="font-size: 4rem; margin-bottom: 10px;">⚠️</div>
                    <h2 style="color: #f87171; margin: 0;">High Churn Risk</h2>
                    <p style="color: #94a3b8; margin-top: 8px;">This customer is likely to churn</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center; border: 2px solid rgba(34, 197, 94, 0.3);">
                    <div style="font-size: 4rem; margin-bottom: 10px;">✅</div>
                    <h2 style="color: #4ade80; margin: 0;">Likely to Stay</h2>
                    <p style="color: #94a3b8; margin-top: 8px;">This customer is likely to remain</p>
                </div>
            """, unsafe_allow_html=True)

    # Probability visualization
    prob_col1, prob_col2, prob_col3 = st.columns([1, 2, 1])
    with prob_col2:
        prob_pct = probability * 100
        prob_color = "#ef4444" if prob_pct > 50 else "#22c55e"

        st.markdown(f"""
            <div class="metric-card" style="margin-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Churn Probability</span>
                    <span style="color: {prob_color}; font-size: 1.5rem; font-weight: 700;">{prob_pct:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="width: {prob_pct}%; height: 100%; background: linear-gradient(90deg, {prob_color}88, {prob_color}); border-radius: 10px; transition: width 1s ease;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <span style="color: #22c55e; font-size: 0.75rem;">0% — Stay</span>
                    <span style="color: #ef4444; font-size: 0.75rem;">100% — Churn</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #f8fafc; margin-top: 0;">💡 Recommendations</h3>
            </div>
        """, unsafe_allow_html=True)

        recommendations = []
        if contract == "Month-to-month":
            recommendations.append("📌 Offer a longer-term contract with discount")
        if tenure < 12:
            recommendations.append("📌 Implement onboarding retention program")
        if payment_method == "Electronic check":
            recommendations.append("📌 Encourage auto-pay setup with incentive")
        if internet_service == "Fiber optic" and online_security == "No":
            recommendations.append("📌 Bundle security services with internet")
        if tech_support == "No":
            recommendations.append("📌 Offer complimentary tech support trial")
        if not recommendations:
            recommendations.append("📌 Continue monitoring customer satisfaction")
            recommendations.append("📌 Offer loyalty rewards program")

        for rec in recommendations[:4]:
            st.markdown(f"""
                <div style="padding: 10px 14px; background: rgba(99, 102, 241, 0.1); border-left: 3px solid #6366f1; border-radius: 0 8px 8px 0; margin-bottom: 8px;">
                    <span style="color: #cbd5e1; font-size: 0.9rem;">{rec}</span>
                </div>
            """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #f8fafc; margin-top: 0;">📊 Key Factors</h3>
            </div>
        """, unsafe_allow_html=True)

        factors = [
            ("Contract Type", contract, "high" if contract == "Month-to-month" else "low"),
            ("Tenure", f"{tenure} months", "high" if tenure < 12 else "low"),
            ("Payment Method", payment_method, "high" if payment_method == "Electronic check" else "low"),
            ("Internet Service", internet_service, "high" if internet_service == "Fiber optic" else "low"),
        ]

        for name, value, impact in factors:
            impact_color = "#ef4444" if impact == "high" else "#22c55e"
            impact_label = "🔴 High Impact" if impact == "high" else "🟢 Low Impact"
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 8px;">
                    <div>
                        <div style="color: #94a3b8; font-size: 0.8rem;">{name}</div>
                        <div style="color: #f8fafc; font-weight: 500;">{value}</div>
                    </div>
                    <div style="color: {impact_color}; font-size: 0.75rem; font-weight: 600;">{impact_label}</div>
                </div>
            """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px; opacity: 0.5;">
            <div style="font-size: 4rem; margin-bottom: 20px;">🔮</div>
            <h3 style="color: #64748b;">Ready to Predict</h3>
            <p style="color: #475569;">Fill in the customer details in the sidebar and click "Predict Churn" to see the AI-powered analysis.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0; color: #475569; font-size: 0.8rem;">
        <hr style="border-color: rgba(148,163,184,0.1); margin-bottom: 20px;">
        Customer Churn Prediction Dashboard • Powered by TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)
