import streamlit as st
import requests

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Supply Chain Intelligence",
    layout="wide",
    page_icon="📦"
)

# =========================
# PREMIUM LUXURY UI STYLE
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Rajdhani:wght@400;600&display=swap');

/* Global */
html, body, [class*="css"] {
    background: radial-gradient(circle at top, #0b0f1a, #05060a);
    color: #e6f1ff;
    font-family: 'Rajdhani', sans-serif;
}

/* Title */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    font-family: 'Orbitron', sans-serif;
    color: #00ffe1;
    text-shadow: 0 0 25px #00ffe1;
    margin-bottom: 25px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(0,255,225,0.3);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(0,255,225,0.08);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00ffe1, #7c4dff);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 18px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #00ffe1;
}

/* Inputs */
input, select {
    background-color: #0d1117 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Divider glow */
hr {
    border: 1px solid rgba(0,255,225,0.2);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='title'>🚀 Supply Chain AI Control Center</div>", unsafe_allow_html=True)

st.markdown("### Smart Demand Forecasting • Multi-Model Intelligence • Real-Time Predictions")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    sku = st.number_input("📦 Enter SKU ID", min_value=1, value=1)

with col2:
    model = st.selectbox(
        "🧠 Select AI Model",
        ["xgb", "lstm", "arima", "tft"]
    )

# =========================
# PREDICTION LOGIC
# =========================
if st.button("⚡ Generate Forecast"):

    url = "http://127.0.0.1:8000/predict"

    payload = {
        "sku": int(sku),
        "model": model
    }

    with st.spinner("Running AI models..."):

        try:
            response = requests.post(url, json=payload)
            result = response.json()

        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    # =========================
    # CLEAN OUTPUT UI
    # =========================
    if "forecast" in result:

        value = result["forecast"]
        model_name = result.get("model", "UNKNOWN")

        try:
            value_float = float(value)
        except:
            value_float = 0

        trend = "📈 Increasing Demand" if value_float > 0 else "📉 Decreasing Demand"

        st.markdown("## 🎯 Prediction Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='card'>
                <h3 style='color:#00ffe1;'>MODEL</h3>
                <h2>{model_name.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='card'>
                <h3 style='color:#7c4dff;'>FORECAST</h3>
                <h1>{value_float:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='card'>
                <h3 style='color:gold;'>TREND</h3>
                <h2>{trend}</h2>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 Raw Response"):
            st.json(result)

    else:
        st.error("Invalid response from API")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("⚡ Powered by XGBoost • LSTM • ARIMA • TFT | Enterprise Supply Chain AI")