import streamlit as st
import requests

st.title("📦 Supply Chain Demand Forecasting System")

sku_id = st.number_input("Enter SKU ID", min_value=1, step=1)
current_stock = st.number_input("Current Stock", min_value=0)

if st.button("Run Forecast"):

    url = f"http://127.0.0.1:8000/full_analysis/{sku_id}"

    payload = {
        "current_stock": current_stock
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()

        st.subheader("📊 Forecast (Next 3 Weeks)")
        st.write(data["forecast"])

        st.subheader("📦 Inventory Analysis")
        st.write(data["inventory_analysis"])

    else:
        st.error("API Error")