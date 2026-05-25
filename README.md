# 📦 Supply Chain AI Forecasting System

## 🚀 Overview
This project is an end-to-end AI-powered supply chain demand forecasting system that predicts SKU-level demand using multiple machine learning and deep learning models. It includes a FastAPI backend and a premium Streamlit dashboard for real-time insights. The system is designed with a multi-model architecture for accuracy comparison and production-style deployment readiness.

## 🧠 Key Features
- Multi-model forecasting using XGBoost, LSTM, ARIMA, and TFT
- Real-time REST API built with FastAPI
- Premium interactive Streamlit dashboard
- SKU-level demand prediction engine
- Temporal Fusion Transformer (SOTA time-series model)
- Clean modular ML pipeline design
- Ready for Docker and cloud deployment

## 🏗️ Architecture
Frontend (Streamlit Dashboard) → FastAPI Backend → ML Models Engine (XGBoost / LSTM / ARIMA / TFT) → Forecast Output (JSON Response)
---
## 📂 Project Structure 
```
supply chainn demand forecasting/
├── api.py (FastAPI backend)
├── dashboard/app.py (Streamlit dashboard)
├── models/
│   ├── xgb_model.pkl
│   ├── tft_model.pth
│   └── lstm_model.py
├── data/sales.csv
├── requirements.txt
└── README.md
```
---
## ⚙️ Tech Stack
Python, FastAPI, Streamlit, Scikit-learn, XGBoost, PyTorch, TensorFlow/Keras, PyTorch Forecasting (TFT), Pandas, NumPy

## 🚀 How to Run Locally

### 1. Clone Repository
git clone https://github.com/your-username/supply-chain-ai-forecasting.git  
cd supply-chain-ai-forecasting  

### 2. Install Dependencies
pip install -r requirements.txt  

### 3. Run Backend
uvicorn api:app --reload  
Backend: http://127.0.0.1:8000  

### 4. Run Dashboard
streamlit run dashboard/app.py  

## 📡 API Usage

Endpoint: POST /predict  

Request Body:
{
  "sku": 1,
  "model": "xgb"
}

Supported Models:
xgb, lstm, arima, tft

## 📊 Example Response
{
  "model": "xgb",
  "forecast": 8.59
}

## 📈 Business Use Cases
Retail demand forecasting, inventory optimization, supply chain planning, stock prediction, and dynamic pricing support.

## 🧠 Future Improvements
Add SHAP explainability, real-time dashboards, Docker deployment, Hugging Face/AWS hosting, and automated model selection.

## 👨‍💻 Author
AI + Supply Chain Engineering Project focused on production-grade ML system design and real-world forecasting systems.

## ⭐ Support
If you like this project, give it a star on GitHub.
