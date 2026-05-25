import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import create_features
from arima_model import arima_forecast


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales.csv")
df = create_features(df)


# =========================
# CONFIG
# =========================
sku = 1

features = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_4",
    "time_idx",
    "week_sin",
    "week_cos",
    "price",
    "price_change",
    "trend"
]


# =========================
# FILTER SKU DATA
# =========================
data = df[df["sku"] == sku].dropna()

y_true = data["weekly_sales"].values[-3:]


# =========================
# ARIMA PREDICTION
# =========================
arima_pred = arima_forecast(pd.read_csv("data/sales.csv"), sku)


# =========================
# XGBOOST PREDICTION
# =========================
model = joblib.load("model/xgb_model.pkl")

x_input = data[features].iloc[-1:]

xgb_preds = []

temp = x_input.copy()

for _ in range(3):
    pred = model.predict(temp)[0]
    xgb_preds.append(pred)

    # update lags for next step
    temp["lag_4"] = temp["lag_2"]
    temp["lag_2"] = temp["lag_1"]
    temp["lag_1"] = pred


# =========================
# METRICS FUNCTION
# =========================
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


# =========================
# EVALUATION OUTPUT
# =========================
print("\n===== MODEL COMPARISON =====")

print("\nARIMA MAE:", mean_absolute_error(y_true, arima_pred))
print("XGB MAE:", mean_absolute_error(y_true, xgb_preds))

print("\nARIMA RMSE:", np.sqrt(mean_squared_error(y_true, arima_pred)))
print("XGB RMSE:", np.sqrt(mean_squared_error(y_true, xgb_preds)))

print("\nARIMA WMAPE:", wmape(y_true, arima_pred))
print("XGB WMAPE:", wmape(y_true, xgb_preds))