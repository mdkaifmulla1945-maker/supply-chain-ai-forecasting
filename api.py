import pandas as pd
import numpy as np
import joblib
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from lstm_model import model as lstm_model
from arima_model import arima_forecast

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


# =========================
# APP
# =========================
app = FastAPI()


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales.csv")

df.columns = df.columns.str.replace(".", "_", regex=False)


# =========================
# FEATURE ENGINEERING (SINGLE SOURCE OF TRUTH)
# =========================
def build_xgb_features(df):
    df = df.copy()

    df["time_idx"] = df.groupby("sku").cumcount()

    df["lag_1"] = df.groupby("sku")["weekly_sales"].shift(1)
    df["lag_2"] = df.groupby("sku")["weekly_sales"].shift(2)
    df["lag_4"] = df.groupby("sku")["weekly_sales"].shift(4)

    df["rolling_mean_4"] = (
        df.groupby("sku")["weekly_sales"]
        .rolling(4).mean()
        .reset_index(level=0, drop=True)
    )

    df["week_sin"] = np.sin(2 * np.pi * df["time_idx"] / 7)
    df["week_cos"] = np.cos(2 * np.pi * df["time_idx"] / 7)

    df["price_change"] = df.groupby("sku")["price"].diff().fillna(0)

    df["trend"] = df["time_idx"]

    df = df.bfill().ffill()

    return df


df = build_xgb_features(df)
df = df.dropna().reset_index(drop=True)


# =========================
# XGBOOST
# =========================
xgb_model = joblib.load("models/xgb_model.pkl")

# =========================
# GLOBAL FEATURE CONFIG (FIXED)
# =========================
XGB_FEATURES = [
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


def build_xgb_input(data):
    data = data.copy()

    # rebuild features (safe for single SKU inference)
    data = build_xgb_features(data)

    # guarantee all columns exist
    for col in XGB_FEATURES:
        if col not in data.columns:
            data[col] = 0

    return data


# =========================
# TFT DATASET
# =========================
tft_dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="weekly_sales",
    group_ids=["sku"],
    max_encoder_length=10,
    max_prediction_length=3,
    time_varying_known_reals=["time_idx", "price", "trend"],
    time_varying_unknown_reals=["weekly_sales"],
)

# 

tft_model = None   # ❗ IMPORTANT FIX (DO NOT CREATE MODEL HERE)

try:
    state = torch.load("models/tft_model.pth", map_location="cpu")

    tft_model = TemporalFusionTransformer.from_dataset(tft_dataset)
    tft_model.load_state_dict(state, strict=False)

    tft_model.eval()

    print("TFT loaded successfully")

except Exception as e:
    print("TFT disabled:", e)



# =========================
# REQUEST MODEL
# =========================
class Request(BaseModel):
    sku: int
    model: str


# =========================
# HELPERS
# =========================
def get_sku_data(sku):
    data = df[df["sku"] == sku].copy()
    if len(data) < 5:
        return None
    return data


# =========================
# API
# =========================
@app.post("/predict")
def predict(req: Request):

    data = get_sku_data(req.sku)

    if data is None:
        return {"error": "Not enough data for SKU"}

    # =====================
    # ARIMA
    # =====================
    if req.model == "arima":
        pred = arima_forecast(df, req.sku)
        return {"model": "arima", "forecast": pred}

    # =====================
    # XGBOOST (FIXED)
    # =====================
    if req.model == "xgb":

       data_xgb = build_xgb_features(data)

       x = data_xgb[XGB_FEATURES].iloc[-1:].fillna(0)

       pred = xgb_model.predict(x)[0]

       return {
            "model": "xgb",
            "forecast": float(pred)
        }

    # =====================
    # LSTM
    # =====================
    if req.model == "lstm":

        seq = data["weekly_sales"].values[-5:]

        if len(seq) < 5:
            return {"error": "Not enough sequence data"}

        seq = np.array(seq).reshape(1, 5, 1)

        pred = lstm_model.predict(seq)[0][0]

        return {"model": "lstm", "forecast": float(pred)}

    # =====================
    # TFT
    # =====================
    if req.model == "tft":

        if tft_model is None:
            return {
                "model": "tft",
                "error": "TFT model not loaded"
            }

        try:
            dl = tft_dataset.to_dataloader(train=False, batch_size=1)
            x, _ = next(iter(dl))

            with torch.no_grad():
                out = tft_model(x)

            return {
                "model": "tft",
                "forecast": out[0].cpu().numpy().tolist()
            }

        except Exception as e:
            return {
                "model": "tft",
                "error": str(e)
            }
