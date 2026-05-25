import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from utils import clean_columns   # IMPORTANT (shared fix)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales.csv")

# =========================
# CLEAN COLUMN NAMES (CRITICAL FIX)
# =========================
df = clean_columns(df)

# =========================
# BASIC PREPROCESSING
# =========================
df["time_idx"] = df.groupby("sku").cumcount()
df = df.sort_values(["sku", "time_idx"])

df["sku"] = df["sku"].astype(str)

# =========================
# FEATURE ENGINEERING
# =========================
def create_features(data):
    data = data.copy()

    data["lag_1"] = data.groupby("sku")["weekly_sales"].shift(1)
    data["lag_2"] = data.groupby("sku")["weekly_sales"].shift(2)
    data["lag_4"] = data.groupby("sku")["weekly_sales"].shift(4)

    data["rolling_mean_4"] = (
        data.groupby("sku")["weekly_sales"]
        .rolling(4)
        .mean()
        .reset_index(0, drop=True)
    )

    data["week_sin"] = np.sin(2 * np.pi * data["time_idx"] / 7)
    data["week_cos"] = np.cos(2 * np.pi * data["time_idx"] / 7)

    data["price_change"] = data.groupby("sku")["price"].diff().fillna(0)

    return data.fillna(0)

df = create_features(df)

# =========================
# LOAD SAME TFT DATASET STRUCTURE (MUST MATCH TRAINING)
# =========================
max_encoder_length = 10
max_prediction_length = 3

dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="weekly_sales",
    group_ids=["sku"],

    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,

    static_categoricals=["sku"],

    time_varying_known_reals=[
        "time_idx",
        "price",
        "trend",
        "week_sin",
        "week_cos",
    ],

    time_varying_unknown_reals=[
        "weekly_sales",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "price_change",
    ],
)

# =========================
# DATALOADER
# =========================
dataloader = dataset.to_dataloader(
    train=False,
    batch_size=1,
    num_workers=0
)

# =========================
# LOAD MODEL
# =========================
from pytorch_forecasting import TemporalFusionTransformer

tft = TemporalFusionTransformer.from_dataset(dataset)

try:
    tft.load_state_dict(
        torch.load("models/tft_model.pth", map_location="cpu")
    )
except Exception as e:
    print("TFT LOAD ERROR:", e)

tft.eval()

# =========================
# PREDICTION / EXPLANATION
# =========================
x, y = next(iter(dataloader))

with torch.no_grad():
    raw_prediction = tft(x)

print("\n=== TFT RAW OUTPUT ===")
print(raw_prediction)

print("\n=== FINAL FORECAST ===")
print(raw_prediction[0].cpu().numpy())