import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import NaNLabelEncoder
from lightning.pytorch import Trainer

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales.csv")

# CLEAN COLUMN NAMES
df.columns = df.columns.str.replace(".", "_", regex=False)
df.columns = df.columns.str.replace(" ", "_", regex=False)

# TIME INDEX
df["time_idx"] = df.groupby("sku").cumcount()

df = df.sort_values(["sku", "time_idx"])

# =========================
# FEATURES
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

df["sku"] = df["sku"].astype(str)

# =========================
# DATASET
# =========================
max_encoder_length = 10
max_prediction_length = 3

training = TimeSeriesDataSet(
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

train_loader = training.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=0
)

# =========================
# MODEL
# =========================
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    loss=QuantileLoss(),
)

# =========================
# TRAINER (FIX)
# =========================
trainer = Trainer(
    max_epochs=20,
    accelerator="cpu"   # change to "gpu" if available
)

trainer.fit(
    tft,
    train_dataloaders=train_loader
)

# =========================
# SAVE MODEL
# =========================
torch.save(tft.state_dict(), "models/tft_model.pth")

print("TFT training complete and saved")