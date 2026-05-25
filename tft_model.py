import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

def train_tft(df):

    df["time_idx"] = df.groupby("sku").cumcount()

    max_encoder_length = 12
    max_prediction_length = 3

    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="weekly_sales",
        group_ids=["sku"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "week_num", "price"],
        time_varying_unknown_reals=["weekly_sales"],
    )

    train_loader = training.to_dataloader(train=True, batch_size=64)

    model = TemporalFusionTransformer.from_dataset(training)

    model.fit(train_loader)

    return model