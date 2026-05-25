import pandas as pd
import numpy as np

def create_features(df):

    df = df.sort_values(["sku", "week"]).reset_index(drop=True)

    # time index
    df["time_idx"] = df.groupby("sku").cumcount()

    # lag features
    df["lag_1"] = df.groupby("sku")["weekly_sales"].shift(1)
    df["lag_2"] = df.groupby("sku")["weekly_sales"].shift(2)
    df["lag_4"] = df.groupby("sku")["weekly_sales"].shift(4)

    # rolling mean
    df["rolling_mean_4"] = df.groupby("sku")["weekly_sales"].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )

    # week features
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["week_num"] = df["week"].dt.isocalendar().week.astype(int)

    df["week_sin"] = np.sin(2 * np.pi * df["week_num"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_num"] / 52)

    # price feature
    df["price_change"] = df.groupby("sku")["price"].pct_change().fillna(0)

    df = df.dropna()

    return df