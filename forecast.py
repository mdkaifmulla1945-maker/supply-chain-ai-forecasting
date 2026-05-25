import joblib
import pandas as pd

model = joblib.load("models/xgb_model.pkl")

def forecast_next_3_weeks(df, sku_id):

    sku_data = df[df["sku"] == sku_id].copy()

    last_row = sku_data.iloc[-1]

    input_data = pd.DataFrame([{
        "lag_1": last_row["weekly_sales"],
        "lag_2": sku_data["weekly_sales"].iloc[-2],
        "lag_4": sku_data["weekly_sales"].iloc[-4],
        "rolling_mean_4": sku_data["weekly_sales"].tail(4).mean(),
        "time_idx": last_row["time_idx"] + 1,
        "week_sin": last_row["week_sin"],
        "week_cos": last_row["week_cos"],
        "price": last_row["price"],
        "price_change": last_row["price_change"],
        "trend": last_row["trend"]
    }])

    pred = model.predict(input_data)[0]

    return {
        "week+1": float(pred[0]),
        "week+2": float(pred[1]),
        "week+3": float(pred[2])
    }