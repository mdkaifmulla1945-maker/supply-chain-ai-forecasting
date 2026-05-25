import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("data/sales.csv")

def arima_forecast(df, sku_id):

    data = df[df["sku"] == sku_id]["weekly_sales"]

    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()

    return model_fit.forecast(steps=3).tolist()


if __name__ == "__main__":
    print(arima_forecast(df, 1))