import joblib
import pandas as pd

model = joblib.load("models/xgb_model.pkl")

def predict(sample):
    return model.predict(sample)

if __name__ == "__main__":

    sample = pd.DataFrame([{
        "lag_1": 100,
        "lag_2": 90,
        "lag_4": 80,
        "rolling_mean_4": 85,
        "time_idx": 120,
        "price": 10.5,
        "trend": 1.2
    }])

    print(predict(sample))