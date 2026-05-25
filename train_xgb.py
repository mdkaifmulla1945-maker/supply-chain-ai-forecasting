import pandas as pd
import joblib
import xgboost as xgb
from preprocess import create_features
import os

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales.csv")
df = create_features(df)

print("DATA SHAPE:", df.shape)

# =========================
# FEATURES
# =========================
features = [
    "lag_1", "lag_2", "lag_4",
    "rolling_mean_4",
    "time_idx",
    "week_sin", "week_cos",
    "price", "price_change",
    "trend"
]

X = df[features]
y = df["weekly_sales"]

# =========================
# CREATE MODEL
# =========================
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X, y)

print("XGBoost training complete")

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgb_model.pkl")

print("Model saved at models/xgb_model.pkl")