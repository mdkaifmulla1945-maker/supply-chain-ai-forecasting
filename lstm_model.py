import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales.csv")
df = df[df["sku"] == 1][["weekly_sales"]].values

# =========================
# SCALE DATA
# =========================
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# =========================
# CREATE SEQUENCES
# =========================
def create_seq(data, step=5):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step])
        y.append(data[i+step])
    return np.array(X), np.array(y)

X, y = create_seq(df_scaled, step=5)

# reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(5,1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# =========================
# TRAIN
# =========================
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# =========================
# PREDICTION (3 STEP FORECAST)
# =========================
input_seq = X[-1]

preds = []

for _ in range(3):
    pred = model.predict(input_seq.reshape(1,5,1))[0][0]
    preds.append(pred)

    input_seq = np.append(input_seq[1:], [[pred]], axis=0)

# inverse scale
preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))

print("\nLSTM Forecast:", preds.flatten())