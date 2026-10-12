import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle

df = pd.read_csv("mixed_sensor_data.csv")

sc = MinMaxScaler()
scaled = sc.fit_transform(df[["vibration", "temperature", "pressure"]])

SEQ_LEN = 10
X = []
y = []

for i in range(len(scaled)-SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(df["label"].iloc[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

model = Sequential([
    LSTM(32, return_sequences=False, input_shape=(SEQ_LEN, 3)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32)

model.save("lstm_model.pkl")  # saved as pickle-like
print("Saved lstm_model.pkl")
