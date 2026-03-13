## Stock_prediction_CNN-BiLSTM

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    df = pd.read_csv(file_path)
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain the columns: {required_columns}")

    # Mixed format handling
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data['Volume'] = data['Volume'].replace(0, np.nan)
    data = data.ffill().dropna()
    return df, data

def scale_data(data):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(scaled_data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])
        y.append(scaled_data[i, 3])  # Close price
    return np.array(X), np.array(y)

from keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Bidirectional

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Bidirectional(LSTM(64, return_sequences=True)),   # Stacked LSTM
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def inverse_transform_close(scaler, data):
    dummy = np.zeros((len(data), 5))
    dummy[:, 3] = data.reshape(-1)
    return scaler.inverse_transform(dummy)[:, 3]

def forecast_next_30_days(model, last_input, scaler, seq_len=60):
    forecasted = []
    forecast_input = last_input
    for _ in range(30):
        pred = model.predict(forecast_input.reshape(1, seq_len, 5), verbose=0)
        new_step = np.append(forecast_input[1:], [[
            *forecast_input[-1][:3], pred[0][0], forecast_input[-1][-1]
        ]], axis=0)
        forecast_input = new_step
        forecasted.append(pred[0][0])

    dummy = np.zeros((30, 5))
    dummy[:, 3] = forecasted
    return scaler.inverse_transform(dummy)[:, 3]

def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === MAIN ===

# Load Stock data
stock_file = "data/TATAMOTORS.NS.csv"

# Step 1: Load and preprocess
df, data = load_and_preprocess_data(stock_file)
scaled_data, scaler = scale_data(data)

# Step 2: Sequence generation
seq_len = 60
X, y = create_sequences(scaled_data, seq_len)
X_train, y_train = X[:-60], y[:-60]
X_test, y_test = X[-60:], y[-60:]

# Step 3: Build and train model
model = build_model((seq_len, 5))
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=110,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Step 4: Plot loss
plot_loss(history)

# Step 5: Predict and evaluate
pred_scaled = model.predict(X_test)
pred_close = inverse_transform_close(scaler, pred_scaled)
actual_close = inverse_transform_close(scaler, y_test)
rmse = sqrt(mean_squared_error(actual_close, pred_close))
print("RMSE:", rmse)

# Step 6: Forecast next 30 business days
forecast_prices = forecast_next_30_days(model, X[-1], scaler)

# Step 7: Save forecast
last_date = df.index[-1]
forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': forecast_prices})
forecast_df.to_csv('forecast_next_30_days.csv', index=False)
print(forecast_df.head())


# Accuracy Score Calculation
actual_direction = np.sign(np.diff(actual_close))
pred_direction   = np.sign(np.diff(pred_close))
Accuracy = accuracy_score(actual_direction, pred_direction)
print(f"Accuracy: {Accuracy*100:.2f}%")

# Map -1, 0, 1 to 0, 1, 2
actual_lbls = actual_direction + 1
pred_lbls   = pred_direction + 1

classes = ['Down', 'No Change', 'Up']

present_labels = sorted(np.unique(np.concatenate([actual_lbls, pred_lbls])))

print(classification_report(
    actual_lbls, pred_lbls,
    labels=present_labels,
    target_names=[classes[int(i)] for i in present_labels],
    digits=4
))
