import os
import numpy as np
from tensorflow.keras.models import load_model
from LoadAndPrepareData import LoadAndPrepareData

import argparse

# --- Configuration ---
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/household_power_consumption.txt"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models/best_model.keras"))

parser = argparse.ArgumentParser(description="Test model prediction with correct window size and preprocessing.")
parser.add_argument('--test_mode', action='store_true', help='Use test mode parameters (match test-mode-trained model)')
args = parser.parse_args()

if args.test_mode:
    window_size = 4
    horizon = 2
    resample_rule = "min"
    print(f"[TestModelPrediction] test_mode: window_size={window_size}, horizon={horizon}, resample_rule={resample_rule}")
else:
    window_size = 24
    horizon = 6
    resample_rule = "h"
    print(f"[TestModelPrediction] full mode: window_size={window_size}, horizon={horizon}, resample_rule={resample_rule}")

# --- Data Preparation (use the same parameters as in training) ---
loader = LoadAndPrepareData(
    filepath=DATA_PATH,
    window_size=window_size,
    horizon=horizon,
    batch_size=1,        # For single-step prediction
    resample_rule=resample_rule
)

train_ds, val_ds, test_ds = loader.get_datasets()
scaler = loader.get_scaler()

# --- Get a single batch from the test set ---
for batch_x, batch_y in test_ds.take(1):
    # batch_x shape: (batch_size, window_size, num_features)
    # batch_y shape: (batch_size, horizon, num_features)
    x_input = batch_x.numpy()
    y_true = batch_y.numpy()
    break
else:
    raise RuntimeError("No test data available for prediction!")

# --- Load the trained model ---
model = load_model(MODEL_PATH)

# --- Make prediction ---
y_pred = model.predict(x_input)
print("Prediction (normalized):")
print(y_pred)
print("Ground truth (normalized):")
print(y_true)

# --- Inverse transform to original scale ---
# Reshape for scaler: (horizon, num_features)
y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
y_true_flat = y_true.reshape(-1, y_true.shape[-1])
y_pred_original = scaler.inverse_transform(y_pred_flat)
y_true_original = scaler.inverse_transform(y_true_flat)

print("\nPrediction (original scale):")
print(y_pred_original)
print("Ground truth (original scale):")
print(y_true_original)
