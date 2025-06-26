from .EdgeDeviceInference import EdgeDeviceInference
import numpy as np
import os
import sys
from src.srv.LoadAndPrepareData import LoadAndPrepareData

if __name__ == "__main__":

    # Laden der Testdaten
    print("Loading test data...")

    # --- Konfiguration ---
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/household_power_consumption.txt"))
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models/model.tflite"))
    QUANTIZED_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models/model_quantized.tflite"))

    # Die Parameter müssen zu deinem Modell passen!
    window_size = 24
    horizon = 6
    resample_rule = "h"

    # --- Daten vorbereiten ---
    loader = LoadAndPrepareData(
        filepath=DATA_PATH,
        window_size=window_size,
        horizon=horizon,
        batch_size=1,
        resample_rule=resample_rule
    )

    _, _, test_ds = loader.get_datasets()

    # Einen Batch aus den Testdaten holen
    for batch_x, batch_y in test_ds.take(1):
        x_input = batch_x.numpy().astype(np.float32)  # TFLite erwartet float32 Eingaben
        y_true = batch_y.numpy()
        break

    # Inferenz mit dem nicht quantisierten TFLite-Modell
    print("Running inference with the non-quantized TFLite model...")
    
    inference = EdgeDeviceInference(MODEL_PATH)
    outputData = inference.run(x_input)
    print("TFLite Prediction", outputData)
    print("Ground truth (normalized):", y_true)
    print("Inference completed.")

    # Inferenz mit dem quantisierten TFLite-Modell
    print("Running inference with the quantized TFLite model...")

    inference_quantized = EdgeDeviceInference(QUANTIZED_MODEL_PATH)
    outputData_quantized = inference_quantized.run(x_input)
    print("Quantized TFLite Prediction", outputData_quantized)
    print("Ground truth (normalized):", y_true)
    print("Inference with quantized model completed.")