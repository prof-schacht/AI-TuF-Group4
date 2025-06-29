import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----- Pfad anpassen für srv-Importe -----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.srv.LoadAndPrepareData import LoadAndPrepareData

# ----- Cache leeren -----
st.cache_resource.clear()

# ----- Konfiguration -----
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/household_power_consumption.txt"))
KERAS_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/srv/models/best_model.keras"))
REEXPORTED_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/srv/models/best_model_reexport.keras"))
TFLITE_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/edgeDevice/models/model.tflite"))
WINDOW_SIZE = 24
DEFAULT_HORIZON = 6

# ----- Reexportiere Modell bei Bedarf -----
if not os.path.exists(REEXPORTED_MODEL_PATH):
    try:
        with st.spinner("Reexportiere Keras-Modell ohne time_major..."):
            original_model = load_model(KERAS_MODEL_PATH, compile=False)
            original_model.save(REEXPORTED_MODEL_PATH)
            st.success("✅ Modell erfolgreich reexportiert.")
    except Exception as e:
        st.error(f"❌ Fehler beim Reexportieren des Modells: {e}")
else:
    st.info("ℹ️ Reexportiertes Modell bereits vorhanden.")

# ----- Streamlit UI Setup -----
st.set_page_config(page_title="Power Forecast Dashboard", layout="wide")
st.title("\U0001F32C️ Household Power Consumption Forecast")

# ----- Sidebar -----
st.sidebar.header("Vorhersage-Steuerung")
forecast_horizon = st.sidebar.slider("Forecast Horizon (Stunden)", 1, 24, DEFAULT_HORIZON)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance (IG)", value=True)

# ----- Verifikation des Modellpfads -----
if not os.path.exists(TFLITE_MODEL_PATH):
    st.error(f"❌ TFLite Model nicht gefunden: {TFLITE_MODEL_PATH}")
else:
    st.success(f"✅ TFLite Model gefunden: {TFLITE_MODEL_PATH}")

# ----- Daten vorbereiten (einmalig) -----
@st.cache_resource
def prepare_data():
    loader = LoadAndPrepareData(
        filepath=DATA_PATH,
        window_size=WINDOW_SIZE,
        horizon=forecast_horizon,
        batch_size=1,
        resample_rule="h"
    )
    train_ds, val_ds, test_ds = loader.get_datasets()
    scaler = loader.get_scaler()
    for batch_x, batch_y in test_ds.take(1):
        x_input = batch_x.numpy()
        y_true = batch_y.numpy()
        break
    return x_input, y_true, scaler

# ----- .tflite-Vorhersagefunktion -----
def predict_tflite(input_tensor):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

# ----- Integrated Gradients (einfach) -----
def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros_like(input_tensor)
    input_tensor = tf.cast(input_tensor, tf.float32)
    baseline = tf.cast(baseline, tf.float32)

    interpolated = [(baseline + (float(i)/steps)*(input_tensor - baseline)) for i in range(steps+1)]
    interpolated = tf.convert_to_tensor(np.concatenate(interpolated, axis=0))

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)  # Explicitly set training=False

    grads = tape.gradient(preds, interpolated)
    avg_grads = tf.reduce_mean(grads, axis=0).numpy()
    attributions = (input_tensor - baseline) * avg_grads
    return attributions.squeeze()  # shape: (window_size, num_features)

# ----- Hauptlogik -----
with st.spinner("Lade Daten und Modelle..."):
    x_input, y_true, scaler = prepare_data()
    y_pred = predict_tflite(x_input)

    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])

    y_pred_orig = scaler.inverse_transform(y_pred_flat)
    y_true_orig = scaler.inverse_transform(y_true_flat)

# ----- Plot Forecast -----
st.subheader(f"Forecast: Nächste {forecast_horizon} Stunden")
df_plot = pd.DataFrame({
    "Stunde": list(range(1, forecast_horizon+1)),
    "Prediction": y_pred_orig[:, 0],
    "Ground Truth": y_true_orig[:, 0]
})
fig = px.line(df_plot, x="Stunde", y=["Prediction", "Ground Truth"], markers=True)
fig.update_traces(selector=dict(name="Prediction"), line=dict(dash="dash", color="blue"))
fig.update_traces(selector=dict(name="Ground Truth"), line=dict(color="red"))
st.plotly_chart(fig, use_container_width=True)

# ----- Metriken -----
st.subheader("\U0001F4CA Modellgüte")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mean_absolute_error(y_true_orig, y_pred_orig):.2f} kW")
rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
col2.metric("RMSE", f"{rmse:.2f} kW")
col3.metric("R²", f"{r2_score(y_true_orig, y_pred_orig):.2f}")

# ----- Feature Importance Heatmap -----
if show_feature_importance:
    st.subheader("\U0001F50D Feature Importance via Integrated Gradients")
    with st.spinner("Berechne IG-Attribution..."):
        keras_model = load_model(REEXPORTED_MODEL_PATH, compile=False)
        ig = integrated_gradients(keras_model, x_input)
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(ig.T, cmap="viridis", aspect="auto")
        ax.set_ylabel("Feature")
        ax.set_xlabel("Zeitschritte (vergangen)")
        ax.set_title("Integrated Gradients")
        st.pyplot(fig)

# ----- Footer -----
st.markdown("---")
st.markdown("AI-TuF Group 4 | Household Power Forecast Dashboard")
