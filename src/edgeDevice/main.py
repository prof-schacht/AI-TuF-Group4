from .EdgeDeviceInference import EdgeDeviceInference
from src.srv.TestOptimizedModels import TestOptimizedModels
import numpy as np
import pandas as pd
import os
import sys
from src.srv.LoadAndPrepareData import LoadAndPrepareData

def make_prediction_with_model(model_path, model_name="TFLite", show_output=True):
    """
    Macht eine Vorhersage für 01.06.2025 12:00-18:00 mit einem spezifischen Modell.
    
    Args:
        model_path (str): Pfad zum TFLite-Modell
        model_name (str): Name des Modells für die Ausgabe
        show_output (bool): Ob die Ausgabe angezeigt werden soll
        
    Returns:
        dict: Enthält prediction_times, feature_names, y_pred_original
    """
    if show_output:
        print(f"\n{'='*80}")
        print(f"  {model_name} VORHERSAGE - 01.06.2025 12:00-18:00")
        print(f"{'='*80}")
    
    # Pfade
    EXAMPLE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/prediction_input_example.csv"))
    TRAINING_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/household_power_consumption.txt"))
    
    window_size = 24
    horizon = 6
    resample_rule = "h"
    
    # 1. Lade den Scaler aus den ursprünglichen Trainingsdaten
    if show_output:
        print("Loading scaler from training data...")
    training_loader = LoadAndPrepareData(
        filepath=TRAINING_DATA_PATH,
        window_size=window_size,
        horizon=horizon,
        batch_size=1,
        resample_rule=resample_rule
    )
    # Scaler wird durch get_datasets() initialisiert
    _, _, _ = training_loader.get_datasets()
    scaler = training_loader.get_scaler()
    
    # 2. Prüfe ob Beispieldaten existieren
    if not os.path.exists(EXAMPLE_DATA_PATH):
        if show_output:
            print(f"Beispieldaten nicht gefunden: {EXAMPLE_DATA_PATH}")
        return None
    
    # 3. Lade und verarbeite die Beispieldaten mit LoadAndPrepareData
    if show_output:
        print("Loading and preprocessing example data...")
        
    example_loader = LoadAndPrepareData(
        filepath=EXAMPLE_DATA_PATH,
        window_size=window_size,
        horizon=1,  # Wir brauchen nur die Input-Daten, nicht die Targets
        batch_size=1,
        resample_rule=resample_rule
    )
    
    # Lade nur die Daten, ohne sie zu splitten (wir wollen alle Daten)
    example_loader.load_and_clean()
    example_loader.resample()
    
    # Verwende den trainierten Scaler anstatt einen neuen zu fitten
    df_resampled = example_loader.df_resampled
    if show_output:
        print(f"Example data shape after resampling: {df_resampled.shape}")
        print(f"Example data time range: {df_resampled.index.min()} to {df_resampled.index.max()}")
    
    # 4. Skalierung mit dem trainierten Scaler
    data_scaled = scaler.transform(df_resampled.values)
    
    # 5. Prüfe ob wir genug Daten haben
    if len(data_scaled) != window_size:
        if show_output:
            print(f"WARNUNG: Benötige genau {window_size} Zeitschritte, aber habe {len(data_scaled)}")
        if len(data_scaled) < window_size:
            if show_output:
                print("Nicht genügend Daten für Vorhersage!")
            return None
        # Falls zu viele Daten, nehme die letzten window_size
        data_scaled = data_scaled[-window_size:]
        if show_output:
            print(f"Verwende die letzten {window_size} Zeitschritte.")
    
    # 6. In die richtige Shape bringen: (1, window_size, num_features)
    x_input = data_scaled.reshape(1, window_size, -1).astype(np.float32)
    
    # 7. Vorhersage mit TFLite-Modell
    if show_output:
        print(f"Making prediction with {model_name}...")
    inference = EdgeDeviceInference(model_path)
    y_pred = inference.run(x_input)
    
    if show_output:
        print("Vorhersage (normalisiert):", y_pred)
    
    # 8. Rücktransformation auf Originalskala
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    y_pred_original = scaler.inverse_transform(y_pred_flat)
    
    # 9. Erstelle Zeitstempel für die Vorhersage
    start_prediction = pd.Timestamp("2025-06-01 12:00:00")
    prediction_times = [start_prediction + pd.Timedelta(hours=i) for i in range(horizon)]
    
    # 10. Ausgabe der Ergebnisse mit Farben (nur wenn gewünscht)
    if show_output:
        feature_names = df_resampled.columns
        
        # ANSI-Farbcodes für verschiedene Farben
        COLORS = [
            '\033[91m',  # Rot
            '\033[92m',  # Grün
            '\033[93m',  # Gelb
            '\033[94m',  # Blau
            '\033[95m',  # Magenta
            '\033[96m',  # Cyan
            '\033[97m',  # Weiß
        ]
        RESET = '\033[0m'    # Farbe zurücksetzen
        BOLD = '\033[1m'     # Fett
        
        # Definiere einheitliche Spaltenbreiten
        time_width = 18
        value_width = 15
        
        # Header erstellen mit Farben
        header = f"{BOLD}{'Zeitpunkt':<{time_width}}{RESET}"
        for i, feature in enumerate(feature_names):
            # Kürze lange Feature-Namen für bessere Darstellung
            short_name = feature[:12] if len(feature) > 12 else feature
            color = COLORS[i % len(COLORS)]  # Zyklisch durch Farben
            header += f"{color}{BOLD}{short_name:>{value_width}}{RESET}"
        
        print(f"\n{header}")
        print("-" * (time_width + len(feature_names) * value_width))
        
        # Datenzeilen mit Farben
        for i, time in enumerate(prediction_times):
            row = f"{time.strftime('%d.%m.%Y %H:%M'):<{time_width}}"
            for j, feature in enumerate(feature_names):
                value = y_pred_original[i, j]
                # Formatiere Werte einheitlich mit 3 Dezimalstellen
                formatted_value = f"{value:.3f}"
                color = COLORS[j % len(COLORS)]  # Gleiche Farbe wie Header
                row += f"{color}{formatted_value:>{value_width}}{RESET}"
            print(row)
        
        print("-" * (time_width + len(feature_names) * value_width))
        print(f"{model_name} Vorhersage erfolgreich abgeschlossen.\n")
    
    # Rückgabe der Ergebnisse
    return {
        'prediction_times': prediction_times,
        'feature_names': df_resampled.columns,
        'y_pred_original': y_pred_original,
        'model_name': model_name
    }

if __name__ == "__main__":
    
    # Pfade zu den Modellen
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models/model.tflite"))
    QUANTIZED_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models/model_quantized.tflite"))
    
    # VORHERSAGE MIT BEISPIELDATEN
    print("\n" + "="*90)
    print("VORHERSAGE MIT BEISPIELDATEN")
    print("="*90)
    print("Hier wird eine Vorhersage für zukünftige Zeitpunkte gemacht (01.06.2025 12:00-18:00)")
    
    # Vorhersage mit Beispieldaten (mit Ausgabe)
    normal_results = make_prediction_with_model(MODEL_PATH, "NORMALES TFLite MODELL", show_output=True)
    quantized_results = make_prediction_with_model(QUANTIZED_MODEL_PATH, "QUANTISIERTES TFLite MODELL", show_output=True)
    