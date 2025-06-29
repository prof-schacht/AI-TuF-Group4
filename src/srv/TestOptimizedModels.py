from src.edgeDevice.EdgeDeviceInference import EdgeDeviceInference
import numpy as np
import pandas as pd
import os
from .LoadAndPrepareData import LoadAndPrepareData
import tensorflow as tf
import random

class TestOptimizedModels:
    """
    Klasse zum Testen und Validieren optimierter TFLite-Modelle.
    
    Diese Klasse bietet Methoden zur:
    - Validierung mit echten Testdaten
    - Bewertung der Modellgenauigkeit
    - Vergleich zwischen normalen und quantisierten Modellen
    """
    
    def __init__(self, training_data_path=None, window_size=24, horizon=6, resample_rule="h"):
        """
        Initialisiert die TestOptimizedModels-Klasse.
        
        Args:
            training_data_path (str, optional): Pfad zu den Trainingsdaten
            window_size (int): Fenstergröße für die Zeitreihen
            horizon (int): Vorhersagehorizont
            resample_rule (str): Regel für das Resampling
        """
        self.window_size = window_size
        self.horizon = horizon
        self.resample_rule = resample_rule
        
        # Standard Pfad zu den Trainingsdaten falls nicht angegeben
        if training_data_path is None:
            self.training_data_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../data/household_power_consumption.txt")
            )
        else:
            self.training_data_path = training_data_path
    
    def make_prediction_with_test_data(self, model_path, model_name="TFLite", show_output=True, num_test_samples=10):
        """
        Macht eine Vorhersage mit echten Testdaten aus dem Trainingsdatensatz.
        
        Args:
            model_path (str): Pfad zum TFLite-Modell
            model_name (str): Name des Modells für die Ausgabe
            show_output (bool): Ob die Ausgabe angezeigt werden soll
            num_test_samples (int): Anzahl der Testbeispiele für robustere Evaluation
            
        Returns:
            dict: Enthält prediction_times, feature_names, y_pred_original, y_true_original, timestamps
        """
        if show_output:
            print(f"\n{'='*80}")
            print(f"  {model_name} VALIDIERUNG MIT ECHTEN TESTDATEN")
            print(f"{'='*80}")
        
        # 1. Lade die Trainingsdaten und erstelle Testset
        if show_output:
            print("Loading training data and creating test set...")
        training_loader = LoadAndPrepareData(
            filepath=self.training_data_path,
            window_size=self.window_size,
            horizon=self.horizon,
            batch_size=1,
            resample_rule=self.resample_rule
        )
        
        # Lade Datasets
        train_ds, val_ds, test_ds = training_loader.get_datasets()
        scaler = training_loader.get_scaler()
        
        # 2. Sammle mehrere Testbeispiele für robustere Evaluation
        all_predictions = []
        all_ground_truth = []
        test_count = 0
        
        if show_output:
            print(f"Collecting {num_test_samples} test samples for robust evaluation...")
        
        for batch_x, batch_y in test_ds:
            if test_count >= num_test_samples:
                break
                
            x_input = batch_x.numpy().astype(np.float32)
            y_true = batch_y.numpy()
            
            # 3. Vorhersage mit TFLite-Modell
            inference = EdgeDeviceInference(model_path)
            y_pred = inference.run(x_input)
            
            all_predictions.append(y_pred)
            all_ground_truth.append(y_true)
            test_count += 1
            
        if show_output:
            print(f"Collected {test_count} test samples")
            
        # Kombiniere alle Vorhersagen
        y_pred_combined = np.concatenate(all_predictions, axis=0)
        y_true_combined = np.concatenate(all_ground_truth, axis=0)
        
        if show_output:
            print(f"Combined test data shape: Input {y_pred_combined.shape}, Target {y_true_combined.shape}")
        
        # 4. Rücktransformation auf Originalskala
        y_pred_flat = y_pred_combined.reshape(-1, y_pred_combined.shape[-1])
        y_pred_original = scaler.inverse_transform(y_pred_flat)
        
        y_true_flat = y_true_combined.reshape(-1, y_true_combined.shape[-1])
        y_true_original = scaler.inverse_transform(y_true_flat)
        
        # 4.1. Post-Processing: Korrigiere negative Energiewerte
        y_pred_original = self._post_process_predictions(y_pred_original, model_name, show_output)
        
        # 5. Erstelle realistische Zeitstempel für alle Vorhersagen
        start_prediction = pd.Timestamp("2023-06-01 12:00:00")
        prediction_times = []
        for sample_idx in range(test_count):
            for hour_idx in range(self.horizon):
                timestamp = start_prediction + pd.Timedelta(hours=sample_idx*24 + hour_idx)
                prediction_times.append(timestamp)
        
        # 6. Ausgabe der Ergebnisse (nur erste paar Beispiele anzeigen)
        if show_output:
            # Zeige nur die ersten 6 Zeitpunkte für bessere Übersicht
            display_limit = min(6, len(prediction_times))
            self._display_prediction_results(
                prediction_times[:display_limit], training_loader.df_resampled.columns, 
                y_pred_original[:display_limit], y_true_original[:display_limit], 
                f"{model_name} (erste {display_limit} von {len(prediction_times)} Vorhersagen)"
            )
        
        # Rückgabe der Ergebnisse
        return {
            'prediction_times': prediction_times,
            'feature_names': training_loader.df_resampled.columns,
            'y_pred_original': y_pred_original,
            'y_true_original': y_true_original,
            'model_name': model_name
        }
    
    def _display_prediction_results(self, prediction_times, feature_names, y_pred_original, y_true_original, model_name):
        """
        Zeigt die Vorhersageergebnisse formatiert an.
        
        Args:
            prediction_times (list): Liste der Zeitstempel
            feature_names (list): Liste der Feature-Namen
            y_pred_original (np.array): Vorhersagewerte
            y_true_original (np.array): Wahre Werte
            model_name (str): Name des Modells
        """
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
        value_width = 12
        
        # Header erstellen mit Farben - Vorhersage vs Ground Truth
        print(f"\n{BOLD}VORHERSAGE vs GROUND TRUTH{RESET}")
        header = f"{BOLD}{'Zeitpunkt':<{time_width}}"
        for i, feature in enumerate(feature_names):
            short_name = feature[:10] if len(feature) > 10 else feature
            color = COLORS[i % len(COLORS)]
            header += f"{color}{BOLD}{short_name + '_pred':>{value_width}}{short_name + '_true':>{value_width}}{RESET}"
        
        print(f"\n{header}")
        print("-" * (time_width + len(feature_names) * value_width * 2))
        
        # Datenzeilen mit Farben - Vorhersage und Ground Truth nebeneinander
        for i, time in enumerate(prediction_times):
            row = f"{time.strftime('%d.%m.%Y %H:%M'):<{time_width}}"
            for j, feature in enumerate(feature_names):
                pred_value = y_pred_original[i, j]
                true_value = y_true_original[i, j]
                
                color = COLORS[j % len(COLORS)]
                row += f"{color}{pred_value:>{value_width}.3f}{true_value:>{value_width}.3f}{RESET}"
            print(row)
        
        print("-" * (time_width + len(feature_names) * value_width * 2))
        
        # Berechne und zeige Fehlermetriken
        print(f"\n{BOLD}FEHLERMETRIKEN:{RESET}")
        mae_values = []
        rmse_values = []
        
        for j, feature in enumerate(feature_names):
            pred_vals = y_pred_original[:, j]
            true_vals = y_true_original[:, j]
            
            mae = np.mean(np.abs(pred_vals - true_vals))
            rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
            mae_values.append(mae)
            rmse_values.append(rmse)
            
            color = COLORS[j % len(COLORS)]
            print(f"{color}{feature[:20]:<20}: MAE = {mae:8.3f}, RMSE = {rmse:8.3f}{RESET}")
        
        overall_mae = np.mean(mae_values)
        overall_rmse = np.mean(rmse_values)
        print(f"\n{BOLD}GESAMT: MAE = {overall_mae:.3f}, RMSE = {overall_rmse:.3f}{RESET}")
        print(f"{model_name} Validierung abgeschlossen.\n")
    
    def make_prediction_with_keras_model(self, model_path, model_name="Keras", show_output=True, num_test_samples=10):
        """
        Macht eine Vorhersage mit dem ursprünglichen Keras-Modell.
        
        Args:
            model_path (str): Pfad zum Keras-Modell (.h5 Datei)
            model_name (str): Name des Modells für die Ausgabe
            show_output (bool): Ob die Ausgabe angezeigt werden soll
            num_test_samples (int): Anzahl der Testbeispiele für robustere Evaluation
            
        Returns:
            dict: Enthält prediction_times, feature_names, y_pred_original, y_true_original, timestamps
        """
        if show_output:
            print(f"\n{'='*80}")
            print(f"  {model_name} VALIDIERUNG MIT ECHTEN TESTDATEN")
            print(f"{'='*80}")
        
        # 1. Lade die Trainingsdaten und erstelle Testset
        if show_output:
            print("Loading training data and creating test set...")
        training_loader = LoadAndPrepareData(
            filepath=self.training_data_path,
            window_size=self.window_size,
            horizon=self.horizon,
            batch_size=1,
            resample_rule=self.resample_rule
        )
        
        # Lade Datasets
        train_ds, val_ds, test_ds = training_loader.get_datasets()
        scaler = training_loader.get_scaler()
        
        # 2. Sammle mehrere Testbeispiele für robustere Evaluation
        all_predictions = []
        all_ground_truth = []
        test_count = 0
        
        if show_output:
            print(f"Collecting {num_test_samples} test samples for robust evaluation...")
            print(f"Loading and predicting with {model_name}...")
        
        # Lade das Keras-Modell einmal
        keras_model = tf.keras.models.load_model(model_path)
        
        for batch_x, batch_y in test_ds:
            if test_count >= num_test_samples:
                break
                
            x_input = batch_x.numpy().astype(np.float32)
            y_true = batch_y.numpy()
            
            # 3. Vorhersage mit Keras-Modell
            y_pred = keras_model.predict(x_input, verbose=0)
            
            all_predictions.append(y_pred)
            all_ground_truth.append(y_true)
            test_count += 1
            
        if show_output:
            print(f"Collected {test_count} test samples")
            
        # Kombiniere alle Vorhersagen
        y_pred_combined = np.concatenate(all_predictions, axis=0)
        y_true_combined = np.concatenate(all_ground_truth, axis=0)
        
        if show_output:
            print(f"Combined test data shape: Input {y_pred_combined.shape}, Target {y_true_combined.shape}")
        
        # 4. Rücktransformation auf Originalskala
        y_pred_flat = y_pred_combined.reshape(-1, y_pred_combined.shape[-1])
        y_pred_original = scaler.inverse_transform(y_pred_flat)
        
        y_true_flat = y_true_combined.reshape(-1, y_true_combined.shape[-1])
        y_true_original = scaler.inverse_transform(y_true_flat)
        
        # 4.1. Post-Processing: Korrigiere negative Energiewerte
        y_pred_original = self._post_process_predictions(y_pred_original, model_name, show_output)
        
        # 5. Erstelle realistische Zeitstempel für alle Vorhersagen
        start_prediction = pd.Timestamp("2023-06-01 12:00:00")
        prediction_times = []
        for sample_idx in range(test_count):
            for hour_idx in range(self.horizon):
                timestamp = start_prediction + pd.Timedelta(hours=sample_idx*24 + hour_idx)
                prediction_times.append(timestamp)
        
        # 6. Ausgabe der Ergebnisse (nur erste paar Beispiele anzeigen)
        if show_output:
            # Zeige nur die ersten 6 Zeitpunkte für bessere Übersicht
            display_limit = min(6, len(prediction_times))
            self._display_prediction_results(
                prediction_times[:display_limit], training_loader.df_resampled.columns, 
                y_pred_original[:display_limit], y_true_original[:display_limit], 
                f"{model_name} (erste {display_limit} von {len(prediction_times)} Vorhersagen)"
            )
        
        # Rückgabe der Ergebnisse
        return {
            'prediction_times': prediction_times,
            'feature_names': training_loader.df_resampled.columns,
            'y_pred_original': y_pred_original,
            'y_true_original': y_true_original,
            'model_name': model_name
        }
    
    def evaluate_model_accuracy(self, model_results):
        """
        Bewertet die Genauigkeit eines Modells gegen Ground Truth.
        
        Args:
            model_results (dict): Ergebnisse des Modells mit y_true_original
            
        Returns:
            dict: Genauigkeitsstatistiken (mae, rmse, mape, assessment)
        """
        print(f"\n{'='*80}")
        print(f"  GENAUIGKEITSBEWERTUNG - {model_results['model_name']}")
        print(f"{'='*80}")
        
        prediction_times = model_results['prediction_times']
        feature_names = model_results['feature_names']
        y_pred = model_results['y_pred_original']
        y_true = model_results['y_true_original']
        
        # Sammle Fehlerstatistiken (ohne detaillierte Ausgabe)
        feature_errors = {feature: [] for feature in feature_names}
        all_absolute_errors = []
        all_relative_errors = []
        
        print(f"Datenbereich-Validierung:")
        print(f"Y_pred range: {y_pred.min():.6f} to {y_pred.max():.6f}")
        print(f"Y_true range: {y_true.min():.6f} to {y_true.max():.6f}")
        
        # Zähle problematische Werte
        near_zero_count = 0
        negative_count = 0
        
        for i, time in enumerate(prediction_times):
            for j, feature in enumerate(feature_names):
                pred_val = y_pred[i, j]
                true_val = y_true[i, j]
                abs_error = abs(pred_val - true_val)
                
                # Verbesserte MAPE-Berechnung mit Schwellenwert
                if abs(true_val) < 1e-6:  # Sehr kleine Werte
                    rel_error = 0  # Ignoriere bei sehr kleinen wahren Werten
                    near_zero_count += 1
                else:
                    rel_error = (abs_error / abs(true_val) * 100)
                    # Begrenze MAPE auf 200% für extreme Ausreißer
                    rel_error = min(rel_error, 200.0)
                
                if true_val < 0:
                    negative_count += 1
                
                feature_errors[feature].append(abs_error)
                all_absolute_errors.append(abs_error)
                all_relative_errors.append(rel_error)
        
        if near_zero_count > 0:
            print(f"⚠️  Warnung: {near_zero_count} Werte nahe Null gefunden (ignoriert für MAPE)")
        if negative_count > 0:
            print(f"⚠️  Warnung: {negative_count} negative wahre Werte gefunden")
        
        # Gesamtstatistiken
        print(f"\n{'='*100}")
        print("GESAMTSTATISTIKEN")
        print(f"{'='*100}")
        
        overall_mae = np.mean(all_absolute_errors)
        overall_rmse = np.sqrt(np.mean([e**2 for e in all_absolute_errors]))
        overall_mape = np.mean(all_relative_errors)  # Mean Absolute Percentage Error
        
        stats_data = [
            ["Mean Absolute Error (MAE)", f"{overall_mae:.3f}"],
            ["Root Mean Square Error (RMSE)", f"{overall_rmse:.3f}"],
            ["Mean Absolute Percentage Error (MAPE)", f"{overall_mape:.2f}%"],
            ["Maximum Absolute Error", f"{max(all_absolute_errors):.3f}"],
            ["Minimum Absolute Error", f"{min(all_absolute_errors):.3f}"],
            ["Standard Deviation", f"{np.std(all_absolute_errors):.3f}"],
        ]
        
        stats_header = f"{'Metrik':<35}{'Wert':<15}"
        print(f"\n{stats_header}")
        print("-" * len(stats_header))
        
        for stat_name, stat_value in stats_data:
            print(f"{stat_name:<35}{stat_value:<15}")
        
        # Feature-spezifische Statistiken
        self._display_feature_statistics(feature_names, feature_errors, prediction_times, y_true)
        
        # Gesamtbewertung
        self._overall_assessment(overall_mape, model_results['model_name'], overall_mae)
        
        return {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'mape': overall_mape
        }
    
    def _display_feature_statistics(self, feature_names, feature_errors, prediction_times, y_true):
        """
        Zeigt feature-spezifische Statistiken an.
        
        Args:
            feature_names (list): Liste der Feature-Namen
            feature_errors (dict): Dictionary mit Fehlern pro Feature
            prediction_times (list): Liste der Zeitstempel
            y_true (np.array): Wahre Werte
        """
        print(f"\n{'='*100}")
        print("FEATURE-SPEZIFISCHE FEHLERSTATISTIKEN")
        print(f"{'='*100}")
        
        feature_stats_header = f"{'Feature':<25}{'MAE':<12}{'RMSE':<12}{'Max Error':<12}"
        print(f"\n{feature_stats_header}")
        print("-" * len(feature_stats_header))
        
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        for feature in feature_names:
            errors = feature_errors[feature]
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            max_error = max(errors)
            
            feature_short = feature[:23] if len(feature) > 23 else feature
            row = f"{feature_short:<25}{mae:<12.3f}{rmse:<12.3f}{max_error:<12.3f}"
            print(row)
    
    def _overall_assessment(self, overall_mape, model_name, overall_mae):
        """
        Erstellt eine Gesamtbewertung des Modells.
        
        Args:
            overall_mape (float): Durchschnittlicher prozentualer Fehler
            model_name (str): Name des Modells
            overall_mae (float): Durchschnittlicher absoluter Fehler
        """
        print(f"\n{'='*100}")
        print("MODELLBEWERTUNG")
        print(f"{'='*100}")
        
        print(f"\nDas {model_name} zeigt einen durchschnittlichen Fehler von {overall_mape:.1f}%")
        print(f"und eine mittlere absolute Abweichung von {overall_mae:.3f}.")
        print(f"\n{'='*100}\n")
    
    def compare_model_accuracies(self, keras_accuracy, normal_accuracy, quantized_accuracy, keras_results, normal_results, quantized_results):
        """
        Vergleicht die Genauigkeitsbewertungen aller drei Modelle.
        
        Args:
            keras_accuracy (dict): Genauigkeitsstatistiken des Keras-Modells
            normal_accuracy (dict): Genauigkeitsstatistiken des normalen Modells
            quantized_accuracy (dict): Genauigkeitsstatistiken des quantisierten Modells
            keras_results (dict): Vollständige Ergebnisse des Keras-Modells
            normal_results (dict): Vollständige Ergebnisse des normalen Modells
            quantized_results (dict): Vollständige Ergebnisse des quantisierten Modells
        """
        # Vergleichstabelle der Hauptmetriken
        print(f"\n{'='*80}")
        print("GENAUIGKEITSVERGLEICH ALLER DREI MODELLE")
        print(f"{'='*80}")
        
        # 3-Modell-Vergleich
        comparison_header = f"{'Metrik':<12}{'Keras':<12}{'Normal TFLite':<15}{'Quantized TFLite':<17}{'Bester':<10}"
        print(f"\n{comparison_header}")
        print("-" * len(comparison_header))
        
        RESET = '\033[0m'
        BOLD = '\033[1m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        
        # MAE Vergleich
        mae_values = [keras_accuracy['mae'], normal_accuracy['mae'], quantized_accuracy['mae']]
        mae_best_idx = mae_values.index(min(mae_values))
        mae_names = ["Keras", "Normal", "Quantized"]
        mae_winner = mae_names[mae_best_idx]
        
        # RMSE Vergleich  
        rmse_values = [keras_accuracy['rmse'], normal_accuracy['rmse'], quantized_accuracy['rmse']]
        rmse_best_idx = rmse_values.index(min(rmse_values))
        rmse_winner = mae_names[rmse_best_idx]
        
        # MAPE Vergleich
        mape_values = [keras_accuracy['mape'], normal_accuracy['mape'], quantized_accuracy['mape']]
        mape_best_idx = mape_values.index(min(mape_values))
        mape_winner = mae_names[mape_best_idx]
        
        comparison_data = [
            ["MAE", f"{keras_accuracy['mae']:.3f}", f"{normal_accuracy['mae']:.3f}", f"{quantized_accuracy['mae']:.3f}", f"{GREEN}{mae_winner}{RESET}"],
            ["RMSE", f"{keras_accuracy['rmse']:.3f}", f"{normal_accuracy['rmse']:.3f}", f"{quantized_accuracy['rmse']:.3f}", f"{GREEN}{rmse_winner}{RESET}"],
            ["MAPE (%)", f"{keras_accuracy['mape']:.2f}", f"{normal_accuracy['mape']:.2f}", f"{quantized_accuracy['mape']:.2f}", f"{GREEN}{mape_winner}{RESET}"],
        ]
        
        for row in comparison_data:
            print(f"{row[0]:<12}{row[1]:<12}{row[2]:<15}{row[3]:<17}{row[4]:<10}")
            
        # Zusammenfassung der Modellkonvertierung
        print(f"\n{BOLD}ZUSAMMENFASSUNG DER MODELLKONVERTIERUNG:{RESET}")
        print(f"Original Keras Modell → TFLite Normal → TFLite Quantized")
        
        keras_to_normal_mae = normal_accuracy['mae'] - keras_accuracy['mae']
        normal_to_quantized_mae = quantized_accuracy['mae'] - normal_accuracy['mae']
        
        print(f"MAE Änderung: Keras→Normal: {keras_to_normal_mae:+.3f}, Normal→Quantized: {normal_to_quantized_mae:+.3f}")
    
    def run_full_validation(self, normal_model_path, quantized_model_path, keras_model_path):
        """
        Führt eine vollständige Validierung aller drei Modelle durch.
        
        Args:
            normal_model_path (str): Pfad zum normalen TFLite-Modell
            quantized_model_path (str): Pfad zum quantisierten TFLite-Modell
            keras_model_path (str): Pfad zum ursprünglichen Keras-Modell
            
        Returns:
            tuple: (keras_accuracy, normal_accuracy, quantized_accuracy) - Genauigkeitsstatistiken aller Modelle
        """
        print("\n" + "="*90)
        print("VOLLSTÄNDIGE MODELLVALIDIERUNG MIT ECHTEN TESTDATEN")
        print("="*90)
        print("Hier wird die tatsächliche Genauigkeit aller drei Modelle gegen bekannte Ground Truth gemessen")
        print("HINWEIS: Für robustere Ergebnisse werden mehrere Testbeispiele verwendet (Standard: 10)")
        print("="*90)
        
        # Verwende den gleichen Random-Seed für alle Modelle
        # um identische Testdaten zu gewährleisten
        random.seed(42)
        np.random.seed(42)
        
        print("🔒 Verwende festen Random-Seed (42) für konsistente Testdaten zwischen allen Modellen")
        
        # Validierung mit echten Testdaten für alle drei Modelle
        keras_test_results = self.make_prediction_with_keras_model(
            keras_model_path, "URSPRÜNGLICHES KERAS MODELL", show_output=True
        )
        
        # Reset Random-Seed für identische Testsamples
        random.seed(42)
        np.random.seed(42)
        
        normal_test_results = self.make_prediction_with_test_data(
            normal_model_path, "NORMALES TFLite MODELL", show_output=True
        )
        
        # Reset Random-Seed für identische Testsamples
        random.seed(42)
        np.random.seed(42)
        
        quantized_test_results = self.make_prediction_with_test_data(
            quantized_model_path, "QUANTISIERTES TFLite MODELL", show_output=True
        )
        
        if keras_test_results and normal_test_results and quantized_test_results:
            # Bewerte jedes Modell einzeln gegen Ground Truth
            print("\n" + "="*90)
            print("EINZELBEWERTUNG ALLER MODELLE GEGEN GROUND TRUTH")
            print("="*90)
            
            keras_accuracy = self.evaluate_model_accuracy(keras_test_results)
            normal_accuracy = self.evaluate_model_accuracy(normal_test_results)
            quantized_accuracy = self.evaluate_model_accuracy(quantized_test_results)
            
            # Vergleiche die Genauigkeiten aller drei Modelle
            self.compare_model_accuracies(
                keras_accuracy, normal_accuracy, quantized_accuracy,
                keras_test_results, normal_test_results, quantized_test_results
            )
            
            return keras_accuracy, normal_accuracy, quantized_accuracy
        else:
            print("Konnte nicht alle drei Modelle mit Testdaten validieren.")
            return None, None, None
    
    def _post_process_predictions(self, y_pred_original, model_name, show_output=True):
        """
        Post-Processing der Vorhersagen um negative Werte für Energiedaten zu korrigieren.
        
        Args:
            y_pred_original (np.array): Ursprüngliche Vorhersagen
            model_name (str): Name des Modells
            show_output (bool): Ob Ausgaben angezeigt werden sollen
            
        Returns:
            np.array: Korrigierte Vorhersagen
        """
        negative_count = np.sum(y_pred_original < 0)
        
        if negative_count > 0 and show_output:
            print(f"🔧 Post-Processing für {model_name}: {negative_count} negative Werte auf 0 gesetzt")
        
        # Setze negative Energiewerte auf 0 (physikalisch sinnvoll)
        y_pred_corrected = np.maximum(y_pred_original, 0)
        
        return y_pred_corrected
