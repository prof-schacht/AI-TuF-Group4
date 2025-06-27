"""
TestOptimizedModels - Klasse zum Testen und Validieren optimierter Modelle

Diese Klasse enthält alle Funktionen zur Validierung von TFLite-Modellen
gegen echte Testdaten und zum Vergleich zwischen normalem und quantisiertem Modell.
"""

from src.edgeDevice.EdgeDeviceInference import EdgeDeviceInference
import numpy as np
import pandas as pd
import os
from .LoadAndPrepareData import LoadAndPrepareData


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
    
    def make_prediction_with_test_data(self, model_path, model_name="TFLite", show_output=True):
        """
        Macht eine Vorhersage mit echten Testdaten aus dem Trainingsdatensatz.
        
        Args:
            model_path (str): Pfad zum TFLite-Modell
            model_name (str): Name des Modells für die Ausgabe
            show_output (bool): Ob die Ausgabe angezeigt werden soll
            
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
        
        # 2. Nimm ein Beispiel aus den Testdaten
        for batch_x, batch_y in test_ds.take(1):
            x_input = batch_x.numpy().astype(np.float32)
            y_true = batch_y.numpy()
            break
        
        if show_output:
            print(f"Test data shape: Input {x_input.shape}, Target {y_true.shape}")
        
        # 3. Vorhersage mit TFLite-Modell
        if show_output:
            print(f"Making prediction with {model_name}...")
        inference = EdgeDeviceInference(model_path)
        y_pred = inference.run(x_input)
        
        # 4. Rücktransformation auf Originalskala
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        y_pred_original = scaler.inverse_transform(y_pred_flat)
        
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_true_original = scaler.inverse_transform(y_true_flat)
        
        # 5. Erstelle realistische Zeitstempel (verwende aktuelles Datum)
        start_prediction = pd.Timestamp("2023-06-01 12:00:00")  # Beispiel-Zeitstempel aus Testdaten
        prediction_times = [start_prediction + pd.Timedelta(hours=i) for i in range(self.horizon)]
        
        # 6. Ausgabe der Ergebnisse mit Farben (nur wenn gewünscht)
        if show_output:
            self._display_prediction_results(
                prediction_times, training_loader.df_resampled.columns, 
                y_pred_original, y_true_original, model_name
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
        
        # Detaillierte Vergleichstabelle
        print(f"\n{'='*100}")
        print("VORHERSAGE vs GROUND TRUTH - DETAILVERGLEICH")
        print(f"{'='*100}")
        
        header = f"{'Zeitpunkt':<18}{'Feature':<20}{'Vorhersage':<15}{'Ground Truth':<15}{'Abs. Fehler':<12}{'Rel. Fehler %':<12}"
        print(f"\n{header}")
        print("-" * len(header))
        
        # Sammle Fehlerstatistiken pro Feature
        feature_errors = {feature: [] for feature in feature_names}
        all_absolute_errors = []
        all_relative_errors = []
        
        for i, time in enumerate(prediction_times):
            for j, feature in enumerate(feature_names):
                pred_val = y_pred[i, j]
                true_val = y_true[i, j]
                abs_error = abs(pred_val - true_val)
                rel_error = (abs_error / abs(true_val) * 100) if true_val != 0 else 0
                
                feature_errors[feature].append(abs_error)
                all_absolute_errors.append(abs_error)
                all_relative_errors.append(rel_error)
                
                # Formatiere Werte
                time_str = time.strftime('%d.%m %H:%M') if j == 0 else ""
                feature_short = feature[:18] if len(feature) > 18 else feature
                
                row = f"{time_str:<18}{feature_short:<20}{pred_val:<15.3f}{true_val:<15.3f}{abs_error:<12.3f}{rel_error:<12.2f}%"
                print(row)
            
            if i < len(prediction_times) - 1:
                print("-" * len(header))
        
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
        
        feature_stats_header = f"{'Feature':<25}{'MAE':<12}{'RMSE':<12}{'Max Error':<12}{'Bewertung':<15}"
        print(f"\n{feature_stats_header}")
        print("-" * len(feature_stats_header))
        
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        for feature in feature_names:
            errors = feature_errors[feature]
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            max_error = max(errors)
            
            # Bewertung basierend auf relativem Fehler
            feature_values = [abs(y_true[i, j]) for i in range(len(prediction_times)) for j, f in enumerate(feature_names) if f == feature]
            avg_value = np.mean(feature_values)
            relative_mae = (mae / avg_value * 100) if avg_value > 0 else 0
            
            if relative_mae < 5:
                assessment = "Sehr gut"
                color = '\033[92m'  # Grün
            elif relative_mae < 10:
                assessment = "Gut"
                color = '\033[93m'  # Gelb
            elif relative_mae < 20:
                assessment = "Befriedigend"
                color = '\033[94m'  # Blau
            else:
                assessment = "Verbesserungsbedarf"
                color = '\033[91m'  # Rot
            
            feature_short = feature[:23] if len(feature) > 23 else feature
            row = f"{feature_short:<25}{mae:<12.3f}{rmse:<12.3f}{max_error:<12.3f}{color}{assessment:<15}{RESET}"
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
    
    def compare_model_accuracies(self, normal_accuracy, quantized_accuracy, normal_results, quantized_results):
        """
        Vergleicht die Genauigkeitsbewertungen beider Modelle.
        
        Args:
            normal_accuracy (dict): Genauigkeitsstatistiken des normalen Modells
            quantized_accuracy (dict): Genauigkeitsstatistiken des quantisierten Modells
            normal_results (dict): Vollständige Ergebnisse des normalen Modells
            quantized_results (dict): Vollständige Ergebnisse des quantisierten Modells
        """
        # Vergleichstabelle der Hauptmetriken
        print(f"\n{'='*80}")
        print("GENAUIGKEITSVERGLEICH")
        print(f"{'='*80}")
        
        comparison_header = f"{'Metrik':<25}{'Normal':<15}{'Quantisiert':<15}{'Unterschied':<15}{'Gewinner':<10}"
        print(f"\n{comparison_header}")
        print("-" * len(comparison_header))
        
        RESET = '\033[0m'
        BOLD = '\033[1m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        
        # MAE Vergleich
        mae_diff = quantized_accuracy['mae'] - normal_accuracy['mae']
        mae_winner = "Normal" if mae_diff > 0 else "Quantisiert"
        mae_color = GREEN if mae_winner == "Normal" else RED
        
        # RMSE Vergleich
        rmse_diff = quantized_accuracy['rmse'] - normal_accuracy['rmse']
        rmse_winner = "Normal" if rmse_diff > 0 else "Quantisiert"
        rmse_color = GREEN if rmse_winner == "Normal" else RED
        
        # MAPE Vergleich
        mape_diff = quantized_accuracy['mape'] - normal_accuracy['mape']
        mape_winner = "Normal" if mape_diff > 0 else "Quantisiert"
        mape_color = GREEN if mape_winner == "Normal" else RED
        
        comparison_data = [
            ["MAE", f"{normal_accuracy['mae']:.3f}", f"{quantized_accuracy['mae']:.3f}", f"{mae_diff:+.3f}", f"{mae_color}{mae_winner}{RESET}"],
            ["RMSE", f"{normal_accuracy['rmse']:.3f}", f"{quantized_accuracy['rmse']:.3f}", f"{rmse_diff:+.3f}", f"{rmse_color}{rmse_winner}{RESET}"],
            ["MAPE (%)", f"{normal_accuracy['mape']:.2f}", f"{quantized_accuracy['mape']:.2f}", f"{mape_diff:+.2f}", f"{mape_color}{mape_winner}{RESET}"],
        ]
        
        for row in comparison_data:
            print(f"{row[0]:<25}{row[1]:<15}{row[2]:<15}{row[3]:<15}{row[4]:<10}")
    
    def _display_quantization_impact(self, normal_accuracy, quantized_accuracy):
        """
        Zeigt die Auswirkungen der Quantisierung an.
        
        Args:
            normal_accuracy (dict): Genauigkeitsstatistiken des normalen Modells
            quantized_accuracy (dict): Genauigkeitsstatistiken des quantisierten Modells
        """
        print(f"\n{'='*80}")
        print("QUANTISIERUNGSAUSWIRKUNG")
        print(f"{'='*80}")
        
        # Berechne Verschlechterungsgrad
        mae_diff = quantized_accuracy['mae'] - normal_accuracy['mae']
        rmse_diff = quantized_accuracy['rmse'] - normal_accuracy['rmse']
        mape_diff = quantized_accuracy['mape'] - normal_accuracy['mape']
        
        mae_degradation = (mae_diff / normal_accuracy['mae'] * 100) if normal_accuracy['mae'] > 0 else 0
        rmse_degradation = (rmse_diff / normal_accuracy['rmse'] * 100) if normal_accuracy['rmse'] > 0 else 0
        mape_degradation = (mape_diff / normal_accuracy['mape'] * 100) if normal_accuracy['mape'] > 0 else 0
        
        avg_degradation = (mae_degradation + rmse_degradation + mape_degradation) / 3
        
        print(f"\nVerschlechterung durch Quantisierung:")
        print(f"MAE: {mae_degradation:+.1f}%")
        print(f"RMSE: {rmse_degradation:+.1f}%") 
        print(f"MAPE: {mape_degradation:+.1f}%")
        print(f"Durchschnitt: {avg_degradation:+.1f}%")
        
        # Gesamtempfehlung
        print(f"\n{'='*80}")
        print("EMPFEHLUNG")
        print(f"{'='*80}")
        
        RESET = '\033[0m'
        BOLD = '\033[1m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        
        if abs(avg_degradation) < 2:
            recommendation = "QUANTISIERUNG EMPFOHLEN - Minimaler Genauigkeitsverlust"
            rec_color = GREEN
        elif abs(avg_degradation) < 5:
            recommendation = "QUANTISIERUNG AKZEPTABEL - Geringer Genauigkeitsverlust"
            rec_color = '\033[93m'  # Gelb
        elif abs(avg_degradation) < 10:
            recommendation = "QUANTISIERUNG ÜBERDENKEN - Moderater Genauigkeitsverlust"
            rec_color = '\033[94m'  # Blau
        else:
            recommendation = "QUANTISIERUNG NICHT EMPFOHLEN - Hoher Genauigkeitsverlust"
            rec_color = RED
        
        print(f"{rec_color}{BOLD}{recommendation}{RESET}")
        
        if avg_degradation < 0:
            print(f"\n✓ Das quantisierte Modell ist überraschenderweise genauer als das normale Modell!")
        else:
            print(f"\n• Das quantisierte Modell verliert {avg_degradation:.1f}% Genauigkeit im Durchschnitt")
        
        print(f"• Modellgröße und Geschwindigkeit vs. Genauigkeit abwägen")
        print(f"• Für Produktionsumgebungen: {recommendation.split(' - ')[0]}")
        print(f"\n{'='*90}\n")
    
    def run_full_validation(self, normal_model_path, quantized_model_path):
        """
        Führt eine vollständige Validierung beider Modelle durch.
        
        Args:
            normal_model_path (str): Pfad zum normalen TFLite-Modell
            quantized_model_path (str): Pfad zum quantisierten TFLite-Modell
            
        Returns:
            tuple: (normal_accuracy, quantized_accuracy) - Genauigkeitsstatistiken beider Modelle
        """
        print("\n" + "="*90)
        print("VOLLSTÄNDIGE MODELLVALIDIERUNG MIT ECHTEN TESTDATEN")
        print("="*90)
        print("Hier wird die tatsächliche Genauigkeit der Modelle gegen bekannte Ground Truth gemessen")
        
        # Validierung mit echten Testdaten
        normal_test_results = self.make_prediction_with_test_data(
            normal_model_path, "NORMALES TFLite MODELL", show_output=True
        )
        quantized_test_results = self.make_prediction_with_test_data(
            quantized_model_path, "QUANTISIERTES TFLite MODELL", show_output=True
        )
        
        if normal_test_results and quantized_test_results:
            # Bewerte jedes Modell einzeln gegen Ground Truth
            print("\n" + "="*90)
            print("EINZELBEWERTUNG DER MODELLE GEGEN GROUND TRUTH")
            print("="*90)
            
            normal_accuracy = self.evaluate_model_accuracy(normal_test_results)
            quantized_accuracy = self.evaluate_model_accuracy(quantized_test_results)
            
            # Vergleiche die Genauigkeiten beider Modelle
            self.compare_model_accuracies(normal_accuracy, quantized_accuracy, normal_test_results, quantized_test_results)
            
            return normal_accuracy, quantized_accuracy
        else:
            print("Konnte nicht beide Modelle mit Testdaten validieren.")
            return None, None
