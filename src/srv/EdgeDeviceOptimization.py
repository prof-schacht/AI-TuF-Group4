import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

class EdgeDeviceOptimization:

    """
    Args:
        model_path: Pfad zum gespeicherten Modell
        export_path: Pfad zum Export des optimierten Modells
    """

    def __init__(self, 
                 model_path: str,
                 export_path: str = "optimized_model"):
        
        self.model_path = model_path
        self.export_path = export_path

        self.modelConverter = None
        self.tfLiteModel = None
        self.quantizedModel = None

    def __loadModel(self):
        """
        Lädt das Keras-Modell aus der angegebenen Datei.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist.")
        
        loaded_model = keras.models.load_model(self.model_path)
        self.modelConverter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
        # Kompatibilität für dynamische Tensoren/LSTM
        self.modelConverter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite-Operationen
            tf.lite.OpsSet.SELECT_TF_OPS  # TensorFlow-Operationen für dynamische Tensoren
        ]
        self.modelConverter._experimental_lower_tensor_list_ops = False  # Deaktiviert die automatische Umwandlung von TensorList-Operationen

    def __ConverToTfLiteWithoutOptimization(self):
        """
        Lädt ein Modell und exportiert es ohne Optimierungen.
        """
        self.tfLiteModel = self.modelConverter.convert()
        with open(self.export_path + "/model.tflite", "wb") as f:
            f.write(self.tfLiteModel)
        
        print(f"Model exported to {self.export_path}/model.tflite without optimizations.")


    def __ConvertToTfLiteWithOptimization(self):
        """
        Führt eine Dynamische Quantisierung des Modells durch.
        Es werden nur die Gewichte von Gleitkommazahlen auf 8-Bit-Ganzzahlen reduziert.
        Dabei verringert sich die Genauigkeit des Modells leicht.
        """
        self.modelConverter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.quantizedModel = self.modelConverter.convert()
        with open(self.export_path + "/model_quantized.tflite", "wb") as f:
            f.write(self.quantizedModel)
        
        print(f"Quantized model exported to {self.export_path}/model_quantized.tflite.")

    
    def __CompareModelSizes(self):
        """
        Vergleicht die Größe des Originalmodells mit der Größe des quantisierten Modells.
        """
        print(f"Original model size: {os.path.getsize(self.model_path)/1024} KB")
        print(f"TFLite model (without optimizations) size: {os.path.getsize(self.export_path + '/model.tflite')/1024} KB")
        print(f"Quantized model size: {os.path.getsize(self.export_path + '/model_quantized.tflite')/1024} KB")

    def ConvertModel(self):
        """
        Führt die Konvertierung des Modells durch.
        """
        # Load the Keras model from the .h5 file
        self.__loadModel()
        self.__ConverToTfLiteWithoutOptimization()
        self.__ConvertToTfLiteWithOptimization()
        self.__CompareModelSizes()


# Temporary Test
# TODO: Remove this after testing
if __name__ == "__main__":
    '''
    # Beispiel: Einfaches Modell für Zeitreihen (ersetzt durch dein finales Modell)
    model = keras.Sequential([
        keras.layers.Input(shape=(10,)),       # <-- Passe Input-Shape an deine Daten an!
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Beispiel-Daten (X: 1000 Zeitschritte à 10 Features, y: 1000 Zielwerte)
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000, 1)
    model.fit(X, y, epochs=5)

    # Speichern als H5
    #model.save("models/my_model.h5")
    '''

    # Testen der Konvertierung
    optimizer = EdgeDeviceOptimization(model_path="models/best_model.h5", export_path="../edgeDevice/models/")
    optimizer.ConvertModel()
    print("Model conversion completed.")