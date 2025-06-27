from .EdgeDeviceOptimization import EdgeDeviceOptimization
from .TestOptimizedModels import TestOptimizedModels


if __name__ == "__main__":

    # Schritt 1: Konvertieren des Modells
    optimizer = EdgeDeviceOptimization(model_path="src/srv/models/best_model.h5", export_path="src/edgeDevice/models/")
    optimizer.ConvertModel()

    # Schritt 2: Testen der Modelle
    test_validator = TestOptimizedModels()
    test_validator.run_full_validation(normal_model_path="src/edgeDevice/models/model.tflite",
                                       quantized_model_path="src/edgeDevice/models/model_quantized.tflite")