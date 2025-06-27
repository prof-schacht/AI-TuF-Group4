import numpy as np
import time

try:
    # Bei der Verwendung mit einem Edge-Device
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("Using tflite_runtime for TFLite model inference.")
except ImportError:
    # Bei Verwendung ohne Edge-Device oder in einer Umgebung mit TensorFlow
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using TensorFlow for TFLite model inference.")
    except ImportError:
        raise ImportError("Neither tflite_runtime nor TensorFlow is available. Please install one of them to run TFLite model inference.")


class EdgeDeviceInference:
    """
    Args:
        model_path: Pfad zum TFLite-Modell
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None

    def __loadModel(self):
        """
        Lädt das TFLite-Modell.
        """
        if not self.model_path:
            raise ValueError("Model path must be provided.")
        
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        print(f"Model loaded from {self.model_path}")

    def __getInputOutputDetails(self):
        """
        Gibt die Eingabe- und Ausgabetensoren des Modells zurück.
        """
        if self.interpreter is None:
            raise RuntimeError("Model must be loaded before getting input/output details.")
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        return input_details, output_details

    def __runInference(self, input_data: np.ndarray):
        """
        Führt die Inferenz mit den gegebenen Eingabedaten durch.
        
        Args:
            input_data: Numpy-Array der Eingabedaten
        """
        if self.interpreter is None:
            raise RuntimeError("Model must be loaded before running inference.")
        
        input_details, output_details = self.__getInputOutputDetails()

        print(f"Input dtype: {input_details[0]['dtype']}")
        
        # Setze die Eingabedaten
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Führe die Inferenz aus
        start_time = time.time()
        self.interpreter.invoke()
        end_time = time.time()
        
        # Hole die Ausgabedaten
        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        print(f"Inference time: {end_time - start_time:.4f} seconds")
        print(f"Output data: {output_data}")

        return output_data

    def __createDummyInput(self):
        """
        Erstellt Dummy-Eingabedaten für die Inferenz.
        
        Args:
            shape: Form der Eingabedaten
        """
        inputDetails, _ = self.__getInputOutputDetails()
        shape = tuple(inputDetails[0]['shape'])
        dataType = inputDetails[0]['dtype']

        return np.random.random_sample(shape).astype(np.float32)

    def run(self, input_data: np.ndarray):
        """
        Führt die Inferenz mit Dummy-Eingabedaten aus.
        Das Input-Shape wird automatisch aus dem Modell gelesen.
        """
        self.__loadModel()
        return self.__runInference(input_data)