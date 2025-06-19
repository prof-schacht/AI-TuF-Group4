import EdgeDeviceInference
import numpy as np

if __name__ == "__main__":

    # Inferenz mit dem nicht quantisierten TFLite-Modell
    print("Running inference with the non-quantized TFLite model...")
    model_path = "models/model.tflite"
    
    inference = EdgeDeviceInference.EdgeDeviceInference(model_path)
    outputData = inference.run()
    print("Inference completed.")

    # Inferenz mit dem quantisierten TFLite-Modell
    print("Running inference with the quantized TFLite model...")
    quantized_model_path = "models/model_quantized.tflite"
    inference_quantized = EdgeDeviceInference.EdgeDeviceInference(quantized_model_path)
    outputData_quantized = inference_quantized.run()
    print("Inference with quantized model completed.")