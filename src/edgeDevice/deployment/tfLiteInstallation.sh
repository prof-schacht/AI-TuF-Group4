#!/bin/bash
set -e  # Bei Fehler abbrechen

# Installiere TensorFlow Lite Runtime für Edge Devices
echo "Installing TensorFlow Lite Runtime for Edge Devices..."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y python3-tflite-runtime
echo "TensorFlow Lite Runtime installation completed."
