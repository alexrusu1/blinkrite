# Blinkrite: Smart Eye-Strain Monitor

Blinkrite is an intelligent lamp that uses computer-based ML to track blink rates and dynamically adjust bias lighting to reduce digital eye strain.

## Core Features
* **ML Tracking:** Uses MediaPipe and a custom-trained model to monitor blink frequency via webcam.
* **Adaptive Lighting:** Computer-processed data triggers bias light adjustments based on fatigue.
* **Communication:** Real-time messaging between the computer and ESP32 to control LED output.
* **Ergonomic Design:** Custom mount fits standard monitor bezels for optimal placement.

## Technical Specs
* **ML:** MediaPipe, custom-trained blink detection model.
* **Hardware:** ESP32, custom PCB, addressable LED array.
* **Firmware:** C++ firmware handling serial/network communication and PWM control.
* **Design:** Autodesk Inventor (CAD), KiCad (Schematic/Layout).
