# Blinkrite: Smart Eye-Strain Monitor

Blinkrite is an intelligent lamp that uses computer-based ML to track blink rates and dynamically adjust bias lighting to reduce digital eye strain.

## Core Features
* **ML Tracking:** Uses MediaPipe and a custom-trained model to monitor blink frequency via webcam.
* **Adaptive Lighting:** Computer-processed data triggers bias light adjustments based on fatigue.
* **Communication:** Real-time messaging between the computer and ESP32 to control LED output.
* **Ergonomic Design:** Custom mount fits standard monitor bezels for optimal placement.

## Running the Monitor

```bash
pip install -r requirements.txt
python3 blink_monitor.py                 # standalone webcam demo
python3 blink_monitor.py --serial COM6   # with the lamp connected
```

A window opens showing live blink detection (eye contours, blinks/min, and a
low-blink-rate warning). Blinks are detected as brief transients of
MediaPipe's eyelid signal relative to a per-user rolling baseline — so quick
partial blinks count, while squints and winks don't. When the blink rate
stays low, the status flips to `LOW BLINK RATE` and (if connected) the ESP32
adjusts the lamp.

## Technical Specs
* **ML:** MediaPipe, custom-trained blink detection model.
* **Hardware:** ESP32, custom PCB, addressable LED array.
* **Firmware:** C++ firmware handling serial/network communication and PWM control.
* **Design:** Autodesk Inventor (CAD), KiCad (Schematic/Layout).
