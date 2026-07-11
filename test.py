import cv2
import mediapipe as mp
import math
import time
import os
import joblib
import numpy as np
import tensorflow as tf
from collections import deque

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Import shared constants and functions from the new utility file
from eye_feature_utils import (
    calculate_ear,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    ALL_EYE_INDICES,
    NOSE_TIP_INDEX
)

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- MediaPipe FaceLandmarker Setup ---
FACEMODEL_PATH = os.path.join(script_dir, "face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=FACEMODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True, # Enable blendshapes for built-in blink detection
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Model & Scaler Loading ---
# We trained separate models for 15fps and 30fps (a fixed-length frame
# sequence encodes a different real-world time window at each rate, so one
# model can't cleanly serve both). Whichever camera fps we measure below,
# pick whichever trained model is closest and, if the camera runs faster
# than that model expects, drop frames so the model sees the same
# frame-to-frame timing it was trained on.
AVAILABLE_MODELS = {
    15: ("blink_model_15fps.keras", "scaler_15fps.joblib"),
    30: ("blink_model_30fps.keras", "scaler_30fps.joblib"),
}
SEQUENCE_LENGTH = 5 # Must match the training script
BLINK_PROB_THRESHOLD = 0.9 # Confidence threshold for blink detection


def measure_camera_fps(cap, num_frames=20):
    """Measure the camera's actual delivered frame rate (not just its
    reported nominal rate, which many webcams misreport)."""
    for _ in range(5):  # warm up / let exposure settle
        cap.read()
    start = time.time()
    count = 0
    for _ in range(num_frames):
        success, _ = cap.read()
        if not success:
            break
        count += 1
    elapsed = time.time() - start
    if elapsed <= 0 or count == 0:
        return 30.0  # sane fallback
    return count / elapsed

# =====================================================================
#  BLINK DETECTION PARAMETERS
# =====================================================================
# To make the custom model LESS sensitive (e.g., to looking up), INCREASE this value.
# To make it MORE sensitive, DECREASE it.
BLINK_PROB_THRESHOLD = 0.95 # Was 0.9. Let's be more strict to avoid false positives.
BLINK_COOLDOWN_S = 0.25       # Ignore re-triggers within this window

# --- MediaPipe's built-in Blendshape detector ---
# These scores are what the `3_process_video_dataset.py` script uses for labeling.
# Scores range from 0.0 (eye open) to 1.0 (eye closed).
# To make this detector MORE sensitive to small blinks, DECREASE this value.
BS_BLINK_THRESHOLD = 0.4      # Was 0.5. Let's try to catch smaller blinks.
BS_OPEN_THRESHOLD = 0.25      # Was 0.3. Reset when eyes are more open.

# =====================================================================
#  STATE
# =====================================================================
feature_history = deque(maxlen=SEQUENCE_LENGTH)

# Unified state for any blink event
blink_event_detected = False
last_blink_time = 0

blink_timestamps = deque()     # Timestamps of recent blinks (last 60s)
current_bpm = 0

# Persists across frames we skip for the MLP (see FRAME_STEP) so the display
# doesn't flicker back to 0 between sampled frames.
blink_prob = 0.0

# =====================================================================
#  CAMERA + MODEL SELECTION
# =====================================================================
cap = cv2.VideoCapture(0)

print("Measuring camera frame rate...")
measured_fps = measure_camera_fps(cap)
chosen_fps = min(AVAILABLE_MODELS.keys(), key=lambda f: abs(f - measured_fps))
model_name, scaler_name = AVAILABLE_MODELS[chosen_fps]
MODEL_MLP_PATH = os.path.join(script_dir, model_name)
SCALER_PATH = os.path.join(script_dir, scaler_name)
# If the camera delivers faster than the chosen model expects (e.g. a 60fps
# camera against the 30fps model), drop frames so the sequence fed to the
# model has the same frame-to-frame timing it was trained on.
FRAME_STEP = max(1, round(measured_fps / chosen_fps))
print(f"Measured camera fps: {measured_fps:.1f} -> using {chosen_fps}fps model "
      f"({model_name}), sampling every {FRAME_STEP} frame(s).")

print("Loading model and scaler...")
try:
    model = tf.keras.models.load_model(MODEL_MLP_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print(f"Please ensure '{MODEL_MLP_PATH}' and '{SCALER_PATH}' are in the same directory.")
    exit()

# =====================================================================
#  MAIN LOOP
# =====================================================================
print("Starting camera... Press 'q' to quit.")
frame_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    current_time = time.time()

    # --- Update BPM (blinks in the last 60 seconds) ---
    while blink_timestamps and blink_timestamps[0] < current_time - 60:
        blink_timestamps.popleft()
    current_bpm = len(blink_timestamps)

    # --- Process face landmarks ---
    # blink_prob intentionally not reset here - it persists across frames
    # skipped by FRAME_STEP (see STATE section above).
    left_blink_score = 0.0
    right_blink_score = 0.0

    if results.face_landmarks:
        # --- MEDIAPIPE'S BUILT-IN BLINK DETECTION ---
        if results.face_blendshapes:
            blendshapes = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}
            left_blink_score = blendshapes.get('eyeBlinkLeft', 0)
            right_blink_score = blendshapes.get('eyeBlinkRight', 0)
            # Use the minimum score to be more robust against winks.
            # If one eye is open (low score), the result will be low.
            blink_score = min(left_blink_score, right_blink_score)

            # If blendshape detects a blink and cooldown is over, trigger a unified event
            if blink_score > BS_BLINK_THRESHOLD and not blink_event_detected and (current_time - last_blink_time > BLINK_COOLDOWN_S):
                blink_event_detected = True
                last_blink_time = current_time # Update the shared cooldown
                blink_timestamps.append(current_time) # Add to BPM counter
                print(f"--- MEDIAPIPE BLINK DETECTED (Score: {blink_score:.2f}) ---")

        # --- CUSTOM MLP-BASED BLINK DETECTION ---
        # Only sample every FRAME_STEP-th real frame into the sequence, so
        # its frame-to-frame timing matches what the chosen model trained on
        # (relevant when the camera runs faster than that model's fps).
        if len(results.face_landmarks) > 0 and frame_counter % FRAME_STEP == 0:
            landmarks = results.face_landmarks[0]

            # --- Feature Extraction (must match training script) ---
            # 1. Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)

            # 2. Extract and normalize all eye landmark coordinates
            eye_landmarks_normalized = []
            nose_tip = landmarks[NOSE_TIP_INDEX]
            for index in ALL_EYE_INDICES:
                lm = landmarks[index]
                norm_x = lm.x - nose_tip.x
                norm_y = lm.y - nose_tip.y
                eye_landmarks_normalized.extend([norm_x, norm_y])

            # Create the feature vector for the current frame
            current_features = [left_ear, right_ear] + eye_landmarks_normalized
            feature_history.append(current_features)

            if len(feature_history) == SEQUENCE_LENGTH:
                # 1. Format the data into a single feature vector
                sequence_data = np.array(feature_history).flatten().reshape(1, -1)

                # 2. Scale the data using the loaded scaler
                sequence_scaled = scaler.transform(sequence_data)

                # 3. Predict the blink probability
                blink_prob = model.predict(sequence_scaled, verbose=0)[0][0]

                # 4. Detect blink event with state machine and cooldown
                if blink_prob > BLINK_PROB_THRESHOLD and not blink_event_detected and (current_time - last_blink_time > BLINK_COOLDOWN_S):
                    blink_event_detected = True
                    last_blink_time = current_time
                    blink_timestamps.append(current_time)
                    print(f"*** BLINK DETECTED (Prob: {blink_prob:.2f}) ***")

    # Reset the unified blink state if both detectors show eyes are open
    if blink_event_detected and blink_prob < 0.5 and min(left_blink_score, right_blink_score) < BS_OPEN_THRESHOLD:
        blink_event_detected = False

    # --- On-screen display ---
    # Custom Model Display
    color_prob = (0, 255, 0) if blink_prob < 0.5 else (0, 165, 255)
    if blink_prob > BLINK_PROB_THRESHOLD:
        color_prob = (0, 0, 255)

    cv2.putText(frame, f"Custom Model Prob: {blink_prob:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_prob, 2)

    # MediaPipe Blendshape Display
    bs_score = (left_blink_score + right_blink_score) / 2.0
    cv2.putText(frame, f"MediaPipe Score: {bs_score:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Unified Blink Indicator
    if blink_event_detected:
        cv2.putText(frame, "BLINK!", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # BPM Display
    cv2.putText(frame, f"BPM: {current_bpm}", (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Blink Tracker', frame)
    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
landmarker.close()
cv2.destroyAllWindows()
