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
MODEL_MLP_PATH = os.path.join(script_dir, "blink_model.keras")
SCALER_PATH = os.path.join(script_dir, "scaler.joblib")
SEQUENCE_LENGTH = 5 # Must match the training script
BLINK_PROB_THRESHOLD = 0.9 # Confidence threshold for blink detection

print("Loading model and scaler...")
try:
    model = tf.keras.models.load_model(MODEL_MLP_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print(f"Please ensure '{MODEL_MLP_PATH}' and '{SCALER_PATH}' are in the same directory.")
    exit()

# --- Landmark Indices (from training scripts) ---
LEFT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
RIGHT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]
LEFT_EYE_CONTOUR_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
ALL_EYE_INDICES = LEFT_EYE_CONTOUR_INDICES + RIGHT_EYE_CONTOUR_INDICES
NOSE_TIP_INDEX = 1

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
blink_detected = False
last_blink_time = 0

# State for MediaPipe's built-in blink detector
bs_blink_active = False
last_bs_blink_time = 0

blink_timestamps = deque()     # Timestamps of recent blinks (last 60s)
current_bpm = 0

# =====================================================================
#  HELPERS
# =====================================================================

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_ear(landmarks, eye_indices):
    v1 = calculate_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    h  = calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

# =====================================================================
#  MAIN LOOP
# =====================================================================
cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")

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
    blink_prob = 0.0
    left_blink_score = 0.0
    right_blink_score = 0.0

    if results.face_landmarks:
        # --- MEDIAPIPE'S BUILT-IN BLINK DETECTION ---
        if results.face_blendshapes:
            blendshapes = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}
            left_blink_score = blendshapes.get('eyeBlinkLeft', 0)
            right_blink_score = blendshapes.get('eyeBlinkRight', 0)
            blink_score = (left_blink_score + right_blink_score) / 2.0

            if blink_score > BS_BLINK_THRESHOLD and not bs_blink_active and (current_time - last_bs_blink_time > BLINK_COOLDOWN_S):
                bs_blink_active = True
                last_bs_blink_time = current_time
                print("--- MEDIAPIPE BLINK DETECTED ---")
            elif blink_score < BS_OPEN_THRESHOLD:
                bs_blink_active = False

        # --- CUSTOM MLP-BASED BLINK DETECTION ---
        if len(results.face_landmarks) > 0:
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
                if blink_prob > BLINK_PROB_THRESHOLD and not blink_detected and (current_time - last_blink_time > BLINK_COOLDOWN_S):
                    blink_detected = True
                    last_blink_time = current_time
                    blink_timestamps.append(current_time)
                    print(f"*** BLINK DETECTED (Prob: {blink_prob:.2f}) ***")
                elif blink_prob < 0.5: # Reset state when eyes are clearly open
                    blink_detected = False

    # --- On-screen display ---
    # Custom Model Display
    custom_model_status_text = "Blink Detected!" if blink_detected else ""
    color_prob = (0, 255, 0) if blink_prob < 0.5 else (0, 165, 255)
    if blink_prob > BLINK_PROB_THRESHOLD:
        color_prob = (0, 0, 255)

    cv2.putText(frame, f"Custom Model Prob: {blink_prob:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_prob, 2)
    if custom_model_status_text:
            cv2.putText(frame, "Custom Blink!", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # MediaPipe Blendshape Display
    bs_score = (left_blink_score + right_blink_score) / 2.0
    bs_status_text = "Blink Detected!" if bs_blink_active else ""
    cv2.putText(frame, f"MediaPipe Score: {bs_score:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if bs_status_text:
            cv2.putText(frame, "MediaPipe Blink!", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow('Blink Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
landmarker.close()
cv2.destroyAllWindows()
