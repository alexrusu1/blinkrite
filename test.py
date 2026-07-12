import cv2
import mediapipe as mp
import math
import statistics
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
    NOSE_TIP_INDEX,
    BS_RISE_DELTA,
    BS_FALL_DELTA,
    MAX_BLINK_DURATION_S,
    BASELINE_FRAMES,
    MIN_BASELINE_SAMPLES,
    EAR_DIP_RATIO,
    EAR_RECOVER_RATIO,
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
# Scores range from 0.0 (eye open) to 1.0 (eye closed), but the resting
# eyes-open level varies by person/lighting, so the blink-vs-squint
# transient thresholds (BS_RISE_DELTA, BS_FALL_DELTA, MAX_BLINK_DURATION_S)
# are relative to a rolling baseline and imported from eye_feature_utils,
# keeping live detection and dataset labeling in sync.
BS_OPEN_THRESHOLD = 0.25      # eyes considered open again (unified reset)

# The EAR-dip transient constants (EAR_DIP_RATIO etc.) are imported from
# eye_feature_utils, shared with the dataset labeler.

# =====================================================================
#  STATE
# =====================================================================
feature_history = deque(maxlen=SEQUENCE_LENGTH)

# Unified state for any blink event
blink_event_detected = False
last_blink_time = 0

# Blendshape transient-detector state. An excursion starts when the score
# crosses baseline + BS_FALL_DELTA and is classified when it ends: blink
# (peaked >= BS_RISE_DELTA above baseline and brief), squint (high but
# long), or a logged near-miss (too small to count).
bs_baseline_hist = deque(maxlen=BASELINE_FRAMES)
bs_event_start = None
bs_event_peak = 0.0

# EAR transient-detector state. The baseline history only collects samples
# while the eyes look open, so blinks don't drag the baseline down.
ear_baseline_hist = deque(maxlen=BASELINE_FRAMES)
ear_event_start = None
ear_event_min = 1.0
# Highest MLP probability seen during the current blendshape event —
# logged with each detected blink to show how close the custom model came.
event_peak_prob = 0.0

blink_timestamps = deque()     # Timestamps of recent blinks (last 60s)
current_bpm = 0

# Persists across frames we skip for the MLP (see FRAME_STEP) so the display
# doesn't flicker back to 0 between sampled frames.
blink_prob = 0.0

# =====================================================================
#  CAMERA + MODEL SELECTION
# =====================================================================
cap = cv2.VideoCapture(0)
# Lower resolution speeds up both capture and (mainly) MediaPipe's landmark
# detection, which is what drags the loop down when a face is in frame.
# Eye landmarks stay accurate at this size.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
loop_fps = 0.0
prev_loop_time = time.time()

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

    # --- Effective loop rate (smoothed). This is the rate frames actually
    # get processed at, which can be far below the camera's nominal fps
    # once landmark detection + model inference time is included.
    loop_dt = current_time - prev_loop_time
    prev_loop_time = current_time
    if loop_dt > 0:
        inst_fps = 1.0 / loop_dt
        loop_fps = inst_fps if loop_fps == 0 else 0.9 * loop_fps + 0.1 * inst_fps

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

            # Collect baseline only while no excursion is in progress, so
            # blinks/squints don't drag the baseline toward "closed".
            if bs_event_start is None:
                bs_baseline_hist.append(blink_score)

            # --- Transient detection, relative to the rolling baseline ---
            # The excursion is tracked from a LOW bar (baseline + FALL delta)
            # and classified when it completes, so near-misses get logged
            # with their actual peak instead of silently vanishing.
            if len(bs_baseline_hist) >= MIN_BASELINE_SAMPLES:
                bs_base = statistics.median(bs_baseline_hist)
                if bs_event_start is None:
                    if blink_score > bs_base + BS_FALL_DELTA:
                        bs_event_start = current_time
                        bs_event_peak = blink_score
                        event_peak_prob = blink_prob
                else:
                    bs_event_peak = max(bs_event_peak, blink_score)
                    event_peak_prob = max(event_peak_prob, blink_prob)
                    if blink_score < bs_base + BS_FALL_DELTA:
                        dur = current_time - bs_event_start
                        rise = bs_event_peak - bs_base
                        if rise >= BS_RISE_DELTA and dur <= MAX_BLINK_DURATION_S:
                            if current_time - last_blink_time > BLINK_COOLDOWN_S:
                                blink_event_detected = True
                                last_blink_time = current_time # shared cooldown
                                blink_timestamps.append(current_time) # BPM counter
                                print(f"--- MEDIAPIPE BLINK DETECTED ({dur*1000:.0f}ms, "
                                      f"peak +{rise:.3f} above base, "
                                      f"MLP peaked at {event_peak_prob:.2f}) ---")
                        elif rise >= BS_RISE_DELTA:
                            print(f"[diag] BS plateau: +{rise:.3f} for {dur*1000:.0f}ms "
                                  f"(too long - squint?)")
                        else:
                            print(f"[diag] BS near-miss: peak +{rise:.3f} above base, "
                                  f"{dur*1000:.0f}ms (below rise delta {BS_RISE_DELTA})")
                        bs_event_start = None

        # --- EAR TRANSIENT DETECTION ---
        # Catches small/fast blinks that the temporally-smoothed blendshape
        # score misses entirely: a brief dip in EAR below the rolling
        # open-eye baseline, recovering within the blink duration window.
        landmarks = results.face_landmarks[0]
        left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)
        # Use the more-open eye: both eyes must dip, so winks don't trigger.
        open_ear = max(left_ear, right_ear)

        # Collect baseline samples only while the eyes look open (no
        # excursion in progress on either signal), so blinks don't drag
        # the baseline down.
        if ear_event_start is None and bs_event_start is None:
            ear_baseline_hist.append(open_ear)

        if len(ear_baseline_hist) >= MIN_BASELINE_SAMPLES:
            ear_baseline = statistics.median(ear_baseline_hist)
            if ear_event_start is None:
                if open_ear < ear_baseline * EAR_RECOVER_RATIO:
                    ear_event_start = current_time
                    ear_event_min = open_ear
            else:
                ear_event_min = min(ear_event_min, open_ear)
                if open_ear >= ear_baseline * EAR_RECOVER_RATIO:
                    dur = current_time - ear_event_start
                    dip_pct = (1.0 - ear_event_min / ear_baseline) * 100
                    is_blink_dip = (ear_event_min < ear_baseline * EAR_DIP_RATIO
                                    and dur <= MAX_BLINK_DURATION_S)
                    if is_blink_dip and (current_time - last_blink_time > BLINK_COOLDOWN_S):
                        blink_event_detected = True
                        last_blink_time = current_time
                        blink_timestamps.append(current_time)
                        print(f"=== EAR BLINK DETECTED ({dur*1000:.0f}ms, dip {dip_pct:.0f}%) ===")
                    elif not is_blink_dip:
                        print(f"[diag] EAR dip: {dip_pct:.0f}% over {dur*1000:.0f}ms (not counted)")
                    ear_event_start = None

        # --- CUSTOM MLP-BASED BLINK DETECTION ---
        # Only sample every FRAME_STEP-th real frame into the sequence, so
        # its frame-to-frame timing matches what the chosen model trained on
        # (relevant when the camera runs faster than that model's fps).
        # left_ear / right_ear are reused from the EAR section above.
        if len(results.face_landmarks) > 0 and frame_counter % FRAME_STEP == 0:

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

                # 3. Predict the blink probability. Calling the model directly
                # avoids the large per-call setup overhead of model.predict(),
                # which can dominate the loop time at webcam frame rates.
                blink_prob = float(model(sequence_scaled, training=False).numpy()[0][0])

                # 4. Detect blink event with state machine and cooldown.
                # The blendshape gate (an excursion must be in progress)
                # vetoes MLP false fires from head motion, where the eyes
                # stay fully open and the score sits at baseline.
                if (blink_prob > BLINK_PROB_THRESHOLD
                        and bs_event_start is not None
                        and not blink_event_detected
                        and (current_time - last_blink_time > BLINK_COOLDOWN_S)):
                    blink_event_detected = True
                    last_blink_time = current_time
                    blink_timestamps.append(current_time)
                    print(f"*** BLINK DETECTED (Prob: {blink_prob:.2f}) ***")

    else:
        # Face lost: abandon any in-progress transient so a stale start
        # time doesn't mislabel the next event.
        bs_event_start = None
        ear_event_start = None

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

    # MediaPipe Blendshape Display — show the min (what the detector
    # actually uses) plus each eye's raw score.
    bs_min = min(left_blink_score, right_blink_score)
    if len(bs_baseline_hist) >= MIN_BASELINE_SAMPLES:
        bs_base_disp = statistics.median(bs_baseline_hist)
        bs_text = (f"BS min: {bs_min:.3f} / base {bs_base_disp:.3f} "
                   f"(trigger {bs_base_disp + BS_RISE_DELTA:.3f})")
    else:
        bs_text = f"BS min: {bs_min:.3f} (calibrating baseline...)"
    cv2.putText(frame, bs_text, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # EAR vs. rolling open-eye baseline (the EAR detector's inputs)
    if results.face_landmarks and len(ear_baseline_hist) >= MIN_BASELINE_SAMPLES:
        ear_text = f"EAR: {open_ear:.3f} / base {statistics.median(ear_baseline_hist):.3f}"
    else:
        ear_text = "EAR: calibrating..."
    cv2.putText(frame, ear_text, (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # Unified Blink Indicator
    if blink_event_detected:
        cv2.putText(frame, "BLINK!", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # BPM Display
    cv2.putText(frame, f"BPM: {current_bpm}", (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Effective processing rate (vs. the camera's nominal fps)
    cv2.putText(frame, f"Loop FPS: {loop_fps:.1f}", (30, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Blink Tracker', frame)
    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
landmarker.close()
cv2.destroyAllWindows()
