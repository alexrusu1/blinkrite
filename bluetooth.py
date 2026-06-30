import cv2
import mediapipe as mp
import math
import time
import os
import numpy as np
import serial
import threading
import queue
from collections import deque

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# --- Bluetooth / Serial Setup ---
# User specified COM6 for outgoing and COM7 for incoming
SERIAL_OUT_PORT = 'COM6'
SERIAL_IN_PORT = 'COM7'
BAUD_RATE = 115200  # Default for most ESP32 Bluetooth projects

ser_out = None
ser_in = None
last_sent_status = None

# Queue for manual messages to be sent
input_queue = queue.Queue()

def manual_input_thread():
    """Thread to capture user input from the console without blocking the main loop."""
    print("\n--- Manual Serial Terminal Active ---")
    print("Type anything and press Enter to send to ESP32.")
    print("--------------------------------------\n")
    while True:
        try:
            msg = input()
            if msg:
                input_queue.put(msg)
        except EOFError:
            break

def send_status(status):
    global last_sent_status
    if ser_out and status != last_sent_status:
        try:
            ser_out.write(status.encode())
            last_sent_status = status
            print(f"Sent status: {status}")
        except Exception as e:
            print(f"Serial write error ({status}): {e}")

try:
    # We attempt to open both ports as requested.
    # Note: Often Bluetooth SPP on Windows uses a single port for both, 
    # but we follow the user's explicit COM6/COM7 configuration.
    ser_out = serial.Serial(SERIAL_OUT_PORT, BAUD_RATE, timeout=0.1)
    ser_in = serial.Serial(SERIAL_IN_PORT, BAUD_RATE, timeout=0.1)
    print(f"Bluetooth Initialized - Out: {SERIAL_OUT_PORT}, In: {SERIAL_IN_PORT}")
    
    # Start the input thread
    t = threading.Thread(target=manual_input_thread, daemon=True)
    t.start()
    
    send_status('O') # Start with 'O' for Off/Warmup
except Exception as e:
    print(f"Warning: Could not connect to Bluetooth/Serial: {e}")
    print("Continuing without Bluetooth functionality.")

# --- Model Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,  # Enables neural-network blink scores
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Eye Landmark Indices (MediaPipe 468-point mesh) ---
LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 380, 374, 263, 386, 385]

# =====================================================================
#  BLINK DETECTION PARAMETERS
# =====================================================================
# Hysteresis thresholds â€” two separate thresholds prevent false triggers
# when the EAR hovers near a single value. The EAR must DROP below the
# close threshold to start a blink, then RISE above the open threshold
# to finish it. The gap between them is a dead zone where noise is ignored.
#
# These are the initial values; they adapt automatically after warmup.
EAR_CLOSE_THRESHOLD = 0.18   # EAR must fall below this â†’ eye is closing
EAR_OPEN_THRESHOLD  = 0.23   # EAR must rise above this â†’ eye is open again

# Adaptive baseline â€” adjusts thresholds to your face and camera
WARMUP_FRAMES = 60            # Frames to collect before adapting (~2s at 30fps)
BASELINE_UPDATE_EVERY = 20    # Recalculate thresholds every N frames
CLOSE_RATIO = 0.82            # Close threshold = baseline Ã— 0.82 (sensitive)
OPEN_RATIO  = 0.90            # Open threshold  = baseline Ã— 0.90
BLINK_DEPTH_RATIO = 0.70      # Absolute depth: min EAR must reach baseline Ã— this
MIN_RELATIVE_DROP = 0.08      # Relative drop: EAR must fall â‰¥8% from last open value
# A blink passes validation if EITHER depth check succeeds.

# Wink filter â€” reject single-eye closures (intentional winks)
# If the OTHER eye's EAR stays above baseline Ã— this ratio during the
# closure, only one eye closed â†’ it's a wink, not a natural blink.
WINK_MAX_EAR_RATIO = 0.92

# Dip detector â€” catches blinks the hysteresis misses at low FPS.
# At low FPS the camera may only see the EAR start to drop and then recover,
# never actually crossing the close threshold. This detector tracks the
# recent EAR peak, watches for any dip, and validates when recovery is
# detected. Works across any number of frames (not just 3).
DIP_MIN_DROP_RATIO = 0.03     # Must drop â‰¥3% of baseline from peak to trough
DIP_MIN_RECOVERY_RATIO = 0.02 # Must recover â‰¥2% of baseline from trough
DIP_MAX_DURATION_S = 0.5      # Peak-to-recovery must complete within this

# Blendshape blink detection â€” uses MediaPipe's neural network blink scores.
# More robust than EAR at low FPS because the model is trained to recognize
# blink patterns even from partially-closed frames. Scores range 0â€“1.
BS_BLINK_THRESHOLD = 0.5      # Score above this = eye is closing
BS_OPEN_THRESHOLD = 0.3       # Score below this = eye is open again

# Eye occlusion â€” detect when a hand covers one eye.
# If one eye's EAR is consistently much lower than the other for several
# frames, it's probably covered. Switch to using only the uncovered eye.
# (Brief disparities like winks are only 1-2 frames, so the frame count
# requirement prevents interference with wink detection.)
OCCLUSION_DISPARITY = 0.5     # Flag if one eye EAR < 50% of the other
OCCLUSION_CONFIRM_FRAMES = 5  # Must persist this many frames to confirm

# Duration constraints â€” a real blink is 50â€“400ms
MIN_BLINK_DURATION_S = 0.05   # Shorter = noise or tracking glitch
MAX_BLINK_DURATION_S = 0.5    # Longer = intentional close or squint
BLINK_COOLDOWN_S = 0.25       # Ignore re-triggers within this window

# Alerting
LOW_BPM_THRESHOLD = 7         # Alert when BPM drops to this
NORMAL_BPM_THRESHOLD = 12     # Clear alert when BPM returns to this
ALERT_WARMUP_S = 60           # Don't alert during the first 60 seconds

# =====================================================================
#  STATE
# =====================================================================
eye_closed = False
eye_close_start_time = None
min_ear_during_close = 1.0    # Tracks how deep the min EAR drops during a closure
max_ear_during_close = 0.0    # Tracks the other eye â€” high = wink, low = both eyes
pre_close_ear = 0.3           # min EAR on the last open frame before closure started
prev_ear = 0.3                # min EAR from the previous frame (for drop calculation)
last_blink_time = 0

# Dip detector state â€” tracks peak, trough, and recovery
dip_peak_ear = 0.0            # Recent local maximum EAR
dip_peak_time = 0.0           # When the peak was recorded
dip_trough_ear = 1.0          # Lowest EAR since peak
dip_trough_max_ear = 1.0      # Other eye's min EAR at trough (wink check)

# Occlusion state â€” per-eye counters for sustained low EAR
left_low_count = 0
right_low_count = 0
left_occluded = False
right_occluded = False

# Blendshape detector state
bs_blink_active = False       # True while blendshape says eye is closing

blink_timestamps = deque()     # Timestamps of recent blinks (last 60s)
current_bpm = 0
low_blink_alert_active = False

ear_history = deque(maxlen=450)  # ~15s at 30fps for baseline calculation
frame_count = 0
baseline_ear = 0.28           # Current baseline (updated adaptively)
blink_depth_threshold = 0.15  # Computed from baseline Ã— BLINK_DEPTH_RATIO
start_time = time.time()

# =====================================================================
#  HELPERS
# =====================================================================

def calculate_distance(p1, p2):
    """2D distance between two normalized landmark points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio for one eye (6 landmark points)."""
    v1 = calculate_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    h  = calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h)


def update_baseline():
    """Recalculate adaptive thresholds from recent EAR history."""
    global EAR_CLOSE_THRESHOLD, EAR_OPEN_THRESHOLD, baseline_ear, blink_depth_threshold
    if len(ear_history) < WARMUP_FRAMES:
        return
    # 90th percentile = "eyes comfortably open" baseline.
    # This naturally ignores blinks, which sit in the lower percentiles.
    baseline_ear = np.percentile(ear_history, 90)
    EAR_CLOSE_THRESHOLD = baseline_ear * CLOSE_RATIO
    EAR_OPEN_THRESHOLD  = baseline_ear * OPEN_RATIO
    blink_depth_threshold = baseline_ear * BLINK_DEPTH_RATIO

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
    frame_count += 1

    # --- Update BPM (blinks in the last 60 seconds) ---
    while blink_timestamps and blink_timestamps[0] < current_time - 60:
        blink_timestamps.popleft()
    current_bpm = len(blink_timestamps)

    # --- Alerting (after warmup period) ---
    if current_time > start_time + ALERT_WARMUP_S:
        if current_bpm <= LOW_BPM_THRESHOLD:
            send_status('A')
            low_blink_alert_active = True
        elif current_bpm >= NORMAL_BPM_THRESHOLD:
            send_status('N')
            low_blink_alert_active = False
    else:
        send_status('O')

    # --- Send manual messages from the input queue ---
    while not input_queue.empty():
        msg = input_queue.get()
        print(f"Debug: Attempting to send manual message: '{msg}'")
        if ser_out:
            try:
                # Many ESP32 sketches expect \r\n or just \n
                data_to_send = msg.encode() + b'\r\n'
                ser_out.write(data_to_send)
                ser_out.flush() # Ensure it's sent immediately
                print(f"Manual send success: {msg}")
            except Exception as e:
                print(f"Manual send error: {e}")
        else:
            print("Debug: Cannot send manual message, ser_out is None")

    # --- Read incoming Bluetooth data (COM7) ---
    if ser_in and ser_in.in_waiting > 0:
        try:
            incoming = ser_in.read(ser_in.in_waiting).decode('utf-8', errors='ignore')
            if incoming:
                print(f"ESP32: {incoming.strip()}")
        except Exception as e:
            print(f"Serial read error: {e}")

    # --- Process face landmarks ---
    if results.face_landmarks:
        for face_landmarks_list in results.face_landmarks:
            landmarks = face_landmarks_list

            left_ear  = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            average_ear = (left_ear + right_ear) / 2.0

            # --- Occlusion detection ---
            # If one eye's EAR is way lower than the other for many
            # consecutive frames, that eye is covered (not a blink).
            if left_ear < right_ear * OCCLUSION_DISPARITY:
                left_low_count += 1
            else:
                left_low_count = 0
                left_occluded = False
            if right_ear < left_ear * OCCLUSION_DISPARITY:
                right_low_count += 1
            else:
                right_low_count = 0
                right_occluded = False

            if left_low_count >= OCCLUSION_CONFIRM_FRAMES:
                left_occluded = True
            if right_low_count >= OCCLUSION_CONFIRM_FRAMES:
                right_occluded = True

            # Choose which eye(s) to use for detection
            if left_occluded and not right_occluded:
                det_ear = right_ear
                max_ear = right_ear
            elif right_occluded and not left_occluded:
                det_ear = left_ear
                max_ear = left_ear
            else:
                # Normal: use the more-closed eye for detection,
                # and the more-open eye for wink filtering.
                det_ear = min(left_ear, right_ear)
                max_ear = max(left_ear, right_ear)

            # Store average EAR for baseline calculation (more stable)
            ear_history.append(average_ear)

            # Periodically adapt thresholds to your face/camera
            if frame_count % BASELINE_UPDATE_EVERY == 0:
                update_baseline()

            # ==========================================================
            #  HYSTERESIS BLINK DETECTION
            #
            #  Uses min(left, right) EAR so that the eye closing more
            #  drives detection. This catches asymmetric blinks that
            #  the average would miss.
            #
            #    OPEN â”€â”€(det_ear < close_threshold)â”€â”€â–º CLOSED
            #    CLOSED â”€(det_ear > open_threshold)â”€â”€â–º OPEN  (+ validate)
            # ==========================================================
            if not eye_closed:
                # Eyes are open â€” watch for either eye to close
                if det_ear < EAR_CLOSE_THRESHOLD:
                    eye_closed = True
                    eye_close_start_time = current_time
                    min_ear_during_close = det_ear
                    max_ear_during_close = max_ear
                    pre_close_ear = prev_ear  # snapshot the last open min EAR
            else:
                # Eyes are closed â€” track depth and the other eye
                if det_ear < min_ear_during_close:
                    min_ear_during_close = det_ear
                # Track the max EAR seen during closure (lowest value = both
                # eyes closing; high value = only one eye closing = wink)
                if max_ear < max_ear_during_close:
                    max_ear_during_close = max_ear

                # Watch for them to reopen
                if det_ear > EAR_OPEN_THRESHOLD:
                    # Eyes reopened â€” validate the blink
                    duration = current_time - eye_close_start_time

                    # Depth check (either path confirms a real blink):
                    deep_enough = min_ear_during_close < blink_depth_threshold
                    relative_drop = (pre_close_ear - min_ear_during_close) / pre_close_ear if pre_close_ear > 0 else 0
                    steep_enough = relative_drop >= MIN_RELATIVE_DROP

                    # Wink filter: if the OTHER eye stayed wide open the
                    # whole time, this was a deliberate wink â€” don't count it.
                    wink_threshold = baseline_ear * WINK_MAX_EAR_RATIO
                    is_wink = max_ear_during_close > wink_threshold

                    if ((deep_enough or steep_enough)
                            and not is_wink
                            and MIN_BLINK_DURATION_S <= duration <= MAX_BLINK_DURATION_S
                            and (current_time - last_blink_time) > BLINK_COOLDOWN_S):
                        blink_timestamps.append(current_time)
                        last_blink_time = current_time
                        print(f"Blink! ({duration*1000:.0f}ms, depth:{min_ear_during_close:.2f}, drop:{relative_drop:.0%})  BPM: {len(blink_timestamps)}")
                    elif is_wink and (deep_enough or steep_enough):
                        print(f"Wink ignored (other eye EAR: {max_ear_during_close:.2f})")

                    eye_closed = False
                    eye_close_start_time = None
                    min_ear_during_close = 1.0
                    max_ear_during_close = 0.0

            # ==========================================================
            #  DIP DETECTION (supplements hysteresis at low FPS)
            #
            #  Tracks the EAR's recent peak and watches for dips.
            #  When the EAR drops from the peak and then recovers,
            #  that's a blink â€” even if it never crossed the close
            #  threshold. Works across any number of frames.
            #
            #  Gaze shifts are gradual (no sharp recovery within the
            #  time window), so they get filtered by the duration and
            #  recovery checks.
            # ==========================================================
            if not eye_closed:
                # Track the trough (lowest point since peak)
                if det_ear < dip_trough_ear:
                    dip_trough_ear = det_ear
                    dip_trough_max_ear = min(dip_trough_max_ear, max_ear)

                drop = dip_peak_ear - dip_trough_ear
                recovery = det_ear - dip_trough_ear
                duration = current_time - dip_peak_time

                min_drop_val = baseline_ear * DIP_MIN_DROP_RATIO
                min_recovery_val = baseline_ear * DIP_MIN_RECOVERY_RATIO

                if (drop >= min_drop_val and recovery >= min_recovery_val
                        and duration <= DIP_MAX_DURATION_S
                        and (current_time - last_blink_time) > BLINK_COOLDOWN_S):
                    # Valid dip with recovery â€” check wink filter
                    wink_threshold = baseline_ear * WINK_MAX_EAR_RATIO
                    is_wink = dip_trough_max_ear > wink_threshold
                    if not is_wink:
                        blink_timestamps.append(current_time)
                        last_blink_time = current_time
                        print(f"Dip-Blink! (drop:{drop:.3f}, rec:{recovery:.3f})  BPM: {len(blink_timestamps)}")
                    # Reset after detection
                    dip_peak_ear = det_ear
                    dip_peak_time = current_time
                    dip_trough_ear = det_ear
                    dip_trough_max_ear = max_ear
                elif duration > DIP_MAX_DURATION_S:
                    # Timeout â€” dip took too long, not a blink. Reset.
                    dip_peak_ear = det_ear
                    dip_peak_time = current_time
                    dip_trough_ear = det_ear
                    dip_trough_max_ear = max_ear
                elif det_ear >= dip_peak_ear:
                    # New peak â€” EAR is higher than before, reset tracking
                    dip_peak_ear = det_ear
                    dip_peak_time = current_time
                    dip_trough_ear = det_ear
                    dip_trough_max_ear = max_ear
            else:
                # Hysteresis is handling this blink â€” reset dip tracker
                dip_peak_ear = det_ear
                dip_peak_time = current_time
                dip_trough_ear = det_ear
                dip_trough_max_ear = max_ear

            prev_ear = det_ear  # save min EAR for next frame's drop calc

            # ==========================================================
            #  BLENDSHAPE BLINK DETECTION (most robust at low FPS)
            #
            #  MediaPipe's neural network outputs blink scores (0â€“1)
            #  for each eye. These are trained to recognize blinks even
            #  from partially-closed frames that EAR can't detect.
            #  Uses min(left, right) to filter winks.
            # ==========================================================
            if results.face_blendshapes and len(results.face_blendshapes) > 0:
                blink_l = 0.0
                blink_r = 0.0
                for bs in results.face_blendshapes[0]:
                    if bs.category_name == 'eyeBlinkLeft':
                        blink_l = bs.score
                    elif bs.category_name == 'eyeBlinkRight':
                        blink_r = bs.score

                # Occlusion-aware score selection
                if left_occluded and not right_occluded:
                    blink_score = blink_r
                elif right_occluded and not left_occluded:
                    blink_score = blink_l
                else:
                    blink_score = min(blink_l, blink_r)  # filters winks

                if blink_score > BS_BLINK_THRESHOLD:
                    if not bs_blink_active:
                        bs_blink_active = True
                elif blink_score < BS_OPEN_THRESHOLD:
                    if bs_blink_active:
                        if (current_time - last_blink_time) > BLINK_COOLDOWN_S:
                            blink_timestamps.append(current_time)
                            last_blink_time = current_time
                            print(f"BS-Blink! (L:{blink_l:.2f}, R:{blink_r:.2f})  BPM: {len(blink_timestamps)}")
                        bs_blink_active = False

            # --- On-screen display ---
            color_ear = (0, 0, 255) if eye_closed else (0, 255, 0)
            cv2.putText(frame, f"EAR: {average_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ear, 2)
            cv2.putText(frame, f"BPM: {current_bpm}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Close<{EAR_CLOSE_THRESHOLD:.2f}  Open>{EAR_OPEN_THRESHOLD:.2f}",
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    cv2.imshow('Blink Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
landmarker.close()
send_status('O')
if ser_out:
    ser_out.close()
if ser_in:
    ser_in.close()
cv2.destroyAllWindows()
