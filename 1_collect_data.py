import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import math
import os
import time
import csv
from collections import deque

# --- Configuration ---
OUTPUT_CSV_FILE = 'blink_data.csv'
# When you press SPACE, we'll go back and label this many previous frames as a blink.
# This accounts for reaction time.
FRAMES_TO_LABEL_PREVIOUSLY = 4
BUFFER_SIZE = 15 # A small buffer to hold frames before writing to CSV.

# --- Landmark Indices ---
# The 6 points used for the classic EAR calculation
LEFT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
RIGHT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]

# The full 16-point contours for each eye, for a richer feature set for the ML model
LEFT_EYE_CONTOUR_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Combine all contour points for the feature vector
ALL_EYE_INDICES = LEFT_EYE_CONTOUR_INDICES + RIGHT_EYE_CONTOUR_INDICES
NOSE_TIP_INDEX = 1 # A stable point for normalization

# --- MediaPipe Setup (using the new Vision API) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1, # We only need to track one face
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False # Not needed for manual labeling
)

# --- Helper Functions ---
def calculate_distance(p1, p2):
    """Calculate L2 norm (Euclidean distance) between two points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(landmarks, eye_ear_indices):
    """Calculate Eye Aspect Ratio (EAR) for one eye."""
    # Vertical distances
    v1 = calculate_distance(landmarks[eye_ear_indices[1]], landmarks[eye_ear_indices[5]])
    v2 = calculate_distance(landmarks[eye_ear_indices[2]], landmarks[eye_ear_indices[4]])
    # Horizontal distance
    h = calculate_distance(landmarks[eye_ear_indices[0]], landmarks[eye_ear_indices[3]])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

# --- Main Data Collection ---
def collect_data():
    """
    Opens the webcam, detects face landmarks, and saves features to a CSV file.
    The user presses the SPACEBAR to label a frame as a blink.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Prepare CSV file for a single person
    csv_header = ['left_ear', 'right_ear']
    # Add normalized landmark coordinates to the header
    for i in range(len(ALL_EYE_INDICES)):
        csv_header.extend([f'eye_lm_{i}_x', f'eye_lm_{i}_y'])
    csv_header.append('is_blink')

    # Use 'w' mode to create a new file each time. Use 'a' to append.
    with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        landmarker = vision.FaceLandmarker.create_from_options(options)
        data_buffer = deque(maxlen=BUFFER_SIZE)

        print("\n--- Data Collection Started ---")
        print(f"Saving data to: {OUTPUT_CSV_FILE}")
        print("Instructions:")
        print("  - Look at the camera and blink normally. Try different head angles.")
        print("  - Press the SPACEBAR exactly when you blink.")
        print("  - Try to capture various blinks (normal, fast, winks).")
        print("  - Press 'q' to quit.")
        print("---------------------------------\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print(f">>> BLINK registered! Labeling previous {FRAMES_TO_LABEL_PREVIOUSLY} frames.")
                # Go back and retroactively label the last few frames in the buffer as a blink
                for i in range(1, min(FRAMES_TO_LABEL_PREVIOUSLY + 1, len(data_buffer))):
                    if len(data_buffer[-i]) > 0:
                        data_buffer[-i][-1] = 1 # The 'is_blink' column is the last one
            elif key == ord('q'):
                break

            if results.face_landmarks:
                # Process the first (and only) detected face
                if results.face_landmarks:
                    landmarks = results.face_landmarks[0]

                    # --- Feature Extraction ---
                    # 1. Calculate EAR for both eyes
                    left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
                    right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)

                    # 2. Extract and normalize eye landmark coordinates
                    eye_landmarks_normalized = []
                    nose_tip = landmarks[NOSE_TIP_INDEX]
                    for index in ALL_EYE_INDICES:
                        lm = landmarks[index]
                        # Normalize by subtracting the nose tip coordinates
                        norm_x = lm.x - nose_tip.x
                        norm_y = lm.y - nose_tip.y
                        eye_landmarks_normalized.extend([norm_x, norm_y])

                    # --- Save to CSV ---
                    row = [left_ear, right_ear] + eye_landmarks_normalized + [0] # Default label is 0
                    data_buffer.append(row)

            # Write to CSV from buffer (outside the face loop)
            # This keeps the data in chronological order, even with multiple faces
            while len(data_buffer) >= BUFFER_SIZE:
                csv_writer.writerow(data_buffer.popleft())

            # Display status on the frame
            cv2.putText(frame, "RECORDING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE on blink | 'q' to quit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Data Collection', frame)

    # After the loop, write any remaining data from the buffer to the file
    with open(OUTPUT_CSV_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        while data_buffer:
            csv_writer.writerow(data_buffer.popleft())
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n--- Data collection finished. Data saved to {OUTPUT_CSV_FILE}. ---")

if __name__ == '__main__':
    collect_data()