import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import math
import os
import csv
import argparse
from tqdm import tqdm

# --- Configuration ---
OUTPUT_CSV_FILE = 'blink_data.csv'
# Use MediaPipe's blendshape score as the "ground truth" for labeling
BLENDSHAPE_BLINK_THRESHOLD = 0.5

# --- Landmark Indices (Copied from 1_collect_data.py for consistency) ---
LEFT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
RIGHT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]
LEFT_EYE_CONTOUR_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
ALL_EYE_INDICES = LEFT_EYE_CONTOUR_INDICES + RIGHT_EYE_CONTOUR_INDICES
NOSE_TIP_INDEX = 1

# --- MediaPipe Setup (using the new Vision API) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1, # Process one face at a time for simplicity
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True # Essential for auto-labeling
)

# --- Helper Functions (Copied from 1_collect_data.py) ---
def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(landmarks, eye_ear_indices):
    v1 = calculate_distance(landmarks[eye_ear_indices[1]], landmarks[eye_ear_indices[5]])
    v2 = calculate_distance(landmarks[eye_ear_indices[2]], landmarks[eye_ear_indices[4]])
    h = calculate_distance(landmarks[eye_ear_indices[0]], landmarks[eye_ear_indices[3]])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

# --- Main Video Processing Function ---
def process_video(video_path):
    """
    Processes a video file, extracts eye features, and auto-labels blinks
    using MediaPipe's blendshape scores. Appends data to the CSV file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Check if CSV exists, if not, create it with a header
    file_exists = os.path.isfile(OUTPUT_CSV_FILE)
    with open(OUTPUT_CSV_FILE, 'a', newline='') as csvfile:
        landmarker = vision.FaceLandmarker.create_from_options(options)

        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_header = ['left_ear', 'right_ear']
            for i in range(len(ALL_EYE_INDICES)):
                csv_header.extend([f'eye_lm_{i}_x', f'eye_lm_{i}_y'])
            csv_header.append('is_blink')
            csv_writer.writerow(csv_header)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {video_path} ({total_frames} frames)...")

        for frame_idx in tqdm(range(total_frames)):
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Calculate timestamp based on frame index and video FPS
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks and results.face_blendshapes:
                landmarks = results.face_landmarks[0]
                blendshapes = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}

                # --- Feature Extraction ---
                left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)

                eye_landmarks_normalized = []
                nose_tip = landmarks[NOSE_TIP_INDEX]
                for index in ALL_EYE_INDICES:
                    lm = landmarks[index]
                    norm_x = lm.x - nose_tip.x
                    norm_y = lm.y - nose_tip.y
                    eye_landmarks_normalized.extend([norm_x, norm_y])

                # --- Auto-Labeling using Blendshapes ---
                blink_score = min(blendshapes.get('eyeBlinkLeft', 0), blendshapes.get('eyeBlinkRight', 0))
                is_blink = 1 if blink_score > BLENDSHAPE_BLINK_THRESHOLD else 0

                # --- Save to CSV ---
                row = [left_ear, right_ear] + eye_landmarks_normalized + [is_blink]
                csv_writer.writerow(row)

    cap.release()
    landmarker.close()
    print(f"Finished processing. Data appended to {OUTPUT_CSV_FILE}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video files to generate blink training data.")
    parser.add_argument("video_files", nargs='+', help="Path(s) to video file(s) to process.")
    args = parser.parse_args()

    for video_file in args.video_files:
        if os.path.exists(video_file):
            process_video(video_file)
        else:
            print(f"Warning: Video file not found at {video_file}")