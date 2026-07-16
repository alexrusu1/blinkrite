import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import os
import csv
import argparse
import statistics
from collections import deque
from tqdm import tqdm

from eye_feature_utils import (
    calculate_ear,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    ALL_EYE_INDICES,
    NOSE_TIP_INDEX,
    BS_VEL_THRESHOLD,
    BS_MIN_RISE,
    BS_FALL_DELTA,
    MAX_BLINK_DURATION_S,
    BASELINE_FRAMES,
    MIN_BASELINE_SAMPLES,
    EAR_DIP_RATIO,
    EAR_RECOVER_RATIO,
    EAR_ASYM_MAX,
    WINK_LOOKBACK_S,
    BLINK_CONFIRM_S,
)

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


def extract_features(landmarks):
    """Build the per-frame feature vector: both EARs followed by the
    nose-normalized eye contour coordinates. Must match test.py exactly."""
    left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)

    features = [left_ear, right_ear]
    nose_tip = landmarks[NOSE_TIP_INDEX]
    for index in ALL_EYE_INDICES:
        lm = landmarks[index]
        features.extend([lm.x - nose_tip.x, lm.y - nose_tip.y])
    return features


def label_blinks(scores, open_ears, ear_asyms, fps):
    """
    Label blinks with the union of the two transient detectors used live
    (same shared thresholds, tuned on marked ground-truth recordings -
    see eye_feature_utils.py):

    1. Blendshape velocity: a blink closes the eyes at 5-40 score/s while
       a squint creeps up at <2, so fire when the score rises at
       >= BS_VEL_THRESHOLD and sits >= BS_MIN_RISE above the rolling
       open-eye baseline, then disarm until the score settles back below
       baseline + BS_FALL_DELTA. This splits rapid blink bursts whose
       score never returns to baseline in between (each blink re-fires),
       which the old excursion-shape detector structurally missed.
       Each firing must also pass the wink/head-shake gate: max EAR
       asymmetry over [firing - WINK_LOOKBACK_S, firing + BLINK_CONFIRM_S]
       stays under EAR_ASYM_MAX (mirrors blink_monitor.py's deferred
       confirmation; offline we can just look at the window directly).
       The whole off-baseline span containing >= 1 confirmed firing is
       labeled 1; spans with none become hard negatives.
    2. EAR dip: the blendshape score is temporally smoothed and can miss a
       1-2 frame blink entirely, so a brief DEEP dip of EAR below the
       rolling open-eye baseline also counts as a blink.

    Returns (labels, bs_blinks, ear_only_blinks, squint_count, wink_count).
    """
    max_blink_frames = max(1, round(MAX_BLINK_DURATION_S * fps))
    cooldown_frames = max(1, round(0.25 * fps))
    lookback_frames = max(1, round(WINK_LOOKBACK_S * fps))
    confirm_frames = max(1, round(BLINK_CONFIRM_S * fps))
    n = len(scores)
    labels = [0] * n
    bs_blinks = 0
    ear_only_blinks = 0
    squint_count = 0
    wink_count = 0

    def symmetric_at(i):
        window = ear_asyms[max(0, i - lookback_frames):min(n, i + confirm_frames + 1)]
        return max(window) <= EAR_ASYM_MAX if window else True

    # --- Detector 1: baseline-relative blendshape velocity ---
    # settled[i] records whether the score was near baseline at frame i, so
    # detector 2 can avoid polluting its EAR baseline with closed-eye
    # frames (mirrors test.py's "collect only while both signals idle").
    settled = [True] * n
    base_hist = deque(maxlen=BASELINE_FRAMES)
    armed = True
    span_start = None      # start of the current off-baseline span
    span_first_fire = None # first/last velocity firings within this span
    span_last_fire = None
    span_peak = 0.0
    last_fire = -cooldown_frames
    pre_fire_frames = max(1, round(0.1 * fps))
    for i, score in enumerate(scores):
        if len(base_hist) >= MIN_BASELINE_SAMPLES:
            base = statistics.median(base_hist)
            is_settled = score < base + BS_FALL_DELTA
            vel = (score - scores[i - 1]) * fps if i > 0 else 0.0
            if (armed and vel >= BS_VEL_THRESHOLD
                    and score >= base + BS_MIN_RISE
                    and i - last_fire > cooldown_frames):
                if symmetric_at(i):
                    bs_blinks += 1
                    if span_first_fire is None:
                        span_first_fire = i
                    span_last_fire = i
                    last_fire = i
                else:
                    wink_count += 1  # asymmetric event: wink or head shake
                armed = False
            if not armed and is_settled:
                armed = True
        else:
            is_settled = True
        settled[i] = is_settled

        # Track the off-baseline span for labeling/squint accounting.
        # Label only the region around the firings, not the whole span: a
        # blink followed by slow droopy reopening would otherwise mark
        # seconds of half-open frames as blink=1.
        if not is_settled:
            if span_start is None:
                span_start = i
                span_peak = score
            span_peak = max(span_peak, score)
        elif span_start is not None:
            if span_first_fire is not None:
                lo = max(span_start, span_first_fire - pre_fire_frames)
                hi = min(i, span_last_fire + max_blink_frames)
                for j in range(lo, hi):
                    labels[j] = 1
            elif len(base_hist) >= MIN_BASELINE_SAMPLES and \
                    span_peak - statistics.median(base_hist) >= BS_MIN_RISE:
                squint_count += 1  # substantial closure with no fast rise
            span_start = None
            span_first_fire = None
            span_last_fire = None

        if is_settled:
            base_hist.append(score)
    # A span still open at the end of the video is dropped: we can't
    # tell whether it was a blink or a sustained closure.

    # --- Detector 2: EAR dip vs. rolling open-eye baseline ---
    baseline_hist = deque(maxlen=BASELINE_FRAMES)
    event_start = None
    event_min = 1.0
    for i, ear in enumerate(open_ears):
        if event_start is None and settled[i]:
            baseline_hist.append(ear)
        if len(baseline_hist) < MIN_BASELINE_SAMPLES:
            continue
        baseline = statistics.median(baseline_hist)
        if event_start is None:
            if ear < baseline * EAR_RECOVER_RATIO:
                event_start = i
                event_min = ear
        else:
            event_min = min(event_min, ear)
            if ear >= baseline * EAR_RECOVER_RATIO:
                # Same wink/head-shake gate as detector 1: both eyes must
                # move together over the whole dip.
                dip_symmetric = max(ear_asyms[event_start:i + 1]) <= EAR_ASYM_MAX
                if (event_min < baseline * EAR_DIP_RATIO
                        and i - event_start <= max_blink_frames
                        and dip_symmetric):
                    if not any(labels[event_start:i]):
                        ear_only_blinks += 1  # detector 1 missed this one
                    for j in range(event_start, i):
                        labels[j] = 1
                event_start = None

    return labels, bs_blinks, ear_only_blinks, squint_count, wink_count


def process_video(video_path, target_fps=None):
    """
    Pass 1: run the landmarker over every frame, collecting the feature
    vector, blendshape blink score, and open-eye EAR per detected frame.
    Pass 2: label blinks from those series with transient detection.

    If target_fps is set and the video runs faster, frames are subsampled
    so consecutive rows have the frame-to-frame timing of target_fps
    (e.g. a 30fps video -> 15fps training data). Returns labeled rows.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = 1
    if target_fps and video_fps > target_fps:
        frame_step = max(1, round(video_fps / target_fps))
    effective_fps = video_fps / frame_step

    landmarker = vision.FaceLandmarker.create_from_options(options)

    features_per_frame = []
    scores_per_frame = []
    open_ears_per_frame = []
    ear_asyms_per_frame = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} ({total_frames} frames @ {video_fps:.1f}fps, "
          f"sampling every {frame_step} frame(s) -> {effective_fps:.1f}fps data)...")

    last_timestamp_ms = -1
    for frame_idx in tqdm(range(total_frames)):
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % frame_step != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # POS_MSEC glitches backwards on some codecs, and MediaPipe's VIDEO
        # mode hard-errors on non-increasing timestamps - clamp to monotonic.
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks and results.face_blendshapes:
            landmarks = results.face_landmarks[0]
            blendshapes = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}

            features = extract_features(landmarks)
            features_per_frame.append(features)
            # min of both eyes, consistent with test.py (robust to winks)
            scores_per_frame.append(min(blendshapes.get('eyeBlinkLeft', 0),
                                        blendshapes.get('eyeBlinkRight', 0)))
            # more-open eye: both must dip for a blink (wink-safe)
            open_ears_per_frame.append(max(features[0], features[1]))
            # left/right EAR gap: the wink/head-shake discriminator
            ear_asyms_per_frame.append(abs(features[0] - features[1]))

    cap.release()
    landmarker.close()

    labels, bs_blinks, ear_only_blinks, squint_count, wink_count = label_blinks(
        scores_per_frame, open_ears_per_frame, ear_asyms_per_frame, effective_fps)
    blink_frames = sum(labels)
    print(f"  {len(features_per_frame)} usable frames | "
          f"{bs_blinks} blendshape blinks + {ear_only_blinks} EAR-only blinks "
          f"({blink_frames} frames labeled 1) | "
          f"{squint_count} squints/closures + {wink_count} winks/shakes rejected")

    # Person ID from the parent folder name (.../<person_id>/<clip>.mov),
    # so training can group-split by person and avoid train/test leakage.
    person_id = os.path.basename(os.path.dirname(os.path.abspath(video_path))) or "unknown"
    return [features + [label, person_id]
            for features, label in zip(features_per_frame, labels)]


def main():
    parser = argparse.ArgumentParser(
        description="Process video files to generate blink training data. "
                    "Blinks are auto-labeled by transient detection on MediaPipe's "
                    "blendshape score plus EAR dips (catches fast blinks, rejects squints).")
    parser.add_argument("video_files", nargs='+', help="Path(s) to video file(s) to process.")
    parser.add_argument("--output", default='blink_data.csv',
                        help="CSV file to append labeled data to (default: blink_data.csv).")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="Subsample videos to this frame rate, e.g. 15 to build "
                             "15fps training data from 30fps recordings.")
    args = parser.parse_args()

    file_exists = os.path.isfile(args.output)
    total_rows = 0
    # Stream rows to disk after each video so a large dataset (many hours
    # of footage) never has to fit in memory, and an interrupted run keeps
    # the videos already processed.
    with open(args.output, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_header = ['left_ear', 'right_ear']
            for i in range(len(ALL_EYE_INDICES)):
                csv_header.extend([f'eye_lm_{i}_x', f'eye_lm_{i}_y'])
            csv_header.extend(['is_blink', 'person_id'])
            csv_writer.writerow(csv_header)

        for video_file in args.video_files:
            if not os.path.exists(video_file):
                print(f"Warning: Video file not found at {video_file}")
                continue
            # One corrupt clip must not abort a multi-hour batch run.
            try:
                rows = process_video(video_file, args.target_fps)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue
            csv_writer.writerows(rows)
            csvfile.flush()
            total_rows += len(rows)

    print(f"Finished. {total_rows} rows appended to {args.output}.")


if __name__ == '__main__':
    main()
