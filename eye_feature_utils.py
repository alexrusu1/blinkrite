import math

# --- Landmark Indices (MediaPipe 468-point mesh) ---
# These are shared across all scripts that process eye data.

# The 6 points for classic EAR calculation
LEFT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
RIGHT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]

# The full 16-point contours for each eye, for a richer feature set
LEFT_EYE_CONTOUR_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Combine all contour points for the feature vector
ALL_EYE_INDICES = LEFT_EYE_CONTOUR_INDICES + RIGHT_EYE_CONTOUR_INDICES
NOSE_TIP_INDEX = 1 # A stable point for normalization

# --- Blink transient detection (shared by live detection and dataset
# auto-labeling, so training labels match runtime behavior) ---
# Blinks and squints are separated by DURATION, not amplitude: a fast blink
# may only peak at ~0.10 on MediaPipe's blendshape score, while a squint can
# sit at ~0.30 indefinitely.
#
# The score's eyes-open resting level varies by person and lighting
# (measured ~0.03-0.05 for one user), so thresholds are RELATIVE to a
# rolling open-eye baseline, not absolute: an excursion starts/ends when
# the score crosses baseline + BS_FALL_DELTA, and counts as a blink if it
# peaked at least BS_RISE_DELTA above baseline and completed within
# MAX_BLINK_DURATION_S.
BS_RISE_DELTA = 0.04
BS_FALL_DELTA = 0.02
# Real blinks run ~100ms (alert) to ~500ms (drowsy); squints and deliberate
# closures plateau for seconds, so 0.5s cleanly separates them while still
# counting slow drowsy blinks.
MAX_BLINK_DURATION_S = 0.5

# Rolling open-eye baseline window, shared by the blendshape and EAR
# detectors. Samples are only collected while no excursion is in progress,
# so blinks don't drag the baseline toward "closed".
BASELINE_FRAMES = 90         # ~3s at 30fps
MIN_BASELINE_SAMPLES = 20

# --- EAR-dip transient detector (also shared) ---
# The blendshape scores are temporally smoothed, so a 1-2 frame small blink
# can vanish from that signal entirely. Raw landmarks (and therefore EAR)
# react faster: a blink is a brief dip in EAR relative to the rolling
# open-eye baseline, with the same duration gate as above.
EAR_DIP_RATIO = 0.82        # blink requires EAR below baseline * this
EAR_RECOVER_RATIO = 0.92    # dip event starts/ends crossing baseline * this


def calculate_distance(p1, p2):
    """
    Calculate L2 norm (Euclidean distance) between two MediaPipe landmark points.
    """
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) for one eye using 6 landmark points.
    The indices correspond to the 6 points for the classic EAR calculation.
    """
    # Vertical distances
    v1 = calculate_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    # Horizontal distance
    h = calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    
    # Avoid division by zero if the horizontal distance is zero
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)