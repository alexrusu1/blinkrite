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
#
# All values below were tuned against a marked ground-truth recording
# (signals_20260712_143700.csv: 43 quick blinks, 30 normal, 11 squints,
# 1 head shake at ~45fps) by replaying the detector over it and sweeping
# (tune_thresholds.py). Result: 65/73 marked blinks caught, 1 squint fire,
# vs 57/73 and 4 fires for the previous hand-guessed constants.
#
# The blendshape detector fires on a fast RISE of the score (velocity),
# not on excursion shape: blinks close the eyes at 5-40 score/s while
# squints creep up at <2, so velocity separates them even when their
# peak heights overlap. This also splits consecutive blinks whose score
# never settles to baseline in between (merged excursions) - each blink
# still has its own velocity spike, which an excursion-shape detector
# structurally cannot see.
#
# The score's eyes-open resting level varies by person and lighting
# (measured ~0.02-0.05), so amplitude gates are RELATIVE to a rolling
# open-eye baseline, never absolute.
BS_VEL_THRESHOLD = 2.5      # score/s rise speed that fires the detector
BS_MIN_RISE = 0.08          # ...and score must be this far above baseline
BS_FALL_DELTA = 0.03        # "settled": score back below baseline + this
                            # (re-arms the detector; ends an excursion)

# Wink rejection. A hard wink sympathetically squeezes the other eye, so
# min(left, right) alone can still rise like a quick blink. Real blinks
# close both eyes together: measured |left - right| at the blink peak was
# median 0.09 / max 0.24 across 73 ground-truth blinks, while a wink
# drives the gap to ~0.5+. Events more asymmetric than this are ignored.
BS_EYE_ASYM_MAX = 0.35

# Partial-reopen re-arm. After firing, the detector re-arms when the score
# settles near baseline OR has fallen this far from its post-fire peak -
# so blinks still count when the eyes never fully reopen between them.
BS_REARM_DROP = 0.15

# Legacy excursion-shape detector constants, still used by
# blink_monitor.py (the MVP). Note: the ground-truth replay showed this
# detector caught only 59/73 marked blinks at best (it can't split
# consecutive blinks that merge into one excursion) - porting the MVP to
# the velocity detector above is the known upgrade.
BS_RISE_DELTA = 0.04

# Duration gate, shared by the EAR-dip detector (below) and the legacy
# excursion detector. Real blinks run ~100ms (alert) to ~500ms (drowsy);
# measured squint dips plateau ~1.9s, so 0.5s cleanly separates them
# while still counting slow drowsy blinks.
MAX_BLINK_DURATION_S = 0.5

# Rolling open-eye baseline window, shared by the blendshape and EAR
# detectors. Samples are only collected while no excursion is in progress,
# so blinks don't drag the baseline toward "closed".
BASELINE_FRAMES = 90         # ~3s at 30fps
MIN_BASELINE_SAMPLES = 20

# --- EAR-dip transient detector (also shared) ---
# The blendshape scores are temporally smoothed, so a 1-2 frame small blink
# can vanish from that signal entirely. Raw landmarks (and therefore EAR)
# react faster: a blink is a brief deep dip in EAR relative to the rolling
# open-eye baseline. Measured dips: normal blinks 96%, quick blinks 36%,
# squints 29%, head shakes 19% - so the 50% depth requirement keeps this a
# high-precision backup for solid closures while the velocity detector
# handles the shallow/fast ones.
EAR_DIP_RATIO = 0.5         # blink requires EAR below baseline * this
EAR_RECOVER_RATIO = 0.85    # dip event starts/ends crossing baseline * this


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