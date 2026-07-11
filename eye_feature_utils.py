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