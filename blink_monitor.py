"""
Blinkrite — real-time blink-rate monitor.

Watches the webcam, counts blinks, and flags when the blink rate drops to
levels associated with screen-induced eye strain. Optionally reports status
to the Blinkrite lamp (ESP32) over serial so it can adjust its bias lighting.

Detection uses MediaPipe FaceLandmarker blendshape blink scores with a
transient detector: a blink is a brief rise-and-fall of the score relative
to a rolling per-user baseline. Separating blinks from squints by DURATION
(blinks are brief, squints plateau) rather than amplitude lets it catch
small, fast blinks without false-firing on squinting or head movement.

Run:   python3 blink_monitor.py [--serial PORT] [--camera N]
Quit:  press 'q' in the video window.
"""

import argparse
import os
import statistics
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from eye_feature_utils import (
    LEFT_EYE_CONTOUR_INDICES,
    RIGHT_EYE_CONTOUR_INDICES,
    BS_RISE_DELTA,
    BS_FALL_DELTA,
    MAX_BLINK_DURATION_S,
    BASELINE_FRAMES,
    MIN_BASELINE_SAMPLES,
)

BLINK_COOLDOWN_S = 0.25    # ignore re-triggers within this window
BLINK_FLASH_S = 0.4        # how long the on-screen blink indicator shows

# Blink-rate alerting. A normal spontaneous rate is ~15-20 blinks/min;
# focused screen use commonly suppresses it below 10.
LOW_BPM_THRESHOLD = 7      # alert when rate drops to this
NORMAL_BPM_THRESHOLD = 12  # clear the alert once rate recovers to this
ALERT_WARMUP_S = 60        # need a full minute of data before judging


class TransientBlinkDetector:
    """Detects blinks as brief transients of the blendshape blink score.

    The score's eyes-open resting level varies by person and lighting, so
    events are tracked relative to a rolling median baseline with
    hysteresis: an event opens when the score exceeds baseline +
    BS_RISE_DELTA and closes when it drops back below baseline +
    BS_FALL_DELTA. Brief events (<= MAX_BLINK_DURATION_S) are blinks;
    longer plateaus (squints, deliberate closures) are rejected.

    If an event stays open far longer than any closure gesture, the score
    has settled at a new resting level (posture or lighting changed), so
    the detector recalibrates its baseline instead of deadlocking.
    """

    EVENT_TIMEOUT_S = 2.0

    def __init__(self):
        self.baseline_hist = deque(maxlen=BASELINE_FRAMES)
        self.event_start = None

    @property
    def calibrated(self):
        return len(self.baseline_hist) >= MIN_BASELINE_SAMPLES

    def reset(self):
        """Abandon any in-progress event (e.g. when the face is lost)."""
        self.event_start = None

    def update(self, score, now):
        """Feed one frame's blink score.

        Returns the blink's duration in seconds when a blink just
        completed, else None.
        """
        if not self.calibrated:
            self.baseline_hist.append(score)
            return None

        base = statistics.median(self.baseline_hist)
        if self.event_start is None:
            if score > base + BS_RISE_DELTA:
                self.event_start = now
            else:
                # Only clearly-open frames feed the baseline, so blinks
                # and squints can't drag it toward "closed".
                self.baseline_hist.append(score)
            return None

        duration = now - self.event_start
        if score < base + BS_FALL_DELTA:
            self.event_start = None
            return duration if duration <= MAX_BLINK_DURATION_S else None
        if duration > self.EVENT_TIMEOUT_S:
            self.baseline_hist.clear()  # adopt the new resting level
            self.event_start = None
        return None


class LampLink:
    """Optional serial link to the Blinkrite lamp (ESP32).

    Status protocol: 'O' = off/warmup, 'N' = normal, 'A' = low-blink-rate
    alert. Deduplicates so each status is sent once per change.
    """

    def __init__(self, port):
        self.ser = None
        self.last_sent = None
        if not port:
            return
        try:
            import serial
            self.ser = serial.Serial(port, 115200, timeout=0.1)
            print(f"Lamp connected on {port}")
        except Exception as e:
            print(f"Warning: no lamp connection on {port} ({e}); running standalone.")

    def send(self, status):
        if self.ser and status != self.last_sent:
            try:
                self.ser.write(status.encode())
                self.last_sent = status
            except Exception as e:
                print(f"Lamp write failed: {e}")

    def close(self):
        if self.ser:
            self.send('O')
            self.ser.close()


def draw_eye_contours(frame, landmarks):
    h, w = frame.shape[:2]
    for contour in (LEFT_EYE_CONTOUR_INDICES, RIGHT_EYE_CONTOUR_INDICES):
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                           for i in contour])
        cv2.polylines(frame, [points], isClosed=True,
                      color=(0, 255, 170), thickness=1)


def draw_hud(frame, bpm, total_blinks, status, status_color, flash):
    """Minimal overlay: status banner, blink rate, and a blink flash."""
    h, w = frame.shape[:2]
    banner = frame[0:74, 0:w]
    cv2.rectangle(banner, (0, 0), (w, 74), (30, 30, 30), -1)
    frame[0:74, 0:w] = cv2.addWeighted(frame[0:74, 0:w], 0.35, banner, 0.65, 0)

    cv2.putText(frame, f"{bpm} blinks/min", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"total {total_blinks}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, status, (w - tw - 20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if flash:
        text = "BLINK!"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
        cv2.putText(frame, text, ((w - tw) // 2, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)


def main():
    parser = argparse.ArgumentParser(description="Blinkrite blink-rate monitor")
    parser.add_argument('--serial', default=None,
                        help="Serial port of the Blinkrite lamp (e.g. COM6 or /dev/tty.usbserial)")
    parser.add_argument('--camera', type=int, default=0, help="Camera index")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=os.path.join(script_dir, "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        ))

    cap = cv2.VideoCapture(args.camera)
    # 640x480 keeps landmark detection fast enough to catch 1-2 frame blinks
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = TransientBlinkDetector()
    lamp = LampLink(args.serial)

    blink_times = deque()
    total_blinks = 0
    last_blink_at = 0.0
    alert_active = False
    start = time.time()

    print("Blinkrite running - press 'q' in the video window to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # selfie view

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.time() * 1000))
        now = time.time()

        while blink_times and blink_times[0] < now - 60:
            blink_times.popleft()
        bpm = len(blink_times)

        if results.face_landmarks and results.face_blendshapes:
            landmarks = results.face_landmarks[0]
            draw_eye_contours(frame, landmarks)

            scores = {b.category_name: b.score for b in results.face_blendshapes[0]}
            # min of both eyes: a wink leaves one score low, so it won't count
            blink_score = min(scores.get('eyeBlinkLeft', 0.0),
                              scores.get('eyeBlinkRight', 0.0))

            blink_dur = detector.update(blink_score, now)
            if blink_dur is not None and now - last_blink_at > BLINK_COOLDOWN_S:
                blink_times.append(now)
                total_blinks += 1
                last_blink_at = now
                print(f"BLINK! ({blink_dur*1000:.0f}ms)  "
                      f"rate: {len(blink_times)}/min  total: {total_blinks}")
            face_status = None
        else:
            detector.reset()
            face_status = ("NO FACE", (120, 120, 240))

        # --- Blink-rate status (with hysteresis so it doesn't flicker) ---
        elapsed = now - start
        if face_status:
            status, status_color = face_status
            lamp.send('O')
        elif not detector.calibrated or elapsed < ALERT_WARMUP_S:
            status, status_color = "measuring...", (200, 200, 200)
            lamp.send('O')
        else:
            if bpm <= LOW_BPM_THRESHOLD:
                alert_active = True
            elif bpm >= NORMAL_BPM_THRESHOLD:
                alert_active = False
            if alert_active:
                status, status_color = "LOW BLINK RATE", (0, 120, 255)
                lamp.send('A')
            else:
                status, status_color = "blink rate OK", (0, 220, 0)
                lamp.send('N')

        draw_hud(frame, bpm, total_blinks, status, status_color,
                 flash=now - last_blink_at < BLINK_FLASH_S)
        cv2.imshow('Blinkrite', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    landmarker.close()
    lamp.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
