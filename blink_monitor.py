"""
Blinkrite — real-time blink-rate monitor.

Watches the webcam, counts blinks, and flags when the blink rate drops to
levels associated with screen-induced eye strain. Optionally reports status
to the Blinkrite lamp (ESP32) over serial so it can adjust its bias lighting.

Detection uses MediaPipe FaceLandmarker blendshape blink scores with a
velocity detector: a blink is a FAST RISE of the score relative to a
rolling per-user baseline (blinks close at 5-40 score/s, squints creep at
<2 - constants tuned on a marked ground-truth recording; 67/73 marked
blinks vs 62/73 for the previous excursion detector). A left/right
symmetry gate rejects winks, a reopening-edge re-arm keeps counting when
consecutive blinks never fully reopen in between, and a head-motion veto
discards events during fast head movement.

Run:   python3 blink_monitor.py [--serial PORT] [--camera N]
Quit:  press 'q' in the video window.
"""

import argparse
import csv
import math
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
    NOSE_TIP_INDEX,
    BS_VEL_THRESHOLD,
    BS_MIN_RISE,
    BS_FALL_DELTA,
    BS_EYE_ASYM_MAX,
    BS_REARM_DROP,
    BASELINE_FRAMES,
    MIN_BASELINE_SAMPLES,
)

BLINK_COOLDOWN_S = 0.25    # ignore re-triggers within this window
BLINK_FLASH_S = 0.4        # how long the on-screen blink indicator shows

# Head-motion veto: fast head movement blurs and shifts the eye landmarks,
# making the blink score spike exactly like a real blink. Motion is nose-tip
# travel per frame, in units of the eye-to-eye distance (distance-invariant);
# a blink detection is discarded if motion during it exceeded this. Blinks
# made while actively shaking the head are sacrificed - acceptable trade.
HEAD_MOTION_VETO = 0.08    # eye-span units per frame
# Left/right outer eye corners, used as the scale reference
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263

# Blink-rate alerting. A normal spontaneous rate is ~15-20 blinks/min;
# focused screen use commonly suppresses it below 10.
LOW_BPM_THRESHOLD = 7      # alert when rate drops to this
NORMAL_BPM_THRESHOLD = 12  # clear the alert once rate recovers to this
ALERT_WARMUP_S = 60        # need a full minute of data before judging


class VelocityBlinkDetector:
    """Detects blinks as fast rises of the blendshape blink score.

    A blink closes the eyes at 5-40 score/s while squints creep up at <2
    (measured on a marked ground-truth recording), so the trigger is RISE
    SPEED with an amplitude floor, both relative to a rolling median
    baseline of the eyes-open resting level (which varies by person and
    lighting). This catches fast/partial blinks a fixed threshold misses
    and rejects squints without needing a duration gate.

    Wink rejection: a hard wink sympathetically squeezes the other eye,
    so even min(left, right) can rise like a quick blink. Real blinks
    close both eyes together (measured |left-right| at the peak <= 0.24),
    so events with a large left/right gap are ignored.

    After firing, the detector disarms until the score either settles
    back near baseline or falls BS_REARM_DROP below its post-fire peak -
    the latter so consecutive blinks still count when the eyes never
    fully reopen in between.

    If the score stays off-baseline too long without firing, it has
    settled at a new resting level (posture or lighting changed), so the
    baseline recalibrates instead of deadlocking.
    """

    STALE_BASELINE_S = 4.0

    def __init__(self):
        self.baseline_hist = deque(maxlen=BASELINE_FRAMES)
        self.armed = True
        self.prev_score = None
        self.prev_time = None
        self.post_fire_peak = 0.0
        self.last_settled_at = None

    @property
    def calibrated(self):
        return len(self.baseline_hist) >= MIN_BASELINE_SAMPLES

    @property
    def base(self):
        return statistics.median(self.baseline_hist) if self.calibrated else None

    def reset(self):
        """Face lost: drop velocity state so re-detection can't produce a
        huge spurious frame-to-frame delta."""
        self.prev_score = None
        self.prev_time = None
        self.armed = True

    def update(self, left_score, right_score, now):
        """Feed one frame's per-eye blink scores.

        Returns the blink's rise velocity (score/s) when a blink was just
        detected, else None.
        """
        # min of both eyes: robust starting point against winks...
        score = min(left_score, right_score)
        # ...but a hard wink still bleeds into the other eye, so reject
        # any event where the eyes disagree too much to be a real blink.
        symmetric = abs(left_score - right_score) <= BS_EYE_ASYM_MAX

        if not self.calibrated:
            self.baseline_hist.append(score)
            self.last_settled_at = now
            return None

        base = statistics.median(self.baseline_hist)
        settled = score < base + BS_FALL_DELTA

        vel = 0.0
        if self.prev_time is not None and now > self.prev_time:
            vel = (score - self.prev_score) / (now - self.prev_time)
        self.prev_score = score
        self.prev_time = now

        fired = None
        if (self.armed and symmetric
                and vel >= BS_VEL_THRESHOLD
                and score >= base + BS_MIN_RISE):
            self.armed = False
            self.post_fire_peak = score
            fired = vel

        if not self.armed:
            self.post_fire_peak = max(self.post_fire_peak, score)
            # Re-arm on full settle OR on the reopening edge, so blinks
            # without a full reopen in between still count individually.
            if settled or score <= self.post_fire_peak - BS_REARM_DROP:
                self.armed = True

        if settled:
            # Only clearly-open frames feed the baseline, so blinks and
            # squints can't drag it toward "closed".
            self.baseline_hist.append(score)
            self.last_settled_at = now
        elif (self.last_settled_at is not None
                and now - self.last_settled_at > self.STALE_BASELINE_S):
            self.baseline_hist.clear()  # adopt the new resting level
            self.last_settled_at = now

        return fired


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

    detector = VelocityBlinkDetector()
    lamp = LampLink(args.serial)

    blink_times = deque()
    total_blinks = 0
    last_blink_at = 0.0
    alert_active = False
    start = time.time()

    # Head-motion tracking for the veto (see HEAD_MOTION_VETO)
    prev_nose = None
    motion_hist = deque()  # (timestamp, motion in eye-spans/frame)

    # Per-frame signal log, written on exit (overwritten each run) so a
    # session that misbehaved can be analyzed afterwards from real data.
    log_rows = []  # (t, score, base, motion, event_open, blink)

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
            left_score = scores.get('eyeBlinkLeft', 0.0)
            right_score = scores.get('eyeBlinkRight', 0.0)
            blink_score = min(left_score, right_score)  # for the session log

            # --- Head motion, in eye-spans per frame ---
            nose = landmarks[NOSE_TIP_INDEX]
            eye_span = math.hypot(
                landmarks[LEFT_EYE_OUTER].x - landmarks[RIGHT_EYE_OUTER].x,
                landmarks[LEFT_EYE_OUTER].y - landmarks[RIGHT_EYE_OUTER].y) or 1e-6
            motion = 0.0
            if prev_nose is not None:
                motion = math.hypot(nose.x - prev_nose[0],
                                    nose.y - prev_nose[1]) / eye_span
            prev_nose = (nose.x, nose.y)
            motion_hist.append((now, motion))
            while motion_hist and motion_hist[0][0] < now - 2.0:
                motion_hist.popleft()

            blink_vel = detector.update(left_score, right_score, now)
            counted = False
            if blink_vel is not None:
                # Peak head motion over the closing edge just detected
                # (velocity fires at the fast rise, so a fixed lookback
                # covers the whole event so far).
                window_start = now - 0.5
                peak_motion = max((m for ts, m in motion_hist if ts >= window_start),
                                  default=0.0)
                if peak_motion > HEAD_MOTION_VETO:
                    print(f"ignored: head motion during event "
                          f"({peak_motion:.2f} eye-spans/frame)")
                elif now - last_blink_at > BLINK_COOLDOWN_S:
                    counted = True
                    blink_times.append(now)
                    total_blinks += 1
                    last_blink_at = now
                    print(f"BLINK! (vel {blink_vel:.1f}/s, motion {peak_motion:.2f})  "
                          f"rate: {len(blink_times)}/min  total: {total_blinks}")

            log_rows.append((round(now - start, 3), round(blink_score, 4),
                             round(detector.base, 4) if detector.base is not None else '',
                             round(motion, 4), int(not detector.armed),
                             int(counted)))
            face_status = None
        else:
            detector.reset()
            prev_nose = None
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

    log_path = os.path.join(script_dir, 'last_session_log.csv')
    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'score', 'base', 'motion', 'event_open', 'blink'])
        w.writerows(log_rows)
    print(f"Session signal log saved to {log_path}")


if __name__ == '__main__':
    main()
