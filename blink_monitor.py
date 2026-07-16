"""
Blinkrite — real-time blink-rate monitor.

Watches the webcam, counts blinks, and flags when the blink rate drops to
levels associated with screen-induced eye strain. Optionally reports status
to the Blinkrite lamp (ESP32) over serial so it can adjust its bias lighting.

Detection uses MediaPipe FaceLandmarker blendshape blink scores with a
velocity detector: a blink is a FAST RISE of the score relative to a
rolling per-user baseline (blinks close at 5-40 score/s, squints creep at
<2 - constants tuned on marked ground-truth recordings; 148/153 marked
blinks). Winks and head shakes are rejected by deferred confirmation on
EAR asymmetry (0/30 winks, 0/5 head shakes counted on the recordings), a
reopening-edge re-arm keeps counting when consecutive blinks never fully
reopen in between, and a head-motion veto discards events during fast
head movement.

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
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    NOSE_TIP_INDEX,
    calculate_ear,
    BS_VEL_THRESHOLD,
    BS_MIN_RISE,
    BS_FALL_DELTA,
    BS_REARM_DROP,
    EAR_ASYM_MAX,
    WINK_LOOKBACK_S,
    BLINK_CONFIRM_S,
    BASELINE_FRAMES,
    MIN_BASELINE_SAMPLES,
)

BLINK_COOLDOWN_S = 0.25    # ignore re-triggers within this window
BLINK_FLASH_S = 0.4        # how long the on-screen blink indicator shows

# Head motion is measured as nose-tip travel per frame, in units of the
# eye-to-eye distance (distance-invariant), and recorded in the session
# log for diagnostics. It is NOT used to veto detections: real blinks
# made while the head moves should count, and motion-without-blink is
# already rejected by the EAR-asymmetry confirmation (head motion shifts
# the two eyes' landmarks differently; a real blink keeps them moving
# together).
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

    Wink and head-shake rejection: a hard wink sympathetically squeezes
    the other eye, so even min(left, right) rises like a quick blink -
    and the smoothed blendshape left/right gap overlaps real blinks, so
    it can't discriminate. Raw-landmark EAR asymmetry can (blinks stay
    under ~0.13, winks/head-shakes exceed ~0.26), but a wink's asymmetry
    can peak BEFORE the trigger, so a triggered event stays PENDING for
    BLINK_CONFIRM_S and is confirmed as a blink only if the max EAR
    asymmetry over [trigger - WINK_LOOKBACK_S, trigger + BLINK_CONFIRM_S]
    stays under EAR_ASYM_MAX.

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
        self.ear_asym_hist = deque()   # (t, |left_ear - right_ear|)
        self.armed = True
        self.prev_score = None
        self.prev_time = None
        self.post_fire_peak = 0.0
        self.last_settled_at = None
        self.pending = None            # (trigger_t, max_ear_asym, trigger_vel)

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
        self.pending = None
        self.ear_asym_hist.clear()

    def update(self, left_score, right_score, left_ear, right_ear, now):
        """Feed one frame's per-eye blink scores and EARs.

        Returns the trigger's rise velocity (score/s) when a blink is
        CONFIRMED - i.e. ~BLINK_CONFIRM_S after the actual blink - else
        None.
        """
        score = min(left_score, right_score)
        ear_asym = abs(left_ear - right_ear)

        self.ear_asym_hist.append((now, ear_asym))
        while self.ear_asym_hist and self.ear_asym_hist[0][0] < now - WINK_LOOKBACK_S:
            self.ear_asym_hist.popleft()

        # Resolve any pending event: confirm as a blink once the window
        # elapses with the eyes staying symmetric; a wink or head shake
        # pushes EAR asymmetry over the gate and cancels it.
        confirmed = None
        if self.pending is not None:
            trigger_t, max_asym, trigger_vel = self.pending
            max_asym = max(max_asym, ear_asym)
            if now - trigger_t >= BLINK_CONFIRM_S:
                if max_asym <= EAR_ASYM_MAX:
                    confirmed = trigger_vel
                self.pending = None
            else:
                self.pending = (trigger_t, max_asym, trigger_vel)

        if not self.calibrated:
            self.baseline_hist.append(score)
            self.last_settled_at = now
            return confirmed

        base = statistics.median(self.baseline_hist)
        settled = score < base + BS_FALL_DELTA

        vel = 0.0
        if self.prev_time is not None and now > self.prev_time:
            vel = (score - self.prev_score) / (now - self.prev_time)
        self.prev_score = score
        self.prev_time = now

        if (self.armed and self.pending is None
                and vel >= BS_VEL_THRESHOLD
                and score >= base + BS_MIN_RISE):
            self.armed = False
            self.post_fire_peak = score
            # Seed with asymmetry already seen in the lookback - a wink's
            # asymmetry peak often precedes the min-score trigger.
            self.pending = (now, max(a for _, a in self.ear_asym_hist), vel)

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

        return confirmed


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
            # Raw-landmark EARs: the wink/head-shake discriminator
            left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)

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

            blink_vel = detector.update(left_score, right_score,
                                        left_ear, right_ear, now)
            counted = False
            if blink_vel is not None and now - last_blink_at > BLINK_COOLDOWN_S:
                # Peak head motion over the event window, logged for
                # diagnostics only - blinks during head movement count,
                # and motion-without-blink was already rejected by the
                # detector's EAR-asymmetry confirmation.
                window_start = now - BLINK_CONFIRM_S - 0.5
                peak_motion = max((m for ts, m in motion_hist if ts >= window_start),
                                  default=0.0)
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
