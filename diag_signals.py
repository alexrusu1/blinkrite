"""
Diagnostic signal recorder for blink detection tuning.

Records every frame's raw signals (blendshape blink scores + EARs) to a CSV
while you mark ground-truth events with keys. Produces a plot on exit so we
can SEE what a quick blink actually looks like in each signal — and whether
it's visible at all — instead of tuning thresholds blind.

Usage:
    python3 diag_signals.py

While it runs, do each action and tag it right after with the key:
    SPACE = I just did a QUICK blink
    n     = I just did a NORMAL blink
    s     = I just squinted
    h     = I just shook my head
    w     = I just WINKED (one eye)
    q     = quit and save

Tips: run it once in your normal lighting and once in the dark conditions
that were struggling, so we can compare. Keep your face at typical distance.
"""

import cv2
import mediapipe as mp
import time
import csv
import os

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from eye_feature_utils import (
    calculate_ear,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
)

script_dir = os.path.dirname(os.path.abspath(__file__))

MARKER_KEYS = {
    ord(' '): 'quick_blink',
    ord('n'): 'normal_blink',
    ord('s'): 'squint',
    ord('h'): 'head_shake',
    ord('w'): 'wink',
}
MARKER_COLORS = {
    'quick_blink': 'red',
    'normal_blink': 'green',
    'squint': 'orange',
    'head_shake': 'purple',
    'wink': 'blue',
}

options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(
        model_asset_path=os.path.join(script_dir, "face_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
)


def main():
    run_id = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(script_dir, f'signals_{run_id}.csv')
    png_path = os.path.join(script_dir, f'signals_{run_id}.png')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    rows = []      # (t, left_bs, right_bs, left_ear, right_ear)
    markers = []   # (t, kind)
    start = time.time()

    print(__doc__)
    print(f"Recording... signals -> {csv_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        t = time.time() - start
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, int((start + t) * 1000))

        if results.face_landmarks and results.face_blendshapes:
            landmarks = results.face_landmarks[0]
            bs = {b.category_name: b.score for b in results.face_blendshapes[0]}
            rows.append((
                t,
                bs.get('eyeBlinkLeft', 0.0),
                bs.get('eyeBlinkRight', 0.0),
                calculate_ear(landmarks, LEFT_EYE_EAR_INDICES),
                calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES),
            ))

        cv2.putText(frame, f"t={t:5.1f}s  frames={len(rows)}  marks={len(markers)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=quick n=normal s=squint h=shake w=wink q=quit",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow('Signal Recorder', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in MARKER_KEYS:
            kind = MARKER_KEYS[key]
            markers.append((t, kind))
            print(f"  [{t:6.1f}s] marked: {kind}")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if not rows:
        print("No frames with a detected face were recorded; nothing saved.")
        return

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'left_bs', 'right_bs', 'left_ear', 'right_ear'])
        w.writerows(rows)
        # markers appended as separate rows for simple re-parsing
        for t, kind in markers:
            w.writerow([t, 'MARK', kind, '', ''])

    # Effective recorded frame rate (frames with a face / elapsed time)
    duration = rows[-1][0] - rows[0][0]
    fps = (len(rows) - 1) / duration if duration > 0 else 0.0
    print(f"Saved {len(rows)} frames over {duration:.1f}s ({fps:.1f} effective fps), "
          f"{len(markers)} markers -> {csv_path}")

    # --- Plot: what each signal did, with ground-truth marks overlaid ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ts = [r[0] for r in rows]
    bs_min = [min(r[1], r[2]) for r in rows]
    ear_max = [max(r[3], r[4]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax1.plot(ts, bs_min, linewidth=0.8)
    ax1.set_ylabel('blendshape score (min of eyes)')
    ax1.axhline(0.07, color='gray', linestyle=':', label='rise threshold')
    ax2.plot(ts, ear_max, linewidth=0.8)
    ax2.set_ylabel('EAR (max of eyes)')
    ax2.set_xlabel('seconds')

    seen_kinds = set()
    for t, kind in markers:
        for ax in (ax1, ax2):
            ax.axvline(t, color=MARKER_COLORS[kind], alpha=0.6,
                       label=kind if kind not in seen_kinds else None)
        seen_kinds.add(kind)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'{fps:.1f} effective fps — markers are pressed AFTER the event, '
                  f'so look just LEFT of each line')

    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    print(f"Plot saved -> {png_path}")


if __name__ == '__main__':
    main()
