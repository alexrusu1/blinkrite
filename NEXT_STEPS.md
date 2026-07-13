# Blink Detection — Next Steps (written 2026-07-10, late night)

## Where things stand

The goal is detecting **small/fast blinks** without counting **squints** or
**head shakes**. Current status:

- `test.py` runs at 30fps with face in frame (was 15 — fixed by 640x480
  capture + replacing `model.predict()` with a direct model call).
- Three detectors run live in `test.py`:
  1. **Blendshape transient** — brief rise-and-fall of MediaPipe's blink
     score (duration-gated, so squints are rejected).
  2. **EAR dip** — brief dip of EAR below a rolling open-eye baseline;
     added because blendshape scores are temporally smoothed and can miss
     1-2 frame blinks entirely.
  3. **Custom MLP** — currently near-useless (trained on old flat-threshold
     labels that excluded fast blinks); gated by blendshape score so it
     can't false-fire on head shakes.
- All detector thresholds are shared in `eye_feature_utils.py` so live
  detection and dataset labeling can't drift apart.
- **Open problems**: EAR detector gives false positives, struggles in the
  dark, and quick blinks are STILL missed. Likely one root cause: dim light
  -> long exposure -> motion blur + silent fps drop, so the camera may not
  even capture the quick blink. Needs data, not more blind threshold tuning.

## Step 1 (do first): record diagnostic signals  (~5 min)

```
python3 diag_signals.py
```

Do ~10 quick blinks, ~5 normal blinks, ~3 squints, a few head shakes —
tagging each RIGHT AFTER with: SPACE=quick blink, n=normal blink,
s=squint, h=head shake, q=quit.

Run it TWICE: once in good lighting, once in the dark conditions that
struggle. Each run saves `signals_<timestamp>.csv` + `.png` plot and
prints the **effective fps**.

Then tell Claude the CSV filenames and ask it to analyze them and set the
detector thresholds from the data. Interpretation:
- Quick blinks visible as EAR dips -> tune thresholds from data (also
  compare what false positives look like vs real blinks).
- Nothing visible in the dark run / fps drops to ~15 -> capture problem;
  software can't fix it — needs more light on the face (note: the product
  IS a lamp; minimum-illumination mode could be a feature).

## Step 2: label the training dataset

Dataset: ~100 people x 3 videos (drowsy / normal / energetic), 10 min each
(~50 hours). The labeler (`3_process_video_dataset.py`) was rewritten to
label by transient detection (blendshape + EAR-dip union) instead of the
old flat 0.5 threshold, and streams to disk per video (won't blow memory,
interruptible).

```
# 30fps training data
python3 3_process_video_dataset.py <videos...> --output blink_data_30fps.csv
# 15fps training data (subsampled from the same footage)
python3 3_process_video_dataset.py <videos...> --target-fps 15 --output blink_data_15fps.csv
```

Notes:
- Do Step 1 FIRST — if it changes thresholds in `eye_feature_utils.py`,
  the labels improve for free before the big 50-hour run.
- Use FRESH output filenames — the old `blink_data.csv` has bad
  (flat-threshold) labels; don't mix.
- ~50h of video takes many hours to process. To parallelize: split the
  video list across several terminals, each with its own `--output`
  part-file, then concatenate (drop duplicate headers). Or ask Claude for
  a parallel driver script.
- **Drowsy videos**: drowsy blinks are slow (300-500ms) and may be
  rejected by the 350ms duration gate (`MAX_BLINK_DURATION_S` in
  `eye_feature_utils.py`). Watch the per-video "squints/closures rejected"
  count on drowsy videos; if it's high, bump to ~0.5 before the big run.
  (For eye-strain monitoring, slow closures probably SHOULD count.)
- Per-video stats print `X blendshape blinks + Y EAR-only blinks` — the
  Y number is how many blinks the old labeler was missing.

## Step 3: retrain both models

```
python3 2_train_model.py --fps 30 --input blink_data_30fps.csv
python3 2_train_model.py --fps 15 --input blink_data_15fps.csv
```

Saves `blink_model_{fps}fps.keras` + `scaler_{fps}fps.joblib` — exactly the
filenames `test.py` auto-selects from measured camera fps.

## Step 4: verify in test.py

Watch the console:
- `--- MEDIAPIPE BLINK DETECTED (...ms transient, MLP peaked at X) ---`
  X near 0.9+ on real blinks means the retrained MLP finally sees them.
- `=== EAR BLINK DETECTED ===` / `[diag]` lines show the other detectors.
- Test: quick blinks, squints (should be silent), head shakes (silent),
  and dark-room behavior.

## Key tunables (all in eye_feature_utils.py)

**Updated 2026-07-12: Step 1 (diagnostic recording) is DONE and the
detector was redesigned from that data.** The recording
(`signals_20260712_143700.csv`: 43 quick blinks, 30 normal, 11 squints,
1 head shake at ~45fps) showed:
- Quick blinks peak at median 0.33 on the blendshape score - amplitude
  thresholds can't separate them from squints (0.1-0.3 plateau overlap).
- But VELOCITY separates them nearly perfectly: blinks close at 5-40
  score/s, squints creep at <2.
- Consecutive blinks merge into one long excursion (score never settles in
  between), so the old classify-the-excursion-at-its-end detector
  structurally missed them. A rising-edge velocity detector fires once
  per blink inside a merged excursion.

`tune_thresholds.py` replays a recording through the detector and sweeps
constants; the deployed values below scored 65/73 marked blinks with 1
squint fire (vs 57/73 and 4 fires for the previous hand-guessed set).

| Constant | Value | Meaning |
|---|---|---|
| `BS_VEL_THRESHOLD` | 2.5 | blink fires at score rise >= this (score/s) |
| `BS_MIN_RISE` | 0.08 | ...and score >= baseline + this (amplitude floor) |
| `BS_FALL_DELTA` | 0.03 | "settled" boundary: re-arms detector, ends excursion |
| `MAX_BLINK_DURATION_S` | 0.5 | EAR-dip duration gate (blink dips ~250ms, squints ~1.9s) |
| `BASELINE_FRAMES` | 90 | rolling baseline window (~3s), shared by both detectors |
| `EAR_DIP_RATIO` | 0.5 | blink needs EAR below baseline x this (deep-dip backup) |
| `EAR_RECOVER_RATIO` | 0.85 | EAR dip event start/end boundary |

These are shared by `test.py` (live) and `3_process_video_dataset.py`
(labeling) — change once, affects both. To re-tune after a new recording:
`python tune_thresholds.py signals_<ts>.csv` (add `--diag` for per-event
shape stats).
