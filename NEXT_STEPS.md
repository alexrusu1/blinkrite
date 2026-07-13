# Blink Detection — Next Steps (written 2026-07-10, late night)

> **Update 2026-07-12:** `blink_monitor.py` is now the clean MVP — one
> baseline-relative transient detector, minimal HUD, optional `--serial`
> lamp link. `test.py` remains the noisy dev harness (3 detectors + diag
> output). The MLP retraining plan below is still the path to
> person-independent detection; the MVP doesn't use the MLP at all.

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

Both detectors are now RELATIVE to a rolling open-eye baseline (the user's
resting blendshape score measured ~0.03-0.05, so absolute thresholds broke:
blink events never "closed" below an absolute 0.05 and got rejected as
too-long squints — fixed 2026-07-11).

| Constant | Value | Meaning |
|---|---|---|
| `BS_RISE_DELTA` | 0.04 | blink needs score >= baseline + this |
| `BS_FALL_DELTA` | 0.02 | excursion starts/ends crossing baseline + this |
| `MAX_BLINK_DURATION_S` | 0.35 | longer events = squint, rejected (maybe 0.5 for drowsy data) |
| `BASELINE_FRAMES` | 90 | rolling baseline window (~3s), shared by both detectors |
| `EAR_DIP_RATIO` | 0.82 | blink needs EAR below baseline x this |
| `EAR_RECOVER_RATIO` | 0.92 | EAR dip event start/end boundary |

These are shared by `test.py` (live) and `3_process_video_dataset.py`
(labeling) — change once, affects both.
