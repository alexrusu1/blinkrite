"""
Replay a diag_signals.py recording through the exact transient-detector
logic used in test.py, score it against the ground-truth markers, and
sweep the detector constants to find data-backed values.

Usage:
    python tune_thresholds.py signals_YYYYMMDD_HHMMSS.csv [more.csv ...]

Scoring: a marker at t was pressed right AFTER its event, so a marked
blink counts as caught if any detection lands in [t-1.8s, t+0.2s]. Any
detection in a squint/head-shake window counts as a false fire. Detections
outside every marker window are reported separately ("unmatched") since
some are real unmarked natural blinks - they are a tiebreaker, not a hard
failure.
"""

import argparse
import csv
import itertools
import statistics
from collections import deque

BLINK_KINDS = {'quick_blink', 'normal_blink'}
REJECT_KINDS = {'squint', 'head_shake'}
MARKER_PRE_S = 1.8   # detection window before a marker key press
MARKER_POST_S = 0.2  # ...and after
BLINK_COOLDOWN_S = 0.25
BASELINE_FRAMES = 90
MIN_BASELINE_SAMPLES = 20


def load(csv_path):
    rows, markers = [], []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if r[1] == 'MARK':
                markers.append((float(r[0]), r[2]))
            else:
                rows.append(tuple(float(v) for v in r))
    return rows, markers


def simulate(rows, p):
    """Run both transient detectors (mirroring test.py) over the recording.
    Returns a list of (t, detector) detection events.

    The blendshape detector has two modes:
      excursion - classify each above-baseline excursion at its END by
                  peak rise + duration (what test.py currently does)
      velocity  - fire on a fast RISE (score/s) with an amplitude floor,
                  re-arming once the score settles back near baseline.
                  Immune to merged excursions (consecutive blinks whose
                  score never returns to baseline in between), which the
                  excursion detector structurally cannot split.
    """
    detections = []
    last_blink_time = -1e9

    bs_baseline = deque(maxlen=BASELINE_FRAMES)
    bs_event_start = None
    bs_event_peak = 0.0
    bs_armed = True
    prev_t, prev_score = None, None

    ear_baseline = deque(maxlen=BASELINE_FRAMES)
    ear_event_start = None
    ear_event_min = 1.0

    for t, lbs, rbs, lear, rear in rows:
        blink_score = min(lbs, rbs)
        open_ear = max(lear, rear)

        if p.get('MODE', 'excursion') == 'velocity':
            # --- blendshape velocity (rising-edge) detector ---
            near_base = True
            if len(bs_baseline) >= MIN_BASELINE_SAMPLES:
                base = statistics.median(bs_baseline)
                near_base = blink_score < base + p['BS_FALL_DELTA']
                if prev_t is not None and t > prev_t:
                    vel = (blink_score - prev_score) / (t - prev_t)
                    if (bs_armed and vel >= p['BS_VEL']
                            and blink_score >= base + p['BS_MIN_RISE']
                            and t - last_blink_time > BLINK_COOLDOWN_S):
                        detections.append((t, 'bs'))
                        last_blink_time = t
                        bs_armed = False
                if not bs_armed and near_base:
                    bs_armed = True
            if near_base:
                bs_baseline.append(blink_score)
            # keep ear-baseline gating (below) working in this mode
            bs_event_start = None if near_base else (bs_event_start or t)
            prev_t, prev_score = t, blink_score
        else:
            # --- blendshape excursion detector ---
            if bs_event_start is None:
                bs_baseline.append(blink_score)
            if len(bs_baseline) >= MIN_BASELINE_SAMPLES:
                base = statistics.median(bs_baseline)
                if bs_event_start is None:
                    if blink_score > base + p['BS_FALL_DELTA']:
                        bs_event_start = t
                        bs_event_peak = blink_score
                else:
                    bs_event_peak = max(bs_event_peak, blink_score)
                    if blink_score < base + p['BS_FALL_DELTA']:
                        dur = t - bs_event_start
                        rise = bs_event_peak - base
                        if (rise >= p['BS_RISE_DELTA']
                                and dur <= p['MAX_BLINK_DURATION_S']
                                and t - last_blink_time > BLINK_COOLDOWN_S):
                            detections.append((t, 'bs'))
                            last_blink_time = t
                        bs_event_start = None

        # --- EAR dip detector ---
        if ear_event_start is None and bs_event_start is None:
            ear_baseline.append(open_ear)
        if len(ear_baseline) >= MIN_BASELINE_SAMPLES:
            ebase = statistics.median(ear_baseline)
            if ear_event_start is None:
                if open_ear < ebase * p['EAR_RECOVER_RATIO']:
                    ear_event_start = t
                    ear_event_min = open_ear
            else:
                ear_event_min = min(ear_event_min, open_ear)
                if open_ear >= ebase * p['EAR_RECOVER_RATIO']:
                    dur = t - ear_event_start
                    if (ear_event_min < ebase * p['EAR_DIP_RATIO']
                            and dur <= p['MAX_BLINK_DURATION_S']
                            and t - last_blink_time > BLINK_COOLDOWN_S):
                        detections.append((t, 'ear'))
                        last_blink_time = t
                    ear_event_start = None

    return detections


def score(detections, markers):
    caught = {}     # marker index -> True/False for blink markers
    false_fires = 0
    matched_det = set()

    # Markers are only seconds apart, so windows overlap: a detection that
    # falls in BOTH a blink window and a squint/shake window is credited to
    # the blink and must not count as a false fire.
    blink_windows = [(mt - MARKER_PRE_S, mt + MARKER_POST_S)
                     for mt, kind in markers if kind in BLINK_KINDS]

    def in_blink_window(dt):
        return any(lo <= dt <= hi for lo, hi in blink_windows)

    for mi, (mt, kind) in enumerate(markers):
        hits = [di for di, (dt, _) in enumerate(detections)
                if mt - MARKER_PRE_S <= dt <= mt + MARKER_POST_S]
        if kind in BLINK_KINDS:
            caught[mi] = bool(hits)
        elif kind in REJECT_KINDS:
            false_fires += sum(1 for di in hits
                               if not in_blink_window(detections[di][0]))
        matched_det.update(hits)

    by_kind = {}
    for mi, (mt, kind) in enumerate(markers):
        if kind in BLINK_KINDS:
            k = by_kind.setdefault(kind, [0, 0])
            k[0] += 1
            k[1] += caught[mi]

    unmatched = len(detections) - len(matched_det)
    return by_kind, false_fires, unmatched


def evaluate(recordings, p):
    total = {k: [0, 0] for k in BLINK_KINDS}
    ff = um = 0
    for rows, markers in recordings:
        by_kind, false_fires, unmatched = score(simulate(rows, p), markers)
        for k, (n, c) in by_kind.items():
            total[k][0] += n
            total[k][1] += c
        ff += false_fires
        um += unmatched
    return total, ff, um


def fmt(total, ff, um):
    parts = []
    for k in ('quick_blink', 'normal_blink'):
        n, c = total.get(k, (0, 0))
        parts.append(f'{k} {c}/{n}')
    return f'{"  ".join(parts)}  squint/shake fires={ff}  unmatched={um}'


def segments(ts, vals, is_active):
    """Contiguous runs where is_active(v) is true -> (start, end, peak).
    Peak is the max of vals in the run; callers negate vals to track dips."""
    segs, start, peak = [], None, None
    for t, v in zip(ts, vals):
        if is_active(v):
            if start is None:
                start, peak = t, v
            else:
                peak = max(peak, v)
        elif start is not None:
            segs.append((start, t, peak))
            start = None
    if start is not None:
        segs.append((start, ts[-1], peak))
    return segs


def diagnose(rows, markers):
    """Measure every contiguous excursion in the whole recording, assign each
    to the nearest following marker, and report per-kind shapes - so we can
    see WHY events pass or fail the duration/amplitude gates."""
    ts = [r[0] for r in rows]
    bs = [min(r[1], r[2]) for r in rows]
    ear = [max(r[3], r[4]) for r in rows]
    bs_base = sorted(bs)[int(len(bs) * 0.3)]      # resting score
    ear_base = sorted(ear)[int(len(ear) * 0.7)]   # resting openness

    def nearest_kind(seg_end):
        # marker key is pressed after the event ends
        cands = [(mt - seg_end, kind) for mt, kind in markers
                 if 0 <= mt - seg_end <= MARKER_PRE_S]
        return min(cands)[1] if cands else None

    print(f'   baselines: bs~{bs_base:.3f} ear~{ear_base:.3f}')

    for name, segs, describe in (
        ('BLENDSHAPE excursions (score > base+0.02)',
         segments(ts, bs, lambda v: v > bs_base + 0.02),
         lambda s: f'rise {s[2] - bs_base:.2f}'),
        ('EAR dips (ear < base*0.92)',
         segments(ts, [-e for e in ear], lambda v: v > -(ear_base * 0.92)),
         lambda s: f'dip {100 * (1 - (-s[2]) / ear_base):.0f}%'),
    ):
        by_kind = {}
        for seg in segs:
            kind = nearest_kind(seg[1])
            by_kind.setdefault(kind or 'unmarked', []).append(seg)

        print(f'\n   {name}:')
        print(f'   {"kind":<14}{"n":>4}{"dur ms med":>12}{"dur ms max":>12}'
              f'{"magnitude med":>16}{">350ms":>8}')
        for kind in ('quick_blink', 'normal_blink', 'squint', 'head_shake', 'unmarked'):
            evs = by_kind.get(kind)
            if not evs:
                continue
            durs = [e[1] - e[0] for e in evs]
            mags = [describe(e) for e in evs]
            mag_med = statistics.median(
                float(m.split()[1].rstrip('%')) for m in mags)
            unit = '%' if mags[0].startswith('dip') else ''
            too_long = sum(1 for d in durs if d > 0.35)
            print(f'   {kind:<14}{len(evs):>4}{1000*statistics.median(durs):>12.0f}'
                  f'{1000*max(durs):>12.0f}{mag_med:>15.2f}{unit}{too_long:>8}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_paths', nargs='+')
    parser.add_argument('--diag', action='store_true',
                        help='print per-event excursion shapes instead of sweeping')
    args = parser.parse_args()

    recordings = [load(cp) for cp in args.csv_paths]

    if args.diag:
        for cp, (rows, markers) in zip(args.csv_paths, recordings):
            print(f'\n{cp}:')
            diagnose(rows, markers)
        return

    # What's actually deployed in eye_feature_utils.py right now, so every
    # future run of this tool scores the live config against the recording.
    import eye_feature_utils as efu
    current = {
        'MODE': 'velocity',
        'BS_VEL': efu.BS_VEL_THRESHOLD,
        'BS_MIN_RISE': efu.BS_MIN_RISE,
        'BS_FALL_DELTA': efu.BS_FALL_DELTA,
        'MAX_BLINK_DURATION_S': efu.MAX_BLINK_DURATION_S,
        'EAR_DIP_RATIO': efu.EAR_DIP_RATIO,
        'EAR_RECOVER_RATIO': efu.EAR_RECOVER_RATIO,
    }
    total, ff, um = evaluate(recordings, current)
    print('CURRENT (deployed) constants:', current)
    print('  ', fmt(total, ff, um), '\n')

    # Grid informed by the --diag shapes: squints run ~1.7s+ so the duration
    # gate can loosen well past 350ms without letting them in, and head
    # shakes bump the score by ~0.07 so the rise threshold must sit above
    # that to reject them.
    grid = {
        'BS_RISE_DELTA': [0.04, 0.06, 0.08, 0.10, 0.12],
        'BS_FALL_DELTA': [0.015, 0.02, 0.03],
        'MAX_BLINK_DURATION_S': [0.35, 0.50, 0.70, 0.90, 1.20],
        'EAR_DIP_RATIO': [0.50, 0.60, 0.70, 0.82],
        'EAR_RECOVER_RATIO': [0.85, 0.88, 0.92],
    }

    vel_grid = {
        'MODE': ['velocity'],
        'BS_VEL': [1.5, 2.5, 4.0, 6.0],       # score/s rise speed to fire
        'BS_MIN_RISE': [0.04, 0.06, 0.08, 0.10],  # amplitude floor above baseline
        'BS_FALL_DELTA': [0.015, 0.02, 0.03],  # re-arm/settle boundary
        'MAX_BLINK_DURATION_S': [0.50, 0.70],  # EAR detector only, in this mode
        'EAR_DIP_RATIO': [0.50, 0.60, 0.70],
        'EAR_RECOVER_RATIO': [0.85, 0.88, 0.92],
    }

    # Objective: each squint/shake fire costs 2 caught blinks. Missing real
    # blinks hurts an eye-strain monitor more than occasionally counting a
    # squint, and some "fires" near squint markers are likely real unmarked
    # blinks right after the squint, so a hard zero-fires rule overtightens.
    def sweep(g):
        results = []
        for combo in itertools.product(*g.values()):
            p = dict(zip(g.keys(), combo))
            total, ff, um = evaluate(recordings, p)
            caught = sum(c for _, c in total.values())
            results.append(((caught - 2 * ff, -um), p, total, ff, um))
        results.sort(key=lambda r: r[0], reverse=True)
        return results

    def show(label, results, describe):
        print(f'{label} — top configs (of {len(results)}), objective = caught - 2*fires:')
        for _, p, total, ff, um in results[:6]:
            print(f'   {describe(p):<58} {fmt(total, ff, um)}')
        print()

    exc_results = sweep(grid)
    show('EXCURSION MODE', exc_results, lambda p: (
        f'rise={p["BS_RISE_DELTA"]} fall={p["BS_FALL_DELTA"]} '
        f'dur={p["MAX_BLINK_DURATION_S"]} dip={p["EAR_DIP_RATIO"]} '
        f'recover={p["EAR_RECOVER_RATIO"]}'))

    vel_results = sweep(vel_grid)
    show('VELOCITY MODE', vel_results, lambda p: (
        f'vel={p["BS_VEL"]} minrise={p["BS_MIN_RISE"]} fall={p["BS_FALL_DELTA"]} '
        f'dur={p["MAX_BLINK_DURATION_S"]} dip={p["EAR_DIP_RATIO"]} '
        f'recover={p["EAR_RECOVER_RATIO"]}'))

    _, p, total, ff, um = max(exc_results[0], vel_results[0], key=lambda r: r[0])
    n_blinks = sum(n for n, _ in total.values())
    caught = sum(c for _, c in total.values())
    print('OVERALL BEST constants:')
    for k, v in p.items():
        print(f'    {k} = {v}')
    print('  ', fmt(total, ff, um))
    print(f'   overall blink recall: {caught}/{n_blinks}')


if __name__ == '__main__':
    main()
