"""
Validate the shipped VelocityBlinkDetector (blink_monitor.py) against
marked diag_signals recordings: blink recall plus wink / squint /
head-shake false fires.

The detector uses deferred confirmation on EAR asymmetry to reject winks
and head shakes: the smoothed blendshape left/right gap overlaps real
blinks (unusable), while raw-landmark EAR asymmetry separates cleanly -
blinks stay under ~0.13, winks/head-shakes exceed ~0.26. Constants live
in eye_feature_utils.py (EAR_ASYM_MAX, WINK_LOOKBACK_S, BLINK_CONFIRM_S);
to re-tune after a new recording, sweep values there and re-run this.

Usage:
    python tune_wink_gate.py signals_*.csv
"""

import argparse
import csv

from blink_monitor import VelocityBlinkDetector
from eye_feature_utils import BLINK_CONFIRM_S

BLINK_COOLDOWN_S = 0.25
BLINK_KINDS = ('quick_blink', 'normal_blink')


def load(path):
    rows, markers = [], []
    for row in csv.reader(open(path, newline='')):
        if row[0] == 't':
            continue
        if row[1] == 'MARK':
            markers.append((float(row[0]), row[2]))
        else:
            rows.append(tuple(float(v) for v in row))
    return rows, markers


def score(rows, markers):
    det = VelocityBlinkDetector()
    last, dets = -1e9, []
    for t, lbs, rbs, lear, rear in rows:
        v = det.update(lbs, rbs, lear, rear, t)
        if v is not None and t - last > BLINK_COOLDOWN_S:
            dets.append(t)
            last = t

    # Markers are pressed AFTER the event; confirmation lands another
    # BLINK_CONFIRM_S after the trigger, so widen the post-window by it.
    win = lambda mt: (mt - 1.8, mt + 0.2 + BLINK_CONFIRM_S)
    kinds = {}
    for mt, k in markers:
        kinds.setdefault(k, []).append(win(mt))
    blink_w = [w for k in BLINK_KINDS for w in kinds.get(k, [])]
    inb = lambda d: any(lo <= d <= hi for lo, hi in blink_w)

    out = {}
    for k, ws in sorted(kinds.items()):
        if k in BLINK_KINDS:
            caught = sum(1 for lo, hi in ws if any(lo <= d <= hi for d in dets))
            out[k] = f'{caught}/{len(ws)}'
        else:
            fires = sum(1 for lo, hi in ws
                        for d in dets if lo <= d <= hi and not inb(d))
            out[k] = f'{fires} fires/{len(ws)}'
    matched = sum(1 for d in dets
                  if any(lo <= d <= hi for ws in kinds.values() for lo, hi in ws))
    out['unmatched'] = len(dets) - matched
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_paths', nargs='+')
    args = parser.parse_args()

    for path in args.csv_paths:
        rows, markers = load(path)
        print(f'{path}:')
        for k, v in score(rows, markers).items():
            print(f'    {k}: {v}')


if __name__ == '__main__':
    main()
