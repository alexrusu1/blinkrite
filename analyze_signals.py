"""
Analyze a diag_signals.py recording: slice a window around every marked
event and overlay them by event type, so the *shape* of each signal per
event kind is visible instead of one long smeared timeline.

Usage:
    python analyze_signals.py signals_YYYYMMDD_HHMMSS.csv

Produces signals_..._analysis.png next to the input CSV and prints a
per-event-kind stats table.

Markers are pressed AFTER the event happens, so each window spans
[-PRE_S, +POST_S] seconds around the key press and the event itself
appears left of t=0.
"""

import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRE_S = 2.0    # seconds shown before the marker key press
POST_S = 0.5   # seconds shown after

KIND_ORDER = ['quick_blink', 'normal_blink', 'squint', 'head_shake', 'wink']
KIND_COLORS = {
    'quick_blink': 'red',
    'normal_blink': 'green',
    'squint': 'orange',
    'head_shake': 'purple',
    'wink': 'blue',
}


def load(csv_path):
    rows, markers = [], []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            if r[1] == 'MARK':
                markers.append((float(r[0]), r[2]))
            else:
                rows.append(tuple(float(v) for v in r))
    return rows, markers


def window(rows, t_mark):
    return [r for r in rows if t_mark - PRE_S <= r[0] <= t_mark + POST_S]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', help='signals CSV produced by diag_signals.py')
    args = parser.parse_args()

    rows, markers = load(args.csv_path)
    out_png = os.path.splitext(args.csv_path)[0] + '_analysis.png'

    duration = rows[-1][0] - rows[0][0]
    fps = (len(rows) - 1) / duration if duration > 0 else 0.0

    kinds = [k for k in KIND_ORDER if any(m[1] == k for m in markers)]

    # Columns: the signal the detector actually uses (min of eye blendshapes),
    # eye openness (max of eye EARs; drops only when BOTH eyes close),
    # blendshape velocity (per-frame delta) since quick blinks are primarily
    # a velocity event, and left/right asymmetry (the wink discriminator).
    n_rows, n_cols = len(kinds), 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 3.2 * n_rows),
                             sharex=True, squeeze=False)

    stats = {}
    for row_i, kind in enumerate(kinds):
        ax_bs, ax_ear, ax_vel, ax_asym = axes[row_i]
        color = KIND_COLORS[kind]
        peaks, ear_mins, widths, asym_peaks = [], [], [], []

        for t_mark, k in markers:
            if k != kind:
                continue
            w = window(rows, t_mark)
            if len(w) < 3:
                continue
            ts = [r[0] - t_mark for r in w]
            bs_min = [min(r[1], r[2]) for r in w]
            ear_max = [max(r[3], r[4]) for r in w]
            vel = [0.0] + [(b - a) / (t2 - t1) if t2 > t1 else 0.0
                           for a, b, t1, t2 in zip(bs_min, bs_min[1:], ts, ts[1:])]
            asym = [abs(r[1] - r[2]) for r in w]

            ax_bs.plot(ts, bs_min, color=color, alpha=0.25, linewidth=0.8)
            ax_ear.plot(ts, ear_max, color=color, alpha=0.25, linewidth=0.8)
            ax_vel.plot(ts, vel, color=color, alpha=0.25, linewidth=0.8)
            ax_asym.plot(ts, asym, color=color, alpha=0.25, linewidth=0.8)

            peak = max(bs_min)
            peaks.append(peak)
            ear_mins.append(min(ear_max))
            asym_peaks.append(max(asym))
            # width of the event: time the signal stays above half its peak
            above = [t for t, v in zip(ts, bs_min) if v >= peak / 2]
            widths.append((above[-1] - above[0]) if len(above) >= 2 else 0.0)

        n = len(peaks)
        stats[kind] = {
            'n': n,
            'peak_bs': statistics.median(peaks) if peaks else 0.0,
            'min_ear': statistics.median(ear_mins) if ear_mins else 0.0,
            'width_ms': 1000 * statistics.median(widths) if widths else 0.0,
            'asym': statistics.median(asym_peaks) if asym_peaks else 0.0,
        }

        ax_bs.set_ylabel(f'{kind}\n(n={n})', fontsize=9)
        ax_bs.axhline(0.4, color='gray', linestyle=':', linewidth=0.8)
        ax_bs.set_ylim(0, 0.85)
        ax_ear.set_ylim(0, 0.85)
        ax_asym.axhline(0.35, color='gray', linestyle=':', linewidth=0.8)
        ax_asym.set_ylim(0, 1.0)
        for ax in (ax_bs, ax_ear, ax_vel, ax_asym):
            ax.axvline(0, color='black', alpha=0.3, linewidth=0.8)
        if row_i == 0:
            ax_bs.set_title('blendshape score (min of eyes)\ndotted = 0.4 detect threshold', fontsize=9)
            ax_ear.set_title('EAR (max of eyes)', fontsize=9)
            ax_vel.set_title('blendshape velocity (score/s)', fontsize=9)
            ax_asym.set_title('|left - right| score asymmetry\ndotted = wink gate 0.35', fontsize=9)
        if row_i == n_rows - 1:
            for ax in (ax_bs, ax_ear, ax_vel, ax_asym):
                ax.set_xlabel('seconds relative to key press (event is LEFT of 0)')

    fig.suptitle(f'{os.path.basename(args.csv_path)} — {fps:.1f} effective fps, '
                 f'{len(markers)} marked events', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_png, dpi=120)
    print(f'Plot saved -> {out_png}\n')

    print(f'{"event":<14}{"n":>4}{"median peak bs":>16}{"median min EAR":>16}'
          f'{"median width":>14}{"median asym":>13}')
    for kind in kinds:
        s = stats[kind]
        print(f'{kind:<14}{s["n"]:>4}{s["peak_bs"]:>16.2f}{s["min_ear"]:>16.2f}'
              f'{s["width_ms"]:>11.0f} ms{s["asym"]:>13.2f}')


if __name__ == '__main__':
    main()
