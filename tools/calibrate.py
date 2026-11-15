#!/usr/bin/env python3
"""Fit this mouse's calibration from recorded scans.

    tools/calibrate.py RECORDING.mjpg [...] [--write] [--no-light]

The built-in values (stitcher.py) were measured on one LSM-100. This fits them to
your unit from recordings in ~/.cache/scannermouse/ and, with --write, stores them
in ~/.config/scannermouse/calibration.json, which the app reads at start-up.
Delete that file to return to the built-in values.

Two things are fitted:

  Motion sensors  How the two sensors' counts translate into movement of the image,
                  from frames where the image itself was tracked reliably.
  LED profile     How brightness falls off across the frame, from spots of the page
                  seen at different places in the frame (needs overlapping sweeps;
                  skipped with --no-light).

Good recordings for this: matte paper with plenty of detail (a printed page or a
newspaper), scanned for 30 s or more in overlapping sweeps in both directions, with
some turning of the mouse. Several recordings can be given. Each fit is reported
with its error next to the error of the values in use; a fit that isn't clearly
better is not worth writing.
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from replay import read, replay  # noqa: E402
from scannermouse import calibration, stitcher  # noqa: E402
from scannermouse.capture import jpeg_to_rgb  # noqa: E402

MIN_INLIERS = 150             # feature matches for a frame pair to count as ground truth
MIN_SAMPLES = 300             # frame pairs needed for the sensor fit
MIN_STEP = 2.0                # px; slower frames say little about the scale
GRID = (12, 16)               # cells (rows, columns) the LED profile is measured in
MIN_CELL_SAMPLES = 40         # segments a cell must be seen in
LIGHT_ROUNDS = 2
LIGHT_DEGREE = (2, 4)         # polynomial degree in u (across) and v (down)
BETTER = 0.9                  # a fit is kept if its error is below this share of the error of the values in use

warnings.filterwarnings('ignore', 'All-NaN slice encountered')   # cells never seen: expected


# -- motion sensors -----------------------------------------------------------

def sensor_samples(frames):
    """(counts (n, 4), image shift (n, 2), rotation (n,)) of the reliably tracked frame pairs."""
    wrap = lambda v: (v + 0x8000) % 0x10000 - 0x8000
    delta = lambda a, b: np.array([wrap(b.x1 - a.x1), wrap(b.y1 - a.y1), wrap(b.x2 - a.x2), wrap(b.y2 - a.y2)], float)
    normalise = stitcher.Stitcher._normalise
    feats = []
    for rec, jpeg in frames:
        img = normalise(None, jpeg_to_rgb(jpeg)[..., 1])
        feats.append(stitcher.Features((img * 255).astype(np.uint8)))
    counts, shifts, turns = [], [], []
    for k in range(1, len(frames) - 1):
        prev, rec, nxt = frames[k - 1][0], frames[k][0], frames[k + 1][0]
        if min(r.quality1 for r in (prev, rec, nxt)) < stitcher.QUALITY_MIN \
                or min(r.quality2 for r in (prev, rec, nxt)) < stitcher.QUALITY_MIN:
            continue
        if (rec.counter - prev.counter) % 256 != 1 or (nxt.counter - rec.counter) % 256 != 1:
            continue                            # a dropped frame in between
        step, inliers = feats[k].relative_to(feats[k - 1], MIN_INLIERS)
        if step is None or np.hypot(step[0], step[1]) < MIN_STEP:
            continue
        c = (1 - stitcher.SENSOR_LAG) * delta(prev, rec) + stitcher.SENSOR_LAG * delta(rec, nxt)
        a, b = np.hypot(*c[:2]), np.hypot(*c[2:])
        if min(a, b) < 0.5 * max(a, b):
            continue                            # one sensor stalled
        counts.append(c)
        shifts.append(-step[:2])                # the content moves against the frame
        turns.append(step[2])
    return np.array(counts).reshape(-1, 4), np.array(shifts).reshape(-1, 2), np.array(turns)


def fit_sensors(counts, shifts, turns):
    solve = lambda X, Y: np.linalg.lstsq(X, Y, rcond=None)[0]
    return {'counts_to_pixels': solve(counts, shifts),
            'counts_to_radians': solve(counts, turns),
            'single_to_pixels': np.array([solve(counts[:, :2], shifts), solve(counts[:, 2:], shifts)])}


def sensor_errors(values, counts, shifts, turns):
    """(both sensors px rms, rotation deg rms, sensor 1 alone px, sensor 2 alone px)."""
    rms = lambda r: float(np.sqrt((r ** 2).sum(axis=-1).mean())) if r.ndim > 1 else float(np.sqrt((r ** 2).mean()))
    return (rms(counts @ values['counts_to_pixels'] - shifts),
            float(np.degrees(rms(counts @ values['counts_to_radians'] - turns))),
            rms(counts[:, :2] @ values['single_to_pixels'][0] - shifts),
            rms(counts[:, 2:] @ values['single_to_pixels'][1] - shifts))


# -- LED profile ----------------------------------------------------------------

def light_residual(frames):
    """Per frame cell: median of (segment / averaged page), 1 where the profile in use is
    right; and the number of segments each cell was measured in."""
    st = replay(frames)
    canvas, mapper = st.canvas, st.mapper
    page = np.where(canvas.weight > 0.05, canvas.sum / np.maximum(canvas.weight, 1e-6), np.nan).astype(np.float32)
    rows, cols = GRID
    cells = []
    for k, seg in enumerate(mapper.segments):
        if seg.reason:
            continue
        h, w = seg.img.shape
        (x0, y0, x1, y1), M = canvas._warp(mapper.pose(k), h, w)
        back = cv2.warpAffine(page[y0:y1, x0:x1], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderValue=np.nan)
        frame = seg.img.astype(np.float32) / 255
        # only smooth, bright enough areas: edges and dark noise bias the ratio
        smooth = cv2.blur(np.abs(cv2.Laplacian(cv2.GaussianBlur(frame, (0, 0), 2), cv2.CV_32F)), (15, 15)) < 0.01
        ratio = np.where((back > 0.15) & (frame < 0.99) & smooth, frame / back, np.nan)
        cells.append(np.nanmedian(ratio.reshape(rows, h // rows, cols, w // cols)
                                  .transpose(0, 2, 1, 3).reshape(rows, cols, -1), axis=2))
    st.close()
    cells = np.array(cells).reshape(-1, rows, cols)
    seen = np.sum(~np.isnan(cells), axis=0)
    median = np.nanmedian(cells, axis=0) if len(cells) else np.full(GRID, np.nan)
    return np.where(seen >= MIN_CELL_SAMPLES, median / np.nanmedian(median), np.nan), seen


def fit_light(cells, h, w):
    """Polynomial through the profile measured per cell; (coefficients, rms of the fit)."""
    rows, cols = GRID
    ys = (np.arange(rows) + 0.5) * h / rows - 0.5
    xs = (np.arange(cols) + 0.5) * w / cols - 0.5
    V, U = np.meshgrid((ys - (h - 1) / 2) / (h / 2), (xs - (w - 1) / 2) / (w / 2), indexing='ij')
    terms = [(i, j) for i in range(LIGHT_DEGREE[0] + 1) for j in range(LIGHT_DEGREE[1] + 1)]
    A = np.stack([(U ** i * V ** j).ravel() for i, j in terms], 1)
    y = cells.ravel()
    ok = ~np.isnan(y)
    weight = np.ones(ok.sum())
    for _ in range(5):                          # robust: outlier cells count less
        coef = np.linalg.lstsq(A[ok] * weight[:, None], y[ok] * weight, rcond=None)[0]
        res = A[ok] @ coef - y[ok]
        weight = 1 / np.maximum(1, np.abs(res) / (2 * (1.4826 * np.median(np.abs(res)) + 1e-6)))
    return {t: float(c) for t, c in zip(terms, coef)}, float(np.sqrt(np.mean(res ** 2)))


def calibrate_light(recordings, h, w):
    """Refine the LED profile in rounds; (coefficients or None, lines of report)."""
    rows, cols = GRID
    report, tried, fitted = [], [], None        # tried: (spread, coefficients or None for those in use)
    for round_no in range(LIGHT_ROUNDS + 1):
        residuals = [light_residual(frames)[0] for frames in recordings]
        residual = np.nanmedian(np.array(residuals), axis=0)
        measured = int((~np.isnan(residual)).sum())
        if measured < 0.6 * rows * cols:
            report.append(f'  only {measured} of {rows * cols} frame cells were seen often enough: '
                          'not enough overlap between sweeps for the LED profile')
            return None, report
        spread = float(np.nanstd(residual))
        tried.append((spread, fitted))
        report.append(f'  {"values in use" if round_no == 0 else f"fit {round_no}":14s} '
                      f'brightness of the same spot varies by {100 * spread:.1f}% across the frame')
        if round_no == LIGHT_ROUNDS:
            break
        current = stitcher._illumination(h, w).reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))
        target = current * residual
        fitted, _ = fit_light(target / np.nanmax(target), h, w)
        stitcher.ILLUMINATION = fitted          # the next round measures what is left
        stitcher._illumination.cache_clear()
    spread, best = min(tried, key=lambda t: t[0])
    if best is None or spread > BETTER * tried[0][0]:
        report.append('  the values in use are as good: nothing to change')
        return None, report
    return best, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recordings', nargs='+')
    parser.add_argument('--write', action='store_true', help=f'store the fitted values in {calibration.PATH}')
    parser.add_argument('--no-light', action='store_true', help='fit the motion sensors only')
    args = parser.parse_args()
    recordings = [read(path) for path in args.recordings]
    values = {}

    print('Motion sensors')
    samples = [sensor_samples(frames) for frames in recordings]
    counts, shifts, turns = (np.concatenate([s[k] for s in samples]) for k in range(3))
    print(f'  {len(counts)} reliably tracked frame pairs')
    if len(counts) < MIN_SAMPLES:
        print(f'  too few (at least {MIN_SAMPLES}): scan a detailed matte page for longer')
    else:
        in_use = {'counts_to_pixels': stitcher.COUNTS_TO_PIXELS, 'counts_to_radians': stitcher.COUNTS_TO_RADIANS,
                  'single_to_pixels': np.array(stitcher.SINGLE_TO_PIXELS)}
        # fitted on one half, judged on the other, so the comparison is fair
        half = np.arange(len(counts)) % 2 == 0
        fitted = fit_sensors(counts[half], shifts[half], turns[half])
        errors = []
        for name, v in (('values in use', in_use), ('fitted', fitted)):
            errors.append(sensor_errors(v, counts[~half], shifts[~half], turns[~half]))
            both, turn, one, two = errors[-1]
            print(f'  {name:14s} error per frame: {both:.2f} px, {turn:.3f} deg; '
                  f'sensor 1 alone {one:.2f} px, sensor 2 alone {two:.2f} px')
        if errors[1][0] <= BETTER * errors[0][0]:
            values.update(fit_sensors(counts, shifts, turns))
        else:
            print('  the values in use are as good: nothing to change')

    if not args.no_light:
        print('LED profile')
        first = jpeg_to_rgb(recordings[0][0][1])
        h, w = first.shape[0] - stitcher.CROP_TOP, first.shape[1]
        light, report = calibrate_light(recordings, h, w)
        print('\n'.join(report))
        if light is not None:
            values['illumination'] = light

    if not values:
        print('Nothing to store.')
    elif args.write:
        stored = calibration.load()
        stored.update(values)
        calibration.save(stored)
        print(f'Written to {calibration.PATH}')
    else:
        print('Not stored. Run again with --write to use these values.')


if __name__ == '__main__':
    main()
