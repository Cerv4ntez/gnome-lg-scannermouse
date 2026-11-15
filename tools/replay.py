#!/usr/bin/env python3
"""Replay recorded scans through the stitcher, without the scanner or the app.

    tools/replay.py RECORDING.mjpg [...] [--out DIR] [--live] [--background]

The app keeps the raw frames of the last three scans in ~/.cache/scannermouse/
(copy the ones worth keeping: older ones are deleted). Each recording is stitched
as the app would, and one line of figures is printed for it; with --out the page
is saved there too (best-frame and averaged).

By default the Mapper runs in step with the frames, so a recording gives the same
result every time and two versions of the code can be compared. --background uses
the background process as the app does (results vary a little from run to run),
--live only the live tracker, without the Mapper's corrections.

Figures:
  link turn   mean rotation by which the tracker's frame-to-frame links disagree with
              the final positions, in degrees per link: how much the tracker drifts
  link shift  the same for their shift, px (median)
  misfit      median disagreement of the overlaps with the final positions, px
"""

import argparse
import hashlib
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scannermouse import align, framedecode, stitcher  # noqa: E402
from scannermouse.capture import jpeg_to_rgb  # noqa: E402


def read(path):
    """The frames of a recording as (MouseRecord, JPEG bytes)."""
    with open(path, 'rb') as f:
        data = f.read()
    frames, start = [], data.find(b'\xff\xd8')
    while start >= 0:
        end = data.find(b'\xff\xd8', start + 2)
        rec, jpeg = framedecode.decode(data[start:end if end > 0 else len(data)])
        if rec is not None:
            frames.append((rec, jpeg))
        start = end
    return frames


def replay(frames, mode='inline'):
    """A Stitcher that has processed `frames` (close it when done)."""
    st = stitcher.Stitcher(inline=mode != 'background')
    if mode != 'live':
        st.start()
    for rec, jpeg in frames:
        st.add(rec, jpeg_to_rgb(jpeg))
    if mode == 'live':
        st.flush()
    else:
        st.finish()
    return st


def figures(st):
    """Tracker drift and overlap misfit of a finished inline replay."""
    m = st.mapper.mapper
    poses = m.poses
    turns, shifts, misfits = [], [], []
    for i, j, meas, weight in m.edges:
        r = align.relative(poses[i], poses[j]) - meas
        if j - i == 1:
            turns.append(abs(math.degrees((r[2] + math.pi) % (2 * math.pi) - math.pi)))
            shifts.append(math.hypot(r[0], r[1]))
        else:
            misfits.append(math.hypot(r[0], r[1]))
    mean = lambda v: float(np.mean(v)) if v else 0.0
    median = lambda v: float(np.median(v)) if v else 0.0
    return mean(turns), median(shifts), median(misfits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recordings', nargs='+')
    parser.add_argument('--out', help='directory to save the pages in')
    parser.add_argument('--live', action='store_true', help='live tracker only')
    parser.add_argument('--background', action='store_true', help='Mapper in the background process, as in the app')
    args = parser.parse_args()
    mode = 'live' if args.live else 'background' if args.background else 'inline'
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for path in args.recordings:
        name = os.path.splitext(os.path.basename(path))[0]
        frames = read(path)
        started = time.perf_counter()
        st = replay(frames, mode)
        took = time.perf_counter() - started
        page = st.page(best=True)
        line = f'{name}: {len(frames)} frames, {st.frames_used} placed, {len(st.mapper.segments)} segments'
        if mode != 'live':
            stats = st.mapper.stats
            line += (f', {stats["overlaps"]} overlaps, {st.misplaced} misplaced, '
                     f'{stats["straight lines"]} lines' + (' (rejected)' if stats['straightening rejected'] else ''))
        if mode == 'inline':
            turn, shift, misfit = figures(st)
            line += f' | link turn {turn:.3f} deg, link shift {shift:.2f} px, misfit {misfit:.2f} px'
        if st.dropped or st.restarts:
            line += f' | {st.dropped} dropped, {st.restarts} restarts'
        if page is not None:
            line += f' | page {page[0].shape[1]}x{page[0].shape[0]} {hashlib.md5(page[0].tobytes()).hexdigest()[:8]}'
            if args.out:
                cv2.imwrite(os.path.join(args.out, f'{name}-best.png'), page[0])
                cv2.imwrite(os.path.join(args.out, f'{name}-avg.png'), st.page(best=False)[0])
        print(line + f' | {took:.0f} s', flush=True)
        st.close()


if __name__ == '__main__':
    main()
