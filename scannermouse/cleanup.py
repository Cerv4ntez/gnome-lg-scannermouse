"""Finishing touches for scanned text pages: straightening and an even background.

- Text detection and skew: rotated to the angle of its lines, a text page has a
  strongly peaked row profile (ink in the lines, none between them). Photos and
  drawings don't, at any angle.
- Orientation: the line profile can't tell upright from upside-down text, so
  Tesseract's orientation detection decides that, when it is installed.
- Background: the paper brightness is estimated by locally removing the text
  (a grey-level closing) and smoothing; dividing by it removes broad blotches
  while leaving the text as it is.
"""

import math
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from scipy import ndimage

INK = 0.75                    # grey level (0..1 of white) below which a pixel counts as ink
SKEW_RANGE = 90               # degrees searched either way (covers sideways text)
TEXT_PEAKINESS = 1.6          # line-profile contrast at the best angle vs a typical angle for "text"
ANALYSE_SIZE = 900            # px; the page is analysed at this size
PROFILE_BAND = 3              # px at that size; text lines there are ~10-15 px apart
BACKGROUND_KERNEL = 25        # px at full resolution; wider than text strokes, narrower than blotches
WHITE = 0.9                   # after flattening, this share of the paper level becomes pure white
OSD_MIN_CONFIDENCE = 2.0      # Tesseract orientation confidence needed to turn the page
# despeckling, in px at full scan resolution
SPECK_INK = 200               # grey level below which a pixel belongs to a dot (fringe included)
SPECK_AREA = 40               # dots up to this area may be removed ...
SPECK_DISTANCE = 12           # ... if they are farther than this from any letter (i-dots never are)
TEXT_AREA = 60                # components at least this big count as letters
CONTRAST_PIVOT = 170          # grey level the contrast setting keeps: darker gets darker, lighter lighter


def analyse(gray, covered):
    """(angle in degrees to turn the page so lines are horizontal, is_text).

    `gray` is the page (uint8), `covered` marks scanned pixels.
    """
    scale = min(1.0, ANALYSE_SIZE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(covered.astype(np.uint8), (small.shape[1], small.shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero((small < INK * 255) & mask)
    if len(xs) < 200:
        return 0.0, False
    if len(xs) > 40000:                         # plenty for a profile
        pick = np.random.default_rng(0).choice(len(xs), 40000, replace=False)
        xs, ys = xs[pick], ys[pick]
    xs = xs - xs.mean()
    ys = ys - ys.mean()

    def peakiness(deg):
        # rows are counted in bands of PROFILE_BAND px: fine enough to separate text
        # lines, coarse enough that the vertical strokes of letters don't count
        t = math.radians(deg)
        rows = np.floor((-xs * math.sin(t) + ys * math.cos(t)) / PROFILE_BAND).astype(int)
        hist = np.bincount(rows - rows.min()).astype(np.float64)
        # contrast of the profile, independent of how long it is (the page's
        # extent in that direction) and of the amount of ink
        return float(hist.var() / hist.mean() ** 2)

    coarse = np.arange(-SKEW_RANGE, SKEW_RANGE, 1.0)
    scores = np.array([peakiness(d) for d in coarse])
    best = coarse[int(np.argmax(scores))]
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    fine_scores = [peakiness(d) for d in fine]
    angle = float(fine[int(np.argmax(fine_scores))])
    is_text = max(fine_scores) / float(np.median(scores)) >= TEXT_PEAKINESS
    return angle, is_text


def rotate(gray, covered, angle):
    """Turn the page by `angle` degrees (counter-clockwise), growing it to fit; new area is white."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    out = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpAffine(covered.astype(np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return out, mask


def upright_turns(gray):
    """Quarter turns (counter-clockwise) that make straightened text upright, from
    Tesseract's orientation detection; 0 if unsure or Tesseract is missing."""
    if shutil.which('tesseract') is None:
        return 0
    scale = min(1.0, 2000 / max(gray.shape))
    img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        cv2.imwrite(f.name, img)
        try:
            out = subprocess.run(['tesseract', f.name, 'stdout', '--psm', '0'],
                                 capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return 0
    rotate_by = re.search(r'Rotate:\s*(\d+)', out)
    confidence = re.search(r'Orientation confidence:\s*([\d.]+)', out)
    if not rotate_by or not confidence or float(confidence.group(1)) < OSD_MIN_CONFIDENCE:
        return 0
    # Tesseract reports the clockwise rotation that makes the text upright
    return (-int(rotate_by.group(1)) // 90) % 4


def flatten_background(gray, covered, scale=1.0):
    """Even out the paper: divide by a local paper-brightness estimate, then make
    near-white pure white, and remove isolated specks. Only for text pages.

    `scale` is the image size relative to the full scan resolution (the size of
    letters depends on the sensor, not on how much of the page was scanned).
    """
    k = max(5, int(BACKGROUND_KERNEL * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)      # text removed
    paper = cv2.GaussianBlur(paper, (0, 0), k)
    flat = gray.astype(np.float32) / np.maximum(paper.astype(np.float32), 1.0)
    out = np.clip(flat / WHITE, 0.0, 1.0) * 255
    out[~covered] = 255
    out = out.astype(np.uint8)
    return despeckle(out, scale)


def despeckle(gray, scale=1.0):
    """Remove tiny isolated dark dots (dust, sensor streaks) from a text page.

    A dot is only removed if it is both tiny and away from letters, so i-dots
    and punctuation next to text stay. `scale` as for flatten_background.
    """
    ink = (gray < SPECK_INK).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return gray
    area = stats[:, cv2.CC_STAT_AREA]
    tiny = area <= SPECK_AREA * scale * scale
    tiny[0] = False                             # background
    letters = (area >= TEXT_AREA * scale * scale) & (np.arange(n) > 0)
    distance = cv2.distanceTransform(1 - letters[labels].astype(np.uint8), cv2.DIST_L2, 3)
    # a dot counts as isolated only if all of it is far from letters
    candidates = np.nonzero(tiny)[0]
    if len(candidates) == 0:
        return gray
    nearest = np.asarray(ndimage.minimum(distance, labels, candidates))
    isolated = np.zeros(n, bool)
    isolated[candidates[nearest > SPECK_DISTANCE * scale]] = True
    if not isolated.any():
        return gray
    remove = cv2.dilate(isolated[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = gray.copy()
    out[remove] = 255
    return out


def contrast(gray, amount):
    """Stretch grey levels around CONTRAST_PIVOT by `amount` (1 = unchanged).
    Paper stays white, pale text turns darker and faint grey on the paper lighter."""
    if amount == 1.0:
        return gray
    lut = np.clip(CONTRAST_PIVOT + (np.arange(256) - CONTRAST_PIVOT) * amount, 0, 255)
    return cv2.LUT(gray, lut.astype(np.uint8))


def crop_box(covered, min_fill=0.6):
    """Rows/columns (top, bottom, left, right) after trimming sparsely covered edges."""
    top, bottom, left, right = 0, covered.shape[0], 0, covered.shape[1]
    while bottom - top > 16 and right - left > 16:
        # coverage of each border line within the current box
        fills = {
            'top': covered[top, left:right].mean(),
            'bottom': covered[bottom - 1, left:right].mean(),
            'left': covered[top:bottom, left].mean(),
            'right': covered[top:bottom, right - 1].mean(),
        }
        worst = min(fills, key=fills.get)
        if fills[worst] >= min_fill:
            break
        if worst == 'top':
            top += 1
        elif worst == 'bottom':
            bottom -= 1
        elif worst == 'left':
            left += 1
        else:
            right -= 1
    return top, bottom, left, right
