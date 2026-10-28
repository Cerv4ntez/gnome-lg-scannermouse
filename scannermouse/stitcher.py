"""Assemble scan frames into one page.

Each frame gets a pose: rotation and position of its centre on the canvas.

The live tracker in this module places every frame as it arrives, for the
preview, using ORB feature matching:

- Tracking: each frame is matched to the previous one, which gives its motion
  including rotation. Where the image has too little detail (blank paper), the
  two mouse sensors fill in.
- Snapping: when a sweep comes back over an area scanned earlier, the frame is
  matched to keyframes there and placed on them.
- Relocation: after the mouse is lifted, frames are matched against all
  keyframes to find the spot again.

Frames that moved are kept as segments by the Mapper (mapper.py), which keeps
re-registering all segments against each other and rebuilding the page from
them in a background process. Each rebuilt page replaces the live one, and the
tracker continues from the corrected positions.

Two pages are kept: one averaging all frames, and one showing for each area the
best frame there (sharpness x distance from the frame edge x confidence).
"""

import collections
import math
import threading

import cv2
import numpy as np

from .mapper import MapperProcess, overlap_agrees

CROP_TOP = 56                 # rows hidden or shadowed by the housing
DPI = 320                     # nominal sensor resolution (LG spec sheet)
GROW = 512                    # canvas growth step

# -- mouse sensors (fallback tracking) -----------------------------------------
QUALITY_MIN = 30              # sensor surface quality below this = not tracking
# Fitted on a 40 s calibration scan. The two sensors sit at different points of
# the mouse, so both are used jointly as a rigid body: inputs are
# [dx1, dy1, dx2, dy2] sensor counts for the frame interval.
# content shift inside the frame (px) = counts @ COUNTS_TO_PIXELS
COUNTS_TO_PIXELS = np.array([[-0.0476, 0.1556],
                             [-0.0637, 0.0677],
                             [0.0470, 0.1602],
                             [-0.2024, -0.0662]])
# change of frame rotation (rad) = counts @ COUNTS_TO_RADIANS
COUNTS_TO_RADIANS = -np.radians(np.array([0.01103, 0.00698, -0.01022, -0.00588]))
# with only one sensor tracking: content shift (px) = that sensor's counts @ SINGLE_TO_PIXELS[n]
SINGLE_TO_PIXELS = (np.array([[-0.0110, 0.2686], [-0.2771, -0.0182]]),
                    np.array([[-0.0045, 0.3268], [-0.2653, 0.0107]]))
# The sensor record in a frame lags its image by about a third of a frame, so a
# frame's motion is blended with the next frame's (frames are processed one late).
SENSOR_LAG = 0.3
SENSOR_MAX_STEP = 200         # px per frame; more is a counter jump (as when scanning starts), not motion

# -- feature matching ------------------------------------------------------------
ORB_FEATURES = 1500
RATIO = 0.8                   # Lowe's ratio test for descriptor matches
RANSAC_PX = 3.0               # reprojection tolerance
TRACK_MIN_INLIERS = 20        # frame-to-frame match accepted
SNAP_MIN_INLIERS = 40         # match to an older keyframe accepted (stricter: a wrong one does damage)
KEYFRAME_SPACING = 80         # px between keyframes
SNAP_MIN_AGE = 30             # frames; younger keyframes are part of the current sweep
SNAP_RADIUS = 350             # px; keyframes this close to the current estimate are tried
SNAP_TRIES = 2                # keyframes tried per frame
SNAP_MAX_JUMP = 200           # px; snaps moving the frame further are wrong matches (repeated words)
MAGNIFICATION_MAX = 2.5       # limit of the scale change tracked while the mouse is lifted
LOST_AFTER = 4                # unplaceable frames in a row before the position counts as lost
RELOCATE_EVERY = 3            # while lost, search all keyframes on every n-th frame
RESUME_AFTER = 8              # tracked frames before painting resumes on new ground after a lift
SEGMENT_SPACING = 8           # px a frame must move to be kept as a segment
SENSOR_LINK_WEIGHT = 0.1      # trust in a segment link that relied on the sensors
# Speed shown while scanning. Fast frames are still used: once tracking is checked
# against the sensors they are placed as well as slow ones, and leaving them out
# made pages worse. Slower scanning gives more overlap between sweeps, though.
SPEED_GOOD = 35               # px/frame (about 8 cm/s)
SPEED_TOO_FAST = 60           # px/frame; false matches on repeated words start around here
SPEED_SMOOTHING = 0.3         # how quickly the displayed speed follows the motion
TRACK_SENSOR_TOLERANCE = 25   # px; feature steps further than this (or half the step) from
                              # the sensors' step are false matches

# -- "best frame per area" compositing ---------------------------------------------
SHARP_WINDOW = 41             # px; local sharpness = mean gradient of the lightly smoothed frame here
SHARP_SMOOTH = 1.0            # Gaussian sigma that keeps sensor noise out of the sharpness measure
SHARP_REDUCE = 4              # sharpness maps are stored at this fraction of the frame size
REPLACE_MARGIN = 1.15         # a frame must score 15% higher to start taking over (no patchwork)
REPLACE_RANGE = 0.3           # ... and takes over fully at 15% + 30% higher
BEST_EDGE = 0.2               # feather weight below which (outer ~6 px) a frame isn't used
# frames not allowed onto the best-frame page
REJECT_SHARPNESS = 0.6        # sharpness below this share of recent frames: defocused / lifted
REJECT_SCALE = 0.03           # image scale off by more than 3%: tilted or lifted
REJECT_SENSOR_QUALITY = 15    # both sensors below this surface quality: lifted
CONFIDENT_INLIERS = 200       # feature matches for full placement confidence
REFINE_MIN_CORRELATION = 0.6  # fine alignment to the averaged page is used if it fits this well ...
REFINE_MAX_SHIFT = 6          # ... and moves the segment at most this many px ...
REFINE_MAX_ANGLE = math.radians(0.6)   # ... and turns it at most this much
CONSENSUS_MIN = 0.4           # correlation with the averaged page a segment needs for the best page
CONSENSUS_TEXTURE = 0.0006    # mean squared detail below which an area counts as blank
REJECTED_WEIGHT = 0.05        # weight of rejected frames on the averaged page
PAPER_LEVEL = 0.92            # paper is scaled to this white level in every frame
WHITE_POINT = 0.95            # output: values above this share of the paper level become white


def _box_blur(img, k):
    """Separable box blur via cumulative sums, edge-padded."""
    r = k // 2
    p = np.pad(img, r, mode='edge')
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = img.shape
    return (c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)


def _edge_weight(h, w, ramp=32):
    """Feathering mask: 1 in the middle, ramping down towards the frame edges."""
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
    return np.minimum.outer(np.minimum(ys, ramp), np.minimum(xs, ramp)).astype(np.float32) / ramp


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def compose(base, rel):
    """Absolute pose from a base pose and a pose relative to it."""
    t = base[:2] + _rot(base[2]) @ rel[:2]
    return np.array([t[0], t[1], base[2] + rel[2]])


def relative(base, pose):
    """`pose` expressed relative to `base` (inverse of compose)."""
    t = _rot(-base[2]) @ (pose[:2] - base[:2])
    return np.array([t[0], t[1], pose[2] - base[2]])


def sharpness(img):
    """(local sharpness map at reduced size, mean sharpness) of a normalised frame."""
    smooth = cv2.GaussianBlur(img, (0, 0), SHARP_SMOOTH)
    grad = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    local = cv2.blur(grad, (SHARP_WINDOW, SHARP_WINDOW)) + 1e-4
    h, w = img.shape
    return cv2.resize(local, (w // SHARP_REDUCE, h // SHARP_REDUCE), interpolation=cv2.INTER_AREA), float(grad.mean())


class Features:
    """ORB keypoints of one frame."""

    _orb = None
    _matcher = None

    def __init__(self, gray):
        if Features._orb is None:
            Features._orb = cv2.ORB_create(ORB_FEATURES, fastThreshold=8)
            Features._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        kps, self.desc = Features._orb.detectAndCompute(gray, None)
        self.last_scale = 1.0
        self.pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        self.centre = np.array([(gray.shape[1] - 1) / 2, (gray.shape[0] - 1) / 2])

    def relative_to(self, ref, min_inliers):
        """This frame's pose relative to `ref`, or None. Also returns the inlier count."""
        if self.desc is None or ref.desc is None or len(self.pts) < min_inliers or len(ref.pts) < min_inliers:
            return None, 0
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.desc, ref.desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < min_inliers:
            return None, len(good)
        src = self.pts[[m.queryIdx for m in good]]
        dst = ref.pts[[m.trainIdx for m in good]]
        A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        inliers = int(mask.sum()) if mask is not None else 0
        if A is None or inliers < min_inliers:
            return None, inliers
        scale = math.hypot(A[0, 0], A[1, 0])
        if not 0.9 < scale < 1.1:                 # the sensor can't zoom: a bad fit
            return None, inliers
        self.last_scale = scale                   # off 1.0 when the mouse is tilted or lifted
        theta = math.atan2(A[1, 0], A[0, 0])
        # ref pixel = R * this pixel + b  ->  centred coords: t = b - c + R c
        t = A[:, 2] - self.centre + _rot(theta) @ self.centre
        return np.array([t[0], t[1], theta]), inliers


class Canvas:
    """The two pages (averaged and best frame per area), growing as needed."""

    def __init__(self):
        self.sum = np.zeros((0, 0), np.float32)
        self.weight = np.zeros((0, 0), np.float32)
        self.best = np.zeros((0, 0), np.float32)
        self.score = np.zeros((0, 0), np.float32)
        self.origin = np.zeros(2)              # canvas index (x, y) of world (0, 0)
        self.bbox = None                       # painted area in canvas index coords (x0, y0, x1, y1)

    def _to_canvas(self, u, v, pose, h, w):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        uu, vv = u - (w - 1) / 2, v - (h - 1) / 2
        return (c * uu - s * vv + pose[0] + self.origin[0], s * uu + c * vv + pose[1] + self.origin[1])

    def _warp(self, pose, h, w):
        """Canvas region (x0, y0, x1, y1) covered by a frame at `pose`, and the
        affine transform from frame pixels into that region."""
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
        cx, cy = self._to_canvas(corners[:, 0], corners[:, 1], pose, h, w)
        x0, y0 = int(math.floor(cx.min())), int(math.floor(cy.min()))
        x1, y1 = int(math.ceil(cx.max())) + 1, int(math.ceil(cy.max())) + 1
        c, s = math.cos(pose[2]), math.sin(pose[2])
        tx = pose[0] + self.origin[0] - x0 - (c * (w - 1) / 2 - s * (h - 1) / 2)
        ty = pose[1] + self.origin[1] - y0 - (s * (w - 1) / 2 + c * (h - 1) / 2)
        return (x0, y0, x1, y1), np.float32([[c, -s, tx], [s, c, ty]])

    def _page_in_frame(self, pose, h, w):
        """The averaged page, resampled into the pixel grid of a frame at `pose`."""
        (x0, y0, x1, y1), M = self._warp(pose, h, w)
        if x0 < 0 or y0 < 0 or x1 > self.sum.shape[1] or y1 > self.sum.shape[0]:
            return None
        s = self.sum[y0:y1, x0:x1]
        wt = self.weight[y0:y1, x0:x1]
        page = np.where(wt > 0.05, s / np.maximum(wt, 1e-6), 1.0).astype(np.float32)
        return cv2.warpAffine(page, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def refine(self, img, pose):
        """Fine-align a frame to the averaged page (sub-pixel), or return `pose` unchanged.

        Segments can sit a pixel or two off; on the best-frame page that shows as
        displaced or doubled letters where the source changes.
        """
        h, w = img.shape
        page = self._page_in_frame(pose, h, w)
        if page is None:
            return pose
        # half resolution is plenty for a small correction, and much faster
        small = lambda a: cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(small(page), small(img), warp, cv2.MOTION_EUCLIDEAN,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4),
                                            None, 5)
        except cv2.error:
            return pose                         # didn't converge
        # warp maps page-grid pixels to frame pixels; the frame belongs at its inverse
        A = cv2.invertAffineTransform(warp)
        A[:, 2] *= 2                            # back to full resolution
        theta = math.atan2(A[1, 0], A[0, 0])
        c = np.array([(w - 1) / 2, (h - 1) / 2])
        t = A[:, :2] @ c + A[:, 2] - c
        if cc < REFINE_MIN_CORRELATION or math.hypot(*t) > REFINE_MAX_SHIFT or abs(theta) > REFINE_MAX_ANGLE:
            return pose
        return compose(pose, np.array([t[0], t[1], theta]))

    def consensus(self, img, pose):
        """Correlation of a frame with the averaged page at its position, or None
        where neither has enough detail to tell (blank paper)."""
        h, w = img.shape
        page = self._page_in_frame(pose, h, w)
        if page is None:
            return None
        a = img - cv2.blur(img, (9, 9))
        b = page - cv2.blur(page, (9, 9))
        ea, eb = float((a * a).mean()), float((b * b).mean())
        if max(ea, eb) < CONSENSUS_TEXTURE:
            return None
        if min(ea, eb) < CONSENSUS_TEXTURE / 4:
            return 0.0                          # blank where the page has text, or the reverse
        return float((a * b).mean()) / math.sqrt(ea * eb)

    def paint(self, img, pose, sharp=None, confidence=0.0, blend=True, weight=1.0):
        """Paint a frame at `pose` onto the averaged page (unless `blend` is False,
        with `weight`) and, given its reduced local sharpness map, onto the
        best-frame page."""
        h, w = img.shape
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
        cx, cy = self._to_canvas(corners[:, 0], corners[:, 1], pose, h, w)
        self._ensure(cx.min(), cy.min(), cx.max(), cy.max())
        cx, cy = self._to_canvas(corners[:, 0], corners[:, 1], pose, h, w)
        x0, y0 = int(math.floor(cx.min())), int(math.floor(cy.min()))
        x1, y1 = int(math.ceil(cx.max())) + 1, int(math.ceil(cy.max())) + 1
        # frame -> this canvas region, as one affine warp of all layers at once
        c, s = math.cos(pose[2]), math.sin(pose[2])
        tx = pose[0] + self.origin[0] - x0 - (c * (w - 1) / 2 - s * (h - 1) / 2)
        ty = pose[1] + self.origin[1] - y0 - (s * (w - 1) / 2 + c * (h - 1) / 2)
        M = np.float32([[c, -s, tx], [s, c, ty]])
        wmask = _edge_weight(h, w)
        layers = [img * wmask, wmask]
        if sharp is not None:
            layers += [img, cv2.resize(sharp, (w, h), interpolation=cv2.INTER_LINEAR)]
        warped = cv2.warpAffine(np.dstack(layers).astype(np.float32), M, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ws = warped[..., 1]
        if blend:
            self.sum[y0:y1, x0:x1] += warped[..., 0] * weight
            self.weight[y0:y1, x0:x1] += ws * weight
            b = self.bbox or (x0, y0, x1, y1)
            self.bbox = (min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1))
        if sharp is None:
            return
        # best-frame page: take over where this frame scores clearly higher,
        # gradually over REPLACE_RANGE so there are no hard seams
        inside = ws > BEST_EDGE                 # the outermost pixels of a frame never win
        new = ws * warped[..., 3] * confidence
        old = self.score[y0:y1, x0:x1]
        ratio = new / np.maximum(old, 1e-9)
        share = np.where(old > 0, (ratio - REPLACE_MARGIN) / REPLACE_RANGE, 1.0)
        share = np.clip(share, 0.0, 1.0) * inside
        best = self.best[y0:y1, x0:x1]
        value = warped[..., 2]
        best += share * (value - best)
        old += share * (new - old)

    def _ensure(self, x0, y0, x1, y1):
        """Grow the canvas so the canvas-index box (x0, y0)-(x1, y1) fits with margin."""
        m = 8
        H, W = self.sum.shape
        left = max(0, -int(math.floor(x0 - m)))
        top = max(0, -int(math.floor(y0 - m)))
        right = max(0, int(math.ceil(x1 + m)) - W)
        bottom = max(0, int(math.ceil(y1 + m)) - H)
        if not (left or top or right or bottom):
            return
        g = lambda v: 0 if v == 0 else (v // GROW + 1) * GROW
        left, top, right, bottom = g(left), g(top), g(right), g(bottom)
        pad = ((top, bottom), (left, right))
        self.sum = np.pad(self.sum, pad)
        self.weight = np.pad(self.weight, pad)
        self.best = np.pad(self.best, pad, constant_values=1.0)
        self.score = np.pad(self.score, pad)
        self.origin += (left, top)
        if self.bbox:
            b = self.bbox
            self.bbox = (b[0] + left, b[1] + top, b[2] + left, b[3] + top)

    def page(self, max_size=None, best=False):
        """(page as 8-bit greyscale, mask of scanned pixels, size relative to full resolution), unmirrored, or None."""
        if self.bbox is None:
            return None
        x0, y0, x1, y1 = self.bbox
        step = 1
        if max_size:
            step = max(1, int(math.ceil(max(y1 - y0, x1 - x0) / max_size)))
        s = self.sum[y0:y1:step, x0:x1:step]
        wt = self.weight[y0:y1:step, x0:x1:step]
        covered = wt > 0.05
        img = np.where(covered, s / np.maximum(wt, 1e-6), 1.0)
        if best:
            # where no segment qualified for the best page, show the averaged one
            chosen = self.score[y0:y1:step, x0:x1:step] > 0
            img = np.where(chosen, self.best[y0:y1:step, x0:x1:step], img)
        if not covered.any():
            return None
        # the sensor sees the page mirrored; paper to white: slight unevenness in
        # the paper clips away, text keeps its contrast
        img = img[:, ::-1] / (WHITE_POINT * PAPER_LEVEL)
        img = np.ascontiguousarray((img * 255).clip(0, 255).astype(np.uint8))
        return img, np.ascontiguousarray(covered[:, ::-1]), 1.0 / step


def render_segments(segments, poses, fit, stats=None):
    """A fresh Canvas from stored segments at the given poses (used by the Mapper).

    First all segments are averaged; that page is the consensus. Then a segment
    only competes for the best-frame page where it agrees with the consensus,
    which keeps out segments that ended up in the wrong place and blank
    segments lying over text.
    """
    canvas = Canvas()
    imgs = [seg.img.astype(np.float32) / 255 for seg in segments]
    for seg, img, pose, ok in zip(segments, imgs, poses, fit):
        # blurred / tilted / lifted / not fitting segments only fill otherwise empty spots
        usable = seg.sharpness is not None and ok is not False
        canvas.paint(img, pose, weight=1.0 if usable else REJECTED_WEIGHT)
    disagreeing = 0
    for seg, img, pose, ok in zip(segments, imgs, poses, fit):
        if seg.sharpness is None or ok is False:
            continue                            # blurred / tilted / lifted, or not fitting
        pose = canvas.refine(img, pose)
        agreement = canvas.consensus(img, pose)
        if agreement is not None and agreement < CONSENSUS_MIN:
            disagreeing += 1
            continue
        canvas.paint(img, pose, seg.sharpness, seg.confidence, blend=False)
    if stats is not None:
        stats['disagreeing'] = disagreeing
    return canvas


class Stitcher:
    def __init__(self):
        self.lock = threading.Lock()
        self.mapper = None
        self.reset()

    def reset(self):
        if self.mapper is None:
            self.mapper = MapperProcess(self._adopt)
        with self.lock:
            self.mapper.reset()
            self.canvas = Canvas()
            self.anchor = None                 # (segment, its pose as the tracker knows it)
            self.pose = np.zeros(3)            # x, y of the frame centre in world coords; rotation (rad)
            self.last = None                   # previous MouseRecord
            self.pending = None                # (record, rgb) held back one frame
            self.illum = None                  # running illumination estimate
            self.prev = None                   # (pose, Features) of the last frame
            self.keyframes = []                # (segment, Features, frame number)
            self.link = None                   # tracked motion since the last segment: [pose, weight] or None
            self.sharp_history = []            # sharpness of recent accepted frames
            self.discarded = collections.Counter()   # frames kept off the best-frame page, by reason
            self.frame_no = 0
            self.frames_used = 0
            self.unsure = 0                    # consecutive frames that couldn't be placed
            self.track_rejected = 0            # feature steps dropped for contradicting the sensors
            self.speed = 0.0                   # recent motion, px per frame (smoothed, for display)
            self.tracked_run = 0               # consecutive frames tracked from the previous one
            self.lost = False                  # position unknown until the page is found again
            self.last_source = None            # what placed the last frame
            self.last_track = None             # inliers of the last frame-to-frame match
            self.last_match = None             # inliers of the last keyframe match
            self.snap_scale = 1.0
            self.magnification = 1.0           # page px per px of the last frame (above 1 while lifted)
            self.velocity = None               # last step, carried on over frames that can't be matched

    def start(self):
        """Start the background re-registration and re-composition."""
        self.mapper.start()

    def close(self):
        """End the background process."""
        self.mapper.close()

    @property
    def pos(self):
        return self.pose[:2]

    @property
    def theta(self):
        return self.pose[2]

    # -- sensors -----------------------------------------------------------

    def _sensor_motion(self, rec, nxt):
        """Pose change reported by the mouse sensors as (content shift px, rotation), or None.

        `nxt` is the following frame's record (or None), used to compensate the
        sensor lag. The laser sensors don't track on glossy or transparent
        surfaces, so the result can be zero even while the mouse moves.
        """
        prev, self.last = self.last, rec
        if prev is None:
            return None
        nxt = nxt or rec
        wrap = lambda v: (v + 0x8000) % 0x10000 - 0x8000

        def delta(a, b):
            return np.array([wrap(b.x1 - a.x1), wrap(b.y1 - a.y1),
                             wrap(b.x2 - a.x2), wrap(b.y2 - a.y2)], float)
        counts = (1 - SENSOR_LAG) * delta(prev, rec) + SENSOR_LAG * delta(rec, nxt)
        ok1 = min(prev.quality1, rec.quality1, nxt.quality1) >= QUALITY_MIN
        ok2 = min(prev.quality2, rec.quality2, nxt.quality2) >= QUALITY_MIN
        if ok1 and ok2:
            return counts @ COUNTS_TO_PIXELS, float(counts @ COUNTS_TO_RADIANS)
        if ok1:
            return counts[:2] @ SINGLE_TO_PIXELS[0], 0.0
        if ok2:
            return counts[2:] @ SINGLE_TO_PIXELS[1], 0.0
        return None

    # -- image processing --------------------------------------------------

    def _normalise(self, green):
        """Flatten the LED illumination so paper comes out uniformly white."""
        img = green[CROP_TOP:].astype(np.float32)
        blur = _box_blur(img, 41)
        if self.illum is None:
            self.illum = blur
        else:
            self.illum += 0.05 * (blur - self.illum)
        flat = img / np.maximum(self.illum, 1.0)
        # bring this frame's paper to the common white level, so neighbouring
        # frames don't differ in brightness (bounded, for frames that are mostly dark)
        paper = float(np.percentile(flat[::4, ::4], 90))
        gain = float(np.clip(PAPER_LEVEL / max(paper, 1e-3), 0.8, 1.25))
        return np.clip(flat * gain, 0.0, 1.0)

    def _snap(self, feats, img, guess):
        """Match older keyframes near `guess` (or all of them if guess is None).

        Keyframe positions come from the Mapper, so they include its corrections.
        Returns the snapped pose or None.
        """
        old = [(self.mapper.pose(k), f, k) for k, f, n in self.keyframes if self.frame_no - n >= SNAP_MIN_AGE]
        if guess is not None:
            old = [k for k in old if np.linalg.norm(k[0][:2] - guess[:2]) < SNAP_RADIUS]
            old.sort(key=lambda k: np.linalg.norm(k[0][:2] - guess[:2]))
            old = old[:SNAP_TRIES]
        best = (None, 0, 1.0)
        for kpose, kfeats, k in old:
            rel, inliers = feats.relative_to(kfeats, SNAP_MIN_INLIERS)
            # repeated text can match features at the wrong place: check the whole overlap
            if rel is not None and inliers > best[1] and overlap_agrees(img, self.mapper.segments[k].img, rel):
                best = (compose(kpose, rel), inliers, feats.last_scale)
        _, self.last_match, self.snap_scale = best
        return best[0]

    def _quality(self, img, rec, inliers, scale):
        """(reduced sharpness map, confidence, reason) for the best-frame page.

        The map is None, with the reason, for frames that must stay off it."""
        sharp_map, sharp = sharpness(img)
        recent = self.sharp_history[-30:]
        reason = None
        if max(rec.quality1, rec.quality2) < REJECT_SENSOR_QUALITY and self.frames_used > 5:
            reason = 'lifted'
        elif abs(scale - 1.0) > REJECT_SCALE:
            reason = 'tilted'
        elif len(recent) >= 10 and sharp < REJECT_SHARPNESS * float(np.median(recent)):
            reason = 'blurred'
        # placement is judged by the Mapper, against all overlapping segments
        confidence = min(1.0, max(inliers, 1) / CONFIDENT_INLIERS)
        if reason:
            return None, confidence, reason
        self.sharp_history = recent + [sharp]
        return sharp_map, confidence, None

    # -- keeping up with the Mapper ----------------------------------------

    def _adopt(self, n, poses, fit, canvas):
        """Adopt a page the Mapper rebuilt from its first n segments, and its
        corrected positions (called from the thread receiving the pages)."""
        scan = self.mapper.scan
        # segments stored after that rebuild started go on top of it, at their
        # corrected positions: most of them here, the last few under the lock
        done = n
        while self.mapper.scan == scan and len(self.mapper.segments) - done > 2:
            end = len(self.mapper.segments)
            for k in range(done, end):
                self._paint_segment(canvas, k, self.mapper.corrected(k, n, poses))
            done = end
        with self.lock:
            if self.mapper.scan != scan:
                return                          # a new scan began meanwhile
            self.mapper.adopt(n, poses, fit)
            for k in range(done, len(self.mapper.segments)):
                self._paint_segment(canvas, k, self.mapper.pose(k))
            self.canvas = canvas
            # continue tracking from the corrected position of the last segment
            if self.anchor is not None:
                k, known = self.anchor
                now = self.mapper.pose(k)
                fix = lambda p: compose(now, relative(known, p))
                self.pose = fix(self.pose)
                if self.prev is not None:
                    self.prev = (fix(self.prev[0]), self.prev[1])
                self.anchor = (k, now)

    def _paint_segment(self, canvas, k, pose):
        seg = self.mapper.segments[k]
        canvas.paint(seg.img.astype(np.float32) / 255, pose, seg.sharpness, seg.confidence,
                     weight=1.0 if seg.sharpness is not None else REJECTED_WEIGHT)

    # -- frame intake ------------------------------------------------------

    def add(self, rec, rgb):
        """Add one decoded frame (HxWx3 uint8) with its mouse record.

        Frames are processed one frame late (see SENSOR_LAG), so this places the
        previous frame. Returns True if that frame was painted. A frame is only
        painted when its position is known; otherwise it is skipped rather than
        smeared.
        """
        if rec is None:
            return False
        pending, self.pending = self.pending, (rec, rgb)
        if pending is None:
            return False
        return self._place(pending[0], pending[1], rec)

    def flush(self):
        """Place the frame still held back, at the end of a scan."""
        pending, self.pending = self.pending, None
        return self._place(pending[0], pending[1], None) if pending else False

    def _place(self, rec, rgb, nxt):
        img = self._normalise(rgb[..., 1])
        gray = (img * 255).astype(np.uint8)
        feats = Features(gray)
        sensor = self._sensor_motion(rec, nxt)
        self.frame_no += 1

        with self.lock:
            # 1. motion since the previous frame: features, else the mouse sensors
            step, source, weight = None, None, 0.0     # pose relative to the previous frame
            track_scale = 1.0
            sensor_step = None
            if sensor is not None and np.any(sensor[0]) and math.hypot(*sensor[0]) <= SENSOR_MAX_STEP:
                sensor_step = np.array([-sensor[0][0], -sensor[0][1], sensor[1]])
            if self.prev is not None:
                step, self.last_track = feats.relative_to(self.prev[1], TRACK_MIN_INLIERS)
                if step is not None and sensor_step is not None:
                    # On text with repeated words, a fast-moving frame can match the
                    # wrong copy of a word and jump by a line or more. The sensors
                    # can't make that mistake, so a step they clearly contradict is
                    # dropped (their own error is ~10% of the step).
                    limit = max(TRACK_SENSOR_TOLERANCE, 0.5 * math.hypot(*sensor_step[:2]))
                    if math.hypot(*(step[:2] - sensor_step[:2])) > limit:
                        step = None
                        self.track_rejected += 1
                if step is not None:
                    source, track_scale = 'features', feats.last_scale
                    weight = float(np.clip(self.last_track / 100.0, 0.4, 3.0))
                    # A lifted mouse sees the page smaller, so a step in the image
                    # covers more of the page (up to ~1.7x in a quick lift between
                    # sweeps). The sensors tell when it is down at the nominal scale;
                    # there, scale changes are tilt and noise and are ignored.
                    if max(rec.quality1, rec.quality2) >= QUALITY_MIN:
                        self.magnification = 1.0
                    else:
                        # where this frame's centre lands in the previous one, scale included
                        shift = step[:2] + (track_scale - 1) * _rot(step[2]) @ feats.centre
                        step = np.array([*(shift * self.magnification), step[2]])
                        self.magnification = float(np.clip(self.magnification * track_scale,
                                                           1 / MAGNIFICATION_MAX, MAGNIFICATION_MAX))
            if step is None and sensor_step is not None and self.prev is not None:
                step, source, weight = sensor_step, 'sensor', SENSOR_LINK_WEIGHT
            if step is not None:
                self.velocity = step
            pose = compose(self.prev[0], step) if step is not None else None
            step_size = math.hypot(step[0], step[1]) if step is not None else 0.0
            self.speed += SPEED_SMOOTHING * (step_size - self.speed)
            self.tracked_run = self.tracked_run + 1 if source == 'features' else 0

            if self.frames_used == 0:
                pose, source = np.zeros(3), 'start'
            elif self.lost:
                # 2a. after a lift: look for this spot among all keyframes
                found = self._snap(feats, gray, None) if self.frame_no % RELOCATE_EVERY == 0 else None
                if found is not None:
                    pose, source, self.lost = found, 'relocated', False
                elif pose is not None and self.tracked_run >= RESUME_AFTER:
                    self.lost = False           # not on the scanned page: new ground, continue
                else:
                    pose = None
            elif pose is not None:
                # 2b. back over an area scanned earlier: place exactly on it, unless
                # that is implausibly far from where tracking says we are
                snapped = self._snap(feats, gray, pose)
                if snapped is not None and math.hypot(*(snapped[:2] - pose[:2])) <= SNAP_MAX_JUMP:
                    pose, source = snapped, source + '+snap'

            # tracked motion since the last segment, for the Mapper; a lift breaks it
            if step is not None and source != 'relocated' and self.link is not None:
                self.link = [compose(self.link[0], step), min(self.link[1], weight)]
            elif pose is not None:
                self.link = None

            # keep tracking relative to this frame even when it isn't painted; until the
            # position counts as lost, assume the motion went on as before
            if pose is not None:
                ref = pose
            elif self.prev is not None and self.velocity is not None and not self.lost and self.unsure < LOST_AFTER:
                ref = compose(self.prev[0], self.velocity)
            else:
                ref = self.prev[0] if self.prev else np.zeros(3)
            self.prev = (ref, feats)
            self.last_source = source if pose is not None else None
            if pose is None:
                self.unsure += 1
                self.link = None
                if self.unsure > LOST_AFTER and self.frames_used:
                    self.lost = True
                return False
            self.unsure = 0
            self.pose = np.array(pose, float)

            # how trustworthy and sharp this frame is, for the best-frame page
            if source == 'start':
                inliers, scale = CONFIDENT_INLIERS, 1.0
            elif 'snap' in source or source == 'relocated':
                inliers, scale = self.last_match, self.snap_scale
            elif source == 'features':
                inliers, scale = self.last_track, track_scale
            else:
                inliers, scale = 0, 1.0         # placed by the sensors alone
            sharp_map, confidence, reason = self._quality(img, rec, inliers, scale)
            if reason:
                self.discarded[reason] += 1
            self.canvas.paint(img, self.pose, sharp_map, confidence,
                              weight=REJECTED_WEIGHT if reason else 1.0)
            self._store(gray, feats, sharp_map, confidence, reason)
            self.frames_used += 1
        return True

    def _store(self, gray, feats, sharp_map, confidence, reason):
        """Hand the frame to the Mapper if it moved enough since the last segment."""
        if self.anchor is not None and self.link is not None:
            moved = self.link[0]
            if math.hypot(moved[0], moved[1]) < SEGMENT_SPACING and abs(moved[2]) < math.radians(0.5):
                return
        link = (self.link[0], self.link[1]) if self.link is not None and self.anchor is not None else None
        k = self.mapper.add(gray, sharp_map, feats, self.pose, link, confidence, reason)
        self.anchor = (k, self.pose.copy())
        self.link = [np.zeros(3), np.inf]
        if all(np.linalg.norm(self.mapper.pose(j)[:2] - self.pose[:2]) >= KEYFRAME_SPACING
               for j, _, _ in self.keyframes):
            self.keyframes.append((k, feats, self.frame_no))

    def finish(self, progress=None):
        """After the scan: final re-registration and rebuild of the whole page."""
        self.flush()
        self.mapper.finish(progress)

    @property
    def misplaced(self):
        """Segments the Mapper found not to fit their neighbours."""
        with self.mapper.lock:
            return sum(1 for f in self.mapper.fit if f is False)

    # -- output -------------------------------------------------------------

    def page(self, max_size=None, best=False):
        """(page as 8-bit greyscale, mask of scanned pixels, size relative to full resolution), unmirrored, or None."""
        with self.lock:
            return self.canvas.page(max_size, best)
