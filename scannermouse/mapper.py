"""Continuous re-registration and re-composition of a scan.

The live tracker (stitcher.py) places each frame once, as it arrives. Small
errors in that placement accumulate, and a sweep that has drifted lays text down
a second time. The Mapper keeps every segment (frame image, features, quality)
and runs two background workers while the scan goes on:

- Registration matches each segment against all segments that may overlap it:
  those near its current position, and any found by feature lookup (which also
  catches large drift). When idle it re-checks older segments. Every couple of
  seconds it solves for the positions that agree best with all matches;
  matches that disagree with the rest are down-weighted, and segments that
  still don't fit their neighbours are marked as misplaced.
- Composition rebuilds the page from all segments at their current positions
  whenever those change: one page averaging all segments, and one showing, for
  each area, the best segment there.

So the more of the page is scanned, the more segments are tied together and
the more accurate the page becomes.
"""

import collections
import math
import threading
import time

import cv2
import numpy as np

from . import align

GAP = 15                      # segments apart before a match counts as an overlap (not a neighbour)
NEAR_RADIUS = 600             # px; segments this close to the current estimate are tried
NEAR_TRIES = 6                # nearby candidates per segment, from different sweeps
SWEEP_SPACING = 15            # segments between those candidates
LOOKUP_TRIES = 3              # candidates per segment from feature lookup
LOOKUP_FEATURES = 300         # strongest features per segment in the lookup table
LOOKUP_EVERY = 3              # every n-th segment goes into the lookup table
LOOKUP_VOTES = 15             # feature votes needed for a lookup candidate
MATCH_INLIERS = 40            # geometric verification for an overlap
OVERLAP_MIN_AREA = 0.15       # share of the frame two segments must overlap to be linked
OVERLAP_MIN_CORRELATION = 0.15  # correlation over the whole overlap for a link to count (fast,
                              # blurred scans give true overlaps as low as ~0.2; misfits stay near 0)
PLAUSIBLE_PX = 200            # an overlap may move a segment at most this far from its current
                              # estimate: on text with repeated words, most feature matches link
                              # different places that share words, typically 400-1000 px off
SOLVE_EVERY = 1.5             # s between position solves
COMPOSE_EVERY = 3.0           # s at least between page rebuilds (they compete with live tracking)
MISPLACED_PX = 12             # median disagreement with neighbours that marks a segment misplaced
ODOMETRY_WEIGHT = 0.1         # weight of sensor-only links between consecutive segments
RATIO = 0.8


def _weight(inliers):
    """Trust in a feature match, growing with its inlier count."""
    return float(np.clip(inliers / 100.0, 0.4, 3.0))


def overlap_agrees(img, ref, rel):
    """Whether frame `img`, placed at `rel` relative to frame `ref`, matches it over
    their whole overlap.

    Feature matches alone can link two different places that happen to show the
    same words (a document repeating an address, say). There only the repeated
    line matches; the lines around it don't, so the correlation over the full
    overlap stays low.
    """
    h, w = ref.shape
    c, s = math.cos(rel[2]), math.sin(rel[2])
    cx, cy = (w - 1) / 2, (h - 1) / 2
    M = np.float32([[c, -s, rel[0] + cx - (c * cx - s * cy)],
                    [s, c, rel[1] + cy - (s * cx + c * cy)]])
    a = img.astype(np.float32)
    b = ref.astype(np.float32)
    a = cv2.warpAffine(a - cv2.blur(a, (9, 9)), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    inside = cv2.warpAffine(np.ones_like(img, np.float32), M, (w, h), borderValue=0) > 0.99
    inside = cv2.erode(inside.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if inside.mean() < OVERLAP_MIN_AREA:
        return False                            # too little overlap to tell
    b = (b - cv2.blur(b, (9, 9)))[inside]
    a = a[inside]
    energy = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if energy < 1e-6:
        return True                             # blank paper: nothing contradicts
    return float((a * b).sum()) / energy >= OVERLAP_MIN_CORRELATION


class Segment:
    __slots__ = ('img', 'sharpness', 'feats', 'confidence', 'reason')

    def __init__(self, img, sharpness, feats, confidence, reason):
        self.img = img                # uint8 normalised frame
        self.sharpness = sharpness    # local sharpness map (reduced size), or None if not usable
        self.feats = feats            # stitcher.Features
        self.confidence = confidence  # placement confidence from the live tracker, 0..1
        self.reason = reason          # why it isn't used on the best-frame page, or None


class Mapper:
    def __init__(self, render):
        """`render(segments, poses, fit, stats)` builds page canvases; called from the composition worker."""
        self.render = render
        self.lock = threading.Lock()
        self.segments = []
        self.linked = []                        # per segment: tracked from the previous one (not after a lift)
        self.poses = []                         # current best positions, (x, y, theta)
        self.edges = []                         # (i, j, pose of j relative to i, weight)
        self.tested = set()
        self.fit = []                           # per segment: True / False (misplaced) / None (unknown)
        self.pending = collections.deque()      # segments not yet searched for overlaps
        self.recheck = 0                        # round-robin position for re-checking old segments
        self.lookup = []                        # (segment, strongest descriptors)
        self.pose_version = 0
        self.rendered = None                    # (pose version, number of segments, canvases)
        self.stats = collections.Counter()
        self.running = False
        self.threads = []

    # -- input from the live tracker ------------------------------------------

    def add(self, img, sharpness, feats, pose, link, confidence, reason=None):
        """Store a segment and return its index.

        `link` = (pose relative to the previous segment, weight) from frame-to-frame
        tracking, or None after a lift. A linked segment starts from the previous
        segment's corrected position; otherwise from `pose`.
        """
        with self.lock:
            i = len(self.segments)
            self.segments.append(Segment(img, sharpness, feats, confidence, reason))
            self.linked.append(link is not None and i > 0)
            if link is not None and i > 0:
                rel = np.array(link[0], float)
                self.poses.append(_compose(self.poses[i - 1], rel))
                self.edges.append((i - 1, i, rel, link[1]))
            else:
                self.poses.append(np.array(pose, float))
            self.fit.append(None)
            self.pending.append(i)
        return i

    def pose(self, index):
        with self.lock:
            return self.poses[index].copy()

    # -- workers -------------------------------------------------------------

    def start(self):
        self.running = True
        self.threads = [threading.Thread(target=self._register_loop, daemon=True),
                        threading.Thread(target=self._compose_loop, daemon=True)]
        for t in self.threads:
            t.start()

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join()
        self.threads = []

    def finish(self, progress=None):
        """After the scan: search all remaining overlaps, solve, and render once more."""
        self.stop()
        total = max(len(self.pending), 1)
        while self.pending:
            self._search(self.pending.popleft())
            if progress:
                progress(0.6 * (1 - len(self.pending) / total))
        for i in range(len(self.segments)):     # one more pass with the corrected positions
            self._search(i, lookup=False)
            if progress and i % 25 == 0:
                progress(0.6 + 0.2 * i / len(self.segments))
        self._solve()
        if progress:
            progress(0.9)
        self._compose()
        if progress:
            progress(1.0)

    def _register_loop(self):
        last_solve = time.monotonic()
        while self.running:
            if self.pending:
                self._search(self.pending.popleft())
            elif self.segments:
                # idle: re-check an older segment against its current neighbours
                self.recheck = (self.recheck + 1) % len(self.segments)
                self._search(self.recheck, lookup=False)
                time.sleep(0.002)
            else:
                time.sleep(0.05)
            if time.monotonic() - last_solve > SOLVE_EVERY:
                self._solve()
                last_solve = time.monotonic()

    def _compose_loop(self):
        last = 0.0
        while self.running:
            stale = self.rendered is None or self.rendered[0] != self.pose_version
            if stale and time.monotonic() - last >= COMPOSE_EVERY:
                last = time.monotonic()
                self._compose()
            else:
                time.sleep(0.05)

    # -- registration -----------------------------------------------------------

    def _search(self, i, lookup=True):
        """Match segment i against older segments that may overlap it."""
        with self.lock:
            if i >= len(self.segments):
                return
            seg = self.segments[i]
            here = self.poses[i][:2].copy()
            centres = np.array([p[:2] for p in self.poses[:max(i - GAP, 0)]])
        if seg.feats.desc is None:
            return
        cands = []
        if len(centres):
            d = np.linalg.norm(centres - here, axis=1)
            for j in np.argsort(d):
                if d[j] > NEAR_RADIUS or len(cands) == NEAR_TRIES:
                    break
                if all(abs(j - c) > SWEEP_SPACING for c in cands):
                    cands.append(int(j))
        if lookup:
            cands += [j for j in self._lookup(i, seg) if j not in cands]
            if i % LOOKUP_EVERY == 0:
                self.lookup.append((i, seg.feats.desc[:LOOKUP_FEATURES]))
        for j in cands:
            if (j, i) in self.tested:
                continue
            self.tested.add((j, i))
            rel, inliers = seg.feats.relative_to(self.segments[j].feats, MATCH_INLIERS)
            self.stats['tested'] += 1
            if rel is None:
                continue
            if self.linked[i]:
                with self.lock:
                    expected = align.relative(self.poses[j], self.poses[i])
                if math.hypot(rel[0] - expected[0], rel[1] - expected[1]) > PLAUSIBLE_PX:
                    self.stats['implausible'] += 1
                    continue
            if not overlap_agrees(seg.img, self.segments[j].img, rel):
                self.stats['repeats rejected'] += 1
                continue
            with self.lock:
                self.edges.append((j, i, rel, _weight(inliers)))
            self.stats['overlaps'] += 1

    def _lookup(self, i, seg):
        """Older segments sharing many features with segment i, wherever they are."""
        entries = [(k, d) for k, d in self.lookup if k < i - GAP]
        if not entries:
            return []
        table = np.vstack([d for _, d in entries])
        owner = np.concatenate([np.full(len(d), k) for k, d in entries])
        votes = collections.Counter()
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(seg.feats.desc[:LOOKUP_FEATURES], table, k=2):
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                votes[int(owner[pair[0].trainIdx])] += 1
        return [k for k, v in votes.most_common(LOOKUP_TRIES) if v >= LOOKUP_VOTES]

    def _solve(self):
        with self.lock:
            n = len(self.poses)
            edges = list(self.edges)
            start = np.array(self.poses)
        if n < 3 or not edges:
            return
        poses = align.solve(start, edges)
        # how well each segment agrees with its overlaps after solving
        errors = collections.defaultdict(list)
        for i, j, meas, w in edges:
            if j - i > 1:                       # overlaps only, not consecutive links
                r = align.relative(poses[i], poses[j]) - meas
                e = math.hypot(r[0], r[1])
                errors[i].append(e)
                errors[j].append(e)
        with self.lock:
            # segments added during the solve keep their live offset to the last solved one
            for k in range(n, len(self.poses)):
                self.poses[k] = _compose(poses[n - 1], align.relative(start[n - 1], self.poses[k]))
            self.poses[:n] = list(poses)
            for k in range(n):
                if errors.get(k):
                    self.fit[k] = float(np.median(errors[k])) <= MISPLACED_PX
            self.pose_version += 1

    # -- composition ---------------------------------------------------------

    def _compose(self):
        with self.lock:
            version = self.pose_version
            n = len(self.segments)
            segments = self.segments[:n]
            poses = [p.copy() for p in self.poses[:n]]
            fit = list(self.fit[:n])
        if n == 0:
            return
        canvases = self.render(segments, poses, fit, self.stats)
        self.rendered = (version, n, canvases)


def _compose(base, rel):
    t = base[:2] + align._rot(base[2]) @ rel[:2]
    return np.array([t[0], t[1], base[2] + rel[2]])
