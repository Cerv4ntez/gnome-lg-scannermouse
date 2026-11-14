"""XFeat keypoints and descriptors (Potje et al., "XFeat: Accelerated Features for
Lightweight Image Matching", CVPR 2024), used by the Mapper to link segments.

ORB finds few reliable matches where two sweeps overlap in dark or low-contrast
areas (photos); XFeat's learned features find many more. The network runs in
OpenCV's DNN module from data/xfeat.onnx, which tools/xfeat_onnx.py builds from
the authors' weights (verlab/accelerated_features, Apache-2.0, see
data/LICENSE-XFeat). Keypoint selection and matching follow their reference code.
"""

import math
import os
import threading

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(__file__), 'data', 'xfeat.onnx')
INPUT_SIZE = (640, 224)        # (w, h) the network was exported for; frames are resized to it
TOP_K = 1500                   # keypoints kept per frame
DETECTION_THRESHOLD = 0.05
MIN_COSSIM = 0.82              # mutual nearest neighbours must be at least this similar
RANSAC_PX = 3.0

_net = None
_lock = threading.Lock()


def available():
    return os.path.exists(MODEL)


def _network():
    global _net
    with _lock:
        if _net is None:
            _net = cv2.dnn.readNetFromONNX(MODEL)
        return _net


def _bilinear(a, fx, fy):
    """Sample a (H, W[, C]) map at fractional positions, clamped at the border."""
    x0 = np.clip(np.floor(fx).astype(int), 0, a.shape[1] - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, a.shape[0] - 1)
    x1, y1 = np.minimum(x0 + 1, a.shape[1] - 1), np.minimum(y0 + 1, a.shape[0] - 1)
    ax, ay = np.clip(fx - np.floor(fx), 0, 1), np.clip(fy - np.floor(fy), 0, 1)
    if a.ndim == 3:
        ax, ay = ax[:, None], ay[:, None]
    return (a[y0, x0] * (1 - ax) * (1 - ay) + a[y0, x1] * ax * (1 - ay)
            + a[y1, x0] * (1 - ax) * ay + a[y1, x1] * ax * ay)


class XFeatures:
    """Keypoints (N, 2, frame pixels) and unit descriptors (N, 64) of one frame."""

    def __init__(self, gray, keep=None):
        """`keep`: optional (h, w) map, keypoints only where it is above 0.5."""
        h0, w0 = gray.shape
        w, h = INPUT_SIZE
        img = cv2.resize(gray.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        img = (img - img.mean()) / math.sqrt(float(img.var()) + 1e-5)
        net = _network()
        with _lock:                             # one forward pass at a time per network
            net.setInput(img[None, None].astype(np.float32))
            feats, logits, heat = net.forward(['feats', 'logits', 'heat'])
        feats, logits, heat = feats[0].transpose(1, 2, 0), logits[0].transpose(1, 2, 0), heat[0, 0]
        feats = feats / (np.linalg.norm(feats, axis=2, keepdims=True) + 1e-12)
        # keypoint logits per 8x8 cell (64 positions + "none") -> full-resolution heatmap
        e = np.exp(logits - logits.max(axis=2, keepdims=True))
        p = (e / e.sum(axis=2, keepdims=True))[..., :64]
        hm = p.reshape(h // 8, w // 8, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)
        ys, xs = np.nonzero((hm == cv2.dilate(hm, np.ones((5, 5), np.uint8))) & (hm > DETECTION_THRESHOLD))
        # feature-map coordinates as in the reference (grid_sample, align_corners=False)
        fx, fy = xs * (w // 8) / (w - 1) - 0.5, ys * (h // 8) / (h - 1) - 0.5
        scores = hm[ys, xs] * _bilinear(heat, fx, fy)
        if keep is not None:
            scores[keep[np.minimum(ys * h0 // h, h0 - 1), np.minimum(xs * w0 // w, w0 - 1)] <= 0.5] = 0
            scores[scores == 0] = -1
        best = np.argsort(-scores)[:TOP_K]
        best = best[scores[best] > 0]
        desc = _bilinear(feats, fx[best], fy[best])
        self.desc = (desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
        self.pts = np.c_[xs[best] * w0 / w, ys[best] * h0 / h].astype(np.float32)
        self.centre = np.array([(w0 - 1) / 2, (h0 - 1) / 2])

    def matches(self, other):
        """Index pairs (mine, other's) of mutual nearest neighbours."""
        if len(self.desc) == 0 or len(other.desc) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        # unit vectors: the nearest by L2 is the most similar, and cos = 1 - d^2 / 2
        pairs = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(self.desc, other.desc)
        limit = math.sqrt(2 * (1 - MIN_COSSIM))
        good = [(m.queryIdx, m.trainIdx) for m in pairs if m.distance < limit]
        if not good:
            return np.zeros(0, int), np.zeros(0, int)
        i, j = np.array(good).T
        return i, j

    def relative_to(self, ref, min_inliers):
        """This frame's pose relative to `ref` (x, y, theta), or None; and the inlier count."""
        i, j = self.matches(ref)
        if len(i) < min_inliers:
            return None, len(i)
        A, mask = cv2.estimateAffinePartial2D(self.pts[i], ref.pts[j], method=cv2.RANSAC,
                                              ransacReprojThreshold=RANSAC_PX)
        inliers = int(mask.sum()) if mask is not None else 0
        if A is None or inliers < min_inliers or not 0.9 < math.hypot(A[0, 0], A[1, 0]) < 1.1:
            return None, inliers
        theta = math.atan2(A[1, 0], A[0, 0])
        c, s = math.cos(theta), math.sin(theta)
        # as stitcher.Features: ref pixel = R * this pixel + b  ->  t = b - c + R c
        t = A[:, 2] - self.centre + np.array([[c, -s], [s, c]]) @ self.centre
        return np.array([t[0], t[1], theta]), inliers
