"""A made-up scan for testing: a page of random text, and the frames and mouse
records the scanner would deliver when swept over it. No recording needed."""

import math

import cv2
import numpy as np

from scannermouse import stitcher
from scannermouse.framedecode import MouseRecord

FRAME_H, FRAME_W = 240, 640   # the frame after cropping the rows under the housing


def page(width=1500, height=760, seed=0):
    """A page in the sensor's view (0..1 float32, mirrored like the camera image):
    random words on slightly uneven paper."""
    rng = np.random.default_rng(seed)
    ink = np.zeros((height, width), np.uint8)   # (text can only be drawn into 8-bit images)
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHKLMNPRSTUVWXYZ0123456789'
    for y in range(24, height, 26):
        x = 10
        while x < width - 40:
            word = ''.join(rng.choice(list(letters), rng.integers(2, 9)))
            cv2.putText(ink, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
            x += cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] + 12
    paper = 0.88 + cv2.GaussianBlur(rng.normal(0, 0.25, ink.shape).astype(np.float32), (0, 0), 6)   # uneven
    return np.clip(paper * (1 - 0.86 * ink.astype(np.float32) / 255), 0.0, 1.0)


def path(sweeps=3, length=820, speed=12.0, spacing=110, start=(340, 150), wobble=math.radians(1.5)):
    """Poses (x, y, theta) of the frame centre on the page: sweeps to and fro, each a
    bit further down, the mouse turning slightly as a hand does."""
    poses = []
    x, y = start
    for s in range(sweeps):
        direction = 1 if s % 2 == 0 else -1
        for _ in range(int(length / speed)):
            x += direction * speed
            poses.append((x, y))
        if s < sweeps - 1:
            for _ in range(int(spacing / speed)):
                y += speed
                poses.append((x, y))
    n = len(poses)
    return np.array([(px, py, wobble * math.sin(2 * math.pi * k / n)) for k, (px, py) in enumerate(poses)])


def frame(img, pose):
    """What the camera sees at `pose`: an RGB frame as the scanner delivers it."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    cx, cy = (FRAME_W - 1) / 2, (FRAME_H - 1) / 2
    M = np.float32([[c, -s, pose[0] - (c * cx - s * cy)], [s, c, pose[1] - (s * cx + c * cy)]])   # frame -> page
    seen = cv2.warpAffine(img, M, (FRAME_W, FRAME_H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.88)
    lit = np.clip(seen * stitcher._illumination(FRAME_H, FRAME_W), 0, 1)       # the LED's uneven light
    rgb = np.zeros((FRAME_H + stitcher.CROP_TOP, FRAME_W, 3), np.uint8)
    rgb[stitcher.CROP_TOP:, :, 1] = np.round(lit * 255)
    return rgb


def records(poses, quality=100):
    """The mouse records for frames at `poses`: sensor counters that add up the motion
    the way the real sensors report it (see stitcher.COUNTS_TO_PIXELS), a third of a
    frame late."""
    A = np.c_[stitcher.COUNTS_TO_PIXELS, stitcher.COUNTS_TO_RADIANS]          # counts (4) -> shift x, y, turn
    to_counts = np.linalg.pinv(A)                                             # (3, 4): the smallest counts doing that
    total = np.zeros((len(poses), 4))
    for k in range(1, len(poses)):
        step = stitcher.relative(poses[k - 1], poses[k])
        total[k] = total[k - 1] + np.array([-step[0], -step[1], step[2]]) @ to_counts
    out = []
    for k in range(len(poses)):
        late = (1 - stitcher.SENSOR_LAG) * total[k] + stitcher.SENSOR_LAG * total[max(k - 1, 0)]
        x1, y1, x2, y2 = (int(v) for v in ((np.round(late).astype(int) + 0x8000) % 0x10000 - 0x8000))
        out.append(MouseRecord(counter=(k + 7) % 256, x1=x1, y1=y1, x2=x2, y2=y2,
                               quality1=quality, quality2=quality, buttons=0))
    return out


def scan(seed=0, **path_options):
    """(page, poses, [(MouseRecord, rgb frame), ...]) of one made-up scan."""
    img = page(seed=seed)
    poses = path(**path_options)
    return img, poses, list(zip(records(poses), (frame(img, p) for p in poses)))
