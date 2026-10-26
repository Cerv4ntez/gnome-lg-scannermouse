"""Decode scan-mode frames from the LG/Dacuda scanner mouse.

In scan mode the firmware hides a mouse record inside each MJPEG frame:
the SOF0 dimension bytes and the last 15 entries of the second quantisation
table carry the data, which makes the JPEG undecodable as delivered. This
module extracts the record and restores the header (logic recovered from
DXCore.dll in the vendor software).
"""

from dataclasses import dataclass

FRAME_WIDTH = 640
FRAME_HEIGHT = 296

BUTTON_LEFT = 0x01
BUTTON_SCAN = 0x02


@dataclass
class MouseRecord:
    counter: int
    x1: int          # sensor 1, signed 16-bit accumulators
    y1: int
    x2: int          # sensor 2
    y2: int
    quality1: int    # surface quality, 0 when the sensor lost tracking
    quality2: int
    buttons: int


def _restore(byte, flag, low):
    """Rebuild one data byte from its DQT entry, a bit-2 flag and two low bits."""
    low &= 3
    base = max(byte - 1, 0) if low >= 2 else byte
    value = base * 4 + low
    return ((value & 0xFB) | ((flag & 1) << 2)) & 0xFF


def _s16(hi, lo):
    v = hi << 8 | lo
    return v - 0x10000 if v & 0x8000 else v


def decode(frame):
    """Return (MouseRecord or None, decodable JPEG bytes)."""
    f = bytearray(frame)
    sof = f.find(b'\xff\xc0')
    dqt1 = f.find(b'\xff\xdb')
    dqt2 = f.find(b'\xff\xdb', dqt1 + 2) if dqt1 >= 0 else -1
    if sof < 0 or dqt2 < 0 or dqt2 + 0x45 > len(f):
        return None, bytes(f)

    counter = f[sof + 5]
    low = f[sof + 6:sof + 9]
    f[sof + 5:sof + 9] = bytes((FRAME_HEIGHT >> 8, FRAME_HEIGHT & 0xFF,
                                FRAME_WIDTH >> 8, FRAME_WIDTH & 0xFF))

    flags = f[dqt2 + 0x36:dqt2 + 0x39]
    raw = f[dqt2 + 0x39:dqt2 + 0x45]
    f[dqt2 + 0x36:dqt2 + 0x45] = b'\x19' * 15

    # Each data byte takes bit 2 from one of the three flag bytes (MSB first,
    # bits 5..1) and its two low bits from one of the three SOF bytes.
    out = []
    for k, b in enumerate(raw):
        flag = flags[k // 5 if k < 10 else 2] >> (5 - (k % 5 if k < 10 else k - 10))
        lo = low[k // 4] >> (6 - 2 * (k % 4))
        out.append(_restore(b, flag, lo))

    rec = MouseRecord(
        counter=counter,
        x1=_s16(out[0], out[1]), y1=_s16(out[2], out[3]),
        x2=_s16(out[4], out[5]), y2=_s16(out[6], out[7]),
        quality1=out[8], quality2=out[9],
        buttons=out[11] & 0x1F,
    )
    return rec, bytes(f)
