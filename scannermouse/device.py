"""Control of the LG/Dacuda scanner mouse (USB 043e:70d7, OmniVision OV550 bridge).

The scanner is a UVC camera plus a HID mouse. Scan mode is switched through
the vendor extension unit: selector 2 gives access to OV550 registers, and
OV550 register 0xd8 is a mailbox to the mouse microcontroller. Sequences are
the ones used by the vendor driver (DLL_OvtApi.dll / OVCamUVC_Ext.dll).
"""

import ctypes
import fcntl
import glob
import os
import time

XU_UNIT = 6
XU_SEL_REGISTER = 2
UVC_SET_CUR = 0x01
UVC_GET_CUR = 0x81
UVCIOC_CTRL_QUERY = (3 << 30) | (16 << 16) | (ord('u') << 8) | 0x21

REG_MAILBOX = 0xd8
REG_INIT = 0x2d
# Exposure: page brightness is roughly linear in this value (about 15 grey levels per
# step); white paper clips above ~0x10. The vendor driver's default is 0x05.
REG_EXPOSURE = 0xe6
EXPOSURE_MIN, EXPOSURE_MAX, EXPOSURE_DEFAULT = 0x02, 0x10, 0x05
MCU_PREPARE, MCU_READY = 0x00, 0xff
MCU_START, MCU_STARTED = 0xe1, 0x99
MCU_STOP, MCU_STOPPED = 0xe0, 0x11


class _Query(ctypes.Structure):
    _fields_ = [('unit', ctypes.c_uint8), ('selector', ctypes.c_uint8),
                ('query', ctypes.c_uint8), ('size', ctypes.c_uint16),
                ('data', ctypes.c_void_p)]


class ScannerError(Exception):
    pass


def find_device():
    """Path of the scanner's video capture node, or None."""
    for node in sorted(glob.glob('/sys/class/video4linux/video*')):
        try:
            with open(node + '/name') as f:
                name = f.read()
            with open(node + '/index') as f:
                index = f.read().strip()
        except OSError:
            continue
        if 'Mouse Scanner' in name and index == '0':
            return '/dev/' + os.path.basename(node)
    return None


class Scanner:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDWR)
        self.scanning = False

    def close(self):
        if self.scanning:
            try:
                self.stop_scan()
            except (OSError, ScannerError):
                pass
        os.close(self.fd)

    # -- extension unit ---------------------------------------------------

    def _xu(self, query, data):
        buf = (ctypes.c_uint8 * len(data))(*data)
        q = _Query(XU_UNIT, XU_SEL_REGISTER, query, len(data), ctypes.cast(buf, ctypes.c_void_p))
        # uvcvideo can briefly report EACCES while the device is busy
        for _ in range(5):
            try:
                fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, q)
                return bytes(buf)
            except PermissionError:
                time.sleep(0.05)
        fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, q)
        return bytes(buf)

    def read_register(self, reg):
        self._xu(UVC_SET_CUR, [0x80, reg, 0, 0, 0])
        r = self._xu(UVC_GET_CUR, [0] * 5)
        if r[1] != reg:
            raise ScannerError(f'register 0x{reg:02x}: unexpected reply {r.hex()}')
        return r[3]

    def write_register(self, reg, value):
        self._xu(UVC_SET_CUR, [0x00, reg, 0, value, 0])

    def _write_verified(self, reg, value):
        for _ in range(10):
            self.write_register(reg, value)
            if self.read_register(reg) == value:
                return
            time.sleep(0.01)
        raise ScannerError(f'register 0x{reg:02x} did not accept 0x{value:02x}')

    def _mailbox(self, command, reply):
        self.write_register(REG_MAILBOX, command)
        for _ in range(100):
            if self.read_register(REG_MAILBOX) == reply:
                return
        raise ScannerError(f'mouse controller did not acknowledge command 0x{command:02x}')

    # -- scan mode --------------------------------------------------------

    def start_scan(self):
        """Switch the sensor and LED on. The video stream must already be running."""
        self._mailbox(MCU_PREPARE, MCU_READY)
        self._write_verified(REG_INIT, 0x60)
        self._mailbox(MCU_START, MCU_STARTED)
        self.scanning = True

    def set_exposure(self, value):
        self._write_verified(REG_EXPOSURE, max(EXPOSURE_MIN, min(EXPOSURE_MAX, int(value))))

    def stop_scan(self):
        """Return to normal mouse operation."""
        self.scanning = False
        self._mailbox(MCU_STOP, MCU_STOPPED)
