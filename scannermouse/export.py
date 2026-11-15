"""Writing the finished page to a file: PNG, JPEG or PDF, each with the scan resolution."""

import struct
import zlib

import cairo
import cv2
import numpy as np

JPEG_QUALITY = 92


def _png_chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


def png_bytes(gray, dpi):
    """`gray` (uint8) as a PNG file that states its resolution (a pHYs chunk)."""
    ok, buf = cv2.imencode('.png', gray)
    if not ok:
        raise OSError('could not encode the image')
    data = buf.tobytes()
    per_metre = int(round(dpi / 0.0254))
    phys = _png_chunk(b'pHYs', struct.pack('>IIB', per_metre, per_metre, 1))
    end_of_header = 8 + 25                      # signature, then the IHDR chunk
    return data[:end_of_header] + phys + data[end_of_header:]


def jpeg_bytes(gray, dpi):
    """`gray` (uint8) as a JPEG file that states its resolution (in the JFIF header)."""
    ok, buf = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError('could not encode the image')
    data = bytearray(buf.tobytes())
    if data[2:4] == b'\xff\xe0' and data[6:11] == b'JFIF\0':
        data[13:18] = struct.pack('>BHH', 1, int(dpi), int(dpi))     # unit: dots per inch
    return bytes(data)


def save_image(gray, path, dpi):
    """Write a PNG, or a JPEG if the name ends in .jpg / .jpeg."""
    jpeg = path.lower().endswith(('.jpg', '.jpeg'))
    with open(path, 'wb') as f:
        f.write(jpeg_bytes(gray, dpi) if jpeg else png_bytes(gray, dpi))


def save_pdf(gray, path, dpi):
    """Write the page as a one-page PDF at its real size."""
    h, w = gray.shape
    rgb = np.empty((h, w, 4), np.uint8)
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = gray
    rgb[..., 3] = 255
    image = cairo.ImageSurface.create_for_data(memoryview(rgb), cairo.FORMAT_RGB24, w, h, w * 4)
    scale = 72.0 / dpi
    surface = cairo.PDFSurface(path, w * scale, h * scale)
    cr = cairo.Context(surface)
    cr.scale(scale, scale)
    cr.set_source_surface(image, 0, 0)
    cr.paint()
    surface.finish()
