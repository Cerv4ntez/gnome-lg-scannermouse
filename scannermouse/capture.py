"""Raw MJPEG frame capture from the scanner via GStreamer."""

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf, Gst  # noqa: E402

import numpy as np  # noqa: E402

Gst.init(None)


def jpeg_to_rgb(data):
    """Decode JPEG bytes into an HxWx3 uint8 array."""
    loader = GdkPixbuf.PixbufLoader.new_with_type('jpeg')
    loader.write(data)
    loader.close()
    pb = loader.get_pixbuf()
    w, h, n = pb.get_width(), pb.get_height(), pb.get_n_channels()
    rows = np.frombuffer(pb.get_pixels(), np.uint8).reshape(h, pb.get_rowstride())
    return rows[:, :w * n].reshape(h, w, n)[..., :3]


class Capture:
    """Streams frames from a V4L2 device; on_frame(bytes) runs on a GStreamer thread."""

    def __init__(self, device, on_frame):
        self.on_frame = on_frame
        self.pipeline = Gst.parse_launch(
            f'v4l2src device={device} ! image/jpeg ! '
            'appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false')
        self.pipeline.get_by_name('sink').connect('new-sample', self._on_sample)

    def _on_sample(self, sink):
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                self.on_frame(bytes(info.data))
            finally:
                buf.unmap(info)
        return Gst.FlowReturn.OK

    def start(self):
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('could not start the video stream')

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
