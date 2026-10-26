# Scanner Mouse

A GNOME app for the LG Smart Scan mouse (USB `043e:70d7`, sold as LSM-100/LSM-150).
LG only ever shipped Windows and macOS software; this app drives the scanner directly
on Linux.

## Install

On Arch Linux and derivatives, build and install the package with the included
`PKGBUILD`:

```sh
git clone https://github.com/Cerv4ntez/gnome-lg-scannermouse.git
cd gnome-lg-scannermouse
makepkg -si
```

This adds **Scanner Mouse** to the app grid and a `scannermouse` command. To run it
from a checkout without installing, use `python3 -m scannermouse`.

Requires GTK 4, libadwaita, GStreamer (`v4l2src`), PyGObject, NumPy, SciPy, OpenCV
(`python-opencv`) and pycairo. Optional: Tesseract with its `osd` data, to turn
sideways or upside-down text pages upright.
Your user must be able to open the scanner's `/dev/videoN` node (the `video` group).

## Use

1. Put the mouse on the page and press **Start Scan**.
2. Sweep slowly over the page. The mouse stops moving the cursor while scanning. A
   dot next to the live view shows the speed: green is fine, yellow is getting
   fast, red means slow down.
3. Press the **Scan** button on the mouse (or Esc) to finish.
4. Turn the page upright with the rotate buttons. **Crop**, on by default, trims the
   ragged edges. Then save as PNG or PDF.

Output is greyscale: the scanner lights the page with a green LED and the sensor
records no usable blue, so colour cannot be recovered.

**Best frame per area** (settings menu, on by default) chooses how frames are
combined. Every scan builds two pages at once, so this can be switched at any time,
even after scanning:

- **On:** each part of the page shows the frame that scored best there. The score is
  local sharpness × distance from the frame edge × placement confidence. A new frame
  only takes over where it scores at least 15% higher, so passing over an area again
  can only improve it.
- **Off:** all frames are averaged. This is smoother but can show faint doubled text.

Frames that look blurred (lifted), show a changed magnification (tilted), come with
both sensors off the surface, or don't match the text already scanned under them
(misplaced) are left out of the best-frame page. The status line counts them by reason.

**Text pages** ([`cleanup.py`](scannermouse/cleanup.py), settings menu, both on by default):

- **Straighten text pages.** If the page is mostly text, it is turned so the lines
  are level. Rotated to the angle of its lines, a text page has a strongly peaked
  row profile. After the scan, Tesseract's orientation detection (`tesseract`
  with `osd` data) turns it upright if it is sideways or upside down. The rotate
  buttons still apply on top.
- **Even out background.** The paper brightness is estimated by removing the text
  locally and smoothing; dividing by it removes broad blotches, and near-white
  becomes white. Tiny dark specks far from any letter are removed, while i-dots
  and punctuation stay. This applies to the finished page; the live preview is only
  straightened.

**Text recognition** ([`ocr.py`](scannermouse/ocr.py)). After a scan, Tesseract reads
the finished page in the background. The result appears in the **Text** tab, where it
can be edited and copied (**Copy All**), or saved as a `.txt` file.
- **Layout:** paragraphs, blank lines, indentation, and tabs for wide gaps such as
  side-by-side signatures.
- **Styles:** bold, measured from stroke thickness, and larger text, from line height.
- **Uncertain words** are dimmed and underlined.
- **Cleanup:** specks read as letters and pure scribbles are dropped.
- **PDF:** when text has been recognised, a saved PDF gets an invisible text layer,
  so it is searchable and copyable in any PDF viewer.
- **Languages:** German and English are used when installed (`tesseract-data-deu`
  for German).
- **Re-reading:** turning the page with the rotate buttons reads it again.

**Exposure** is in the settings menu. In automatic mode (the default) it is measured
during the first quarter second of each scan and then held fixed. Manual values run
from 2 to 16; the vendor driver used 5, which leaves the image dark and noisy.
Exposure is OV550 register `0xe6`. The other vendor setting, gain (`0xdc`), has no
visible effect, so the app doesn't offer it.

## How it works

The hardware is an OmniVision OV550 UVC camera with a mouse microcontroller.
The protocol below was recovered from the vendor driver DLLs.

- **Scan mode.** UVC extension unit 6 (`dd880f8a-1cba-4954-8a25-f7875967f0f7`),
  selector 2, reads and writes OV550 registers. Register `0xd8` is a mailbox to the
  mouse controller: write `0x00` and wait for `0xff`, write-verify register `0x2d = 0x60`,
  then write `0xe1` and wait for `0x99` to start. Write `0xe0` and wait for `0x11` to stop.
- **Motion data.** Scan-mode frames are 640×296 MJPEG with a record hidden in the
  JPEG header: the SOF0 dimension bytes and the end of the second quantisation table.
  It holds a frame counter, X/Y counters from two motion sensors, their surface quality
  and the button state ([`framedecode.py`](scannermouse/framedecode.py)).
- **Stitching** ([`stitcher.py`](scannermouse/stitcher.py)) uses OpenCV ORB features.
  - **Tracking.** Each frame is matched to the previous one, which gives its movement
    and rotation. The two laser sensors are the fallback where there is too little
    detail, for example on blank paper. They often report no motion on glossy surfaces.
  - **Snapping.** Keyframes are kept as the scan proceeds. When a sweep comes back
    over an area scanned earlier, the frame is matched to those keyframes and placed
    exactly, so drift doesn't accumulate.
  - **Unknown position.** A frame is only painted when its position is known.
    After the mouse is lifted, it is matched against all keyframes to find the spot
    before painting resumes.
  - **Sensor lag.** The sensor reading in a frame lags its image by about a third of
    a frame, so frames are processed one frame late to interpolate.
- **Continuous re-registration** ([`mapper.py`](scannermouse/mapper.py)). Every
  frame that moved is kept as a segment, and two background workers run during
  the scan.
  - **Registration.** Each segment is matched against all older segments that may
    overlap it: those near its current position, and any found by feature lookup.
    When idle, the worker re-checks older segments. Every 1.5 s it solves for the
    positions that agree best with all matches
    ([`align.py`](scannermouse/align.py)), down-weighting outliers. Segments that
    still disagree with their neighbours are marked as not fitting.
  - **Repeated text.** Documents repeat words, and most feature matches between
    non-neighbouring segments link *different* places that share words: 83% on one
    test scan. Three checks keep them out:
    - an overlap may move a segment at most 200 px from its current estimate;
    - the two segments must agree over their whole overlap;
    - each frame-to-frame step is checked against the mouse sensors, which can't
      confuse two copies of a word. A step jumping by a line or more is replaced by
      the sensor step.
  - **Composition.** Whenever positions change (at most every 3 s), the page is
    rebuilt from all segments, and the preview switches to it. The averaged page
    serves as the consensus: a segment only competes for the best-frame page where
    it agrees with it. That keeps out stray patches and blank segments lying over
    text, and where no segment qualifies, the averaged page shows through. Live tracking continues from the
    corrected positions, so the page gets more accurate as the scan goes on.
  - **On stop,** a final registration pass and rebuild run over everything.

Selector 3 of the same unit reads and writes the firmware flash. The app never uses it.

## Limitations

- The background workers compete with live tracking for Python's interpreter lock,
  so 10–25% of frames are dropped on a fast scan. Coverage is unaffected, because
  consecutive frames overlap heavily, but the preview is less smooth.
- The sensor fallback's constants were calibrated from one 40 s test scan. They are
  not read from the calibration data stored in the mouse's EEPROM.
- Keep the mouse on the page. If it is lifted and set down somewhere the keyframes
  don't cover, scanning continues from a best guess.
- The raw frames of the last 3 scans are kept in `~/.cache/scannermouse/`, so a problem
  scan can be replayed and diagnosed.
- Set `SCANNERMOUSE_DEBUG=/some/dir` to record the raw frames and a per-frame log of
  each scan, for diagnosing stitching problems.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE).
