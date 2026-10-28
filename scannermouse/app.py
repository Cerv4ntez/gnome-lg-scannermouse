"""GNOME application for the LG scanner mouse."""

import collections
import json
import os
import sys
import threading
import time

import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
gi.require_version('Gdk', '4.0')
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

import cairo  # noqa: E402
import numpy as np  # noqa: E402

from . import cleanup, framedecode, ocr, stitcher  # noqa: E402
from .capture import Capture, jpeg_to_rgb  # noqa: E402
from .device import (EXPOSURE_DEFAULT, EXPOSURE_MAX, EXPOSURE_MIN,  # noqa: E402
                     Scanner, ScannerError, find_device)

APP_ID = 'io.github.scannermouse.ScannerMouse'
PREVIEW_SIZE = 1600
WARMUP = 0.5          # seconds of stream before switching the sensor on
DARK_FRAME = 10       # mean green level below which a frame is unlit
# Automatic exposure, measured at the start of each scan and then locked (changing it
# mid-scan would leave brightness bands). Brightness ~ 15.5 * exposure - 16.
AUTO_TARGET = 150     # mean grey level aimed for
AUTO_CLIP = 0.005     # share of clipped pixels that forces a lower exposure
AUTO_SETTLE = 4       # frames to wait after changing exposure
AUTO_ROUNDS = 4       # adjustments at most
ANALYSE_EVERY = 1.0   # s between text/skew measurements of the live preview
PREVIEW_EVERY = 0.4   # s between preview updates while scanning
CONTRAST_MAX = 2.5    # contrast setting: 1 leaves the page as scanned
CONTRAST_DEFAULT = 1.5
ZOOM_MAX = 8.0        # times the size that fits the window
ZOOM_STEP = 1.25      # per mouse-wheel notch
SETTINGS_FILE = os.path.join(GLib.get_user_config_dir(), 'scannermouse', 'settings.json')


RECORDINGS_DIR = os.path.join(GLib.get_user_cache_dir(), 'scannermouse')
RECORDINGS_KEPT = 3   # raw recordings of the latest scans, for replaying a problem scan


def _new_recording():
    """A file for the raw frames of a new scan; only the latest few are kept."""
    try:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        old = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.startswith('scan-') and f.endswith('.mjpg'))
        for name in old[:max(0, len(old) - (RECORDINGS_KEPT - 1))]:
            os.remove(os.path.join(RECORDINGS_DIR, name))
        return open(os.path.join(RECORDINGS_DIR, time.strftime('scan-%Y%m%d-%H%M%S.mjpg')), 'wb')
    except OSError:
        return None


def _load_settings():
    settings = {'auto_exposure': True, 'exposure': EXPOSURE_DEFAULT, 'best_frames': True,
                'straighten': True, 'clean_background': True, 'contrast': CONTRAST_DEFAULT}
    try:
        with open(SETTINGS_FILE) as f:
            settings.update(json.load(f))
    except (OSError, ValueError):
        pass
    return settings


def _save_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)
    except OSError:
        pass


def _texture(gray):
    h, w = gray.shape
    return Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.G8,
                                 GLib.Bytes.new(np.ascontiguousarray(gray).tobytes()), w)


class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title='Scanner Mouse', default_width=1000, default_height=760)
        self.stitcher = stitcher.Stitcher()
        self.scanner = None
        self.capture = None
        self.state = 'idle'            # idle | starting | scanning | stopping | aligning
        self.started_at = 0.0
        self.last_buttons = 0
        self.live = None
        self.dirty = False
        self.debug_dir = os.environ.get('SCANNERMOUSE_DEBUG')   # record raw frames + per-frame log
        self.debug_files = None
        self.recording = None          # raw frames of the current scan, kept for diagnosis

        header = Adw.HeaderBar()
        self.scan_button = Gtk.Button(label='Start Scan')
        self.scan_button.add_css_class('suggested-action')
        self.scan_button.connect('clicked', lambda *_: self.toggle_scan())
        header.pack_start(self.scan_button)
        self.new_button = Gtk.Button(icon_name='document-new-symbolic', tooltip_text='New Page')
        self.new_button.connect('clicked', lambda *_: self.new_page())
        header.pack_start(self.new_button)
        self.save_button = Gtk.Button(icon_name='document-save-symbolic', tooltip_text='Save…')
        self.save_button.connect('clicked', lambda *_: self.save())
        header.pack_end(self.save_button)

        # page orientation (quarter turns, counter-clockwise) and cropping, applied
        # to the preview and to saved files
        self.turns = 0
        self.page_turns = 0            # quarter turns found by the orientation check
        self.analysis = None           # (skew angle, is text, measured at)
        self.last_preview = 0.0
        self.ocr_version = 0           # identifies the latest text recognition
        self.ocr_timer = 0
        self.preview_timer = 0
        self.rotate_right = Gtk.Button(icon_name='object-rotate-right-symbolic', tooltip_text='Rotate Right')
        self.rotate_right.connect('clicked', lambda *_: self._rotate(-1))
        self.rotate_left = Gtk.Button(icon_name='object-rotate-left-symbolic', tooltip_text='Rotate Left')
        self.rotate_left.connect('clicked', lambda *_: self._rotate(1))
        self.crop_button = Gtk.ToggleButton(label='Crop', active=True,
                                            tooltip_text='Trim ragged edges of the scanned area')
        self.crop_button.connect('toggled', lambda *_: self._show_page())
        for widget in (self.crop_button, self.rotate_right, self.rotate_left):
            header.pack_end(widget)

        # scanner settings
        self.settings = _load_settings()
        self.metering = None                   # auto exposure state during the first frames of a scan
        self.auto_switch = Gtk.Switch(active=self.settings['auto_exposure'], valign=Gtk.Align.CENTER)
        self.exposure_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL,
                                                       EXPOSURE_MIN, EXPOSURE_MAX, 1)
        self.exposure_scale.set_value(self.settings['exposure'])
        self.exposure_scale.set_digits(0)
        self.exposure_scale.set_size_request(220, -1)
        self.exposure_scale.add_mark(EXPOSURE_DEFAULT, Gtk.PositionType.BOTTOM, None)
        self.auto_switch.connect('notify::active', lambda *_: self._settings_changed())
        self.exposure_scale.connect('value-changed', lambda *_: self._settings_changed())
        grid = Gtk.Grid(row_spacing=12, column_spacing=12, margin_top=12, margin_bottom=12,
                        margin_start=12, margin_end=12)
        grid.attach(Gtk.Label(label='Automatic exposure', xalign=0, hexpand=True), 0, 0, 1, 1)
        grid.attach(self.auto_switch, 1, 0, 1, 1)
        grid.attach(Gtk.Label(label='Exposure', xalign=0), 0, 1, 2, 1)
        grid.attach(self.exposure_scale, 0, 2, 2, 1)
        hint = Gtk.Label(label='Higher is brighter and less noisy.\nAutomatic sets it at the start of each scan.',
                         xalign=0, wrap=True)
        hint.add_css_class('dim-label')
        hint.add_css_class('caption')
        grid.attach(hint, 0, 3, 2, 1)
        self.best_switch = Gtk.Switch(active=self.settings['best_frames'], valign=Gtk.Align.CENTER)
        self.best_switch.connect('notify::active', lambda *_: (self._settings_changed(), self._show_page()))
        grid.attach(Gtk.Separator(), 0, 4, 2, 1)
        grid.attach(Gtk.Label(label='Best frame per area', xalign=0, hexpand=True), 0, 5, 1, 1)
        grid.attach(self.best_switch, 1, 5, 1, 1)
        best_hint = Gtk.Label(label='Each part of the page shows the sharpest frame scanned there, '
                                    'so going over it again can only improve it. Blurred, tilted '
                                    'and misplaced frames are left out. Off: all frames are averaged.',
                              xalign=0, wrap=True, max_width_chars=36)
        best_hint.add_css_class('dim-label')
        best_hint.add_css_class('caption')
        grid.attach(best_hint, 0, 6, 2, 1)
        self.straighten_switch = Gtk.Switch(active=self.settings['straighten'], valign=Gtk.Align.CENTER)
        self.clean_switch = Gtk.Switch(active=self.settings['clean_background'], valign=Gtk.Align.CENTER)
        for switch in (self.straighten_switch, self.clean_switch):
            switch.connect('notify::active', lambda *_: (self._settings_changed(), self._show_page()))
        grid.attach(Gtk.Separator(), 0, 7, 2, 1)
        grid.attach(Gtk.Label(label='Straighten text pages', xalign=0, hexpand=True), 0, 8, 1, 1)
        grid.attach(self.straighten_switch, 1, 8, 1, 1)
        grid.attach(Gtk.Label(label='Even out background', xalign=0, hexpand=True), 0, 9, 1, 1)
        grid.attach(self.clean_switch, 1, 9, 1, 1)
        page_hint = Gtk.Label(label='For pages that are mostly text: turn the lines level and upright, '
                                    'and remove blotches from the paper.',
                              xalign=0, wrap=True, max_width_chars=36)
        page_hint.add_css_class('dim-label')
        page_hint.add_css_class('caption')
        grid.attach(page_hint, 0, 10, 2, 1)
        self.contrast_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.0, CONTRAST_MAX, 0.1)
        self.contrast_scale.set_value(self.settings['contrast'])
        self.contrast_scale.set_digits(1)
        self.contrast_scale.set_format_value_func(lambda _, v: f'{v:.1f}×')
        self.contrast_scale.add_mark(1.0, Gtk.PositionType.BOTTOM, None)
        self.contrast_scale.add_mark(CONTRAST_DEFAULT, Gtk.PositionType.BOTTOM, None)
        self.contrast_scale.connect('value-changed', lambda *_: (self._settings_changed(), self._page_changed()))
        grid.attach(Gtk.Separator(), 0, 11, 2, 1)
        grid.attach(Gtk.Label(label='Contrast', xalign=0), 0, 12, 2, 1)
        grid.attach(self.contrast_scale, 0, 13, 2, 1)
        contrast_hint = Gtk.Label(label='Darkens pale text and lightens faint grey on the paper. '
                                        '1.0× leaves the page as scanned.',
                                  xalign=0, wrap=True, max_width_chars=36)
        contrast_hint.add_css_class('dim-label')
        contrast_hint.add_css_class('caption')
        grid.attach(contrast_hint, 0, 14, 2, 1)
        self.settings_button = Gtk.MenuButton(icon_name='emblem-system-symbolic', tooltip_text='Scanner Settings',
                                              popover=Gtk.Popover(child=grid))
        header.pack_end(self.settings_button)

        self.page = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN, hexpand=True, vexpand=True)
        # Ctrl + mouse wheel zooms: the picture grows inside the scroller
        self.zoom = 1.0                # times the size that fits the window
        self.zoom_anchor = None        # (page point 0..1, pointer) to keep in place once laid out
        self.pointer = None
        self.render_scale = 1.0        # resolution of the preview relative to the full page
        self.render_size = PREVIEW_SIZE  # longest side the preview was asked for
        self.scroller = Gtk.ScrolledWindow(child=self.page)
        wheel = Gtk.EventControllerScroll(flags=Gtk.EventControllerScrollFlags.VERTICAL)
        wheel.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)   # before the scroller scrolls
        wheel.connect('scroll', self._on_wheel)
        self.scroller.add_controller(wheel)
        motion = Gtk.EventControllerMotion()
        motion.connect('motion', lambda c, x, y: setattr(self, 'pointer', (x, y)))
        motion.connect('leave', lambda c: setattr(self, 'pointer', None))
        self.scroller.add_controller(motion)
        for adj in (self.scroller.get_hadjustment(), self.scroller.get_vadjustment()):
            adj.connect('changed', lambda adj: self._apply_zoom_anchor())
        self.empty = Adw.StatusPage(
            icon_name='scanner-symbolic', title='No Scan Yet',
            description='Press <b>Start Scan</b>, then slowly sweep the mouse over the page.\n'
                        'Press the <b>Scan</b> button on the mouse (or Esc) to finish.')
        self.stack = Gtk.Stack()
        self.stack.add_named(self.empty, 'empty')
        self.stack.add_named(self.scroller, 'page')

        self.live_view = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN, width_request=320, height_request=122)
        self.live_view.add_css_class('card')
        self.status = Gtk.Label(xalign=0)
        self.status.add_css_class('dim-label')
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, halign=Gtk.Align.END,
                       valign=Gtk.Align.END, margin_end=12, margin_bottom=12)
        self.progress = Gtk.ProgressBar(visible=False)
        # movement speed while scanning: a coloured dot and a word
        self.speed_dot = Gtk.Label(label='●', margin_start=10, margin_top=6, margin_bottom=6)
        self.speed_label = Gtk.Label(xalign=0, margin_end=12)
        self.speed_box = Gtk.Box(spacing=6, halign=Gtk.Align.START, visible=False)
        self.speed_box.add_css_class('osd')           # readable on top of the page
        self.speed_box.append(self.speed_dot)
        self.speed_box.append(self.speed_label)
        side.append(self.speed_box)
        side.append(self.live_view)
        side.append(self.progress)
        side.append(self.status)
        overlay = Gtk.Overlay(child=self.stack)
        overlay.add_overlay(side)

        self.views = Adw.ViewStack()
        self.views.add_titled_with_icon(overlay, 'scan', 'Scan', 'image-x-generic-symbolic')
        self.views.add_titled_with_icon(self._build_text_view(), 'text', 'Text', 'text-x-generic-symbolic')
        header.set_title_widget(Adw.ViewSwitcher(stack=self.views, policy=Adw.ViewSwitcherPolicy.WIDE))

        self.toasts = Adw.ToastOverlay(child=self.views)
        view = Adw.ToolbarView(content=self.toasts)
        view.add_top_bar(header)
        self.set_content(view)

        keys = Gtk.ShortcutController()
        keys.add_shortcut(Gtk.Shortcut.new(Gtk.ShortcutTrigger.parse_string('Escape'),
                                           Gtk.CallbackAction.new(lambda *_: self.stop_scan() or True)))
        self.add_controller(keys)
        self.connect('close-request', self._on_close)
        GLib.timeout_add(150, self._refresh)
        self._update_controls()

    # -- scanning ---------------------------------------------------------

    def toggle_scan(self):
        if self.state == 'scanning':
            self.stop_scan()
        elif self.state == 'idle':
            if self.stitcher.frames_used and self.dirty:
                self._confirm_discard(self.start_scan)
            else:
                self.start_scan()

    def start_scan(self):
        path = find_device()
        if path is None:
            self._toast('Scanner mouse not found. Is it plugged in?')
            return
        self.stitcher.reset()
        self.analysis, self.page_turns = None, 0
        self._clear_text()
        self._set_zoom(1.0)
        self.stitcher.start()                   # background re-registration and rebuilding
        self.dirty = False
        self.stack.set_visible_child_name('empty')
        self.last_buttons = 0
        self.state = 'starting'
        self._update_controls()
        threading.Thread(target=self._start_worker, args=(path,), daemon=True).start()

    def _start_worker(self, path):
        try:
            self.scanner = Scanner(path)
            self.capture = Capture(path, self._on_frame)
            self.capture.start()
            time.sleep(WARMUP)
            self.scanner.start_scan()
            exposure = int(self.settings['exposure'])
            self.scanner.set_exposure(exposure)
            # with automatic exposure, the first frames meter the page before painting starts
            self.metering = ({'exposure': exposure, 'wait': AUTO_SETTLE, 'rounds': 0}
                             if self.settings['auto_exposure'] else None)
        except (OSError, ScannerError, RuntimeError) as e:
            GLib.idle_add(self._scan_failed, str(e))
            return
        self.recording = _new_recording()
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            stamp = time.strftime('%Y%m%d-%H%M%S')
            self.debug_files = (open(os.path.join(self.debug_dir, f'{stamp}.mjpg'), 'wb'),
                                open(os.path.join(self.debug_dir, f'{stamp}.jsonl'), 'w'))
        self.started_at = time.monotonic()
        self.state = 'scanning'
        GLib.idle_add(self._update_controls)

    def _scan_failed(self, message):
        self._teardown()
        self.state = 'idle'
        self._update_controls()
        self._toast(f'Could not start scanning: {message}')

    def stop_scan(self):
        if self.state != 'scanning':
            return
        self.state = 'stopping'
        self._update_controls()
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self):
        error = None
        try:
            self.scanner.stop_scan()
        except (OSError, ScannerError) as e:
            error = str(e)
        self._teardown()
        # final re-registration of all segments and rebuild of the page
        self.state = 'aligning'
        GLib.idle_add(self._update_controls)
        try:
            self.stitcher.finish(lambda f: GLib.idle_add(self.progress.set_fraction, f))
        except Exception as e:              # keep the live result if the final pass fails
            error = error or f'final alignment failed: {e}'
        self.state = 'idle'
        GLib.idle_add(self._stopped, error)

    def _stopped(self, error):
        self.analysis = None                    # measure the finished page afresh
        self._update_controls()
        if self.settings['straighten']:
            # recognition follows once the page is turned upright
            threading.Thread(target=self._find_orientation, daemon=True).start()
        else:
            self._start_ocr()
        if error:
            self._toast(f'Scanner did not confirm stop: {error}')

    def _teardown(self):
        if self.recording:
            self.recording.close()
            self.recording = None
        if self.debug_files:
            for f in self.debug_files:
                f.close()
            self.debug_files = None
        self.recording = None          # raw frames of the current scan, kept for diagnosis
        if self.capture:
            self.capture.stop()
            self.capture = None
        if self.scanner:
            self.scanner.close()
            self.scanner = None

    def _on_frame(self, data):
        """Runs on the GStreamer thread."""
        if self.state != 'scanning':
            return
        if self.recording:
            self.recording.write(data)
        rec, jpeg = framedecode.decode(data)
        if rec is None:
            return
        pressed = rec.buttons & ~self.last_buttons
        self.last_buttons = rec.buttons
        if pressed & framedecode.BUTTON_SCAN and time.monotonic() - self.started_at > 1.0:
            GLib.idle_add(self.stop_scan)
            return
        try:
            rgb = jpeg_to_rgb(jpeg)
        except GLib.Error:
            return
        brightness = float(rgb[..., 1].mean())
        added = False
        if self.metering is not None:
            self.live = np.ascontiguousarray(rgb[stitcher.CROP_TOP:, ::-1, 1])
            self._meter(rgb[stitcher.CROP_TOP:, :, 1])
            brightness = 0.0                    # nothing is painted while metering
        if brightness >= DARK_FRAME:
            self.live = np.ascontiguousarray(rgb[stitcher.CROP_TOP:, ::-1, 1])
            t0 = time.monotonic()
            added = self.stitcher.add(rec, rgb)
            self.dirty = self.dirty or added
        if self.debug_files:
            st = self.stitcher
            self.debug_files[0].write(data)
            self.debug_files[1].write(json.dumps({
                't': round(time.monotonic() - self.started_at, 3), 'rec': rec.__dict__,
                'brightness': round(brightness, 1), 'added': added,
                'ms': round((time.monotonic() - t0) * 1000, 1) if brightness >= DARK_FRAME else None,
                'pos': [round(float(v), 1) for v in st.pos], 'theta_deg': round(float(np.degrees(st.theta)), 2),
                'source': st.last_source, 'lost': st.lost, 'track': st.last_track,
                'match': st.last_match}, default=float) + '\n')

    def _meter(self, green):
        """One automatic-exposure step (GStreamer thread)."""
        m = self.metering
        m['wait'] -= 1
        if m['wait'] > 0:
            return                              # the last change hasn't reached the image yet
        mean = float(green.mean())
        clipped = float((green >= 250).mean())
        exposure = m['exposure']
        # brightness ~ k * exposure - 16, with k depending on the material
        k = max(mean + 16, 1.0) / exposure
        target = int(round((AUTO_TARGET + 16) / k))
        if clipped > AUTO_CLIP:
            target = min(target, exposure - 1)
        target = max(EXPOSURE_MIN, min(EXPOSURE_MAX, target))
        m['rounds'] += 1
        if target == exposure or m['rounds'] > AUTO_ROUNDS:
            self.metering = None                # settled: start painting
            self.settings['exposure'] = exposure
            GLib.idle_add(self._show_exposure, exposure)
            return
        try:
            self.scanner.set_exposure(target)
        except (OSError, ScannerError):
            self.metering = None
            return
        m['exposure'], m['wait'] = target, AUTO_SETTLE

    def _show_exposure(self, exposure):
        self.exposure_scale.set_value(exposure)
        _save_settings(self.settings)

    def _settings_changed(self):
        self.settings['auto_exposure'] = self.auto_switch.get_active()
        self.settings['exposure'] = int(self.exposure_scale.get_value())
        self.settings['best_frames'] = self.best_switch.get_active()
        self.settings['straighten'] = self.straighten_switch.get_active()
        self.settings['clean_background'] = self.clean_switch.get_active()
        self.settings['contrast'] = round(self.contrast_scale.get_value(), 1)
        _save_settings(self.settings)

    # -- UI ---------------------------------------------------------------

    # -- recognised text ---------------------------------------------------

    def _build_text_view(self):
        self.text_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, editable=True, monospace=False,
                                      left_margin=24, right_margin=24, top_margin=18, bottom_margin=18)
        buf = self.text_view.get_buffer()
        buf.create_tag('bold', weight=700)
        buf.create_tag('large', scale=1.3)
        buf.create_tag('uncertain', foreground='#8a8a8a', underline=Pango.Underline.ERROR)
        scrolled = Gtk.ScrolledWindow(child=self.text_view, vexpand=True)

        self.text_status = Gtk.Label(xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.END)
        self.text_status.add_css_class('dim-label')
        copy = Gtk.Button(label='Copy All', tooltip_text='Copy the recognised text')
        copy.connect('clicked', lambda *_: self._copy_text())
        again = Gtk.Button(icon_name='view-refresh-symbolic', tooltip_text='Recognise Again')
        again.connect('clicked', lambda *_: self._start_ocr())
        bar = Gtk.Box(spacing=6, margin_start=12, margin_end=12, margin_top=6, margin_bottom=6)
        for widget in (self.text_status, again, copy):
            bar.append(widget)
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.append(bar)
        page.append(Gtk.Separator())
        page.append(scrolled)

        self.text_stack = Gtk.Stack()
        self.text_stack.add_named(Adw.StatusPage(icon_name='text-x-generic-symbolic', title='No Text Yet',
                                                 description='The text is recognised when a scan is finished.'),
                                  'empty')
        busy = Adw.StatusPage(title='Recognising Text…', child=Adw.Spinner(width_request=32, height_request=32))
        self.text_stack.add_named(busy, 'busy')
        self.text_stack.add_named(page, 'text')
        return self.text_stack

    def _start_ocr(self):
        """Recognise the finished page as it is currently shown (background thread)."""
        if self.state != 'idle' or self.stitcher.frames_used == 0:
            return
        langs = ocr.languages()
        if langs is None:
            self.text_stack.set_visible_child_name('empty')
            self._toast('Install Tesseract (and tesseract-data-deu for German) to recognise text')
            return
        self.ocr_version += 1
        version, crop = self.ocr_version, self.crop_button.get_active()
        self.text_stack.set_visible_child_name('busy')

        def work():
            page = self._render(crop=crop)
            lines = ocr.recognise(page, langs) if page is not None else []
            GLib.idle_add(self._show_text, lines, langs, version)
        threading.Thread(target=work, daemon=True).start()

    def _show_text(self, lines, langs, version):
        if version != self.ocr_version:
            return                              # a newer recognition is running
        buf = self.text_view.get_buffer()
        buf.set_text('')
        uncertain = 0
        for line in lines:
            if line.blank_before:
                buf.insert(buf.get_end_iter(), '\n')
            line_tags = ['large'] if line.large else []
            buf.insert(buf.get_end_iter(), ' ' * line.indent)
            for k, word in enumerate(line.words):
                if k:
                    buf.insert(buf.get_end_iter(), line.gaps[k - 1])
                tags = line_tags + (['bold'] if word.bold else [])
                if word.confidence < ocr.UNCERTAIN:
                    tags.append('uncertain')
                    uncertain += 1
                buf.insert_with_tags_by_name(buf.get_end_iter(), word.text, *tags) if tags else \
                    buf.insert(buf.get_end_iter(), word.text)
            buf.insert(buf.get_end_iter(), '\n')
        names = {'deu': 'German', 'eng': 'English'}
        language = ' + '.join(names.get(l, l) for l in langs.split('+'))
        self.text_status.set_label(f'{len(lines)} line{"s" if len(lines) != 1 else ""} · {language}'
                                   + (f' · {uncertain} uncertain words underlined' if uncertain else ''))
        self.text_stack.set_visible_child_name('text' if lines else 'empty')

    def _clear_text(self):
        self.ocr_version += 1                   # drop any recognition still running
        self.text_view.get_buffer().set_text('')
        self.text_stack.set_visible_child_name('empty')

    def _text(self):
        buf = self.text_view.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

    def _copy_text(self):
        self.get_clipboard().set(self._text())
        self._toast('Text copied')

    def _render(self, max_size=None, crop=None):
        """The page as it will be saved: straightened, cleaned, cropped and rotated
        as chosen, or None."""
        page = self.stitcher.page(max_size, best=self.settings['best_frames'])
        if page is None:
            return None
        img, covered, scale = page
        self.render_scale = scale
        turns = self.turns
        if self.settings['straighten'] or self.settings['clean_background']:
            angle, is_text = self._analyse(img, covered)
            if is_text and self.settings['straighten']:
                img, covered = cleanup.rotate(img, covered, angle)
                turns += self.page_turns
            # evening out takes a moment, so the live preview only gets it once the scan is done
            if is_text and self.settings['clean_background'] and self.state != 'scanning':
                img = cleanup.flatten_background(img, covered, scale)
        if self.crop_button.get_active() if crop is None else crop:
            top, bottom, left, right = cleanup.crop_box(covered)
            img = img[top:bottom, left:right]
        img = cleanup.contrast(img, self.settings['contrast'])
        return np.ascontiguousarray(np.rot90(img, turns))

    def _analyse(self, img, covered):
        """Text detection and skew angle, re-measured at most every ANALYSE_EVERY s
        while scanning and once for the finished page."""
        now = time.monotonic()
        if self.analysis is None or (self.state == 'scanning' and now - self.analysis[2] > ANALYSE_EVERY):
            angle, is_text = cleanup.analyse(img, covered)
            self.analysis = (angle, is_text, now)
        return self.analysis[:2]

    def _find_orientation(self):
        """After a scan: let Tesseract decide whether the straightened page needs
        quarter turns to be upright (background thread)."""
        page = self.stitcher.page(2000, best=self.settings['best_frames'])
        turns = 0
        if page is not None:
            img, covered, _ = page
            angle, is_text = cleanup.analyse(img, covered)
            if is_text:
                turns = cleanup.upright_turns(cleanup.rotate(img, covered, angle)[0])
        def apply():
            self.page_turns = turns
            self._show_page()
            self._start_ocr()
        GLib.idle_add(apply)

    def _show_page(self):
        self.render_size = self._preview_size()
        page = self._render(self.render_size)
        if page is not None:
            self.page.set_paintable(_texture(page))
            self._size_page()
            self.stack.set_visible_child_name('page')

    # -- zoom -------------------------------------------------------------

    def _on_wheel(self, controller, dx, dy):
        if not controller.get_current_event_state() & Gdk.ModifierType.CONTROL_MASK:
            return False                        # plain scrolling
        self._set_zoom(self.zoom * ZOOM_STEP ** -dy, self.pointer)
        return True

    def _layout(self, zoom):
        """(width, height, left, top) of the shown page at `zoom`, in scroller content
        pixels, or None. A page smaller than the window is centred in it."""
        texture = self.page.get_paintable()
        vw, vh = self.scroller.get_width(), self.scroller.get_height()
        if texture is None or vw <= 0 or vh <= 0:
            return None
        iw, ih = texture.get_intrinsic_width(), texture.get_intrinsic_height()
        fit = min(vw / iw, vh / ih)
        w, h = iw * fit * zoom, ih * fit * zoom
        return w, h, max(0.0, (vw - w) / 2), max(0.0, (vh - h) / 2)

    def _set_zoom(self, zoom, pointer=None):
        """Zoom to `zoom` (1 = fit the window), keeping the page point under the
        pointer (or the middle of the view) where it is."""
        zoom = min(max(zoom, 1.0), ZOOM_MAX)
        layout = self._layout(self.zoom)
        if zoom == self.zoom or layout is None:
            self.zoom = zoom
            self._size_page()
            return
        w, h, left, top = layout
        x, y = pointer or (self.scroller.get_width() / 2, self.scroller.get_height() / 2)
        if self.zoom_anchor is None:
            hval = self.scroller.get_hadjustment().get_value()
            vval = self.scroller.get_vadjustment().get_value()
        else:                                   # the last step isn't laid out yet: go on from its target
            au, av, ax, ay = self.zoom_anchor
            hval, vval = au * w + left - ax, av * h + top - ay
        u = (hval + x - left) / w
        v = (vval + y - top) / h
        self.zoom = zoom
        self.zoom_anchor = (min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0), x, y)
        self._size_page()
        self._apply_zoom_anchor()               # in case the size didn't change
        # zoomed in beyond the preview's resolution: render it sharper
        if self.render_scale < 1.0 and self._preview_size() > self.render_size and not self.preview_timer:
            self.preview_timer = GLib.timeout_add(150, self._delayed_preview)

    def _size_page(self):
        layout = self._layout(self.zoom) if self.zoom > 1.0 else None
        if layout is None:
            self.page.set_size_request(-1, -1)
        else:
            self.page.set_size_request(int(layout[0]), int(layout[1]))

    def _apply_zoom_anchor(self):
        """Scroll so the zoom anchor is back under the pointer, once the scroller
        has taken on the new page size."""
        layout = self._layout(self.zoom)
        if self.zoom_anchor is None or layout is None:
            return
        w, h, left, top = layout
        hadj, vadj = self.scroller.get_hadjustment(), self.scroller.get_vadjustment()
        if (abs(hadj.get_upper() - max(w, hadj.get_page_size())) > 1.5
                or abs(vadj.get_upper() - max(h, vadj.get_page_size())) > 1.5):
            return                              # not laid out yet
        u, v, x, y = self.zoom_anchor
        self.zoom_anchor = None
        hadj.set_value(u * w + left - x)
        vadj.set_value(v * h + top - y)

    def _preview_size(self):
        """Longest side to render the preview at: enough for the zoom, in device pixels."""
        layout = self._layout(self.zoom) if self.zoom > 1.0 else None
        if layout is None:
            return PREVIEW_SIZE
        return max(PREVIEW_SIZE, int(max(layout[:2]) * self.get_scale_factor()))

    def _rotate(self, turns):
        self.turns = (self.turns + turns) % 4
        self._show_page()
        self._schedule_ocr()

    def _page_changed(self):
        """A look setting changed: update the preview, at most every 150 ms while the
        slider moves, and recognise the text again."""
        if not self.preview_timer:
            self.preview_timer = GLib.timeout_add(150, self._delayed_preview)
        if self.state == 'idle':
            self._schedule_ocr()

    def _delayed_preview(self):
        self.preview_timer = 0
        self._show_page()
        return GLib.SOURCE_REMOVE

    def _schedule_ocr(self):
        """Recognise the text again once the changes (turning, sliders) have stopped."""
        if self.ocr_timer:
            GLib.source_remove(self.ocr_timer)
        self.ocr_timer = GLib.timeout_add(1200, self._delayed_ocr)

    def _delayed_ocr(self):
        self.ocr_timer = 0
        self._start_ocr()
        return GLib.SOURCE_REMOVE

    def _refresh(self):
        if self.state == 'scanning':
            if time.monotonic() - self.last_preview >= PREVIEW_EVERY:
                self.last_preview = time.monotonic()
                self._show_page()
            if self.live is not None:
                self.live_view.set_paintable(_texture(self.live))
            self._show_speed()
            if self.metering is not None:
                self.status.set_label('Measuring exposure…')
            elif self.stitcher.lost:
                self.status.set_label('Position lost — move back over a part that is already scanned')
            else:
                self.status.set_label(f'Scanning — {self.stitcher.frames_used} frames · '
                                      f'press Scan or Esc to finish')
        return GLib.SOURCE_CONTINUE

    def _show_speed(self):
        speed = self.stitcher.speed
        if speed < stitcher.SPEED_GOOD:
            style, text = 'success', 'Good speed'
        elif speed < stitcher.SPEED_TOO_FAST:
            style, text = 'warning', 'A bit fast'
        else:
            style, text = 'error', 'Too fast – slow down'
        for name in ('success', 'warning', 'error'):
            self.speed_dot.remove_css_class(name)
        self.speed_dot.add_css_class(style)
        self.speed_label.set_label(text)

    def _update_controls(self):
        labels = {'idle': 'Start Scan', 'starting': 'Starting…', 'scanning': 'Stop Scan',
                  'stopping': 'Stopping…', 'aligning': 'Aligning…'}
        self.progress.set_visible(self.state == 'aligning')
        if self.state == 'aligning':
            self.progress.set_fraction(0.0)
            self.status.set_label('Aligning the page…')
        self.scan_button.set_label(labels[self.state])
        self.scan_button.set_sensitive(self.state in ('idle', 'scanning'))
        if self.state == 'scanning':
            self.scan_button.remove_css_class('suggested-action')
            self.scan_button.add_css_class('destructive-action')
        else:
            self.scan_button.remove_css_class('destructive-action')
            self.scan_button.add_css_class('suggested-action')
        has_page = self.stitcher.frames_used > 0
        self.save_button.set_sensitive(self.state == 'idle' and has_page)
        self.new_button.set_sensitive(self.state == 'idle' and has_page)
        for widget in (self.rotate_left, self.rotate_right, self.crop_button):
            widget.set_sensitive(has_page or self.state == 'scanning')
        # exposure is fixed for the duration of a scan; the compositing choice is
        # only a view, so it can be switched at any time
        for widget in (self.auto_switch, self.exposure_scale):
            widget.set_sensitive(self.state == 'idle')
        self.live_view.set_visible(self.state == 'scanning')
        self.speed_box.set_visible(self.state == 'scanning')
        if self.state == 'idle':
            self._show_page()
            self.status.set_label(self._summary() if has_page else '')
        return GLib.SOURCE_REMOVE

    def _summary(self):
        mapper = self.stitcher.mapper
        text = (f'{self.stitcher.frames_used} frames · {len(mapper.segments)} segments, '
                f'{mapper.stats["overlaps"]} overlaps')
        left_out = collections.Counter(self.stitcher.discarded)
        if self.stitcher.misplaced:
            left_out['not fitting'] = self.stitcher.misplaced
        if left_out:
            reasons = ', '.join(f'{n} {why}' for why, n in left_out.most_common())
            text += f' · left out of best page: {reasons}'
        return text

    def _toast(self, text):
        self.toasts.add_toast(Adw.Toast(title=text, timeout=5))

    def _confirm_discard(self, then):
        dialog = Adw.AlertDialog(heading='Discard Current Scan?',
                                 body='The current page has not been saved.')
        dialog.add_response('cancel', 'Cancel')
        dialog.add_response('discard', 'Discard')
        dialog.set_response_appearance('discard', Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.connect('response', lambda d, r: then() if r == 'discard' else None)
        dialog.present(self)

    def new_page(self):
        def clear():
            self.stitcher.reset()
            self.analysis, self.page_turns = None, 0
            self._clear_text()
            self.dirty = False
            self._set_zoom(1.0)
            self.page.set_paintable(None)
            self.stack.set_visible_child_name('empty')
            self._update_controls()
        if self.dirty:
            self._confirm_discard(clear)
        else:
            clear()

    # -- saving -----------------------------------------------------------

    def save(self):
        dialog = Gtk.FileDialog(title='Save Scan', initial_name='scan.png')
        filters = Gio.ListStore.new(Gtk.FileFilter)
        for name, pattern in (('PNG image', '*.png'), ('PDF document (searchable when text was recognised)', '*.pdf'),
                              ('Recognised text', '*.txt')):
            f = Gtk.FileFilter(name=name)
            f.add_pattern(pattern)
            filters.append(f)
        dialog.set_filters(filters)
        dialog.save(self, None, self._on_save)

    def _on_save(self, dialog, result):
        try:
            file = dialog.save_finish(result)
        except GLib.Error:
            return                          # cancelled
        path = file.get_path()
        if path.lower().endswith('.txt'):
            try:
                with open(path, 'w') as f:
                    f.write(self._text())
            except OSError as e:
                self._toast(f'Could not save: {e}')
                return
            self._toast(f'Saved {GLib.path_get_basename(path)}')
            return
        page = self._render()
        try:
            if path.lower().endswith('.pdf'):
                # with a recognised text, the PDF gets an invisible text layer
                searchable = self._text().strip() and ocr.searchable_pdf(page, path, stitcher.DPI)
                if not searchable:
                    _save_pdf(page, path)
            else:
                if not path.lower().endswith('.png'):
                    path += '.png'
                _texture(page).save_to_png(path)
        except (OSError, cairo.Error, GLib.Error) as e:
            self._toast(f'Could not save: {e}')
            return
        self.dirty = False
        self._toast(f'Saved {GLib.path_get_basename(path)}')

    def _on_close(self, *_):
        if self.scanner:
            # restore normal mouse operation before quitting
            try:
                self.scanner.stop_scan()
            except (OSError, ScannerError):
                pass
            self._teardown()
        self.stitcher.close()
        return False


def _save_pdf(gray, path):
    h, w = gray.shape
    rgb = np.empty((h, w, 4), np.uint8)
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = gray
    rgb[..., 3] = 255
    image = cairo.ImageSurface.create_for_data(memoryview(rgb), cairo.FORMAT_RGB24, w, h, w * 4)
    scale = 72.0 / stitcher.DPI
    surface = cairo.PDFSurface(path, w * scale, h * scale)
    cr = cairo.Context(surface)
    cr.scale(scale, scale)
    cr.set_source_surface(image, 0, 0)
    cr.paint()
    surface.finish()


class Application(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_startup(self):
        Adw.Application.do_startup(self)
        # running from a checkout: the icon isn't installed into the system theme
        icons = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'icons')
        if os.path.isdir(icons):
            Gtk.IconTheme.get_for_display(Gdk.Display.get_default()).add_search_path(icons)
        Gtk.Window.set_default_icon_name(APP_ID)

    def do_activate(self):
        win = self.get_active_window() or Window(self)
        win.present()


def main():
    return Application().run(sys.argv)
