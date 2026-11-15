"""The parts that can be checked on their own."""

import math
import struct

import cv2
import numpy as np
import pytest

from scannermouse import align, calibration, cleanup, export, ocr


# -- align ----------------------------------------------------------------------

def test_solve_removes_drift_with_a_loop_closure():
    """A square walked with a drifting tracker: one overlap between the first and the
    last segment pulls all positions back to where they were."""
    rng = np.random.default_rng(0)
    true = np.array([(100 * math.cos(a), 100 * math.sin(a), a) for a in np.linspace(0, 2 * math.pi, 41)])
    edges, drifted = [], [true[0]]
    for k in range(1, len(true)):
        link = align.relative(true[k - 1], true[k]) + rng.normal(0, [0.3, 0.3, 0.004])
        edges.append((k - 1, k, link, 1.0))
        t = drifted[-1][:2] + align._rot(drifted[-1][2]) @ link[:2]
        drifted.append(np.array([t[0], t[1], drifted[-1][2] + link[2]]))
    drifted = np.array(drifted)
    assert np.hypot(*(drifted[-1] - true[-1])[:2]) > 3          # the drift to remove
    edges.append((0, len(true) - 1, align.relative(true[0], true[-1]), 3.0))
    solved = align.solve(drifted, edges)
    assert np.hypot(*(solved[-1] - true[-1])[:2]) < 0.5
    assert np.abs(solved[:, :2] - true[:, :2]).max() < 3


def test_solve_limits_the_damage_of_a_wrong_match():
    """One match between two different places, 230 px off, among true ones: it is
    down-weighted, so it moves the positions by a few px, not by its full error."""
    true = np.array([(10.0 * k, 0.0, 0.0) for k in range(30)])
    edges = [(k - 1, k, align.relative(true[k - 1], true[k]), 1.0) for k in range(1, 30)]
    edges += [(k, k + 5, align.relative(true[k], true[k + 5]), 1.0) for k in range(25)]
    wrong = (2, 20, np.array([400.0, 60.0, 0.0]), 1.0)
    start = true + np.random.default_rng(1).normal(0, 1, true.shape) * [1, 1, 0.002]
    start[0] = true[0]                          # the first segment is where the page starts
    assert np.abs(align.solve(start, edges)[:, :2] - true[:, :2]).max() < 0.1
    assert np.abs(align.solve(start, edges + [wrong])[:, :2] - true[:, :2]).max() < 5


# -- cleanup ---------------------------------------------------------------------

def test_contrast_keeps_black_white_and_the_pivot():
    ramp = np.arange(256, dtype=np.uint8)[None]
    out = cleanup.contrast(ramp, 2.0)[0]
    assert out[0] == 0 and out[255] == 255 and out[cleanup.CONTRAST_PIVOT] == cleanup.CONTRAST_PIVOT
    assert (np.diff(out.astype(int)) >= 0).all()
    assert out[100] < 100 and out[220] > 220
    assert cleanup.contrast(ramp, 1.0) is ramp


def test_text_is_recognised_as_text_and_levelled():
    page = np.full((700, 900), 255, np.uint8)
    for y in range(60, 660, 30):
        cv2.putText(page, 'the quick brown fox jumps over the lazy dog', (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    covered = np.ones(page.shape, bool)
    tilted, mask = cleanup.rotate(page, covered, 7.0)
    angle, is_text = cleanup.analyse(tilted, mask)
    assert is_text and abs(angle + 7.0) < 0.3
    noise = (np.random.default_rng(0).random(page.shape) * 255).astype(np.uint8)
    assert not cleanup.analyse(cv2.GaussianBlur(noise, (0, 0), 8), covered)[1]


def test_photo_edges_are_found():
    """A dark photo on a light ground, turned by 4 degrees: the angle to turn it back."""
    scene = np.full((800, 1100), 235, np.uint8)
    cv2.rectangle(scene, (200, 150), (900, 650), 60, -1)
    turned, mask = cleanup.rotate(scene, np.ones(scene.shape, bool), 4.0)
    turned = cv2.GaussianBlur(turned, (0, 0), 1.0)
    assert abs(cleanup.edge_angle(turned, mask) + 4.0) < 0.3
    flat = np.full((400, 600), 200, np.uint8)
    assert cleanup.edge_angle(flat, np.ones(flat.shape, bool)) is None


def test_crop_trims_ragged_edges():
    covered = np.zeros((300, 400), bool)
    covered[40:260, 30:380] = True
    covered[10:40, 100:140] = True              # a stray tongue at the top
    assert cleanup.crop_box(covered) == (40, 260, 30, 380)


# -- export ------------------------------------------------------------------------

@pytest.fixture
def picture():
    return (np.random.default_rng(0).random((120, 200)) * 255).astype(np.uint8)


def test_png_keeps_every_pixel_and_states_the_resolution(picture, tmp_path):
    path = str(tmp_path / 'page.png')
    export.save_image(picture, path, 320)
    assert np.array_equal(cv2.imread(path, cv2.IMREAD_GRAYSCALE), picture)
    data = open(path, 'rb').read()
    at = data.index(b'pHYs')
    assert struct.unpack('>IIB', data[at + 4:at + 13]) == (12598, 12598, 1)       # 320 dpi in px per metre


def test_jpeg_states_the_resolution(picture, tmp_path):
    path = str(tmp_path / 'page.jpg')
    export.save_image(picture, path, 320)
    data = open(path, 'rb').read()
    assert data[:2] == b'\xff\xd8' and struct.unpack('>BHH', data[13:18]) == (1, 320, 320)
    assert cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape == picture.shape


def test_pdf_is_written(picture, tmp_path):
    path = str(tmp_path / 'page.pdf')
    export.save_pdf(picture, path, 320)
    assert open(path, 'rb').read(5) == b'%PDF-'


# -- calibration -----------------------------------------------------------------------

def test_calibration_round_trip(tmp_path):
    path = str(tmp_path / 'calibration.json')
    values = {'counts_to_pixels': np.arange(8.0).reshape(4, 2), 'counts_to_radians': np.arange(4.0),
              'single_to_pixels': np.arange(8.0).reshape(2, 2, 2),
              'illumination': {(0, 0): 0.6, (2, 4): -0.01}, 'glare_spots': ((132, 5), (226, 4))}
    calibration.save(values, path)
    loaded = calibration.load(path)
    assert set(loaded) == set(values)
    assert np.array_equal(loaded['counts_to_pixels'], values['counts_to_pixels'])
    assert loaded['illumination'] == values['illumination'] and loaded['glare_spots'] == values['glare_spots']


def test_missing_or_broken_calibration_is_ignored(tmp_path, capsys):
    assert calibration.load(str(tmp_path / 'none.json')) == {}
    broken = tmp_path / 'broken.json'
    broken.write_text('{"counts_to_pixels": [1, 2, 3]}')
    assert calibration.load(str(broken)) == {}
    assert 'ignoring' in capsys.readouterr().err


# -- text recognition ----------------------------------------------------------------------

def test_language_choice(monkeypatch):
    monkeypatch.setattr(ocr, 'installed_languages', lambda: ('deu', 'eng', 'fra'))
    monkeypatch.setattr(ocr.locale, 'getlocale', lambda: ('fr_FR', 'UTF-8'))
    assert ocr.languages() == 'fra+deu+eng'                 # the system's language first
    assert ocr.languages(['eng']) == 'eng'
    assert ocr.languages(['spa']) == 'fra+deu+eng'          # chosen but not installed: as without a choice
    monkeypatch.setattr(ocr.locale, 'getlocale', lambda: ('en_US', 'UTF-8'))
    assert ocr.languages() == 'deu+eng'
    monkeypatch.setattr(ocr, 'installed_languages', lambda: ())
    assert ocr.languages() is None


def test_layout_of_recognised_lines():
    word = lambda text, left, top: ocr.Word(text, (left, top, 12 * len(text), 20), 95.0)
    lines = [ocr.Line([word('Dear', 50, 100), word('reader', 110, 100)]),
             ocr.Line([word('indented', 110, 130), word('far', 600, 130)]),
             ocr.Line([word('after', 50, 160), word('a', 122, 160)]),
             ocr.Line([word('gap', 50, 260), word('here', 98, 260)])]
    ocr._layout(lines)
    assert lines[0].indent == 0 and lines[1].indent > 0
    assert lines[1].gaps == ['\t'] and lines[0].gaps == [' ']
    assert lines[3].blank_before and not lines[1].blank_before
    assert ocr.plain_text(lines).splitlines()[0] == 'Dear reader'
