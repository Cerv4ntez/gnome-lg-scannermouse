"""The whole pipeline on a made-up scan: tracker, Mapper, rebuilt page."""

import cv2
import numpy as np
import pytest

import synthetic
from scannermouse import stitcher


def run(frames, mapper=True, inline=True):
    st = stitcher.Stitcher(inline=inline)
    if mapper:
        st.start()
    for rec, rgb in frames:
        st.add(rec, rgb)
    if mapper:
        st.finish()
    else:
        st.flush()
    result = {'page': st.page(best=True), 'average': st.page(best=False), 'placed': st.frames_used,
              'segments': len(st.mapper.segments), 'overlaps': st.mapper.stats['overlaps'],
              'misplaced': st.misplaced, 'dropped': st.dropped, 'restarts': st.restarts}
    st.close()
    return result


@pytest.fixture(scope='module')
def made_up():
    return synthetic.scan()


@pytest.fixture(scope='module')
def result(made_up):
    return run(made_up[2])


def offsets(page, truth, size=160):
    """Where patches of the stitched page are found on the true page, minus where
    they were cut from: all the same if the page is stitched without distortion.
    Returns (offsets (n, 2), correlation of the worst patch)."""
    truth = (np.clip(truth[:, ::-1], 0, 1) * 255).astype(np.uint8)     # the page is delivered unmirrored
    h, w = page.shape
    found, worst = [], 1.0
    for y in np.linspace(40, h - size - 40, 3).astype(int):
        for x in np.linspace(40, w - size - 40, 5).astype(int):
            score = cv2.matchTemplate(truth, page[y:y + size, x:x + size], cv2.TM_CCOEFF_NORMED)
            _, best, _, where = cv2.minMaxLoc(score)
            found.append((where[0] - x, where[1] - y))
            worst = min(worst, best)
    return np.array(found), worst


def test_every_frame_is_placed(made_up, result):
    assert result['placed'] >= len(made_up[2]) - 2
    assert result['restarts'] == 0 and result['dropped'] == 0


def test_sweeps_are_tied_together(result):
    assert result['overlaps'] >= 20
    assert result['misplaced'] == 0


@pytest.mark.parametrize('which', ['page', 'average'])
def test_page_matches_the_original(made_up, result, which):
    page = result[which][0]
    found, worst = offsets(page, made_up[0])
    assert worst > 0.8, 'a part of the page does not show what was scanned there'
    spread = np.abs(found - np.median(found, axis=0)).max()
    assert spread <= 2, f'the page is distorted: parts are up to {spread} px out of place'


def test_page_covers_the_scanned_area(made_up, result):
    poses = made_up[1]
    page, covered, scale = result['page']
    assert scale == 1.0
    assert abs(page.shape[1] - (np.ptp(poses[:, 0]) + synthetic.FRAME_W)) < 40
    assert abs(page.shape[0] - (np.ptp(poses[:, 1]) + synthetic.FRAME_H)) < 40
    assert covered.mean() > 0.9


def test_same_scan_gives_the_same_page(made_up, result):
    again = run(made_up[2])
    assert np.array_equal(again['page'][0], result['page'][0])


def test_background_process_gives_the_same_quality(made_up):
    """As the app runs it: the Mapper in its own process (its timing varies, so the
    page is compared with the original, not with the in-step one)."""
    result = run(made_up[2], inline=False)
    assert result['overlaps'] >= 20 and result['misplaced'] == 0
    found, worst = offsets(result['page'][0], made_up[0])
    assert worst > 0.8
    assert np.abs(found - np.median(found, axis=0)).max() <= 2


def test_tracker_alone_stays_close(made_up):
    live = run(made_up[2], mapper=False)
    found, worst = offsets(live['page'][0], made_up[0])
    assert worst > 0.4
    assert np.abs(found - np.median(found, axis=0)).max() <= 12


def test_dropped_frames_are_counted(made_up):
    frames = made_up[2][:60]
    del frames[30:32]
    assert run(frames, mapper=False)['dropped'] == 2


def test_blank_paper_is_tracked_by_the_sensors():
    """Without anything to see, the sensors' motion alone places the frames."""
    img = np.full((500, 1200), 0.9, np.float32)
    poses = synthetic.path(sweeps=1, length=480)
    frames = list(zip(synthetic.records(poses), (synthetic.frame(img, p) for p in poses)))
    live = run(frames, mapper=False)
    assert live['placed'] >= len(frames) - 2
    assert abs(live['page'][0].shape[1] - (np.ptp(poses[:, 0]) + synthetic.FRAME_W)) < 40
