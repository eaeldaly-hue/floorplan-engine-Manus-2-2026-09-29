"""N.2: local jamb verification, repeated-line patterns and the wall tone model."""

import cv2
import numpy as np

from engine.structure import analyze_structure
from engine.wall_tone import learn_wall_tone

WALL = (90, 90, 90)          # gray walls (a coloured plan)
T = 10


def _room(img):
    cv2.rectangle(img, (60, 60), (700, 460), WALL, T)
    cv2.line(img, (380, 60), (380, 460), WALL, T)
    cv2.rectangle(img, (375, 230), (385, 290), (255, 255, 255), -1)          # door between the rooms


def _gaps_near(s, x, y, r=40):
    return [c for c in s.candidates if abs(c.center[0] - x) < r and abs(c.center[1] - y) < r]


def test_gap_to_an_off_tone_furniture_block_is_not_an_opening_and_does_not_partition():
    img = np.full((520, 760, 3), 255, np.uint8)
    _room(img)
    cv2.line(img, (380, 380), (380, 460), (255, 255, 255), T + 2)          # wall ends at y=380 ...
    cv2.rectangle(img, (366, 400), (396, 422), (20, 80, 20), -1)           # ... a dark green plant pot in the gap
    s = analyze_structure(img)
    assert not _gaps_near(s, 381, 390, 12) and not _gaps_near(s, 381, 440, 12), [(c.start, c.end, c.jambs) for c in s.candidates]
    assert any("tone differs" in " ".join(c.jambs) for c in s.rejected_candidates)
    assert _gaps_near(s, 380, 260)                                         # the real door stays


def test_same_tone_piece_is_kept_without_tone_evidence():
    """A short detached piece drawn exactly like the walls (a column, a pier) is never rejected
    on geometry alone."""
    img = np.full((520, 760, 3), 255, np.uint8)
    _room(img)
    cv2.line(img, (380, 380), (380, 460), (255, 255, 255), T + 2)
    cv2.rectangle(img, (375, 405), (385, 417), WALL, -1)                   # wall-tone post in the gap
    s = analyze_structure(img)
    assert not s.rejected_candidates
    assert any("kept: tone matches the walls" in j for c in s.candidates for j in c.jambs)


def test_piers_between_windows_are_supported_jambs():
    img = np.full((520, 760, 3), 255, np.uint8)
    _room(img)
    for x0, x1 in ((120, 200), (212, 292), (304, 360)):                    # three windows, short piers between
        img[56:65, x0:x1] = 255
        for dy in (-3, 0, 3):
            cv2.line(img, (x0, 60 + dy), (x1 - 1, 60 + dy), (0, 0, 0), 1)
    s = analyze_structure(img)
    assert not s.rejected_candidates
    found = [c for c in s.candidates if abs(c.center[1] - 60) < 8 and c.center[0] < 370]
    assert len(found) == 3, [(c.start, c.end, c.jambs) for c in found]
    assert all(c.jambs and all(not j.startswith(("short", "piece")) for j in c.jambs) for c in found)


def test_stair_treads_between_two_walls_are_not_openings():
    img = np.full((520, 760, 3), 255, np.uint8)
    _room(img)
    cv2.line(img, (520, 60), (520, 460), (0, 0, 0), T)                     # stairwell wall
    for y in range(120, 400, 22):                                          # treads wall to wall
        cv2.line(img, (385, y), (515, y), (0, 0, 0), 1)
    s = analyze_structure(img)
    treads = [c for c in s.candidates if 390 < c.center[0] < 515 and 110 < c.center[1] < 410 and c.source != "wall_gap"]
    assert not treads, [(c.start, c.end, c.source) for c in treads]


def test_wall_tone_learns_one_or_two_tones_and_is_neutral_on_black_and_white():
    img = np.full((520, 760, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 60), (700, 460), (0, 0, 0), T)
    cv2.line(img, (380, 60), (380, 460), (150, 150, 150), T)               # interior walls in gray
    cv2.line(img, (60, 260), (380, 260), (150, 150, 150), T)
    mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 200).astype(np.uint8) * 255
    tone = learn_wall_tone(img, mask, T)
    assert tone.reliable and len(tone.tones) == 2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    assert tone.distance(lab[60:64, 100:200]) < 1 and tone.distance(lab[258:262, 100:200]) < 1
    bw = np.full((520, 760, 3), 255, np.uint8)
    cv2.rectangle(bw, (60, 60), (700, 460), (0, 0, 0), T)
    tone_bw = learn_wall_tone(bw, (cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255, T)
    black_blob = np.zeros((10, 3), np.uint8) + np.array([0, 128, 128], np.uint8)
    assert tone_bw.distance(black_blob) < 1                                # black furniture: no tone evidence against it
