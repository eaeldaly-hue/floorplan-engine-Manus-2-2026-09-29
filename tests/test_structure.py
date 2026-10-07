"""Structural geometry: walls, gaps, sealed spaces, topology (deterministic synthetic drawings)."""

import cv2
import numpy as np
import pytest

from engine.structure import EXTERIOR, analyze_structure, estimate_wall_kernel, _binarize

T = 20          # wall thickness (px)
W, H = 900, 700


def canvas():
    return np.full((H, W, 3), 255, np.uint8)


def wall(img, x0, y0, x1, y1, t=T):
    """Axis-aligned wall centred on the segment (x0, y0)–(x1, y1)."""
    h = t // 2
    cv2.rectangle(img, (min(x0, x1) - h, min(y0, y1) - h), (max(x0, x1) + h, max(y0, y1) + h), (0, 0, 0), -1)


def box(img, x0, y0, x1, y1):
    wall(img, x0, y0, x1, y0); wall(img, x0, y1, x1, y1)
    wall(img, x0, y0, x0, y1); wall(img, x1, y0, x1, y1)


def clear(img, x0, y0, x1, y1):
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)


def space_at(s, x, y):
    return int(s.space_labels[y, x])


def test_closed_room_is_one_space_with_no_openings():
    img = canvas(); box(img, 100, 100, 700, 550)
    s = analyze_structure(img)
    assert len(s.spaces) == 1
    assert s.candidates == []
    interior = (s.space_labels == 1).sum()
    assert interior >= 0.95 * (600 - T) * (450 - T)


def test_door_gap_between_two_rooms_is_a_candidate_and_rooms_stay_separate():
    img = canvas(); box(img, 100, 100, 800, 550); wall(img, 450, 100, 450, 550)
    clear(img, 440, 250, 460, 340)                     # 90 px door gap in the shared wall
    s = analyze_structure(img)
    assert len(s.spaces) == 2
    assert space_at(s, 250, 300) != space_at(s, 650, 300)
    [c] = s.candidates
    assert c.relation == "between_rooms"
    assert abs(c.width - 90) <= 4 and c.orientation == "vertical"
    assert s.adjacency_pairs() == {tuple(sorted((space_at(s, 250, 300), space_at(s, 650, 300))))}


def test_small_crack_is_sealed_but_not_reported_as_an_opening():
    img = canvas(); box(img, 100, 100, 700, 550)
    clear(img, 300, 90, 304, 110)                      # 5 px break in the top wall
    s = analyze_structure(img)
    assert len(s.spaces) == 1
    assert s.candidates == []


def test_no_wall_pixels_are_fabricated():
    img = canvas(); box(img, 100, 100, 800, 550); wall(img, 450, 100, 450, 550)
    clear(img, 440, 250, 460, 340); clear(img, 250, 540, 360, 560)
    s = analyze_structure(img)
    ink = cv2.dilate(s.ink, np.ones((3, 3), np.uint8)) > 0
    assert int(((s.wall_mask > 0) & ~ink).sum()) == 0   # walls only where the drawing has ink


def test_l_shaped_room_is_one_space():
    img = canvas()
    wall(img, 100, 100, 700, 100); wall(img, 700, 100, 700, 350); wall(img, 700, 350, 400, 350)
    wall(img, 400, 350, 400, 600); wall(img, 400, 600, 100, 600); wall(img, 100, 600, 100, 100)
    s = analyze_structure(img)
    assert len(s.spaces) == 1
    truth = np.zeros((H, W), np.uint8)
    cv2.fillPoly(truth, [np.array([(110, 110), (690, 110), (690, 340), (390, 340), (390, 590), (110, 590)])], 1)
    pred = s.space_labels == 1
    iou = (pred & (truth > 0)).sum() / (pred | (truth > 0)).sum()
    assert iou >= 0.95


def test_corridor_between_rooms_is_its_own_space():
    img = canvas(); box(img, 100, 100, 800, 600)
    wall(img, 100, 300, 800, 300); wall(img, 100, 400, 800, 400)   # corridor between y=300 and y=400
    wall(img, 450, 100, 450, 300)
    for x0 in (250, 600):
        clear(img, x0, 290, x0 + 80, 310)                            # doors into the corridor
    s = analyze_structure(img)
    corridor = space_at(s, 450, 350)
    assert corridor > 0
    assert len({space_at(s, 250, 200), space_at(s, 650, 200), corridor, space_at(s, 450, 500)}) == 4
    assert sum(1 for c in s.candidates if c.relation == "between_rooms") == 2


def test_exterior_window_gap_connects_room_to_exterior():
    img = canvas(); box(img, 100, 100, 700, 550)
    clear(img, 300, 90, 420, 110)
    for y in (90, 100, 110):                                        # triple-line window
        cv2.line(img, (300, y), (420, y), (0, 0, 0), 1)
    s = analyze_structure(img)
    assert len(s.spaces) == 1                                      # the window seals the room
    [c] = s.candidates
    assert c.relation == "room_to_exterior"
    assert EXTERIOR in c.sides
    assert c.line_coverage >= 0.9


def test_wall_kernel_stays_below_the_thinnest_walls():
    img = canvas(); box(img, 100, 100, 800, 600, ); wall(img, 450, 100, 450, 600, t=12)
    cv2.putText(img, "BEDROOM 12'x10'", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    ink, _ = _binarize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    kernel, line_width, thinnest = estimate_wall_kernel(ink)
    assert 3 <= kernel <= 12
    s = analyze_structure(img)
    assert s.wall_mask[350, 450] > 0                               # the 12 px wall survives
    assert s.wall_mask[340:360, 150:400].sum() == 0                # text does not become wall


@pytest.mark.parametrize("degrees", [-2.5, 1.5, 3.0])
def test_skew_is_estimated_and_results_are_in_original_coordinates(degrees):
    img = canvas(); box(img, 150, 150, 750, 550); wall(img, 450, 150, 450, 550)
    clear(img, 440, 300, 460, 390)
    m = cv2.getRotationMatrix2D((W / 2, H / 2), degrees, 1.0)
    rotated = cv2.warpAffine(img, m, (W, H), borderValue=(255, 255, 255))
    s = analyze_structure(rotated)
    assert abs(s.skew_degrees + degrees) <= 0.3
    assert len(s.spaces) == 2
    [c] = [c for c in s.candidates if c.relation == "between_rooms"]
    centre = m @ np.array([450, 345, 1.0])
    assert np.hypot(c.center[0] - centre[0], c.center[1] - centre[1]) <= 8


def test_scan_speckle_does_not_create_spaces_or_openings():
    img = canvas(); box(img, 100, 100, 700, 550)
    rng = np.random.default_rng(0)
    noise = rng.random((H, W)) < 0.004
    img[noise] = 0
    s = analyze_structure(img)
    assert len(s.spaces) == 1
    assert s.candidates == []



def test_thin_interior_walls_close_to_line_width_are_kept():
    """Short kernel plateau: 5.5 px interior walls, 12 px exterior walls and 3 px symbol
    strokes leave only two flat kernel steps for the interior walls (fresh-validation case)."""
    from benchmark.evaluate import StructureStats
    from benchmark.generator import family_specs, generate
    plan = generate(family_specs("door_exterior", 3, salt=7)[0])
    s = analyze_structure(plan.image)
    assert s.wall_kernel <= 5
    st = StructureStats()
    st.add(s.space_labels, plan.room_labels, s.wall_mask, plan.wall_mask, s.adjacency_pairs(), plan.adjacency)
    m = st.metrics()
    assert m["wall_iou"] >= 0.95 and m["room_recall_iou70"] >= 0.9


# --- Multi-scale wall inference -------------------------------------------------------

def _thin_wall_plan(interior_t=5, line_px=1):
    """Thick exterior (12 px), thin interior partitions only a few px thicker than the lines,
    plus dimension lines with filled arrowheads and text (all 1-2 px)."""
    img = canvas()
    for (x0, y0, x1, y1) in ((100, 100, 800, 100), (100, 600, 800, 600), (100, 100, 100, 600), (800, 100, 800, 600)):
        wall(img, x0, y0, x1, y1, t=12)
    wall(img, 450, 100, 450, 600, t=interior_t)                     # partitions
    wall(img, 100, 350, 450, 350, t=interior_t)
    clear(img, 440, 420, 460, 500)                                   # door in the vertical partition
    clear(img, 250, 340, 320, 360)                                   # door in the horizontal partition
    cv2.line(img, (100, 60), (800, 60), (0, 0, 0), line_px)          # dimension line with arrowheads
    for x, s in ((100, 1), (800, -1)):
        cv2.fillPoly(img, [np.array([(x, 60), (x + 12 * s, 56), (x + 12 * s, 64)])], (0, 0, 0))
    cv2.putText(img, "Bedroom", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv2.putText(img, "Kitchen", (560, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return img


def test_thin_interior_walls_are_recovered_as_a_second_wall_class():
    s = analyze_structure(_thin_wall_plan())
    assert s.wall_mask[230, 450] > 0 and s.wall_mask[350, 180] > 0  # 5 px partitions kept
    assert len(s.spaces) == 3
    rooms = {space_at(s, 250, 200), space_at(s, 250, 480), space_at(s, 650, 350)}
    assert len(rooms) == 3 and 0 not in rooms
    assert sum(c.relation == "between_rooms" for c in s.candidates) == 2
    assert s.wall_mask[50:70, 150:750].sum() == 0                    # dimension line is not wall


def test_heavy_door_leaf_at_a_jamb_is_not_a_partition():
    """A heavy leaf hinged at a wall end must not be taken as a thin wall (it would close the door)."""
    img = canvas(); box(img, 100, 100, 800, 550); wall(img, 450, 100, 450, 550)
    clear(img, 440, 250, 460, 340)
    cv2.rectangle(img, (460, 250), (545, 254), (0, 0, 0), -1)        # 5 px leaf, opened 90°
    s = analyze_structure(img)
    assert len(s.spaces) == 2
    assert any(c.relation == "between_rooms" for c in s.candidates)


def test_glazing_in_a_thick_wall_gap_is_not_a_partition():
    """Heavy (4-5 px) glazing bars inside a 24 px wall's gap continue that wall's axis: they
    are a window symbol, not a thin wall that would close the opening."""
    img = canvas()
    for (x0, y0, x1, y1) in ((100, 100, 700, 100), (100, 550, 700, 550), (100, 100, 100, 550), (700, 100, 700, 550)):
        wall(img, x0, y0, x1, y1, t=24)
    clear(img, 300, 87, 420, 113)
    for y in (90, 100, 110):
        cv2.line(img, (300, y), (420, y), (0, 0, 0), 4)
    for x in range(160, 660, 60):                                     # 1 px drawing lines (furniture)
        cv2.rectangle(img, (x, 300), (x + 40, 340), (0, 0, 0), 1)
    s = analyze_structure(img)
    assert s.wall_mask[100, 330:390].sum() == 0
    assert [c.relation for c in s.candidates] == ["room_to_exterior"]


def test_line_weight_walls_are_recovered_when_the_interior_collapses():
    """Walls drawn with the same weight as the drawing lines: recovered from structure
    (straight lines meeting walls at junctions), not from thickness. Stair treads are not walls."""
    img = canvas()
    for (x0, y0, x1, y1) in ((100, 100, 800, 100), (100, 600, 800, 600), (100, 100, 100, 600), (800, 100, 800, 600)):
        wall(img, x0, y0, x1, y1, t=6)
    cv2.line(img, (450, 103), (450, 400), (0, 0, 0), 2)              # partitions, 2 px like the lines
    cv2.line(img, (450, 460), (450, 597), (0, 0, 0), 2)              # (door gap 400-460)
    cv2.line(img, (103, 350), (450, 350), (0, 0, 0), 2)
    cv2.line(img, (450, 300), (797, 300), (0, 0, 0), 2)
    for y in range(420, 580, 12):                                     # stair treads
        cv2.line(img, (600, y), (700, y), (0, 0, 0), 2)
    s = analyze_structure(img)
    assert len(s.spaces) >= 3
    assert space_at(s, 250, 200) != space_at(s, 250, 500)
    assert space_at(s, 600, 200) != space_at(s, 250, 200)
    w = s.walls
    assert w.line_walls is not None
    assert w.line_walls[420:580, 610:690].sum() == 0                  # treads rejected
