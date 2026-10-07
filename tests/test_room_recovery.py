"""Room recovery inside a space that holds several room labels (engine.room_recovery)."""

import cv2
import numpy as np

from engine.room_recovery import partition_space


def _space(width_passage):
    """Two 300x300 rooms joined by a passage of the given width; returns (region, strokes)."""
    region = np.zeros((500, 800), bool)
    region[100:400, 50:350] = True
    region[100:400, 450:750] = True
    mid = 250
    region[mid - width_passage // 2:mid + width_passage // 2, 350:450] = True
    strokes = np.zeros(region.shape, np.uint8)
    cv2.rectangle(strokes, (120, 160), (220, 220), 255, 1)       # furniture outline inside a room
    return region, strokes


def test_two_labelled_rooms_joined_by_a_doorway_are_split_at_the_doorway():
    region, strokes = _space(60)
    parts = partition_space(region, strokes, [(200, 300), (600, 300)], min_area=2000)
    assert parts is not None and len(parts) == 2
    left = next(mask for mask, members in parts if members == [0])
    right = next(mask for mask, members in parts if members == [1])
    assert left[250, 100] and not left[250, 700]
    assert right[250, 700] and not right[250, 100]
    # the cut lies in the passage, not inside either room
    assert left[:, 50:350].sum() >= 0.97 * region[:, 50:350].sum()


def test_labels_in_one_open_room_are_not_split():
    region = np.zeros((500, 800), bool)
    region[100:400, 50:750] = True
    strokes = np.zeros(region.shape, np.uint8)
    assert partition_space(region, strokes, [(200, 250), (600, 250)], min_area=2000) is None


def test_wide_opening_between_rooms_keeps_them_together():
    """A 260 px opening between 300 px rooms is an open plan, not a doorway."""
    region, strokes = _space(260)
    assert partition_space(region, strokes, [(200, 250), (600, 250)], min_area=2000) is None


def test_split_does_not_depend_on_where_the_label_text_sits():
    """Labels placed close to the doorway (not at room centres) still split at the doorway:
    each label climbs to the interior of its own room before the rooms are grown."""
    region, strokes = _space(60)
    parts = partition_space(region, strokes, [(330, 240), (470, 260)], min_area=2000)
    assert parts is not None and len(parts) == 2
    left = next(mask for mask, members in parts if members == [0])
    assert left[:, 50:350].sum() >= 0.97 * region[:, 50:350].sum()
    assert left[:, 450:750].sum() == 0


def test_two_labels_in_one_room_share_it():
    region, strokes = _space(60)
    parts = partition_space(region, strokes, [(150, 200), (250, 330), (600, 250)], min_area=2000)
    assert parts is not None and sorted(sorted(m) for _, m in parts) == [[0, 1], [2]]
