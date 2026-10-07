# Physical spaces, functional zones and open plans

## Representation

```
cells            structure only: regions closed by walls and by every sealed wall gap
  │
physical space   one or more cells; cells are joined across an *open connection*
  │
functional zone  a named function (kitchen, living, bedroom ...) inside a physical space
```

* A room with walls of its own is one cell = one physical space = one zone. Its boundary is
  the cell (`wall-region`).
* An **open plan** is one physical space holding several zones with no wall between them.
  Each zone's `boundary` is its estimated extent (`open-plan-zone`): the part of the space
  geodesically nearest its label (`room_recovery.functional_zones`). The space itself is in
  `room.physical_space` (`kind: "open-plan"`). Zone borders are estimates, never walls.
* Several **enclosed** room types (bedroom, bath, closet, garage ...) in one cell are a
  structural failure (a wall or door was missed), reported as `physical_space.kind: "merged"`
  with `merged-space-zone` extents. They are not presented as an open plan.

Names and printed sizes never change cells. They are used in two interpretation steps only:

1. **Open connection** (`analyzer._join_open_connections`): two cells are one physical space
   when the opening between them is wider than two of this plan's doors (door width = median of
   doors drawn with a swing arc), carries no door symbol (no swing arc, no double door, no
   glazing), and every room label in both cells is an open-plan function. Locally such an
   opening looks exactly like a wide doorway; the rooms on both sides tell them apart.
2. **Label partition** (`analyzer._split_shared_spaces`): a cell is split at a narrowing between
   its labels only when it holds an enclosed room type (a likely structural merge); a cell
   whose labels are all open-plan functions stays one physical space.

Printed sizes are metadata (`zone_extent`, area); implausible sizes (a side under 2.5 ft /
0.75 m) are ignored.

## Measuring it

`benchmark/physical_eval.py` (fixture `physical`: `must_share` = zones with no wall between them,
`may_share` = zones joined by a doorway or cased opening; labelled from the raw drawings) reports
per room: ok / split / merged / missed, must-share sets held as one space, and how many returned
room extents claim each room point (several = overlapping rooms on screen). `--compare` lists
previously correct, newly fixed and newly broken rooms. The older structural `separation_rate`
does not penalise splitting an open plan and must not be used to judge open space.

Diagnostics: `scripts/diagnostics/open_space_audit.py` (what splits each open-plan group),
`space_audit.py` (structure vs API per room), `leak_finder.py` (where merged rooms connect).

## Investigation log (2026-10-06, development set = `Test Cases`)

* Regression seen in the Workbench: zones of an open plan displayed as the whole shared
  space (test_floorplan Living/Hall/Hallway, 20.jpg nook) after the 'physical first' change.
  Cause: one geometry slot per room. Fixed by the zone representation above.
* Rejected: sealing fewer wide gaps by width + no arc alone (S2) - wide door-less gaps are as
  often real separations (double doors, symbol-less doors, terrace doors): 0 fixed, 5-8 broken.
* Rejected: boundary 'openness' ratio - real doors score like open connections.
* Rejected: ridge-flood zone partition - one label takes the whole space.
* Kept: open-connection join with the semantic guard (6.png, 11.png fixed, nothing broken at
  2.0-2.5 door widths; 1.5 breaks 21.png where a merged cell propagates).
* Ground truth correction: 21.png Kitchen/Dining are separated by a real wall with a door-sized
  opening (must_share -> may_share).

## False merges: doors at wall corners (2026-10-06)

Leak analysis of every falsely merged room (`leak_finder.py`, wide space maps) showed the
dominant cause: **doorways whose jamb is a wall corner** (19.jpg storage / A-V doors, 21.png
hallway passages, 15.png laundry door, 11.png). Where a wall turns at the jamb there is no end
face of its own, so the gap scan never starts there and the two rooms flood together.

`structure._seal_corner_doors` adds such doors, conservatively:

* candidates come from terminal ends of wall bands (`_terminal_band_ends`), scanned like end
  faces but exempt from the corner guards;
* a corner gap is sealed only when the drawing shows a door on it: a swing arc (or double swing)
  of a door-sized width measured in this plan's own door width (single 0.6-1.5, double 1.2-2.5);
* the gap is clean or confirmed from both jambs (a seal along fixture lines is not a doorway);
* both jambs are drawn like the walls (plan wall tone, when reliable);
* the seal separates two rooms and does not cut off a piece smaller than a closet;
* a candidate that repeats an opening already found is not added again (`_same_opening`).

Bare corner gaps (no door symbol) stay open: they also bound open areas (hall into living room,
test_floorplan, 4.jpg), and an unconditional version split those open plans.

Rejected on the way: sealing every corner gap (C1: 13 rooms fixed, 11 broken, open plans split);
door symbol alone (C2: fabricated partitions through rooms on 13.png, bathrooms on 16.png - the
point metric did not see them, the drawings did); without the clean-gap / tone checks (seals
through bathroom fixtures on 8.png / 16.png and along a kitchen counter on 12.png).

## Label assembly: degenerate OCR boxes

11.png's 'LIVING' was boxed 5 px tall next to 'ROOM' (13 px), so the height test kept them as
two labels and the open plan got a spurious 'Room' zone. Words whose boxes are vertically
contained in each other, side by side (not overlapping), now join one line
(`room_labels._contained_vertically`). On all real plans only 11.png's labels change.

## Ground-truth corrections

* 8.png 'Hall' point moved from (176,217) to (215,232): the old point lay inside Bedroom 1's door
  swing (bedroom floor).
