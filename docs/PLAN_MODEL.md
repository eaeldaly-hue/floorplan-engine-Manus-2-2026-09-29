# Architectural Plan Model (`engine/plan/`)

An explicit model of the building a drawing represents, built deterministically from the ink
(OpenCV; no learned model, no per-file tuning). Status: **experimental, in production only as a
fallback** (see *Integration*).

| Element | Representation | Built in |
|---|---|---|
| Stroke | every straight drawn line: centerline, width, darkness, roles (negative evidence) | `primitives.py` |
| Wall | centerline + thickness + style (`filled`, `hollow`, `hatched`, `thin`), confidence, ink support, provenance | `walls.py`, `reconstruct.py` |
| Wall network | L/T junctions, collinear pieces merged, door leaves removed, connected components | `reconstruct.py` |
| Opening | interval closing a gap between walls: `door` / `window` / `opening`, evidence, the spaces it connects | `reconstruct.py` |
| Space | face enclosed by walls and opening closures: polygon, area, labels found inside | `reconstruct.py` |
| Topology | space - space / exterior adjacency through openings | `reconstruct.py` |
| Status | `ok` / `partial` (walls, no space) / `no_structure`, warnings | `model.py` |

## Pipeline

1. **Strokes.** LSD line segments on the engine's binarised ink. The width of each one is measured
   by walking perpendicular into the ink from the first ink pixel (a thin line's edge can sit a pixel
   off). Then:
   - The width is the lower quantile over 24 samples, because lines that touch the stroke only make
     it look wider.
   - Where the paper position wanders along the stroke (a dense hatch), slits of 1-2 px are bridged.
   - The two edges of one stroke, and its collinear pieces, are merged (circular mean of doubled
     angles).
2. **Wall hypotheses.** The same wall in any drawing convention:
   - Two parallel faces with paper or hatch between them (face pairs, scored against the plan's
     dominant separations, the wall classes).
   - A filled band.
   - A single line in the plan's wall pen.

   Negative evidence:
   - Lattices of 4 or more equally spaced parallel strokes (tiles, treads, hatching).
   - Compact solid blocks (furniture).
   - Thin bars in a plan whose walls are face pairs (door leaves).
3. **Analysis by synthesis.** Each hypothesis is redrawn in its own style and must match the ink
   (support of at least 0.7).
4. **Network.**
   - Collinear pieces merge when the break is shorter than the wall is thick (a door or window is
     wider) or when a crossing wall fills it.
   - Wall ends snap to crossing walls.
   - Door leaves (hinged on a wall, with a swing arc of their length) are removed.
5. **Openings, only with evidence.** For each free wall end, the nearest collinear partner or the
   crossing wall ahead is classified from the drawing: swing arc, double arc, leaf, sliding panels,
   or window lines.
   - Gap limits are relative to the plan's own door width, measured from symbol-backed doors.
   - A closure never crosses a wall.
   - Short walls attached to nothing (glyph pairs, fixtures) carry no opening.
   - Glazing (thin line or lines spanning between two walls, in a wall line, with no wall running
     alongside it) closes as a window.
6. **Spaces.**
   - Walls are rendered as bands (extended by half their thickness at the ends) plus the opening
     closures, then joined at a fraction of the wall class.
   - Faces touching the image border are the exterior.
   - A face that holds two or more other faces in its holes is the outside closed by a sheet frame
     or title block, not a room.
7. **Labels and topology.** Room labels are attached to the space they fall in. **A label never
   creates a wall or a space.**

## Integration (`engine/plan/adapter.py`)

`FLOORPLAN_PLAN_MODEL` = `off` | `fallback` (default) | `shadow`.

- **fallback:** the Plan Model runs only when the legacy structure found no wall or no enclosed
  space. When it finds spaces, its walls and spaces replace the empty legacy ones in the
  structure handed to the analyzer (`adapter.as_structure`, legacy format). The regular steps then
  run on them: label association, room records, open-plan zones, unlabeled spaces, overlays and
  topology. They appear in the Workbench like any other rooms, with a warning that says where
  they come from. The full model is also returned as `plan_model`.
  - Paper outside every Plan Model space is left unclassified, not exterior. A unit that leaks
    through an unclosed opening therefore keeps its label (as a room without a boundary) instead of
    having it discarded as text outside the building.
  - Openings (`openings`, door and window counts) are still the legacy engine's. Plan Model openings
    are only in `plan_model`.
  - Plans the legacy engine reads are untouched: same output, no extra time.
- **shadow:** runs on every plan and attaches `plan_model` only (comparison and benchmarks).
- Errors in the Plan Model never break an analysis; they become a warning.

The server must be restarted to pick up engine changes (no auto-reload).

Development plans that trigger the fallback: 18.jpg, 9.jpg, 22.pdf p5-p8, 3.pdf p4.

## Measurements (2026-10-07)

Synthetic wall styles, 12 layouts x 5 styles, room recall at IoU 0.7 (`false` = false spaces):

| Style | Legacy | Plan Model | Plan Model merged / split / false |
|---|---|---|---|
| filled | 1.000 | 0.911 | 1 / 2 / 0 |
| hollow | 0.000 | 0.878 | 5 / 0 / 0 |
| hatched | 0.000 | 0.744 | 8 / 1 / 0 |
| thin | 0.000 | 0.856 | 3 / 6 / 0 |
| filled_furniture | 0.778 | 0.711 | 6 / 6 / 0 |

Development set (`real_plans --full`, 17 plans): summary identical to the Phase 1 baseline (the
fallback does not trigger on any scored plan).

Real plans the legacy engine cannot read (spaces found by the Plan Model; legacy finds 0 on all):

| Plan | Spaces | Time | Notes |
|---|---|---|---|
| 22.pdf p5 | 37 | 8 s | bedrooms, baths and closets found. Unit living areas leak through unclosed openings. Dimension areas form spaces. |
| 22.pdf p6-p8 | 64-67 | 14 s | most unit rooms found. Many false spaces outside the building (setback and dimension cells). |
| 3.pdf p4 | 31 | 5 s | WIC, baths and bedroom #3 found. Some false splits (shelving, counters). Bedrooms #1/#2 merge with the hall. |
| 9.jpg | 3 | 5 s | finely hatched historic walls are mostly missed. |
| 18.jpg | 2 | 0.2 s | — |

Tests: `tests/test_plan_model.py` (19). Full suite: 182 passed.

## Known gaps (next)

- **Plan region.** Spaces outside the building envelope (dimension and setback cells, title-block
  panels) need an envelope test.
- **Openings on real CAD sheets.** Wide sliders and storefronts, and doors whose jambs are not
  walls.
- **Plan Model openings in the response.** `openings` and the openings overlay still come from the
  legacy engine (0 on the fallback plans).
- **Fine hatch and historic drawings** (9.jpg). Dimensions and scale on the model. Space kinds and
  uncertainty flags.
- **Speed.** Stroke measurement (about 6 s on a 24 MP sheet) is the next hot spot.
