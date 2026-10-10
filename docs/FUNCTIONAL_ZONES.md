# Functional zones and objects (2026-10-08, updated 2026-10-09)

**Space** = structural region bounded by walls and openings. **Zone** = functional region inside a
Space (kitchen, dining, living), bounded by implied edges such as counters, partial walls and
furniture groups.

## The reported case: 3.pdf page 5

An electrical sheet with no room names. The kitchen, dining and living areas share one open space.

| | Before (28f3572) | After |
|---|---|---|
| Structural space | `space_1`, one open space, unnamed | `space_1`, unchanged (one open space) |
| Zones | none | Kitchen 0.99 (17.5 m²), Dining 0.63 (15.8 m²), Living 0.80 (19.5 m²); circulation 2.5 m² |
| Kitchen evidence | — | range (4 burners in a grid), 2 sink basins, counter, dishwasher (word 'WASHER' among kitchen fixtures), refrigerator (word), counter run (assembly holding the fixtures) |
| Dining evidence | — | dining set (table with 6 chairs on 2+ sides) |
| Living evidence | — | sofa (245×89 cm, cushions), loveseat, coffee table |
| Boundaries | — | kitchen \| dining: counter / fixture edge; dining \| living: implied (open) |

The result is identical with and without the cleaning toggle.
`tests/test_zones.py::test_real_unlabelled_open_plan_has_kitchen_dining_living_zones` keeps it solved.
The ground truth is in `benchmark/fixtures/zones.json`.

## How it works

1. **Objects** (`engine/arch/objects.py`). The cleaner's non-architectural vector geometry is
   preserved and typed. Scale comes from printed dimensions or the plan's own doors (~82 cm).
   - **Primitives.** Circles fitted to drawn arcs; a 2×2 grid of equal circles not inside a basin
     is a range. Rounded basins are matched quadruples of outward corner arcs (sinks). Counter
     fronts are lines parallel to a wall at counter depth.
   - **Components.** Non-architectural ink is cut away from walls, so furniture against a wall stays
     its own object. Standard dimensions and sub-loops identify:
     - a dining set: a table with ≥ 3 adjacent chairs on ≥ 2 sides, or separate pieces arranged so;
     - sofas: seat rows of touching cushions;
     - loveseats, armchairs, coffee and side tables, media units;
     - beds (against a wall), tubs and toilets (curved bowl).
   - **Words.** OCR or text-layer words: REFRIGERATOR, DISH / WASHER, RANGE, DRYER…
   - **Context resolves ambiguity.** A sink or a "washer" among decisive kitchen fixtures is a
     kitchen sink or dishwasher. A large drawn assembly holding ≥ 2 kitchen fixtures is a counter run.
2. **Zones** (`engine/arch/zones.py`).
   - **Evidence.** Typed objects and printed labels inside a space each propose function weights.
     A label contradicted by geometry wins (the object is down-weighted).
   - **Acceptance.** Items cluster per function. A cluster is accepted with enough support: several
     items, a label, or one decisive object. Repeats of one kind count with diminishing returns.
   - **Which spaces get zones.** Zones are made for kitchen / dining / living, and only when a space
     holds at least two accepted clusters. With one function the space gets that function (only if
     unnamed, and only on decisive evidence). Bedrooms, baths and laundries characterise whole
     spaces, never zones.
   - **Extent.** Zones grow from their items through the floor, at the same pace, up to 2.2 m.
     Floor no zone reaches stays circulation.
   - **Boundary cues.** Each boundary is recorded as counter / fixture edge, partial wall or implied.
3. **Building model.** `building.spaces[i].zones` (function, confidence, polygon, area, evidence,
   boundaries), `building.spaces[i].function`, `building.objects`, `building.zones_summary`. The
   Workbench Building view tints zones and draws dashed implied boundaries, the function and
   confidence, and the supporting objects.

## Measurement (`python -m benchmark.zones_eval`)

Ground truth: `fixtures/zones.json` (3.pdf p5), plus the open-plan groups of `pdf_plans.json`
(22.pdf p5–p8: each unit's Living / Dining / Kitchen; 3.pdf p4) and `real_plans.json` (dev images).

| | Zone points correct | Wrong function | No zone | Zones without GT open plan |
|---|---|---|---|---|
| All (PDF + dev images), labels + objects | **87 / 99** | 4 | 8 | 0 |
| 3.pdf p5 (reported case, objects only: no labels on the sheet) | **3 / 3** | 0 | 0 | 0 |
| PDF pages, objects only (`FLOORPLAN_ZONE_LABELS=0`) | **64 / 77** | 9 | 4 | 0 |

Before this work: 0 zones anywhere.

Rooms, names and openings are unchanged (dev benchmark identical; PDF GT 66/74, 18/19). Objects add
about 0.4 s on a 19 MP sheet.

## Limits

- **Raster plans** have no typed objects yet; there, zones come from labels only.
  - 8.png, 14.png and 21.png ("no zone"): a second open-plan label was not read, so only one
    function is supported.
  - 12.png: its other function lies in a separate structural space.
- **Wrong functions (4):** in one unit per 22.pdf page, a Dining GT point falls in the kitchen or
  living zone, because that unit's dining evidence (table, chairs, label) was not recognised.
- **Zone extents** are geodesic Voronoi cells around the evidence. They do not yet follow
  circulation paths or ceiling / floor changes.
- **Object typing** uses standard furniture dimensions. Unusual furniture (round dining tables,
  sectional sofas drawn as one outline) or very schematic drawings will give fewer objects. The
  rules return "no zone" in that case rather than a wrong one.

## One reconstruction, four views (2026-10-09)

**The problem.** The zones appeared in Building only. The four Workbench views read three different
representations:

| View | Read before | Reads now |
|---|---|---|
| Building | server PNG of the structured plan (`building`) | unchanged |
| Rooms | `rooms` / `unlabeled_spaces`, the legacy room records | the same records, plus `zones` and `objects` projected from the structured plan |
| Cleaned | the cleaned skeleton, plus the legacy room records drawn over it | the same, plus the zones (a layer that can be switched off) |
| Elements | the cleaner's typed-element image only | that image, plus the typed objects (boxes coloured by function, linked to the Objects table) |

The legacy records had their own label-only zone extents (`_assign_functional_zones`), a second zone
implementation. Objects existed only inside `building`.

**The decision.** The structured plan is the canonical reconstruction.
- `engine/arch/projection.py` projects it onto the flat records instead of inferring anything again:
  - `result.zones` (each with its `space_id`, and the room ids that name it);
  - `result.objects` (each with its `space_id` and `zone_id`);
  - `unlabeled_spaces[i].function` / `.zone_ids`;
  - `rooms[i].zone_id`.
- A labelled open-plan room (method `open-plan-zone`) takes its zone's extent as its boundary, so a
  zone has one geometry in every view. The room is still a zone, never a walled room.
- Three region kinds stay distinct:
  - a structural space (walls and openings);
  - a room (a named space, or a named part of a shared space);
  - a zone (an inferred functional part of a space, bounded by implied edges).
- No wall is invented and no structural space is split.

**Each view keeps its own job:**
- **Rooms:** spaces and rooms, with zones as dashed, tinted, interactive regions. A "Functional zones"
  table gives function, confidence, evidence, boundary cues and parent space. Unnamed spaces show
  "open plan: 3 functional zones" or "probably bedroom (bed)".
- **Cleaned:** the skeleton the walls were recognised on, with the same spaces and zones. It shows the
  zones sharing one wall-bounded space.
- **Elements:** typed vector elements and the objects read from the suppressed ink, plus an "Objects"
  table.
- **Building:** unchanged.
- **Scale:** the metrics panel shows the structured plan's scale estimate (e.g. "≈ 1.83 px/cm,
  median of 8 door widths") where printed dimensions give none.

**Furniture no longer depends on the wall reading.**
- **Before:** objects were read only when the reconstruction chose the vector-cleaned reading. They
  also needed the cleaner to type the page's walls.
- **Now:** the page's vectors are read for furniture whichever reading wins. They run in the
  structural job, alongside OCR.
- **Walls the cleaner cannot type** (e.g. solid poché, 27.pdf): the vectors are kept untyped, and the
  walls come from the structural wall mask.
- **Elements view:** it is then available with an elements-only image. The Cleaned view says why it
  has no cleaned plan.

## Object recognition, measured (`python -m benchmark.objects_eval`)

Hand-labelled boxes are in `fixtures/objects.json`.
- A detection is **correct** when it overlaps a box of a compatible kind.
- It is **wrong kind** when it overlaps a box of another kind (a washbasin typed as a toilet).
- It is **false** when it overlaps nothing (legend symbols, lamps, electrical symbols, tags).

| Plan | Role | Before (ea6865f) precision / recall | Now precision / recall |
|---|---|---|---|
| 3.pdf p5 (35 objects) | development | 0.31 / 0.43 | **0.94 / 0.83** |
| 22.pdf p6, unit 201 (31) | development (labelled before tuning) | 0.19 / 0.23 | **0.87 / 0.65** |
| 27.pdf p1 (12) | **held out**: labelled blind, never tuned on | 0 objects (no vector layer read) | 0.29 / 0.17 |

What changed (general drafting conventions, no drawing-specific values):

- **Placement:** only objects inside the walls count. The convex hull of each building's walls drops
  legend symbols, title-block words and equipment outside the exterior walls.
- **Basins:**
  - a rounded rectangle is a basin only with its drain inside, and is at least 22 cm across;
  - an oval bowl with a drain is a basin;
  - a basin among kitchen fixtures is a kitchen sink;
  - a single oval bowl, or a basin beside a toilet, shower or tub, is a washbasin.
- **Toilet vs. wall basin:** both are a rounded bowl against a wall. The bowl runs away from the
  wall for a toilet, and is round or wider along the wall for a basin.
- **Showers:** the crossed-rectangle symbol, or the word SHOWER / TUB.
- **Beds:**
  - small pieces at a bed's corners are nightstands;
  - pieces inside a bed outline (pillows, blanket folds) are part of the bed.
- **Wardrobes:** a closet-deep box (≥ 45 cm) with many cross strokes is a wardrobe (hangers on a rod).
  A shallower one is a dimension string's ticks, not furniture.
- **Seats:**
  - a lone seat-sized square is a symbol;
  - seats count only beside a table, counter or another seat.
- **Tags:** a box-like table or chair enclosing a coded annotation (KIT-1, A-3.7, 2 % SLOPE) is a
  tag. Room names (LIVING) are not codes.
- **Furniture groups:** inside a group (a living set on a rug), a seat frame is an armchair and a
  small square holding a lamp is an end table.
- **Range:** the burner grid is limited to a cooktop's span, so outlet symbols beside it no longer join.
- **Duplicates:** one object per drawn thing.

**Effect on zones (`zones_eval`):**
- **With labels:** unchanged at 87 / 99 correct, 0 unsupported.
- **From objects only** (`FLOORPLAN_ZONE_LABELS=0`, PDF pages): 64 → **69 / 77** correct.
  - Wrong functions fell from 9 to 4; 3.pdf p4 is now 5 / 5.
  - Part of the earlier object-only support came from false objects: tag boxes read as armchairs
    held up some living zones. Two changes now carry those zones:
    - **Seating arrangements:** end tables beside a sofa or loveseat count as living evidence.
    - **Kitchen sink by context:** a bowl among kitchen fixtures is a kitchen sink.

**Runtime** (A/B with OCR from the cache, two runs each):
- **3.pdf p5:** 43.0 → 43.5 s.
- **22.pdf p6:** 73.8 → 74.0 s.
- **27.pdf p1:** about +1 s, now that its vectors are read for furniture.

**Limits (measured).**
- **Held-out page (27.pdf, UK drafting style):** recall is low. Bath fixtures against solid-poché
  walls and a dresser are missed, and a stair is read as a sofa. This style needs its own work.
- **22.pdf:**
  - the toilets (≈ 40 px, a glyph inside the bowl) are missed;
  - tub/showers are missed: they merge with neighbouring fixtures, and their words are vector glyphs
    that OCR misreads;
  - bar stools touching the island are missed.
- **Raster plans:** still no objects.

## Furniture, second round (2026-10-09/10)

Precision / recall (`python -m benchmark.objects_eval`):

| Plan | Role | P / R |
|---|---|---|
| 3.pdf p5 | development | 0.92 / 0.89 |
| 22.pdf p6 (unit 201) | development | 0.88 / 0.71 |
| 27.pdf p1 | development since this round (first blind score 0.29 / 0.17) | 0.75 / 0.50 |
| 25.pdf p4 | held out; diagnosed once, not rule-tuned (first score 0.14 / 0.09) | 0.21 / 0.27 |
| 25.pdf p5 | **held out, labelled blind, scored once** | **0.00 / 0.00 (0 of 21)** |

What changed:
- **Fixtures from closed outlines:** tub, shower tray, basin with a drain, and a toilet bowl with
  its tank at a wall.
- **Context rules:**
  - an outline on an appliance is a detail of it;
  - a piece inside a fixture is part of it;
  - a "basin" at a bed's corner is a lamp;
  - a basin without kitchen context is a washbasin.
- **Untyped vectors:** read when the cleaner cannot type the walls.
- **Small-scale sheets:** read at a canonical 1 px/cm.

**Why 25.pdf p5 scores zero: scale, not recognition.**
- That sheet's poché layer types no doors, so the scale falls back to the analysis' wall-gap widths.
- Their median (35 px) is inflated by merged / spurious openings; the same set's p4 measures doors
  of ~23 px.
- The scale is therefore over-estimated ~1.5×, every fixture appears 2/3 of its size, and every
  size rule fails.

The next step is a more robust scale: door swing-arc radii or standard fixture sizes, validated on
every page with a known scale.
