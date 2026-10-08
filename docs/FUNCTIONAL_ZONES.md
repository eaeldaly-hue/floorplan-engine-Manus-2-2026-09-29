# Functional zones (2026-10-08)

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
