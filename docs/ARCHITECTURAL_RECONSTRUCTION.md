# Architectural reconstruction (2026-10-08, phase 2)

The engine no longer commits to a single reading of a plan. It reconstructs the architectural model
that best explains the evidence, and reports that model as a building with walls, openings and
spaces.

```
evidence (raster, PDF vectors, PDF text layer, OCR)
  -> readings (hypotheses)   engine.reconstruction
       default | normalised scale | wall class from the erosion cliff | Plan Model | page vectors cleaned
  -> scored on evidence the structure did not use (room labels, text blocks)
  -> chosen structure        engine.structure (walls, gaps, sealed spaces)
  -> openings                engine.opening_detection (+ engine.arch.symbol_scale, engine.arch.openings)
  -> rooms, names            engine.analyzer, engine.analysis.context_ocr
  -> structured plan         engine.arch.building (+ envelope, wall runs, door / window objects)
  -> views                   engine.arch.building_view (Workbench 'Building'), building.json
```

## Readings (engine.reconstruction)

Branching only on structural signals. Easy plans run once.

| Signal | Alternative reading | Why |
|---|---|---|
| Walls ≤ 6 px (under-resolved) | The whole plan at a normalised scale (walls ~12 px, from the plan's own wall thickness), mapped back by `engine.rescale` | Below ~6 px, walls, door leaves, furniture and text differ by a pixel. 7.png: 2 px partitions next to 4 px exterior walls |
| No space / one space holds most of the floor / wall mask floods or vanishes | Wall class re-read from the *cliff* of the erosion curve (`structure.cliff_wall_class`) | Anti-aliased / JPEG walls erode gradually. The plateau rule can latch onto a furniture tail (5.jpeg at ×1.2: kernel 20, 0 spaces) |
| Default reading found no space | Plan Model (now competes; no longer overrides) | Hollow / hatched walls |
| A PDF page whose raster reading is broken | The page's own vector geometry, cleaned (`engine.cleaning`) | CAD sheets. Used without the cleaning toggle when it explains the page better |

The score uses evidence the structure did not use. A label of a normally walled room (bedroom,
bath…) should not share a space with another label (open-plan names may). No label should land in
a wall or outside. Geometric text blocks (read or not) should sit one or two per space.
`score = -(2·merged label pairs + 1.5·lost labels + 0.5·merged blocks + 0.25·lost blocks)`.
The default wins ties. The report is in `response['reconstruction']`. `FLOORPLAN_HYPOTHESES=0`
disables the search.

Cost control:
- The wall-class re-read runs only up to 8 MP (on a 24 MP CAD sheet it took 73 s and was never chosen).
- When the page's vectors give a reading, neither the raster re-read nor the Plan Model is prepared.

## Openings as architectural objects

- **Symbols read where they are legible** (`arch.symbol_scale`). On an under-resolved plan, each
  opening keeps the default reading's geometry but takes its type from the normalised reading of
  the same opening. That's one primitive, two representations, and the legible one decides.
- **Dominant window symbol.** Suppose a drawing marks doors with door symbols, and outline frames
  outnumber its glazing-line windows. Then its outline-only exterior openings are windows. Where
  glazing-line windows dominate, an outline exterior gap is an open side.
- **Swing arcs at the drawn hinge.** Hinges sit on the wall face, on the axis or in between, so
  arcs are tested at all three offsets and ±6% radius.
- **Openings → walls.**
  - A long gap found from one wall end is kept when a door is drawn in it.
  - Doors at T-junction jambs are confirmed by their swing arcs.
- **Hatch is not an opening.** A line-bridged opening at an angle no wall takes must be
  door-sized (doors across a 45° corner). Longer ones are hatch or tiles.
- **Objects.**
  - Door: operation (hinged / double hinged / sliding / unspecified), leaves, hinge points, the
    space the leaf swings into, leaf angle.
  - Window: exterior or onto an outdoor space (terrace, balcony, porch), adjacent space, glazing.
  - Passage.

## The structured plan (engine.arch.building)

| Element | Contents |
|---|---|
| Buildings | Footprints (envelope: the sealed wall network that carries openings; frames, title blocks and leader-line appendages excluded) |
| Walls | Wall bands and **wall runs**: collinear pieces joined across the openings they host. "One wall with a door", with solid and open length and an exterior flag |
| Openings | Host wall run, the two places connected (found geometrically on both sides), role (exterior / interior / to an outdoor space / inside one space), door / window / passage object |
| Spaces | Names, kind (room / open-plan / outdoor / unnamed), area, building, doors / windows / passages, exterior openings, neighbours through openings and walls |
| Graph | Spaces plus the exterior; edges are door / window / opening / wall |

## Measurement and diagnosis

| Harness | Measures |
|---|---|
| `benchmark.real_plans --full` | Rooms (structure and API), names, probes |
| `benchmark.openings_eval [--analyzer]` | Door / window precision and recall, confusions, passages, false openings (9 plans with opening GT) |
| `benchmark.topology_eval` | Opening → wall, door → spaces, window → exterior, swing side |
| `benchmark.scale_robustness` | The same plans resampled (×0.8 / ×1.25 / ×1.5): API room recovery |
| `benchmark.pdf_plans --variants A,F,E` | PDF GT: default path (A), default path with the vector hypothesis (F), toggle (E) |
| `scripts.diagnostics.explain_plan <plans>` | Per-plan error report with the engine's evidence: missed / false / wrong-type openings, passages as doors, merged / missed rooms, unnamed rooms, ambiguous readings, plus an image |
| Workbench 'Building' view | The structured plan over the drawing |
