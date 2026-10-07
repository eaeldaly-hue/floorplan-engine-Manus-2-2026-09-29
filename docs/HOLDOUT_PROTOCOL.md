# Real-plan hold-out: labelling and measurement protocol (N.3a)

The hold-out is a sealed set of real floor plans that is never used for development, tuning,
debugging or benchmark decisions. It measures how the engine generalises; it is reported
separately from the 17-plan development set (`benchmark/fixtures/real_plans.json`).

Files:

* plans: `../floorplan_test_dataset_01/` next to the project (`pdf/`, `images/`; override with
  `FLOORPLAN_HOLDOUT_PLANS`)
* ground truth: `benchmark/fixtures/holdout_plans.json` (keys are paths inside the dataset)
* runner: `python -m benchmark.real_plans --holdout [--full]`
* metrics: `benchmark/holdout_metrics.py`
* how a plan reaches the engine: `benchmark/holdout_inputs.py` (below)
* labelling aid: `scripts/diagnostics/label_grid.py` (the analysis frame + coordinate grid only)
* review aid: `scripts/diagnostics/holdout_review.py` (ground truth drawn on the plan, no engine output)

### Analysis frame

Labels are pixel coordinates in exactly the image the engine analyses
(`benchmark/holdout_inputs.py`, shared by the labelling tool and the runner):

* PDF pages: the production PDF path — the page rendered at its analysis DPI (150, lowered for
  large sheets to stay within 24 MP) — and the page's text layer passed to the analyzer as
  text evidence, as the Workbench does (`page` in the fixture).
* Images above 24 MP: reduced with INTER_AREA by the factor recorded in the fixture
  (`analysis_scale`); the upload path would reject them (> 25 MP), which would measure nothing.
  Smaller images are used as they are. Original files are never modified.

Freezing records the hash of every plan file and of every analysis frame (`frozen.frames`), so
a change of renderer or reduction that would move pixels under the labels is reported.

### Labelled region

A plan may be labelled only inside `roi` ([[x0, y0, x1, y1], ...]) when the rest cannot be
labelled with confidence (dense multi-unit sheets, several plans on one sheet). Engine spaces
and opening candidates outside the region are neither credited nor counted as false; every
physical space inside it is labelled.

## 1. Order of work (no step may be skipped or reordered)

1. **Novelty check.** Each new plan is compared with the development set: file hash, and a
   perceptual comparison (downscaled, normalised images) against every development plan.
   Duplicates and near-duplicates (same plan at another size, crop or format) are excluded.
2. **Label** every plan once, from the raw image only (section 2).
3. **Save** the ground truth in `holdout_plans.json`.
4. **Freeze**: `python -m benchmark.real_plans --holdout --freeze` records a hash of the
   ground truth and of every plan file. Freezing is possible only once.
5. **Verify** the freeze (the runner re-hashes both and reports any difference).
6. Only then run the engine baseline, once.
7. No threshold, rule or code is changed because of hold-out results. Labels are not edited
   after the freeze; a genuine labelling mistake found later is recorded as an erratum in a
   separate field, never by editing the frozen labels.

## 2. Ground-truth sources

Allowed: the raw plan image (or PDF page) and the coordinate grid of `label_grid.py`, which adds
coordinates only.

Not allowed: engine output of any kind (overlays, wall masks, spaces, polygons, opening
candidates), OCR output, previous benchmark results, the Workbench.

Label only what can be identified confidently on the drawing. When in doubt, leave it out
and say so in `gt_notes`. Never assume that a compact or isolated component is furniture.

## 3. What is labelled

### Physical spaces (`spaces`)

A physical space is an area bounded by structure (walls, partitions, openings in walls, the
building outline). Every physical space gets an id and at least one interior point:

```json
"spaces": [
  {"id": "S1", "kind": "open_plan", "points": [[410, 220], [520, 300]]},
  {"id": "S2", "kind": "room", "points": [[150, 120]]}
]
```

`kind`: `room` (one functional room), `open_plan` (several functional areas in one physical
space), `circulation` (hall, corridor, landing), `closet`, `wet` (bath, WC, laundry),
`outdoor` (terrace, balcony, porch inside the drawn outline), `other`.

A boundary between two physical spaces needs structural evidence on the drawing: a wall or
partition (with or without a door or opening in it). A change of floor finish, a rug, a
counter, an island, a sofa or a dashed "area" line is **not** a physical boundary.

### Functional rooms / zones (`rooms`)

Each named or visually identifiable function gets an entry pointing to its physical space:

```json
"rooms": [
  {"name": "Kitchen", "printed": "KITCHEN", "point": [430, 210], "space": "S1"},
  {"name": "Living room", "printed": "LIVING", "point": [530, 310], "space": "S1"},
  {"name": "Bedroom", "printed": "BEDROOM 2", "point": [150, 120], "space": "S2"},
  {"name": "Closet", "printed": null, "point": [90, 60], "space": "S3"}
]
```

* `printed` is the name as printed on the drawing (null when nothing is printed). Only rooms
  with a printed name count for named-room recall.
* Several entries with the same `space` are functional zones of one open-plan space. No wall
  or polygon is drawn between them: **physical space ≠ functional room / zone**.
* `point` is the location of the zone (its label when printed, else the middle of the area
  used for that function).
* A physical space without any function entry is an unnamed physical space; it is still a
  physical space.

### Openings (`openings`)

`door`, `window`, or `opening` (real but ambiguous: passage, glazed door, folding wall), as two
end points on the wall axis and the wall thickness (same format as the development set).

### Non-structural elements (`nonstructural`)

Elements that could be mistaken for structure, labelled only when identifiable with confidence:

```json
"nonstructural": [
  {"kind": "furniture", "what": "sofa", "point": [300, 400], "bbox": [270, 380, 340, 420]},
  {"kind": "stairs", "what": "stair run", "point": [600, 200], "bbox": [580, 120, 640, 300]}
]
```

`kind`: `furniture`, `fixture` (bath, shower, WC, sink, appliance), `counter` (kitchen counter,
island, vanity), `shelving`, `stairs`, `hatching` (tile / floor pattern), `marker` (coloured
symbol, sensor, legend dot), `text`, `other`. `bbox` = [x0, y0, x1, y1] around the element
when its extent is clear; `point` lies on the element's ink.

### Protected structural pieces (`protected`)

Points on short real structure that must stay wall: piers between windows, mullions, wall
stubs, free-standing columns.

### Per-plan notes

`style` (resolution, colour, wall drawing style, scanned / CAD / render) and `gt_notes`
(anything ambiguous or deliberately left out).

## 4. Metrics (kept separate; `benchmark/holdout_metrics.py`)

Each measure is reported for structure (the engine's segmentation) and, where it applies,
for the API response (named rooms + unnamed spaces).

1. **Physical-space recall**: a GT physical space is *recovered* when all its points fall in
   one engine space that holds no point of another GT space. Otherwise it is *split* (points in
   several engine spaces), *merged* (shares an engine space with another GT space) or *missed*
   (a point is not in any space). Unnamed spaces count like any other.
2. **Named-room recall**: printed names recognised and associated (development-set rules).
3. **Unnamed physical spaces**: GT spaces with no printed name, and how many the API returns
   as unnamed spaces; an unnamed space is never counted as missing.
4. **Room point → physical space**: each function point lies in the engine space that
   recovers its GT physical space.
5. **Open-plan groups**: GT spaces with two or more functional zones.
6. **Correct open-plan grouping**: all zones of the group in one engine space, holding no
   other GT space.
7. **Incorrect split of an open-plan space** (subset of 1).
8. **Incorrect merge of separated rooms**: engine spaces holding points of two or more GT
   physical spaces.
9. **Functional zones identified**: zones of open-plan spaces whose printed name is
   recognised, associated with the shared space, and reported as sharing it.
10. **Furniture-induced false spaces**: engine spaces with no GT point that lie inside a
    labelled non-structural element or are bounded mainly by one.
11. **Furniture-induced false boundaries**: sealed gaps with an end on a labelled
    non-structural element, and boundaries between two engine spaces of one GT space that run
    mainly through labelled non-structural elements (including the open-plan splits they cause).
12. **Non-structural ink as wall**: labelled non-structural points / bboxes covered by the wall
    mask, by kind.
13. **False opening candidates by cause**: candidates not matching any GT opening, attributed to
    a labelled non-structural element (by kind) when an end or the centre lies on it.
14. **Door / window candidate recall and classification** (development-set matching).
15. **Unsupported / fabricated boundaries**: API boundaries estimated from wall rays or
    dimension boxes, and engine boundaries that split one GT physical space.

Not counted as failures: an unnamed physical space; an open-plan physical space recovered
as one space; a functional zone without its own walls. Counted as failures: any false physical
boundary or false space, including those created by non-structural elements.
