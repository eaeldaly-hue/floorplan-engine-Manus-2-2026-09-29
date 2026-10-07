# Structural simplification / semantic element layer (experimental)

**Question:** does a structurally simplified representation (walls, doors, windows, structural
elements only), built before space reconstruction, help the reconstruction? The original page,
text and metadata stay untouched.

**Answer (2026-10-07, 22.pdf p5-p8):** yes, materially, where the elements can be typed reliably.
- Spaces outside the building went from 24-38 per page to 0.
- Merges went from 1-2 per page to 0-1.
- Every room the simplified pipeline draws is bounded by the real walls; no furniture holes, no
  bleed into setbacks.

It loses whole units, though. Balcony sliders and storefronts are not typed yet, so those units
open to the outside and their labels are dropped. The simplification must be **semantic**: a plain
"keep only walls" image made the result worse (B0 below).

## 1. Where the pipeline builds its representations today

| Stage | Code | Representation |
|---|---|---|
| PDF page | `ingest/pdf.py` `render_page` | raster at analysis DPI (crop box) |
| PDF text | `ingest/pdf_text.py` `text_boxes` | words with boxes in the render frame (merged with OCR) |
| Image text | `engine/analysis/ocr.py` | OCR lines / boxes (room names, dimensions) |
| Legacy geometry | `engine/structure.py` `analyze_structure` | binarised ink → wall mask → spaces, opening candidates |
| Plan Model | `engine/plan/` | strokes (with roles: lattice, block, thin bar) → walls, openings, spaces |
| PDF vectors | `ingest/pdf_vectors.py` (new) | every stroked/filled path with its pen, in the render frame |

Before this experiment, nothing in production used the PDF vectors. Every geometric stage saw all
the ink: walls, furniture, text, dimensions, hatch and frames together.

## 2. Is there enough information to separate structure from annotation?

| Source | What it separates | Limits |
|---|---|---|
| PDF text layer | real text: exact words and boxes | CAD text exported as strokes (SHX) is not in it (22.pdf: dimension strings, W.I.C.) |
| OCR | text in the image | slow; no geometry |
| PDF vectors | pens (width, colour), arcs as curves, fills | no layers (OCGs) or semantics in the test files. Pens map to roles per drawing, not universally (below) |
| Plan Model (raster) | walls by evidence; lattice / block / thin-bar roles | furniture and dimension lines can still pair up as walls; no text knowledge |

22.pdf p5, pen by pen (`output/structural_layer/` and the discovery notes):

| Pen | Content |
|---|---|
| 1.98 pt | the walls and nothing else |
| 0.57 | fixtures |
| 0.84 | stroked text and dimension strings |
| 0.27 | dimension and extension lines |
| 0.27 grey | furniture together with door and window parts |
| 0.12 | ticks |
| 0.12 grey | grid lines |

So one pen can isolate walls. Doors share a pen with furniture and need geometry.

On 3.pdf the walls (outline and hatch) share the general 0.72 pt pen with almost everything, so
pen separation fails there. 23.pdf has a raster drawing.

## 3. Can walls, doors and windows be isolated reliably?

**Walls: yes, by evidence, not by pen width.** Each pen is drawn alone and reconstructed by the Plan
Model. The wall pen is the one that reconstructs as walls over most of its ink (cover ≥ 0.6) *and*
as the richest wall network: on 22.pdf 93-116 walls, against 16 for the sheet frame and title-block
rules, which also pair up.

Fixed-threshold variants failed and are recorded in the module: overlap with the raster walls is
circular, and the frame passes a plain cover test.

**Doors: partly.**
- Swing doors are reliable: a quarter arc whose centre (the hinge) sits on a wall band, and the
  straight leaf of the arc's radius from the hinge. 22.pdf: 99-121 door elements per page.
- Bifold, sliding and pocket doors have no arc and are not typed yet.

**Windows: partly.**
- Lines in a wall line between two wall ends, with the wall continuing beyond both ends, are found.
- Glazing between tiny jamb stubs, and balcony sliders, are not: 2-19 window elements per page.
  This is the main remaining gap (section 6).

**When it can't be typed, the layer says so.** On 3.pdf and 23.pdf no pen draws a wall network on
its own: the layer reports itself not applicable and nothing is simplified.

## 4. Preprocessing image, vector filter, semantic layer, or a combination?

**A semantic element layer, with a structural image as one rendering of it** (combination):

- Every element gets a type, geometry, source, confidence and evidence:
  `engine/semantic/layer.py` `Element` (WALL, COLUMN, DOOR, WINDOW, TEXT, DIMENSION, OTHER).
- The structural image is rendered from the WALL / COLUMN / DOOR / WINDOW elements, so the existing
  geometric stages run unchanged on it.
- Text, dimensions and the original image stay available to everything else.

A pure preprocessing filter on pixels can't tell a door swing from a furniture arc, or a hollow
wall from two parallel lines. A pure vector filter only exists for vector PDFs.

Experiment B0 below shows the structural image must carry meaning, not just a subset of lines.

## 5. What is lost if text and annotations are removed too early?

- **Room names** (labels).
- **Dimensions and scale** (pixel scale, room sizes).
- **Unit numbers and areas.**
- **Door and window tags** (10, 11, 12), which link to the schedules.
- **Annotations used as negative evidence** (setback lines, grid).
- **Stroked text, if typed wrongly.** It is OTHER today and stays in the original.

Here nothing is removed: the analyzer reads text, dimensions and labels from the original page, and
only the geometry input changes.

## 6. Extract room labels first?

Yes: labels, dimensions and scale come from the original page (text layer + OCR) before and
independently of the simplification. They are then associated with the spaces built from the
structural layer (`FloorPlanAnalyzer.analyze(..., structure_image=...)`).

Labels must not create geometry, but they are a signal: a label that lands in "exterior" marks a
leak, i.e. a missing opening closure, not text outside the building.

## 7. Dimensions and scale before simplification, kept as metadata?

Yes, unchanged. They are read from the original page and kept in the response. DIMENSION elements
(lines along a dimension string) are typed so they can later tie a dimension value to its
measured span. They are excluded from the structure.

## The experiment

- **Code:**
  - `ingest/pdf_vectors.py`: vector paths in the render frame (crop box, rotated pages).
  - `engine/semantic/layer.py`: element typing, wall-body fill, structural and debug images.
  - `FloorPlanAnalyzer.analyze(..., structure_image=None)`: the geometry input. When it is None,
    behaviour is unchanged.
  - The app's PDF page route, behind the flag `FLOORPLAN_STRUCTURAL_LAYER=on` or the request body
    `{"structural_layer": true}`. **Off by default.**
- **Comparison:** `python -m scripts.diagnostics.structural_layer_ab "<pdf>" 5,6,7,8`. Same render,
  same text layer, same OCR (benchmark cache); only the geometry input differs.
- **Outputs:** `output/structural_layer/` holds per page:
  1. the original;
  2. the typed elements;
  3. the structural image;
  4. the spaces (A and B);
  5. the final overlays (A and B).
- **Reference for scoring** (not used by either pipeline):
  - The building footprint is the convex hull of the main wall drawing. "Outside" is therefore a
    lower bound.
  - "Merged" means a space holding labels of two or more separately walled rooms.
  - Splits have no ground truth. "Interior unlabeled spaces" is a proxy; the overlays are audited by
    eye.

### B0: walls, doors and windows exactly as drawn (no semantic fill)

The legacy engine reads each hollow wall's two face lines as thin walls. The narrow paper inside
every wall then becomes a "space" (46-50 slivers per page). Because legacy "succeeds", the Plan Model
fallback does not run and 0-1 rooms are labelled.

**Lesson:** a structural image is only useful if it encodes what the elements *are*. A hollow
wall's interior is wall: wall bodies are filled from the wall pen's own closed outlines, narrower
than half a wall thickness and lying at reconstructed walls. Faces, thickness and door gaps are
unchanged.

### A vs B (22.pdf, OCR cached; seconds = analysis + structural layer)

| Page | | Engine | Walls | Doors | Windows | Spaces | Rooms (labels) | Rooms with boundary | …inside building | Unlabeled | Outside building | Merged | Interior unlabeled | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p5 | A | Plan Model | 310 | 85 | 30 | 37 | 34 | 13 | 2 | 31 | 24 | 2 | 11 | 31 |
| p5 | B | legacy | 200 | 28 | 0 | 34 | 14 | 12 | 12 | 26 | 0 | 1 | 26 | 29 |
| p6 | A | Plan Model | 534 | 160 | 66 | 67 | 31 | 20 | 16 | 56 | 33 | 1 | 23 | 42 |
| p6 | B | legacy | 185 | 32 | 3 | 43 | 13 | 13 | 13 | 34 | 0 | 0 | 34 | 27 |
| p7 | A | Plan Model | 529 | 162 | 61 | 64 | 29 | 19 | 14 | 53 | 32 | 1 | 22 | 43 |
| p7 | B | legacy | 185 | 32 | 3 | 43 | 11 | 11 | 11 | 36 | 0 | 0 | 36 | 26 |
| p8 | A | Plan Model | 768 | 193 | 56 | 64 | 27 | 23 | 14 | 53 | 38 | 2 | 17 | 43 |
| p8 | B | legacy | 186 | 30 | 2 | 40 | 11 | 11 | 11 | 33 | 0 | 0 | 33 | 27 |

Notes on the table:
- Walls, doors and windows are counted by different engines (A: Plan Model; B: legacy wall bands
  and classified gaps), so they are not directly comparable.
- A's door and window counts include gaps that are not openings.
- Live OCR adds the same time to both. App route on p6: B 68 s, A about 93 s.

**Visual audit, p6 units 201/202/203:**
- **A:** furniture is carved out of living rooms; 202's living zone bleeds through the balcony into
  the setback; BDRM.#2 is merged with a bath and the W/D closet; baths are unlabeled fragments.
- **B:** every space follows the walls: bedrooms, W.I.C., each bath, and the living/dining open
  plans. B's missing rooms are whole units (203-205 on p6), whose balcony sliders are not closed.
  Their labels are then dropped as "outside the building" (legacy rule).

**Regression:** with the flag off, the development benchmark summary is identical to the Phase 1
baseline. Full test suite: 191 passed (`tests/test_semantic_layer.py`: 9).

## Recommendation

**Adopt the semantic element layer as the direction, and keep developing it before it becomes a
default.** In priority order:

1. **Openings as first-class typed elements.** Balcony sliders, storefront glazing, bifold and
   pocket doors: parallel lines or panels spanning a wall gap, in the wall line. This is what
   decides whether B keeps the units it loses today.
2. **Labels as a leak signal.** A room label that falls into the exterior marks an unclosed opening
   to look for nearby, rather than being dropped.
3. **A raster source for the same layer.** Most real inputs are images or image PDFs (survey V1).
   - The Plan Model's strokes can feed the same element types: text from the text layer and OCR
     boxes, dimensions from dimension strings, lattice, blocks.
   - The vector route is the high-precision special case.
4. **Path-level typing for drawings without a wall pen** (hatch conventions like 3.pdf): wall
   outline plus hatch inside it, typed per element instead of per pen.
5. **Reconstruction directly over typed elements** (the Plan Model reading WALL / DOOR / WINDOW
   elements rather than a re-rendered image), once the types are reliable.

---

# Phase 2 (2026-10-07): from proxies to ground truth, and openings as the bottleneck

## Measurement

The phase-1 A/B was scored with proxies (convex-hull footprint, label merges). This phase adds
ground truth and a correctness-first benchmark:

- **Fixture:** `benchmark/fixtures/pdf_plans.json`, five pages, every point checked by eye on the
  drawing:
  - 22.pdf p5: 66 interior points, 5 patios / balconies.
  - 22.pdf p6, p7, p8: 74 points + 6 balconies each. p7 and p8 are the same typical floor, checked
    by ink difference.
  - 3.pdf p4: 19 points.
- **Points:**
  - A point is placed in every room, including unnamed closets, W/D closets, corridors, stairs and
    the vestibules inside bedroom doors.
  - Several points share a `group` when they are one room or one open plan (splits are detectable).
- **Benchmark:** `python -m benchmark.pdf_plans --variants A,E [--images DIR]`.
  - A room counts as **separated** only if its space holds no other room.
  - It also reports splits, balconies merged with interiors, correctly *named* rooms, and spaces
    outside the building.

## What the evidence showed

1. **The walls were never the main problem on 22.pdf; the openings were.**
   - B (phase 1) had clean walls but separated only 38 of 74 points.
   - Most misses were units whose balcony glazing, sliders and swing doors were left open, so the
     whole unit leaked to the outside. Most merges were rooms joined through open doors.
2. **Re-running the Plan Model on the clean layer (variant C) was worse** (12 of 74). Its gap rules
   (plan door width, crossing limits) are tuned for noisy rasters and reject real wide openings.
   Recorded and abandoned.
3. **Hollow walls must be solid in the structural image.** Faces closed by another pen were left
   hollow, and the inside of a wall became a "space". Wall bodies are now also filled between
   paired parallel faces of the wall pen.
4. **Room names were the second bottleneck.** CAD sheets tag rooms with drafting abbreviations
   (`BDRM.#1`, `BA-3`, `KIT-2`, `W/D`, `STAIR#1`), which the lexicon did not know. That explains 2 of
   50 named on p6 even with good spaces.

## What changed (all behind the flag; production unchanged when it is off)

- **Openings closed as drawn** (`SemanticLayer.closures`; `structural_image(closures=True)`):
  - **Swing doors:** from the hinge to the arc end that lands on the opposite jamb, i.e. the
    closed leaf (not the open leaf lying along a wall).
  - **Glazing / sliders:** two or more parallel thin lines spanning a wall gap, with no wall beside
    them. At least one end is on a wall; the other end is on a wall, a closed door or another
    glazing run (corner windows).
  - **Small closed symbols** (tags, fixtures) are never glazing.
  - **Remaining gaps:** wall gaps with an opening symbol on the page (Plan Model gap classifier on
    the full page as evidence, `reconstruct(..., evidence_image=)`). These are gated by symbol
    strength against the plan's own door width, measured from its swing arcs.
  - **New Plan Model gap rule:** pocket and bi-parting panels (ink continuous from both jambs).
  - **Passages without a symbol stay open.**
- **Wall poché for drawings without a wall pen** (3.pdf):
  - Hatch is found as families of 150+ short parallel off-axis segments whose closed region forms
    long bands. Isolated dimension ticks never form a band.
  - The band is the wall body. Outline and hatch segments on it are typed WALL. Furniture in the
    same pen is left out.
- **Safety gate:** with the page image, the typed walls must lie on the page's ink (alignment
  ≥ 0.9), otherwise the layer is not used. Development pages: 1.00.
  - 05.pdf p9 (a design document whose vectors do not reproduce the render) is refused.
  - 05.pdf p17 and Richmond have no wall pen or poché, so they are not applicable and unchanged.
- **Drafting abbreviations:** `room_lexicon.ABBREVIATIONS` and room tags (term + number). They are
  context-scoped (`room_lexicon.abbreviations()`) and enabled only with a structural image. The
  default lexicon is unchanged.
- **Analyzer:**
  - `structure_engine="legacy"` is the default with a structural image (variant E).
  - `"plan"` (variant C) is kept for comparison.
- **App:** the flagged PDF route uses variant E (layer with closures; gap closures from the page).

## Results (OCR cached; A = production, E = flag on)

| Page | | Separated | Merged | Missed | Splits | Balcony merged | Named correct | Outside building | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| 22.pdf p5 | A | 8 / 66 | 12 | 46 | 1 | 0 | 0 / 43 | 23 | 35 |
| | E | **54 / 66** | 8 | 4 | 0 | 0 | **30 / 43** | 3 | 36-100 |
| 22.pdf p6 | A | 17 / 74 | 49 | 8 | 5 | 3 | 2 / 50 | 31 | 47 |
| | E | **66 / 74** | 8 | 0 | 0 | 0 | **40 / 50** | 0 | 40-85 |
| 22.pdf p7 | A | 16 / 74 | 47 | 11 | 5 | 3 | 2 / 50 | 30 | 45 |
| | E | **66 / 74** | 8 | 0 | 0 | 0 | **40 / 50** | 1 | 39 |
| 22.pdf p8 | A | 17 / 74 | 44 | 13 | 3 | 0 | 1 / 50 | 34 | 54 |
| | E | **66 / 74** | 8 | 0 | 0 | 0 | **40 / 50** | 1 | 39 |
| 3.pdf p4 | A | 7 / 19 | 11 | 1 | 1 | — | 4 / 12 | 17 | 23 |
| | E | **18 / 19** | 0 | 1 | 0 | — | **12 / 12** | 2 | 82 |

Intermediate variants on 22.pdf p6 (separated / merged / missed):
- B (layer, no closures): 47 / 12 / 15.
- C (Plan Model on layer): 12 / 33 / 29.
- D (+ typed door and glazing closures): 61 / 12 / 1.
- E (+ gap closures from page symbols): 66 / 8 / 0.

Regression:
- Flag off: development benchmark summary identical to the Phase 1 baseline.
- Synthetic wall styles unchanged.
- Tests: 195 passed (`tests/test_semantic_layer.py`: 13).

## Still failing

- **Kitchen / living / dining of two units (p5 103 and 105, p6 204 and 205)** share a space with an
  adjacent W/D or closet: the bifold doors (10) of the W/D closets are not closed.
- **Lobby (p5)** is not reconstructed: its storefront and entry glazing is not typed.
- **3.pdf laundry point** (on the washer drawing) is missed.
- **Time:** gap closures run the Plan Model on the full page (+20-60 s on 24 MP sheets).
- **Applicability:** only vector PDFs with a wall pen or poché. Raster plans and other conventions
  (solid fills mapped through clip paths, thin grey pens) keep the production path.
