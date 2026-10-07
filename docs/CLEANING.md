# Architectural Cleaning Mode (experimental)

```
Cleaning OFF:  Original ───────────────────────────────▶ Recognition engine ──▶ Result   (unchanged)
Cleaning ON:   Original ─▶ Architectural Cleaner ─▶ Cleaned plan ─▶ Recognition engine ──▶ Result
                  │                                                     ▲
                  └──────── original page: text, labels, dimensions ────┘   (always the evidence layer)
```

The cleaner does not recognise rooms. It decides what is architecture:

- **Kept:** walls (faces and solid wall bodies), columns, doors, windows and glazing, and the
  openings they close.
- **Suppressed:** text and room names, dimensions and dimension lines, furniture, fixtures, symbols,
  tags, notes, title blocks, legends and grids.

Nothing is deleted. Every element keeps its type in the representation, and labels, dimensions and
scale are still read from the original page.

## Where it lives

| Part | File |
|---|---|
| Cleaner stage (sources, result, artifacts) | `engine/cleaning/cleaner.py` |
| Semantic element typing (vector) | `engine/semantic/layer.py` (see `docs/STRUCTURAL_LAYER.md`) |
| PDF vector extraction | `ingest/pdf_vectors.py` |
| Recognition on a cleaned plan | `FloorPlanAnalyzer.analyze(..., structure_image=)` |
| App integration and artifact routes | `app.py` (`run_cleaner`, `/api/results/<id>/cleaning/<file>`) |
| Workbench | "Clean plan first" checkbox; **Cleaned** and **Elements** views (use *Side by side*) |
| Evaluation | `scripts/diagnostics/cleaning_eval.py`, `benchmark/pdf_plans.py` |
| Tests | `tests/test_cleaning.py`, `tests/test_semantic_layer.py` |

## Sources (first that applies)

1. **vector-wall-pen:** the PDF pen(s) that reconstruct on their own as a wall network.
2. **vector-poche:** walls drawn as outline + hatch in a shared pen (hatch families form wall bands).
3. **raster:** Plan Model walls and openings. Only in the explicit `raster` mode: measured worse,
   see below.
4. **none:** the original is analysed unchanged, with a warning.

Safety: the typed walls must lie on the page's ink (alignment ≥ 0.9), otherwise the source is
refused.

## Running it

- **Workbench:** tick *Clean plan first*, analyse, then open **Cleaned** or **Elements** (and *Side
  by side* for Original vs Cleaned).
- **API:**
  - Request: `POST /api/pdf/<id>/pages/<n>/analyze` with `{"cleaning": "on"}`, or `/api/analyze`
    with the form field `cleaning=on`.
  - Response: a `cleaning` block (source, reason, kept / suppressed counts, sealed openings, URLs).
- **Environment:** `FLOORPLAN_CLEANING=on|vector|raster|off` (default off).
  `FLOORPLAN_STRUCTURAL_LAYER=on` still works and maps to `vector`.
- **Artifacts per result:** `instance/results/<id>/cleaning/`
  - `cleaned.png`: the skeleton. Walls black, doors green, windows orange, sealed openings red.
  - `recognition-input.png`: what recognition ran on.
  - `elements.png`: every element by type.
  - `comparison.png`: 2×2 sheet of Original | Cleaned / Elements | Recognised rooms.
  - `cleaning.json`
- **Evaluation:** `python -m scripts.diagnostics.cleaning_eval --plans "22.pdf:6,3.pdf:4"` writes the
  same artifacts plus both room overlays to `output/cleaning/<page>/`.

## Results (2026-10-07)

Rooms against `benchmark/fixtures/pdf_plans.json`, same render, text and OCR:

| Page | | Separated | Merged | Missed | Split | Outside building | Named correct |
|---|---|---|---|---|---|---|---|
| 22.pdf p6 | original | 17 / 74 | 49 | 8 | 5 | 31 | 2 / 50 |
| | **cleaned** | **66 / 74** | **8** | **0** | **0** | **0** | **40 / 50** |
| 3.pdf p4 | original | 7 / 19 | 11 | 1 | 1 | 17 | 4 / 12 |
| | **cleaned** | **18 / 19** | **0** | 1 | **0** | 2 | **12 / 12** |

Cleaner integrity:
- **Artificial ink** (cleaned ink on no ink of the original, sealed openings and wall bodies
  excluded): 0.0000 on both pages.
- **Openings sealed:**
  - 22.pdf p6: 49 doors, 52 glazing, 6 from page symbols.
  - 3.pdf p4: 8 doors, 17 glazing, 13 from page symbols.

Opening preservation, audited by eye on 3.pdf against every door and window tag of the drawing:
- **Doors:** 14 / 14 preserved: 1, 2 ×3, 3 ×3, 5, 6 pocket, 7 and 8 bypass sliders, 9, 10 bifold.
- **Windows:** 11 / 11 preserved: A ×4, B ×4, C ×2, D.
- **Incorrectly closed:** the 2 shower glass screens (they split the shower stalls off the baths).
- **Wall continuity:** one exterior wall segment under the concrete-landing stipple is lost from
  the poché and replaced by a glazing seal (no leak, but the wall itself is missing).
- **False geometry:** one symbol near the north arrow kept as wall, outside the building.

22.pdf p6 visual audit (units 204 / 205):
- Doors, balcony doors and glazing are preserved.
- The diamond window tags no longer produce seals (rule: glazing follows the building's wall
  directions).
- The title-block wall legend is suppressed (rule: wall drawing far from the building's networks
  is not structure).
- **Not sealed:** the bifold doors of the W/D closets (tag 10). This is the cause of all 8
  remaining merges.

Raster source on the 17 development images (legacy structure, room points):

| | Separated | Merged | Missed |
|---|---|---|---|
| original | 157 | 21 | 3 |
| raster-cleaned | 108 | 35 | 38 |

Better on 1 image (19.jpg), worse on 11. **The raster cleaner is not used by `on`.**

Regression (cleaning off):
- Full test suite: 201 passed.
- Development benchmark summary identical to the Phase 1 baseline.

## Rules introduced in this phase, and their generality

| Rule | General? |
|---|---|
| Glazing must follow one of the building's dominant wall directions | Yes: windows are built in walls. Weak only for plans with walls at the same angle as their tags |
| Wall drawing far (> 20 wall thicknesses) from every large wall network is not structure | Yes for legends, details and key plans; several buildings of similar size all stay |
| Openings belong to the network their jambs sit on | Yes |
| Bifold V-detector | **Rejected:** fired on tags and clearance marks and missed the double-line leaves of 22.pdf; it would have needed drawing-specific tuning |

## Known limits

- **Openings in the API response:** door / window counts and the openings overlay still come from
  the legacy gap finder. On a cleaned plan the openings are sealed, so it finds few. The cleaner's
  openings are in `cleaning.openings` only.
- **Bifold doors** are not sealed.
- **Glass shower screens** are sealed as glazing.
- **Coverage:** vector PDFs with a wall pen or poché only. Raster plans keep the original path.
