# PDF discovery: how much more can a PDF give us?

Status: investigation only. No production code was changed; every experiment ran as a scratch
script against the unchanged engine. The Test Cases PDFs (3.pdf, 22.pdf, 23.pdf) are the
development set; none of these numbers is a generalization claim. The holdout was not used.

## Experiments (Hypothesis → Test → Evidence → Conclusion)

| # | Hypothesis | Test | Evidence | Conclusion |
|---|---|---|---|---|
| E1 | The text layer contains the room names, dimensions and scale more reliably than OCR | `pdftotext -bbox -cropbox` on 3.pdf p4, 22.pdf p5, 23.pdf | 3.pdf p4: 14 room words and 28 dimension strings in ~0.1 s; engine OCR found 1 room name on the same page. 23.pdf (raster drawings) still has a vector text layer | **Confirmed.** Exact text and positions for free on CAD exports, including "raster" sheets that carry a text layer |
| E2 | Text coordinates map onto the rendered page without a transform | Room words checked against the regions they name | 10/10 words fall inside the same-named region with the identity mapping (CropBox, rotation already applied) | **Confirmed.** No coordinate work is needed beyond points→pixels (×DPI/72) |
| E3 | The scale is recoverable from the text | Title strings such as `3/8" = 1'-0"` | 27 pt/ft → 56.25 px/ft at 150 DPI; the engine found no scale on that page | **Confirmed** where a scale note exists |
| E4 | Printed room dimensions validate the detected rooms | Printed W×L vs the polygon size at the PDF scale | Correct rooms agree within ~1%; disagreement flags merges and zone splits | **Confirmed.** Becomes an independent per-room check, not just a scale source |
| E5 | Hatched walls (3.pdf) become solid walls the engine already understands | 45°/135° segments of the dominant length → solid band, plan region from the hatch extent | Plain crop: 0 rooms, 0 doors, 0 windows (39 s). Hatch→solid: 11 named + 6 unlabeled spaces, 11 doors, 16 windows (36 s); overlay visually correct | **Confirmed — the biggest single win.** Caveats: legend swatches enter the region; hatch clusters fragment at openings |
| E6 | OCR dominates PDF analysis time | Stage timing on the E5 crop | OCR 29.7 s of 36 s (83%); structure 4.5 s | **Confirmed.** Using the text layer instead of OCR gives ~5–7× on vector sheets |
| E7 | Clip paths separate the drawing from the sheet furniture | Clip inventory | Only 4 clips on the sheet | **Rejected.** Region detection needs geometry and text, not PDF structure |
| E8 | Wall layers can be found without layer names (OCGs are absent) | Stroke inventory by pen width | 22.pdf: the heaviest pen used for real drawing (1.98 pt) is walls only; 3.pdf: walls are hatch, not pen | **Partly.** "Heaviest pen" works on one convention; hatch needs its own strategy, so it must be chosen by evidence |
| E9 | Hollow double-line walls (22.pdf p5) filled from the wall pen fix the plan | 1.98 pt pen, narrow enclosed bands (≤ 6 pt inscribed radius) filled → over the plain render | Plain: 19 names, 0 doors/windows. Wall layer: 20 named + 56 unlabeled, 126 doors, 31 windows; unit/room structure visually right, but false seals along dimension and dashed lines | **Confirmed with a defect.** The walls are correct; the annotation pens that stay in the image create false structure |
| E10 | Drawing only the walls plus a symbol pen removes the E9 false seals | Walls (E9) + the 0.57 pt black pen only (dimension, text and thin pens dropped); structure + openings only | 76 → 49 spaces; the false seals on dimension/dashed lines disappear and the rooms are clean in the overlay (`e10_overlay_zoom.png`). Doors fall to 8 (61 untyped openings): the door swings are not on that pen | **Confirmed for structure; doors need a role-detected symbol layer** (arcs found by geometry, not by pen width alone). Structure-only run: 3.2 s |

Curve inventory (door-arc candidates): 22.pdf mostly the 0.57 pt black pen (1595 curves, also
fixtures); 3.pdf the 0.72 pt pen (431).

Scratch code and outputs: the session scratchpad `pdfx/` (`pdfx.py`, `exp_hatch.py`,
`exp_pen.py`, `e5_*`, `e9_*`, `e10_*`).

## 1. What we can already do

* Upload a PDF, inspect pages (size, rotation, vector/scan), pick pages from thumbnails, render at
  a bounded DPI, run the unchanged engine, show the result over the exact page image.
* Works well when the page looks like the images the engine was built for (solid walls, plan
  fills most of the page). Fails or degrades on CAD sheets: hatched or hollow walls, title blocks,
  schedules and notes on the same sheet, small text read by OCR, ARCH D sheets taking ~2.5 min.

## 2. What the PDF gives that raster images don't

| Information | Raster | Vector PDF |
|---|---|---|
| Text (names, dimensions, scale, title) | OCR, slow, error-prone on small text | Exact strings + boxes, ~0.1 s |
| Scale | Inferred from OCR'd dimensions, often absent | Scale note text; printed dimensions to cross-check |
| Wall geometry | Thresholded pixels | Exact segments, pen widths, hatch patterns, fills |
| Layers of meaning | All ink is equal | Pen width / colour / dash separate walls, symbols, annotations |
| Door arcs | Detected as pixels | Bezier curves with exact centres and radii |
| Resolution | Fixed | Any DPI, any crop, re-renderable per region |

Not available in these files: named layers (OCGs), semantic tags, room objects. "Raster PDFs"
(23.pdf) give no geometry, but may still give the text layer.

## 3. What we can realistically add

1. **Text-layer extraction** feeding names, dimensions and scale into the engine instead of OCR
   (fallback to OCR when a page has no text layer).
2. **Scale from text** plus **printed-dimension cross-check** of each room.
3. **Vector wall reconstruction** as a preprocessing step with several strategies, chosen by
   evidence on the page: hatch→solid (E5), heavy-pen hollow fill (E9), solid fills (already fine).
4. **Annotation filtering**: render only the wall layer and a symbol layer; drop dimension, text,
   grid and thin pens (E10).
5. **Plan-region detection** from the wall-geometry clusters plus text anchors ("FLOOR PLAN",
   scale title, "LEGEND", schedule tables), with a manual crop override in the Workbench.

All of these sit in front of the engine; the engine keeps receiving an image (plus optional
text/scale hints), so the existing image path and its benchmarks are unchanged.

## 4. Difficult or not worth pursuing now

* **Full vector-native recognition** (rooms from segment graphs instead of pixels): the most
  accurate in theory, but a second engine, CAD-convention-specific, and the raster engine's room
  logic (labels, zones, openings) would have to be rebuilt. Not justified by the evidence yet.
* **Relying on PDF layers/OCGs or clip paths**: absent or meaningless in the test files (E7).
* **Automatic page classification and cross-page merging**: useful later; the manual picker
  already solves the user problem at low cost.
* **Raster-only PDFs**: no gain beyond the text layer; they stay on the image path.
* **Door classification from pen width alone** (E10): door symbols share pens with fixtures;
  arcs must be found by geometry (quarter-circle Beziers hinged on a wall end).

## 5. Highest-value opportunities

| Rank | Opportunity | Why |
|---|---|---|
| 1 | Hatch / hollow wall reconstruction | Turns 0-room results into usable plans (E5: 0 → 17 spaces; E9: 0 → 31 windows and a correct structure) |
| 2 | Text layer instead of OCR | Correct names on CAD sheets (1 → 14 words on 3.pdf p4) and ~5–7× faster |
| 3 | Scale + printed-dimension validation | Real areas; an independent merge/zone check per room |
| 4 | Annotation filtering | Removes the false seals that the wall layer otherwise brings (E10) |
| 5 | Plan-region detection | Removes title blocks/legends; cuts pixels and time further |

## 6. Recommended architecture

Compared options:

| | A. Render + engine (today) | B. Vector-native engine | C. Hybrid: vector features fused into the raster engine | D. Vector preprocessing → clean image → engine | E. PDF-aware layer: text + scale hints + D |
|---|---|---|---|---|---|
| Accuracy on CAD sheets | Low | Potentially highest | High | High for walls | High for walls, names, scale |
| Generalization | Same as images | Per CAD convention | Medium | Strategy per convention, fallback to A | Same, with fallback |
| Performance | Slow (OCR) | Fast | Medium | Same as A | ~5–7× faster (no OCR on vector) |
| Complexity / risk | None | Very high | High (touches engine internals) | Low–medium, isolated | Medium, isolated plus one hint interface |
| Engineering drawings | Poor | Good | Good | Good | Good |
| Information kept | Pixels only | All vectors | Most | Walls, symbols | Walls, symbols, text, scale |
| Maintainability | — | Two engines | Coupled | One engine, plug-in strategies | One engine, plug-in strategies |

**Recommendation: E — a PDF-aware preprocessing layer.** It extracts the text layer and scale,
reconstructs walls with evidence-selected strategies, filters annotations, proposes a plan region,
and hands the existing engine a clean image plus optional text/scale hints. Scans and unknown
conventions fall back to today's path (A). B stays a long-term option if E plateaus.

## 7. Roadmap

| Phase | Content | Accuracy impact | Performance impact | Effort | Risk | Dependencies |
|---|---|---|---|---|---|---|
| P1 Text layer | Positioned words → engine label/dimension input; OCR fallback | Names on CAD sheets from near 0 to most rooms (3.pdf p4: 1 → 14 words available) | −80% analysis time on vector pages | S–M (engine needs a "provided text" input) | Low; image path untouched | None |
| P2 Scale + validation | Scale note parsing; printed dimensions vs polygons; warnings | Correct areas; flags merges/splits | Negligible | S | Low (warnings only) | P1 |
| P3 Wall reconstruction | Hatch→solid, hollow fill, solid passthrough; evidence-based choice; fallback | Largest: 0-room CAD sheets become usable (E5, E9) | Neutral | M | Medium: wrong strategy fabricates walls → needs an applicability test and holdout check | None (benefits from P4) |
| P4 Layer filtering | Wall + symbol layers only; door arcs by geometry | Removes false seals (E9 → E10); restores doors once arcs are role-detected | Slightly faster | M | Medium: dropping a pen that carries walls | P3 |
| P5 Plan region | Wall clusters + text anchors; manual crop override in Workbench | Removes legend/title-block false spaces | 2–4× fewer pixels on ARCH sheets | M | Low with manual override | P1, P3 |
| P6 Validation | Freeze PDF ground truth on held-out PDFs before tuning; compare A vs E | Measures, doesn't change | — | S–M | — | Holdout PDFs |

Order: P6 ground truth first (or in parallel with P1), then P1 → P2 → P3 → P4 → P5. Each phase is
gated on: unchanged results for image inputs and scans, no regression on the protected baseline,
and improvement on held-out PDFs, not only on 3/22/23.pdf.

## Major capability upgrades this unlocks

* CAD exports with hatched or hollow walls go from "no rooms" to analysable.
* Exact room names and dimensions without OCR errors.
* True scale and areas, with a printed-dimension check of every room.
* Multi-minute sheets become tens of seconds.

## Validation round 2 (independent PDFs) — running record

Independent set: `~/Downloads/Test Files` minus every file that is in the holdout
(`floorplan_test_dataset_01`): copeland, lowell, telford, seattle, ribble and the mango image are
holdout copies and were not opened. Used: `05.pdf` (60 p), `fpc_cambridge_jan_pack.pdf` (35 p),
`fpc_haringey_plevna.pdf` (11 p), `fpc_richmond_3f.pdf` (1 p). `fpc_planningorg_gf.pdf` is an
HTML file named .pdf (correctly rejected as `not_pdf`).

| # | Tested | Learned | Decision |
|---|---|---|---|
| V1 | Format survey of 107 independent pages | Cambridge (PowerPoint export) and Haringey (Word export) hold every plan as an embedded image, with almost no text layer (3–8 words per page). Only Richmond (A1 CAD sheet, site scale) and 05.pdf p9/p17 (small plan vignettes in a design statement) have vector plans | The vector path applies to CAD exports only; presentation and document packs stay on the raster path |
| V2 | Wall conventions on the 5 vector plan pages (dev + independent) | 5 pages, 4+ conventions: hatch (3.pdf), heavy-pen hollow (22.pdf), solid black fills (05 p9), thin double lines (05 p17), grey-scale thin pens (Richmond). The E5 hatch selector also fires on 22.pdf (non-wall diagonals) | **E5/E9 are convention-specific: their gains do not transfer.** Each new convention needs a new strategy |
| V3 | Raster-only wall normalization vs the vector walls (E11) | Hollow fill: best recall 0.75 at precision 0.43; hatch closing: best 0.72 at 0.47 | Vectors carry real wall information that raster heuristics can't match, so the vector route is not useless, just narrow |
| V4 | One convention-agnostic vector strategy: parallel-line pairs at a page-dominant spacing (E12) | No dominant wall spacing on any page; text glyph fills, hatch and furniture swamp the histogram (3.pdf: 757k pair candidates) | Rejected as tested; a general vector wall extractor is a research project, not a phase |
| V5 | Text-layer value on independent pages | Room words: 05 p9 128, Richmond 19, 05 p17 4 (elevations page), plus scale notes (1:100, 1:250). Pages that hold the plan as an image have none (Cambridge, Haringey), as expected. Minor garbling: mirrored stamp text on Richmond, non-ASCII symbols (², ”) | **The text layer generalizes** wherever the PDF has one; it needs a quality gate and an OCR fallback |

Decision: reorder. Text layer + scale (P1/P2) are supported by independent evidence and are low
risk. Vector wall reconstruction (P3/P4) is not: it showed overfitting to the conventions of
3.pdf and 22.pdf. It stays experimental until a labelled PDF set covering several conventions
exists.

## Text-layer phase — running record

| # | Hypothesis | Experiment | Evidence | Decision |
|---|---|---|---|---|
| T1 | The text layer can replace OCR on PDF pages | E13: engine on 6 full pages, OCR vs text-only | Faster everywhere (5–15×). Names: 05 p9 0→21, Richmond 10→19, 23.pdf equal, **22.pdf 33→16** | Rejected as a replacement: CAD programs export part of the text as strokes (AutoCAD SHX), which only OCR reads |
| T2 | OCR only on what the text layer doesn't cover (text words blanked) keeps speed and names | E13b, hybrid with blanked words | 2–4× faster, but 22.pdf lost 5 bedrooms that full-page OCR finds; OCR voting depends on the whole page | Rejected |
| T3 | Union: OCR on the unchanged page, text layer wins where both read | E13c on 4 pages | Never lost a valid OCR name; names: 05 p9 0→21, Richmond 10→21, 22.pdf 33→34, 23.pdf equal; the one dropped OCR "name" was a title-block address | **Adopted.** Same speed as OCR; accuracy gain only |
| T4 | Text-layer quality is a risk | Survey of 79 pages with text in 7 PDFs | Median 0.99 ordinary words, minimum 0.89; risks: broken encodings, invisible OCR layers of scans | Quality gate (≥ 0.8) and invisible-layer check, both tested |
| T5 | Printed room dimensions come through | 3.pdf p4 | CAD exports `12'-0"`, `x`, `7'-0"` as 3 words → joined (9/9 room dimensions). **Engine bug found**: `14'-8" x 9'-6"` parses as 8 in × 9 ft and `12'-0" x 7'-0"` not at all — the feet-inch hyphen is unsupported, for OCR'd images too | Join implemented; the parser fix is an engine change for every input, handled separately with regression checks |

Integration: `ingest/pdf_text.py`, `merge_document_text` (engine/analysis/ocr.py), optional
`text_evidence` in `FloorPlanAnalyzer.analyze`, PDF page endpoint. See docs/PDF_TEXT_LAYER.md.
Open: skipping OCR safely (needs a signal for stroked text) for the 5–15× speed-up.
