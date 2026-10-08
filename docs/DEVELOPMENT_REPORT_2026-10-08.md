# Development report - 2026-10-08 (overnight)

Evaluation: canonical `../Test Cases` only. Dev = 17 scored images (181 GT room points, 154 printed
names), `python -m benchmark.real_plans --full`. PDF = 22.pdf p5-p8 + 3.pdf p4 (`benchmark.pdf_plans`,
A = production default, E = Architectural Cleaning). Baseline = 775fcf7 (start of the night).

## Commits (all on main, pushed)

| Commit | What |
|---|---|
| 775fcf7 | OCR: 8 upright page passes + scale-normalised text lines (start of night, H1 work) |
| 4319cac | Object outlines on a wall axis are not opening symbols (H2) |
| 8026597 | Context-aware room names; fused term+number; stack guard |
| 1297e18 | Glazing bands of several parallel lines seal as openings |
| 82ea789 | Structured plan: Building -> envelope -> walls / openings -> spaces graph |

## Hypothesis 1 - cleaning before OCR

**Rejected (measured).** Details: `docs/OCR_ARCHITECTURE.md`. A text-only image costs about the same
Tesseract time and loses words that touch lines (3.pdf: 19 -> 17 room words). Tesseract's cost is
area x passes, not clutter. What limited recall was text *size*. Scale-normalised text lines fixed
that, and cut the passes from 32 to 8: Analyze 406 -> 230 s over 26 inputs, names 106 -> 113.

## Hypothesis 2 - furniture interpreted as spaces

Measured against GT points (`nonstructural`, `must_share`, open-plan groups) and visual overlays:

- **Furniture rarely becomes a space.** The 51 "spaces without a GT point" are, when inspected,
  almost all real unlabelled rooms: closets, WICs, extra baths, stair halls, linen. GT lists one
  BATH on 10.png, which prints five. Furniture blobs that enter the wall mask (5.jpeg: 14 of 19
  nonstructural points) are isolated components. They do not split rooms.
- **Where furniture does damage, it is as false opening evidence.** A counter, fridge or stair
  box with one edge on a wall's axis made a wide interior gap look "drawn" (line coverage), so
  it was sealed into a fake partition. These were the only 2 GT `must_share` violations
  (6.png kitchen | living, 11.png kitchen | foyer).
- **Fix (general):** a covering line that belongs to a closed outline with a parallel back edge
  >= 3 wall thicknesses into the room is an object, not an opening symbol. An outline spanning the
  gap jamb to jamb (overhead garage door) still is one.
  - must_share violations 2 -> 0.
  - Partitions inside one open-plan group 10 -> 6.
  - Open-plan groups split 10 -> 8.
  - Cost: 7.png's bonus-room double door had been sealed only by a bathtub edge on its axis.

## Other changes

- **Context-aware room names.** Once the structure is known, the text lines inside *unnamed*
  spaces are read one at a time (scale-normalised crop, single-line mode, grey + Otsu). They are
  accepted only with a specific room term at >= 60 confidence. Names that lost their number
  ('BED' for 'BED 4') are re-read. Reading uses parallel batches with a yield budget: 22.pdf p6
  was 60 reads / 7 s with no yield, now 1.3 s.
- **Room labels.** 'BATH1' is read as 'BATH 1'. A stacked second line must carry room vocabulary, a
  number or a confident read, so misread dimension lines ('Txs', 'sans') no longer join names.
- **Glazing bands.** Sliding-glass walls drawn as 3+ parallel lines were rejected as
  "text/hatching". Now, each inked row between the pair must be one or two long runs on the
  snapped wall axis. Text, hatching, and patterns that continue beyond the band (planks, treads)
  are still rejected.
- **Structured plan** (`engine/arch/building.py`, `engine/arch/envelope.py`):
  - What it contains:
    - buildings (footprints)
    - walls (exterior / interior)
    - openings, with what they connect, found geometrically on both sides
    - spaces: names, room / open-plan / unnamed, area, building, neighbours through openings and walls
    - a space graph
  - Where it appears: in every response as `building`, and as `/api/results/<id>/building.json`.
    Visual check: `scripts/diagnostics/building_view.py`.
  - Envelope evidence: a building is the sealed wall network that carries openings. A network
    enclosing a busier one is a frame. Leader-line appendages are cut, and clusters under 5 % of
    the main footprint are dropped.
  - Envelope vs GT hulls:
    - dev: 0 spaces wrongly put outside a building.
    - PDF Plan Model path: 0 wrong; 12 of 135 bogus sheet spaces recognised.

## Metrics, baseline (775fcf7) -> now (82ea789)

| Dev (17 images) | Before | After |
|---|---|---|
| Room separation (structure) | 0.867 | 0.867 |
| GT must_share violations | 2 / 10 | **0 / 10** |
| Partitions inside one open-plan group | 10 | **6** |
| Open-plan groups split | 10 / 16 | **8 / 16** |
| False opening candidates | 31 | 30 |
| Names correct (of 154 printed) | 113 | **119** |
| Label recall / precision | 0.753 / 0.899 | **0.792 / 0.910** |
| Wrong names | 7 | **5** |
| False labels (vs GT) | 6 | 7 (the extra is a second HALL that 10.png prints but GT omits) |
| API room recovery | 0.890 | 0.884 |

Per plan:
- Separation: 12.png +1, 7.png -1.
- Names: 5.jpeg +1, 8.png +2, 15.png +2, 16.png +1.
- No plan lost a name.

| PDF | Before | After |
|---|---|---|
| E, 22.pdf p5 separated / names | 51 / 28 | **55 / 32** |
| E, 22.pdf p6-p8, 3.pdf p4 | 66 / 66 / 66 / 18; names 40 / 40 / 40 / 12 | unchanged |
| A (default) | 8-17 separated per page | unchanged |

Runtime: +0.2-0.4 s per image and +1.3 s on a 24 MP sheet (context reading); the Building model
takes ~10 ms. Tests: 212 -> 226 passed. Desktop app: GUI smoke tests 26/26 + 26/26 on 82ea789,
and the app is running on 82ea789.

Correction: `docs/STRUCTURAL_LAYER.md` listed 22.pdf p5 (E) as 54/66. Bisecting showed the
committed code has given 51/66 since 7048ae1, so 54 was a pre-commit state. The table now shows
the measured value (55/66 after the glazing fix).

## Tried and rejected tonight

| Idea | Result |
|---|---|
| Plan Model as primary on raster plans | 58 vs 157 separated: far worse, stays a fallback |
| Global upscaling of small images (to 1200 / 1800 px) | 152 / 147 vs 157; it helps 7.png and 14.png but hurts 4, 5, 8 and 11. There is no reliable selector yet |
| Envelope = network ∩ hull of its openings | The Plan Model "finds" openings among dimension lines; 5 real rooms were flagged outside |
| Largest-part-only envelope trimming | Cut real wings (22.pdf p5: 8 GT spaces flagged) -> replaced by reconstruction from the body |

## Still difficult

- **7.png:** thin interior walls (2 px) next to 4 px exterior walls on a 545 px image. 10 of 12
  points are merged. The line-wall recovery rejects chains of thin walls that connect through
  each other. Upscaling x3.3 separates 10 of 12, so the information is there, but no
  scale-selection rule is safe yet.
- **Tiny text** (14.png, 7.png: 4-5 px glyphs) stays unreadable even per line, and room numbers
  are often dropped.
- **Production PDF path A** (no cleaning) separates 8-17 of 66-74 points per page. With
  cleaning it is 55-66. Recommendation, not changed tonight: make cleaning the default for
  vector PDFs where the layer is applicable (alignment gate). The toggle stays yours to decide.
- **Envelope polygon** can keep a strip enclosed by dimension extension lines (3.pdf p4). No
  spaces are affected.
- **GT gaps to review (labels unchanged):**
  - 10.png prints 5 BATH and 2 HALL; GT lists 1 BATH and 1 HALL.
  - 14.png has a second GARAGE.

## Next breakthrough

Reasoning at the level of **space hypotheses rather than pixels**:
1. Generate alternative structures per plan: wall classes, scales, line-wall chains, gap seals.
2. Score each structure with evidence the engine already has: one name per space, text lines
   per space, door count per space, envelope coverage, wall-graph closure.
3. Keep the best-scoring structure.

7.png (scale) and the remaining merges (unsealed glazing, double doors) are exactly the cases
where one alternative is right and today's single pass is not. Next steps:
- the Building model above is the representation that scoring needs;
- `benchmark.real_plans` plus the new probes (must_share, partitions, orphans) is the harness.
