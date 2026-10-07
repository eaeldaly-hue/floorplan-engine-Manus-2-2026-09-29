# Structure, doors and windows

How the engine finds walls, spaces, openings and their types, and how this is
measured. Everything is deterministic OpenCV/NumPy code: there is no machine
learning, no vision model and no network access. The previous CNN experiment
stays in `experimental/` and is not imported.

| Module | Role |
|---|---|
| `engine/structure.py` | walls → gaps (opening candidates) → sealed spaces → topology |
| `engine/wall_inference.py` | multi-scale wall inference: primary and secondary wall classes, line-weight walls |
| `engine/room_recovery.py` | splits a space holding several room labels at drawn constrictions |
| `engine/opening_detection.py` | evidence per candidate → door / window / unknown |
| `engine/analyzer.py` | API path (`/api/analyze`): OCR + structure + openings + rooms |
| `engine/pipeline.py` | standalone geometry run (`python -m engine.pipeline`) |
| `benchmark/` | synthetic plan generator, evaluator, runners |
| `scripts/diagnostics/opening_evidence_view.py` | development-only evidence view |
| `scripts/diagnostics/trace_pipeline.py` | development-only stage-by-stage trace with wall evidence images |

## 1. Structure (`analyze_structure`)

1. **Binarise and despeckle.** Otsu ink, plus a permissive ink mask that keeps
   light-grey symbol lines. Specks are removed only when they are still isolated
   after a 3×3 closing, so dashed and broken lines survive.
2. **Deskew.** The dominant wall angle comes from a Hough transform on wall edges
   (θ step 0.125°, weighted median folded to ±45°). From 0.35° up, the plan is
   rotated, analysed, and every result is mapped back to original coordinates.
   The rotated "work" frame is kept for the classifier.
3. **Wall mask by thickness.** The image is eroded, dilated back and ANDed with
   the ink. The erosion kernel is not a constant. It is read from the curve of
   surviving area against kernel size: thin strokes vanish first, then the curve
   flattens while only walls are left. The kernel is taken on the earliest
   plateau of at least 3 flat steps that ends in a drop. A 2-step plateau is
   accepted when it is very flat, ends in a strong drop and starts clearly above
   the line width; this covers thin interior walls. Solid, medium-thickness
   blocks attached to walls (jambs) are kept. **No wall pixel is ever drawn that
   is not in the image.** The old "connect disconnected pieces" step drew lines
   between components, which fabricated walls across door openings. It was
   removed; walls broken by openings really are disconnected.
   **Multi-scale (secondary) wall classes** (`engine/wall_inference.py`). A plan can
   hold 1–2 px lines, 5 px interior walls and 12 px exterior walls at once. When
   the thinner walls are only a few pixels thicker than the lines, their plateau is
   too short to be chosen, so the primary kernel would erase them. Below the primary
   kernel, the same curve is searched for a stable class that still holds clearly
   more ink (≥ 1 % of all ink). Its strokes become wall only when they behave like
   architecture:
   * they contain a long straight run;
   * they connect to the primary walls, directly or through other accepted pieces;
   * they are not collinear fill inside a thicker wall's gap (glazing, frames,
     sliding panels);
   * they are not hinged at a wall's end face (a door leaf).

   Each rejected candidate keeps its reason. When no stroke of the class forms a long
   straight run, the search stops after one cheap check.

   **Line-weight walls (fallback).** Some plans draw walls with ordinary line
   weight. If one sealed space holds most of the floor area, straight lines are
   tested as walls. Each end must be a junction or a door jamb, and no end may be
   free. A line is rejected if it is collinear fill at both ends, hinged at a wall's
   end face, or one of a repeated set of close parallel lines (stairs, shelving).
   The accepted network must connect to the walls. The fallback is kept only if it
   adds at least two substantial rooms.

   No wall pixel is ever invented: accepted walls are the drawn strokes themselves,
   and line walls are widened by at most one pixel inside the stroke's own outline.
4. **Wall bands.** Axis-aligned runs come from the mask. Diagonal walls come from
   tiled Hough on mask edges, re-centred on the medial axis. Each band has a
   thickness.
5. **Opening candidates.** These come from three sources:
   * **Wall gaps.** From every free wall end, a ray is cast along the wall to
     the next aligned wall end. A gap shorter than 1.1 × the wall thickness
     (at least 6 px) is a crack: it is sealed, not reported. Long gaps are only accepted when they are
     found from both ends or are bridged by drawn lines.
   * **Line pairs.** Two thin parallel lines that join two walls and enclose
     empty paper, as in windows drawn as outlines in hollow wall runs.
   * **Single lines.** A single thin line that continues a wall line, with paper
     on both sides, as in open edges or glazed walls.

   A gap is split where a lone frame line crosses the band, or where glazing
   starts or stops, unless the area is cluttered (text, hatching).
6. **Spaces.** Each candidate is closed with a wall-thickness bridge. The
   exterior is flood-filled from the border, and the remaining regions of at
   least (3t)² with a usable clear width become spaces. Rooms are not invented
   from text.
   **Room recovery** (`engine/room_recovery.py`, used by the analyzer). When OCR
   finds two or more room labels in one sealed space, a marker-controlled flood on
   the free floor's distance transform, seeded at the labels, places boundaries along
   the narrowest passages between them. Text is removed from the strokes first. A
   boundary is kept only if it is a real constriction, shorter than 0.45 × the
   smaller room's size. Labels open to each other (open plans) stay one region.
   Recovered rooms report the boundary method `passage-partition`, which the
   Workbench shows as an estimate.
7. **Topology.** Each candidate gets the labels of the regions on its two sides:
   `between_rooms`, `room_to_exterior`, `same_space` or `unknown`. Wall
   adjacency between spaces comes from a dilation inside each space's bounding
   box. The API returns this as `topology` (spaces, connections,
   wall_adjacency, wall_thickness_px).

## 2. Classification (`classify_openings`)

Each candidate is measured in its own wall-aligned frame (u along the wall, v
across it), so diagonal openings use the same code.

| Signal | What is measured |
|---|---|
| band lines | thin lines parallel to the wall inside the band: wall-face lines vs interior glazing lines (full or dashed); merged double lines are split |
| swing arc | quarter circle of radius ≈ width from either jamb, either side; drawn fraction, discounted unless continuous |
| double arc | two half-width arcs from both jambs |
| door leaf | straight leaf from a hinge at 30–90°; or a heavy leaf the wall filter kept, accepted only as a free-standing stub about the opening width long and thinner than the wall |
| sliding | two partial offset panels, or a shift of the ink centroid between the two ends |
| relation | topology from the structure pass |
| width | in wall thicknesses; in feet when a scale is known (door ≤ 4.6 ft, garage ≥ 8 ft) |

Decision order, in short:

* sliding → **door**;
* swing / double / leaf ≥ 0.55 → **door**;
* glazing lines → **window** (a single faint line between two rooms → unknown);
* outline only → **door** between rooms, or on an exterior wall when door-sized
  with two face lines, or a very wide exterior gap (garage);
* single exterior line, bare gap or mixed evidence → **unknown**
  (`type: "opening"`).

**Plan-level symbol conventions.** Every plan has windows. A drawing may show no
glazing-line window anywhere, draw its doors with door symbols (swing, leaf,
double, sliding), and never use a bare outline for a door between rooms. Its
repeated outline-only openings in exterior walls (at least two) are then that
drawing's window symbol, and they are classified as windows with a reason saying
so. Drawings that mark windows with glazing, or use outlines for doors, are
unchanged.

**An exterior position alone never makes a window**, and a door does not need a
swing arc. Every opening carries its evidence and a one-line `reason`.

## 3. Measuring

```bash
.venv/bin/python -m benchmark.run --engine new                 # tuning families + real sample
.venv/bin/python -m benchmark.run --engine baseline            # frozen pre-Phase-3 engine
.venv/bin/python -m benchmark.run --families holdout_mixed,holdout_scan,holdout_rotated,holdout_large --count 6
.venv/bin/python -m benchmark.run --salt 11 --count 3 ...      # fresh, never-seen plans
.venv/bin/python -m benchmark.candidates --families door_swing --errors
.venv/bin/python -m benchmark.inspect_errors door_leaf_only --salt 8
.venv/bin/python -m scripts.diagnostics.opening_evidence_view test_floorplan.png --ppu 41.73
```

The generator (`benchmark/generator.py`) draws plans with ground truth for
openings, rooms, walls and adjacency. Its layouts include BSP rooms, corridors,
L-rooms and dropped corners. Openings come in swing, leaf-only, double,
outline, sliding and garage door styles, and in triple, double, sample-style and
weak window styles. Plans can add text, furniture, dimension lines, speckle,
blur, JPEG artefacts and rotation. There are 16 tuning families and 4 hold-out
families, which use other scales, line weights, noise and skew. `--salt N`
regenerates every family with new seeds, for one-off unbiased checks. The real
sample has a hand-labelled ground truth in
`benchmark/fixtures/test_floorplan_openings.json`.

Matching is one-to-one by 1-D overlap along the wall: IoU ≥ 0.2, with an overlap
of at least 40 % of the shorter opening, on the same wall line. Rooms match by mask IoU ≥ 0.7.

## 4. Results (2026-10-05)

"Baseline" is the frozen pre-Phase-3 engine (`benchmark/legacy_openings.py`).
"Fresh" sets are drawn with `--salt` and were never used for tuning. Salt 7 was
evaluated once; two weaknesses it exposed were then fixed using other seeds. Salt
9 was evaluated once at the very end and not tuned on.

| Set | Engine | Cand. recall | Door P / R | Window P / R | Accuracy | UNKNOWN | Rooms (IoU ≥ 0.7) | Wall IoU |
|---|---|---|---|---|---|---|---|---|
| Tuning, 64 plans | baseline | 0.150 | 0.894 / 0.105 | 0.783 / 0.032 | 0.411 | – | 2.8 % | – |
| Tuning, 64 plans | new | 0.997 | 0.997 / 0.990 | 0.998 / 0.998 | 0.998 | 0.1 % | 98.7 % | 0.990 |
| Hold-out, 24 plans | new | 0.987 | 0.963 / 0.957 | 1.000 / 0.991 | 0.990 | 0.8 % | 96.9 % | 0.982 |
| Fresh salt 7, 60 plans (before the two fixes) | baseline | 0.156 | 0.929 / 0.067 | 0.825 / 0.092 | 0.523 | 40 % | 7.5 % | 0.932 |
| Fresh salt 7, 60 plans (before the two fixes) | new | 0.987 | 0.989 / 0.949 | 0.998 / 0.998 | 0.990 | 0.8 % | 95.8 % | 0.980 |
| **Fresh salt 9, 60 plans (final, untuned)** | baseline | 0.151 | 0.974 / 0.100 | 0.851 / 0.072 | 0.553 | 39 % | 5.9 % | 0.933 |
| **Fresh salt 9, 60 plans (final, untuned)** | **new** | **0.987** | **0.997 / 0.961** | **0.998 / 1.000** | **0.997** | **0.2 %** | **95.2 %** | **0.981** |

Real sample (`test_floorplan.png`, hand-labelled): baseline found 21 % of the
openings, with 0/17 doors and 3/11 windows plus 4 false windows. The new engine
finds 31/31 openings, 17/17 doors and 11/11 windows with no false positives. The
3 ambiguous openings are returned as unknown. The structure pass gives 16 spaces
covering all 17 room-name centres.

Fabricated wall pixels (wall mask outside the drawn walls): baseline 5.7 %, new
0.4 %. Space adjacency on salt 9: baseline P/R 0.81 / 0.03, new 0.997 / 0.960.

Time on the sample: geometry plus openings take 2.7 s (structure 2.6 s, of which
deskew estimation is 0.7 s; classification 0.1 s). This replaces 19.9 s for two
bridged wall masks, the old opening detector and the space segmenter. The full
`/api/analyze` call went from 37.9 s to 26.9 s on the same machine state; OCR is
now 90 % of it. See `PERFORMANCE_BASELINE.md`.

### Generalization phase (multi-scale structure), 2026-10-05

Real plans: `python -m benchmark.real_plans [--full]` (hand-labelled room points for
17 real plans in `benchmark/fixtures/real_plans.json`; openings labelled for 2.jpg,
4.jpg and test_floorplan.png). "Before" is the engine with this phase's changes switched
off.

| | Before | After |
|---|---|---|
| Real plans: room points separated by structure | 128/181 (0.707) | 143/181 (0.790) |
| Real plans: room points recovered in the API response | 125/181 (0.691) | 141/181 (0.779) |
| Real plans: rooms reported without a boundary | 9 | 4 |
| Catastrophic plans (interior collapse) | 2.jpg, 7.png, 12.png | 7.png |
| 2.jpg rooms / doors / windows | 0/5, 1/5, 0/7 | 5/5, 5/5, 7/7 |
| Synthetic tuning (64): door P/R, window P/R, rooms | 0.997/0.990, 0.998/0.998, 0.987 | identical |
| Synthetic hold-out (24) | 0.963/0.957, 1.000/0.991, 0.969 | identical |
| Fresh salt 9 (60, untuned) | 0.997/0.961, 0.998/1.000, 0.952 | 0.997/0.963, 0.998/1.000, 0.959 |
| test_floorplan.png | 17/17 doors, 11/11 windows | identical |

### N.2: jamb verification and repeated-line patterns, 2026-10-05

Non-structural ink (furniture, plants, markers, title text, fixtures) can become wall in the
mask. The harm comes when such a piece ends a gap: the gap is sealed and becomes a partition
that is not on the drawing. N.2 breaks this chain at the gap, without deleting wall components.

* **Jamb verification** (`structure._jamb_support`). Each end of every gap is checked for
  structural support: the piece belongs to the wall network, continues as a straight wall run,
  meets a wall at an angle (junction / corner), is a wall face, or is a short wall-width piece
  on a wall line (pier, mullion; the scan passes over further piers and openings).
* **Tone as supporting evidence** (`engine/wall_tone.py`). Missing structural evidence alone
  never rejects a gap: free-standing columns, bay-window piers and posts of outlined walls have
  none. An unsupported jamb is rejected only when the plan's own wall tone(s) — learned from the
  core pixels of long, thin wall runs; up to three tones, e.g. black exterior and gray interior
  walls — say the piece is drawn differently. Black-and-white plans, and plans whose furniture
  shares the wall tone, give no tone evidence, and nothing is rejected there.
* **Repeated-line patterns** (`structure._repeated_lines`). Line-bridged candidates (outlines
  between two walls) are dropped when they belong to four or more parallel, overlapping lines
  with no wall between them that spread over more than five wall thicknesses (stair treads,
  tub / shower hatching, tile lines).
* Rejected gaps are not sealed and not reported. They are kept in
  `StructureResult.rejected_candidates`, and every candidate carries `jambs` (also in the
  opening evidence as `jamb_support`) explaining why each end was or was not accepted.

### Physical space first, 2026-10-06

An audit of every labelled room point (`scripts/diagnostics/space_audit.py`: structure vs the
API's final geometry) and of every merge (`scripts/diagnostics/leak_finder.py`: the bottleneck
where two rooms connect) showed where geometry went wrong:

* **Door jambs with frame ticks on thin walls were discarded.** A 1-2 px jamb tick makes the end
  face look wider than the 3-4 px wall behind it; the band test sampled +-0.4 x the face length
  and fell outside the wall, so the jamb was "not a band", no opening was scanned, and the two
  rooms merged. The test now samples the wall's core (one pixel inside its edges).
* **Room names and printed sizes changed physical geometry.** Open-plan functional zones got
  dimension rectangles as boundaries inside one physical space, and misread sizes (`108"` read
  as 1'8") replaced correct rooms. Now: zones of an open-plan space keep the shared physical
  polygon (`open-plan-shared`; the printed size is `zone_extent` metadata only); implausible
  printed sizes (a side under 2.5 ft / 0.75 m) are not used for geometry or scale; compact
  inches prefer room-sized values.
* **Structural merges were reported as open plans.** A shared space holding an enclosed room
  type (bedroom, bath, closet, garage ...) is reported as `merged-space` (rooms not separated),
  not as an open plan.

Development set (these 17 plans are not an independent test set): structural separation
0.796 -> 0.829, API room recovery 0.807 -> 0.834, merged room points 32 -> 28, unsupported
boundaries 2 -> 0; synthetic tuning / hold-out / stress unchanged.

## 5. Known limits

* Walls drawn with the same line weight as everything else at low resolution
  (7.png: 2 px partitions among 1–2 px fixtures). The line-wall fallback recovers
  part of the structure (1 → 4 spaces), but walls that end at many door jambs
  are hard to tell from door leaves, so most rooms still merge.
* Very small synthetic plans with ≈ 5 px interior walls and 1 px lines, where the
  line-width estimate lands on the walls themselves: partly improved (fresh salt 9
  door_exterior rooms 0.55 → 0.70) but window_clear is unchanged (rooms 0.79).
* Doors in thin walls drawn diagonally, and door gaps at very low resolution, are
  still missed sometimes; rooms on both sides then merge, unless OCR labels allow
  a split at the passage.
* Plans drawn as 3D renders, and WEBP/AVIF files (rejected by the upload format
  check), are out of scope.
* Heavily degraded scans at low resolution (blur + JPEG + speckle): thin symbol
  lines break up, and sliding and leaf-only doors fall back to unknown.
* A leaf opened flat against an adjoining wall merges into that wall's face, so
  the opening is reported as an unknown bare gap.
* Double doors drawn without arcs, at one-sided wall ends.
* Hollow (double-line) wall drawing styles are not generated and are not
  specifically handled.
* Open-plan areas that have no wall between named rooms are one space. The
  analyzer then uses dimension boxes for those rooms (orange in the Workbench).
* Spaces are not labelled here. Room names come from OCR in `engine/analyzer.py`.
* Furniture drawn in the wall tone (gray desks and counters on a gray-wall plan, black
  furniture on black-and-white plans) is not caught by jamb verification, and non-structural
  pieces stay in the wall mask (N.2 changes which gaps are sealed, not the mask).
