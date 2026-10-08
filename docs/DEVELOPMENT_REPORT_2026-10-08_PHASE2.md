# Development report – phase 2: architectural reconstruction (2026-10-08)

Canonical evaluation: `../Test Cases` only.
- Dev: 17 images, 181 GT room points, 154 printed names.
- Openings GT: 9 plans, 77 doors, 62 windows, 27 passages.
- PDF GT: 22.pdf p5–p8 and 3.pdf p4.

Phase start: 172acdb. Architecture overview: `docs/ARCHITECTURAL_RECONSTRUCTION.md`.

## Commits

| Commit | What |
|---|---|
| 415ffa3 | Hypothesis-based reconstruction (normalised scale, erosion-cliff wall class, Plan Model as a competitor); opening symbols read at a legible scale; door-sized rule for off-axis line openings; openings and scale-robustness harnesses |
| 054e722 | Openings as architectural objects (door operation, hinge, swing-into space; window exterior / outdoor; wall runs hosting openings; spaces list doors / windows); dominant-window-symbol convention; topology harness; self-diagnosis (`explain_plan`) |
| 680b9b2 | Vector PDFs: the page's own geometry is a reading without the toggle; door evidence keeps one-sided long gaps; cost caps |
| 2c5b883 | Workbench 'Building' view; swing arcs searched at drawn hinge offsets (double doors at T-junction jambs) |

## Results (phase start → now)

### Reconstruction

| | Before | After |
|---|---|---|
| API room recovery (dev) | 0.884 | **0.934** |
| Structural separation, single reading (dev) | 0.867 | 0.873 |
| Merged GT points (structure) | 21 | 20 |
| 7.png (2 px partitions) API recovery | 2 / 12 | **9 / 12** |
| 10.png API recovery | 11 / 13 | 13 / 13 |
| Same plans resampled ×0.8 / ×1.25 / ×1.5 | 0.669 / 0.878 / 0.878 | **0.801 / 0.928 / 0.919** |
| Building envelope: spaces wrongly put outside a building | 0 (dev) | 0 (dev), 0 (PDF) |

The `catastrophic` flag on 7.png in the structural benchmark is the single default reading. The
analyzer now reads 7.png at its normalised scale.

### PDF, default path (no cleaning toggle)

| Page | Separated before → after | Names before → after | Doors / windows after |
|---|---|---|---|
| 22.pdf p5 | 8 → **55** / 66 | 0 → 32 / 43 | 65 / 22 |
| 22.pdf p6 | 17 → **66** / 74 | 1 → 40 / 50 | 72 / 28 |
| 22.pdf p7 | 16 → **66** / 74 | 1 → 40 / 50 | 74 / 27 |
| 22.pdf p8 | 17 → **66** / 74 | 1 → 40 / 50 | 72 / 27 |
| 3.pdf p4 | 7 → **18** / 19 | 4 → 12 / 12 | 19 / 12 |

These numbers are identical to the cleaning toggle (variant E), at +6–10 s per page.

### Openings (9 plans with opening GT, through the analyzer)

| | Before | After |
|---|---|---|
| Door precision / recall | 0.612 / 0.961 | **0.721 / 0.974** |
| Window precision / recall | 1.000 / 0.581 | 0.982 / **0.887** |
| Classification accuracy (matched doors and windows) | 0.809 | **0.949** |
| Windows typed as doors | 20 | 5 |
| Passages typed as doors | 15 | 11 |
| False doors / windows | 27 | 25 |
| Missed doors / windows | 3 | 2 |

### Opening topology (new; `benchmark.topology_eval`)

| | Rate |
|---|---|
| Opening → host wall run | 0.912 |
| Door → two different spaces, or a space and the exterior | 1.000 |
| Window → exterior or outdoor space, from a room | 0.952 |
| Hinged door: swing side resolved to one of its spaces | 0.981 |

### Semantics

| | Before | After |
|---|---|---|
| Names correct (of 154) | 119 | 121 |
| Label recall / precision | 0.792 / 0.910 | 0.799 / 0.911 |
| Label-to-space association | 0.975 | 0.984 |

### Performance (26 test-case inputs, battery power, same conditions)

| | Hypotheses off | On |
|---|---|---|
| Total | 247.7 s | 284.4 s (+15 %) |

- Easy plans: +0.1 s.
- Ambiguous plans read twice: +1 to +3 s (7.png 0.7 → 1.9 s, 15.png 1.2 → 4.0 s).
- PDF pages: +2 to +6 s.
- The vector reading of a broken CAD page adds 6–10 s against the toggle path.
- Cost caps: the raster wall-class re-read is limited to 8 MP (73 s on a 24 MP sheet, never chosen
  there). When the page's vectors give a reading, the Plan Model is not prepared.

Tests: 226 → 237 passed. Desktop GUI smoke tests passed earlier in the phase.

## What genuinely improved

- **Hypotheses instead of irreversible early choices.** The plan's own wall thickness sets the
  working scale, and label/text evidence picks the reading. This fixed the thin-wall plan (7.png),
  made results far steadier across resolutions, and removed the 5.jpeg-at-×1.2 collapse.
- **Openings.** Window typing went from about 58% to 89% recall. Two principles drove it: symbols
  are read at a legible scale, and a drawing's dominant window symbol is learned from the drawing.
  Double doors with junction jambs are now confirmed.
- **Topology.** Doors and windows are objects on walls between spaces. A wall with a door is one
  wall run. Terraces and balconies are outdoor spaces.
- **PDFs without the toggle** reach the toggle's accuracy automatically.
- **Self-diagnosis.**
  - `explain_plan` lists every error with the engine's own evidence.
  - The 'Building' view shows what the engine believes the building is.

## Tried and rejected (evidence recorded)

| Idea | Result |
|---|---|
| Hinge-anchored swing-arc detector (doors found from wall-outline vertices without gap candidates) | 0 of 16 candidate-free detections were real doors; overall precision 0.67, below the existing classifier. Not adopted for discovery |
| "Openings lie along wall directions" as a hard rule | Removed real 45° corner doors (20.jpg 9 → 1 rooms). Replaced by "off-axis openings must be door-sized" |
| Leaf-tip and wall-face checks on door leaves | No measurable effect: the leaf test finds another angle that still scores ~1.0 on small openings. Reverted |
| Dropping the 30° leaf angle | Classification 0.88 → 0.79. Some doors are drawn barely open |
| Outline-window convention without the glazing guard | test_floorplan's open sides became windows. Replaced by the dominant-symbol rule |
| Uniform upscaling of every small plan | Hurts plans whose walls are already resolved. Replaced by thickness-triggered hypotheses scored on evidence |

## What did not improve / still difficult

- **False doors and passages typed as doors (25 / 11).** The leaf evidence is unspecific on small
  openings: ink lies within tolerance almost everywhere. A learned per-plan door appearance
  (self-supervised from confident doors) is the next step.
- **Unnamed rooms (about 26 per the diagnosis).** Tiny text (4–5 px) in 7.png and 14.png is still
  unreadable. OCR is not yet re-run at the normalised scale.
- **Remaining merges.** 7.png (bonus room double door: its jamb end faces are not produced at
  that scale), 11.png and 14.png (one each).
- **Envelope.** On the Plan Model PDF path the footprint follows dimension strings attached to
  walls. With the vector reading now chosen there, this matters less.
- **Hypothesis scoring needs evidence.** A plan with neither labels nor text blocks scores every
  reading equally, and the default wins.

## Recommended next steps

1. **Per-plan learned opening appearance.** Use confident doors and windows (strong arcs, glazing,
   topology) as self-supervised examples of this drawing's symbols, and classify the ambiguous ones
   by similarity. Target: the 25 false doors and 11 passages.
2. **OCR on the chosen reading.** When the normalised scale wins, re-read labels there, starting
   with the context OCR of unnamed spaces.
3. **More hypothesis dimensions with the same scoring.** Seal-tolerance variants for remaining
   merges; per-region readings for plans that mix wall pens.
4. **Opening topology as a scoring term.** Doors that connect two distinct spaces and windows
   on exterior walls are cheap, strong evidence for choosing between readings.

Ground-truth notes (labels unchanged):
- 10.png prints 5 BATH and 2 HALL labels; GT lists one of each.
- 14.png has a second GARAGE.
