# Learned perception: transfer experiment (2026-10-08)

Question: can learned floor-plan perception generalise to OUR drawings well enough to justify a
learned evidence layer? Research only. No production behaviour changed by this experiment; the
code lives in `research/perception/`, and data, weights and the training environment stay outside
the repository.

## 1. Pretrained models and licences

Every downloadable pretrained floor-plan model found is non-commercial:

| Candidate | Licence | Status |
|---|---|---|
| CubiCasa5k (repo, dataset, weights `model_best_val_loss_var.pkl`) | CC BY-NC 4.0 | Not used |
| OsamaMo/2dplan2strct (RF-DETR: wall / room / door / window boxes) | "non-commercial research and evaluation only" | Not used |
| FloorPlanCAD | CC BY-NC 4.0 | Not used |
| MLSTRUCT-FP | Unclear dataset rights; walls only | Not used |
| hallelu/floorplan-segmentation | MIT code, no weights; trained on CubiCasa | Not used |

An evaluation run to steer a commercial product is arguably "directed towards commercial
advantage", so under the conservative rule none of them was downloaded or run.

Instead, the experiment tests the route a commercial product would actually take: a model trained
**only on our own synthetic generator** (exact ground truth, no licence issue), evaluated on the
real canonical Test Cases. It never trains on them. This also directly tests the main risk of that
route, the synthetic-to-real gap.

## 2. Setup

- **Data** (`research/perception/synth_data.py`): 2,400 plans from `benchmark.generator` with
  randomised conventions.
  - Wall rendering: filled, hollow, hatched, thin.
  - Scale: 0.35–0.75 px/cm.
  - Line weights 1–3 and grey tones.
  - All door and window styles.
  - Text, furniture outlines, dark furniture, hatch, stairs, markers, dimensions.
  - Blur, JPEG, speckle, background tint, small rotations.
  - Per-pixel classes: exterior, interior, wall, door span, window span, passage span, other ink
    (furniture / fixtures / text). Door-symbol ink is ignored.
- **Model** (`model.py`): U-Net, 3.5 M parameters. Trained for 5,000 iterations on 320 px crops
  (scale jitter 0.6–1.6×), about 1 hour on an Apple M4 GPU.
  - Synthetic validation IoU: wall 0.90, door 0.84, window 0.75, passage 0.77, other 0.77,
    interior 0.80.
- **Inference** (`infer.py`): each real image is resampled so the engine's measured wall
  thickness is about 12 px, then run in tiles. 0.1–2.2 s per image on the Apple GPU (CPU/ONNX
  timing not measured yet).
- **Evaluation** (`eval_real.py`, `eval_zones.py`): the 17 dev images with room / opening /
  furniture / wall-piece ground truth, plus PDF pages.
  - **A** = the current production analyzer.
  - **B** = the model alone (rooms by flood fill with predicted walls and openings as barriers;
    openings from predicted components).
  - **C** = hybrid: the analyzer with the model's evidence in its wall layer (model walls added
    where ink exists; wall components the model calls furniture removed) and opening types from
    the model.

## 3. Results

### Walls and furniture (GT points on 17 dev images)

| | A engine | B model |
|---|---|---|
| Real short wall pieces (piers, stubs) kept as wall, of 24 | 22 | 21 |
| **Furniture / fixture points read as wall, of 26** | **21** | **0** |
| Furniture points read as furniture, of 26 | n/a | 22 |

Thin partitions: on 7.png (2 px interior walls) the model recovers the partitions the engine's
thickness filter misses. Visual, plus room results below.

### Rooms (17 dev images, 181 GT points: recovered / merged / missed)

| | Recovered | Rate |
|---|---|---|
| A engine | 169 / 7 / 5 | 0.934 |
| B model alone | 124 / 34 / 23 | 0.685 |
| C hybrid (add model walls + remove model-furniture walls) | 167 / 6 / 8 | 0.923 |
| C, add model walls only | 169 / 6 / 6 | 0.934 |
| C, remove furniture walls only | 164 / 7 / 10 | 0.906 |

Per plan, C beats A on 7.png, 12.png, 15.png and 21.png. It loses on 20.jpg (−5) and 6.png:
the model calls the 45° walls "other", because there are no diagonal walls in the synthetic data.

### Openings (9 plans; 77 doors, 62 windows, 27 passages)

| | Door P / R / F1 | Window P / R / F1 | Type accuracy | False | Missed | Passages typed as doors |
|---|---|---|---|---|---|---|
| A engine | 0.721 / 0.974 / **0.829** | 0.982 / 0.887 / **0.932** | **0.949** | 25 | 2 | 11 |
| B model alone | 0.496 / 0.766 / 0.602 | 0.547 / 0.758 / 0.635 | 0.876 | 84 | 18 | 11 |
| C, model retypes all | 0.686 / 0.909 / 0.782 | 0.820 / 0.806 / 0.813 | 0.889 | 29 | 4 | 11 |
| C, model retypes only weak-evidence openings | — / — / 0.820 | — / — / 0.897 | 0.927 | 25 | 2 | — |

The model is complementary per plan:
- 8.png: door precision 0.56 → 0.83, window recall 0.57 → 1.0.
- 11.png: door precision 0.62 → 0.83.
- 14.png: windows typed correctly.

It is worse on others: 16.png windows are lost, and 5.jpeg gets false windows on interior walls.
Glazing has no separate ground truth; it is part of the window class.

### Scale sensitivity (model alone, rooms)

| Input scale | Rooms |
|---|---|
| Engine-normalised scale (walls ~12 px) | 0.685 |
| Half | 0.398 |
| Double | 0.558 |

Normalising the input by the measured wall thickness is essential.

### PDFs / CAD sheets

The model fails on CAD sheets:
- 22.pdf p6: 0 / 80 rooms.
- 3.pdf p4: 5 / 19.
- Sheet frames and legend tables become walls.
- Dashed lighting circuits become windows.
- At 0.24× the walls are 1–2 px.

The production vector reading (66 / 74, 18 / 19) is far better: for vector PDFs the learned raster
model adds nothing.

### Open-plan zones (8 GT groups: several functions in one space)

Clustering objects in the space and splitting the floor by nearest cluster resolved **0 of 8**
groups, with raw ink objects (A) and with the model's furniture class (B) alike.
- In 5 of 8 groups the plan draws no furniture in the open area at all: the zones exist only as
  printed labels.
- Where furniture exists, untyped clusters do not follow the functions.

## 4. Failure modes found (trying to disprove it)

- **Missing from the synthetic data:**
  - diagonal walls and corner doors (20.jpg);
  - CAD sheet context (frames, title blocks, legend tables, schedules, dashed electrical circuits);
  - UI widgets next to plans (16.png);
  - grey paper and coloured fills (exterior / interior confusion on 5.jpeg);
  - drawing conventions with interior windows.
- **Opening false positives:** furniture edges and dashed curves read as door or window spans
  (84 false openings for B alone).
- **Room extraction:** the model's walls have small gaps, so rooms leak without the engine's gap
  sealing (B rooms 0.685).
- **No functional typing:** generic "other" cannot tell kitchen fixtures from a sofa, so no zone
  inference.

## 5. Verdict

- **Generalisation:** partial, and strong exactly where the hand-built engine is weakest.
  - Furniture vs architecture: 0 vs 21 false walls out of 26.
  - Thin partitions.
  - These came from 2,400 untuned synthetic plans and an hour of training.
  - The failures map to concrete gaps in the generator, not to a limit of the approach.
- **Not ready for production.** C ≈ A on rooms and C < A on opening typing with simple fusion.
  The model alone (B) is far below the engine.
- **Recommendation: build our own model**, as an evidence generator feeding one shared evidence
  layer. Train on an extended synthetic generator plus a small set of our own annotated real
  plans for validation and fine-tuning.
  - Gate it on C > A on the canonical set.
  - Do not use it for vector PDFs; the vector reading already wins there.
- **Spaces:** the Building model should hold enclosed rooms and functional zones. Zones need
  *typed* object evidence (and labels) that neither the engine nor this model provides yet.
