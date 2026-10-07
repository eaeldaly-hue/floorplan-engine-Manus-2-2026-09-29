# experimental/ — archived, NOT part of production

Nothing in this folder is imported by `app.py`, `engine/`, or the test suite.
The production execution path (`app.py` → `engine.analyzer.FloorPlanAnalyzer`)
must never import from here. Code here is kept because it contains ideas worth
revisiting, not because it works; treat every result it produces as unverified.

Run anything here from the repository root with `python -m`, e.g.
`python -m experimental.topology.run_wall_topology_analysis`, so that
`engine.*` resolves to the production engine.

## topology/

Recovered from a duplicate project copy that used to live in
`floorplan-engine-Manus-2-2026-09-29/` (removed in Phase 0; still in git
history at commit `a9928d7`). These were the only files in that copy not
present in the production tree or in earlier history.

| File | Origin (nested copy) | What it is |
|---|---|---|
| `topology.py` | `engine/rooms/topology.py` | Wall-topology rasterisation with virtual bridges across group gaps, flood-fill regions, large-region subdivision (V5/V5.1). No importers. |
| `reconstruction_engine.py` | `engine/rooms/reconstruction_engine.py` | Planar wall graph: split at H/V intersections, node snapping, half-edge face walk → room faces. |
| `reconstruction_v2.py` | `engine/rooms/reconstruction.py` | "V2" of `engine/rooms/reconstruction.py` (endpoint snapping, collinear merge, morphological cleanup). The production tree still has V1 at `engine/rooms/reconstruction.py`. |
| `run_wall_topology_analysis.py` | `run_wall_topology_analysis.py` | Diagnostic that produced "53 segments / 38 groups / 13 gaps" on `test_floorplan.png`. Writes to `output_v2/`. |
| `run_room_reconstruction.py` | `run_room_reconstruction.py` | Driver for `reconstruction_engine.py`. Writes to `output_v2/`. Only its import path was updated when it was moved. |

Known issues (found in the Phase 0 audit, deliberately not fixed here):

- Both `run_*.py` scripts monkey-patch `WallSegmentExtractor.detect(self)`
  without accepting keyword arguments. The production
  `engine/opening_detection.py` calls `detect(include_diagonal=True)`, so
  against the current engine they fail with
  `TypeError: ... unexpected keyword argument 'include_diagonal'`. They only
  ran against the older nested engine.
- `reconstruction_engine.py` does not reject the unbounded outer face, so the
  building outline is reported as a room; its collinearity tolerance is applied
  to a cross product (≈0.1 px effective).
- Grouping (from `engine/geometry_v2/groups.py`) is greedy and order-dependent.

## cnn_opening_classifier/

A small PyTorch CNN that classifies a door/window symbol patch, plus its
training script, weights, training notes and the PERDAW dataset licence.
It was removed from the production path in Phase 0 because the engine must be
deterministic and fully algorithmic. Before removal it only ran when PyTorch
was installed; in the default environment it was already inactive, so removing
it did not change detection results.

- `opening_symbol_model.py` — model definition, patch cropping, inference.
  Weights default to `models/opening_symbol_classifier.pt` next to this file
  (override with `FLOORPLAN_OPENING_MODEL`).
- `train_opening_symbol_model.py` — training script; see `MODEL_TRAINING.md`.
- `third_party/perdaw/LICENSE` — attribution for the training data, kept with
  the weights derived from it.

PyTorch is intentionally not a project dependency. Install it separately
(`pip install 'torch>=2.2,<3'`) only if you want to experiment with this code.
