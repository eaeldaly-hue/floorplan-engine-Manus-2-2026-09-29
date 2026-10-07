# Diagnostics

Visual/debug scripts for the geometry_v2 opening code, moved out of
`engine/geometry_v2/` in Phase 0 (algorithms unchanged). They are not tests
and are not part of the production path.

Run from the repository root so `test_floorplan.png` and `engine.*` resolve:

```bash
python -m scripts.diagnostics.integration_test
```

| Script | Writes |
|---|---|
| `integration_test.py` | `output_v2/opening_XX_crop.png`, `output_v2/opening_evidence.json`, `output_v2/wall_mask_v2_source.png` |
| `integration_overlay.py` | `output_v2/opening_candidates_overlay.png`, `output_v2/opening_overlay_data.json` |
| `local_opening_overlay.py` | `output_v2/local_opening_candidates.png` (runs without the wall mask) |
| `opening_debug_visualizer.py` | `output/opening_candidates_debug.png` (create `output/` first) |
| `local_opening_test.py`, `candidate_test_v2.py` | console only |

`output/` and `output_v2/` are git-ignored.

## Door/window evidence view (Phase 3)

`opening_evidence_view.py` draws, for every opening candidate of the current engine,
the wall band, the lines found in it, the best-scoring swing arc / leaf, and the
decision; it also writes an enlarged contact sheet and the evidence JSON. It is a
development aid only (not used by the app or the API).

```bash
python -m scripts.diagnostics.opening_evidence_view test_floorplan.png --ppu 41.73
```

Writes `output/evidence/<name>_evidence.png`, `<name>_evidence_sheet.png`, `<name>_evidence.json`.

## Pipeline trace for failed plans

`trace_pipeline.py` re-runs the production path stage by stage (upload decoder, structure
stages, full analyzer with OCR, and the real Flask endpoint) and records counts and
statistics for each stage, plus diagnostic images. Measurement only. Its "what-if kernel"
lines are counterfactual wall masks and never affect the production result.

```bash
python -m scripts.diagnostics.trace_pipeline path/to/plan.jpg --out output/diagnostics/failed_plan
python -m scripts.diagnostics.trace_pipeline --summary path/to/folder     # one line per plan
```

Writes `<out>/<name>/trace.json` and `01_gray.png` … `09_ocr_boxes.png`.

OCR evidence (written by `trace_pipeline.py` when the analyzer runs):
`10_ocr_all_pass_observations.png` (every word from every pass, coloured by rotation),
`11_ocr_regions_accepted_rejected.png` (resolved text regions: green accepted with support,
red rejected), `12_room_labels_and_association.png` (final room labels and the boundary each
refers to; magenta = open-plan shared space), `13_upright_vs_rotated_vs_final.png`, and
`ocr_regions.json` (every region with its readings, support, decision and reason).

## Benchmark N.1 overlays

`python scripts/diagnostics/n1_overlays.py [--only 5.jpeg,16.png]` writes `output/realplans/n1/<plan>.png`:
wall mask (gray), GT openings (green door / blue window / orange ambiguous), room points (red ring = its
group shares a space with another group), nonstructural points (red X = covered by wall), protected
points (orange square; red outer square = at risk under the compact/isolated-jamb probe), and false
opening candidates by probable cause (magenta furniture jamb, yellow pattern, brown small isolated jamb,
red other). The numbers come from `benchmark/real_plan_probes.py` (`python -m benchmark.real_plans --full`
prints them in the SUMMARY `n1` block). Synthetic stress families: `python -m benchmark.run --families
stress_dark_furniture,stress_markers,stress_hatch,stress_stairs,holdout_stress --no-sample`.
