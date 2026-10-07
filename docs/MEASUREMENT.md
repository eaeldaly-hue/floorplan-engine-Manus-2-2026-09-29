# Measurement layer (Phase 1)

Three benchmarks, reported separately — synthetic results are never real-world accuracy:

| Benchmark | What it measures | Command | Time (battery, Low Power Mode) |
|---|---|---|---|
| Wall styles (synthetic) | the same 12 layouts drawn in 5 wall conventions; walls, rooms, openings | `python -m benchmark.wall_styles [--count N --styles a,b --images DIR --save F]` | 2 min (no OCR) |
| Development set (17 real plans) | room points, physical spaces, names, openings (existing) | `python -m benchmark.real_plans --full`, `python -m benchmark.physical_eval --api` | 99 s cold, 48 s with OCR cache |
| Hold-out (9 real plans, sealed) | generalisation; physical spaces, zones, false spaces, openings | `python -m benchmark.real_plans --holdout --full` (after the freeze) | — |

## Wall-style benchmark (`benchmark/wall_styles.py`)

The generator draws the wall band in a chosen convention (`Style.wall_render`: filled, hollow,
hatched, thin; deterministic, no randomness). Layout, openings, room names and all ground truth are
identical across styles — checked on every run (`gt_digest`); the original families are
byte-identical to before (75/75 plans). `filled_furniture` adds solid dark furniture against walls,
furniture outlines and a stair run to filled walls.

Wall metrics (`benchmark/wall_metrics.py`): pixel precision / recall against the drawn wall
(band, or the line of a thin wall; 2 px tolerance), centerline recall (wall axes covered by
predicted wall, opening gaps excluded — style-independent), segment recall (pieces covered over
>= 80 %, also length-weighted), fabricated share (predicted wall farther than 2 px from any band),
phantom components, fragmentation, thickness ratio. Rooms and openings use the existing
`StructureStats` / `OpeningStats`.

### Baseline (2026-10-07, 12 layouts x 5 styles, 90 rooms per style)

| Style | wall P | wall R | axis R | seg R | fabricated | phantom/plan | room R (IoU 0.7) | merged / split / false | door R | window R |
|---|---|---|---|---|---|---|---|---|---|---|
| filled | 0.994 | 1.000 | 1.000 | 1.000 | 0.006 | 0.5 | **1.000** | 0 / 0 / 0 | 0.987 | 1.000 |
| hollow | 0.810 | 0.023 | 0.003 | 0.000 | 0.190 | 1.7 | **0.000** | 0 / 0 / 8 | 0.026 | 0.000 |
| hatched | 0.872 | 0.038 | 0.047 | 0.066 | 0.128 | 1.3 | **0.000** | 0 / 0 / 0 | 0.051 | 0.103 |
| thin | 0.801 | 0.105 | 0.091 | 0.146 | 0.186 | 1.8 | **0.000** | 0 / 0 / 0 | 0.013 | 0.000 |
| filled_furniture | 0.798 | 1.000 | 1.000 | 1.000 | 0.202 | 9.0 | **0.778** | 4 / 10 / 0 | 0.897 | 0.916 |

9 of 12 hollow and thin plans, and 7 of 12 hatched plans, produce no wall at all.

## Benchmark OCR cache (`benchmark/ocr_cache.py`)

Benchmarks only (never the app). `extract_ocr(image)` and the dimension re-reads `ocr_lines(crop,
psm)` are served from `benchmark/.ocr_cache` when the same pixels were read by the same OCR code;
the key includes a hash of the OCR source files, the Tesseract version and the language. On by
default in `real_plans --full` and `physical_eval --api` (`--no-ocr-cache` to disable,
`FLOORPLAN_OCR_CACHE=off|DIR`). Cached output is identical to uncached (tested).

| Input | Cold | Warm | Output |
|---|---|---|---|
| Development set, `real_plans --full` (17 plans) | 99 s | 48 s (108 s of OCR served in 0.1 s) | identical summary |
| 22.pdf p5 (CAD sheet, Workbench flow) | 77.8 s | 13.6 s (5.7x) | identical |

## Hold-out preparation

`docs/HOLDOUT_PROTOCOL.md`. Dataset: `../floorplan_test_dataset_01`. Novelty check: 9 of 10 plans
are new; `images/09.png` is the development plan `9.jpg` (similarity 1.00) and is excluded. Labels:
`benchmark/fixtures/holdout_plans.json` (9 plans, 88 physical spaces, 101 functional zones, 109
openings, 110 non-structural elements, 6 protected columns), review images in
`output/holdout_review/`. Not frozen; the baseline runs once, after the freeze.
