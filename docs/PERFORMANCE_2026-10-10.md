# Analysis speed, measured end to end (2026-10-09/10)

Tool: `python -m benchmark.profile_app [--cases ...] [--ocr-cache] [--rss]`.
- It runs exactly the Workbench path (PDF: render + text layer + analysis with the vector-cleaned
  candidate; image: decode + analysis), one process per case.
- It reports wall time, time after OCR, per-stage time with a timeline, and peak memory (with
  `--rss`, a memory trace and the stages active at the peak).

## Baseline vs now (Apple M4, 10 cores, same session)

Each version ran cold twice, interleaved; the rounds agree within ~1 %. Warm runs serve OCR from the
cache, as when a page is analysed again in the app.

| Input | MP | HEAD 6808f40 cold | Now cold | Saved | Warm: HEAD → now |
|---|---|---|---|---|---|
| 3.pdf p5 | 19.4 | 51.5 s | 31.0 s | 40 % | 51.1 → 23.7 s |
| 3.pdf p4 | 19.4 | 67.5 s | 45.4 s | 33 % | 67.4 → 30.4 s |
| 22.pdf p6 | 24.0 | 89.1 s | 53.7 s | 40 % | 89.8 → 38.3 s |
| 24.pdf p1 | 17.4 | 62.5 s | 37.5 s | 40 % | 62.3 → 37.5 s |
| 29.pdf p1 | 4.4 | 33.4 s | 29.3 s | 12 % | 33.2 → 29.4 s |
| 27.pdf p1 | 4.4 | 6.5 s | 6.0 s | 7 % | 6.4 → 6.3 s |
| 26.jpg | 24.0 | 48.0 s | 22.1 s | 54 % | 47.9 → 21.9 s |
| 9.jpg | 24.0 | 32.3 s | 20.7 s | 36 % | 32.3 → 20.6 s |
| 1.png | 2.8 | 15.0 s | 8.3 s | 45 % | 15.1 → 8.3 s |
| 20.jpg | 12.0 | 8.2 s | 7.3 s | 11 % | 11.1 → 5.1 s |
| 7.png | 0.2 | 3.4 s | 2.4 s | 30 % | 3.4 → 2.2 s |
| **Total** | | **417 s** | **264 s** | **37 %** | **420 → 224 s (47 %)** |

**Results.** Rooms, spaces, openings and zones are identical for every input. The dev benchmark
summary, PDF ground truth, openings and zones benchmarks are unchanged.

**An earlier "regression".** One full-suite run measured slower than mid-session runs. The paired
A/B showed it was the machine: a first round ran 1.5–2× slower for *both* versions. Only interleaved,
same-session comparisons are reported.

## Where the time went, and what changed

All structural and cleaning changes are **exact**: fingerprints of the cleaned layer (every element
type, closure, wall body) on 8 vector pages and of the structure (wall mask, spaces, gaps) on
8 inputs are identical, and `tests/test_speed_equivalence.py` checks each rewrite against the
original algorithm.

| Stage (critical path) | Cause | Change |
|---|---|---|
| Vector cleaning, 27-40 s per sheet | `_drop_isolated_walls`: one 20 × wall-thickness elliptical dilation of the whole page (12 s) | an exact Euclidean distance transform decides clusters clearly inside / outside the reach; only boundary clusters are dilated locally (the element is symmetric) |
| | `_glazing`: all pairs of candidate lines compared in Python (20 M tests) | pairs only within a spatial grid of margin-grown boxes, after a vectorised 2° angle test (the same pairs) |
| | 12 pen tests (wall reconstruction per pen), serial | 4 threads, results in pen order |
| | `primitives._measure`: per-edge NumPy loop (~24 k edges) | all edges measured in batches; `_measure_reference` kept as the test oracle |
| Structure, 7-43 s | `_line_bridged_gaps`: 130 k Python traces of Hough pieces, each sampling the full 32 t search length | all pieces traced at once in NumPy, in chunks only as far as lines go |
| | Hough over ~500 tiles, serial | tiles on 4 threads (OpenCV releases the GIL; HoughLinesP seeds its RNG per call) |
| Room records, 23 s on 24.pdf | a Tesseract re-read near every label (378 labels) looking for printed dimensions | skipped when page OCR found no dimension-like text at all; measured on 22 plans: 0 of 427 such re-reads ever found a dimension, and all successful re-reads were on pages with dimension text |
| OCR | page passes waited for the text-line mosaics | page passes start as soon as the variants exist; mosaics are built meanwhile (identical readings) |
| Scheduling | the vector cleaning ran after the default reading | it starts at t = 0, beside the default reading and OCR (`FLOORPLAN_VECTORS_EARLY=0` restores serial): about 1.5 s (5 %) |

## Rejected, with measurements

**OCR pass pruning** (`python -m benchmark.ocr_passes record|evaluate`). All observations were
recorded once and every subset of the 8 page passes + text-line passes re-aggregated. Subsets were
chosen on 17 dev images and validated on 5 PDF pages:
- **Dropping one page pass:** no loss, −7 % OCR CPU. Not worth the risk on unseen styles.
- **Dropping 2–4 page passes:** the images show no change (they cannot discriminate), but the PDFs
  lose 1–11 of 46 room names.

All 8 passes are kept, as concluded in docs/OCR_ARCHITECTURE.md.

## Memory

Peak resident memory is 20–35 % higher (e.g. 3.pdf p5 2.2 → 3.0 GB, 22.pdf p6 2.6 → 3.3 GB),
because more work overlaps:
- the pen tests run in parallel;
- the OCR page passes run during mosaic building;
- the cleaning runs during the structure pass.

`--rss` traces show the peak during cleaning's full-resolution gap reconstruction, with no leak.
About 2.4 GB stays resident to the end, because every alternative reading keeps its structure
arrays until the result is built. Traced allocation of the structure pass alone is unchanged.

## Remaining time (now)

- **PDF sheets:** the critical path is cleaning (10–15 s alone, ~20 s beside OCR) → structure of
  the cleaned reading (~5 s) → model, objects, overlays (~4 s).
- **29.pdf:** OCR-bound (21 s of Tesseract).
- **Large images:** structure (Hough, gap tracing, kernel search).

## Second cycle (2026-10-10) and final A/B

Additional changes:
- **Text lines:** found once per page (memo by pixels), and deduplicated and merged on grids.
- **Vectorised loops:** wall runs, object circles and glazing candidates are vectorised (all exact,
  with equivalence tests).
- **Memory:** the alternative readings are released once the openings are typed.
- **Tesseract's inverted-text retry is off** (`tessedit_do_invert=0`, `FLOORPLAN_OCR_INVERT=1`
  restores it).
  - On the 22 inputs with ground-truth names, the retry costs 26 % of the Tesseract CPU (830 → 615 s
    summed) and changes no room name.
  - Dimension readings: 623 → 619. One real wall dimension on 22.pdf is lost; one reading on 21.png
    improves.
  - The dev benchmark and PDF ground truth are identical with it off.

Measured and reverted:
- **Speculative object reading beside the room records:** no gain; both are Python and share the
  GIL.
- **6 instead of 8 OCR workers:** 29.pdf ~3 s slower, sheets unchanged.

**Final A/B** against 6808f40 (same session, interleaved, 11 inputs). Outputs (rooms, spaces,
openings, zones) are identical on every input. Machine speed drifted between rounds, so each round
is paired:

| | Total HEAD 6808f40 | Total now | Saved | Median per input |
|---|---|---|---|---|
| Round 1, cold | 417 s | 230 s | 45 % | 43 % (range 20-58 %) |
| Round 2, cold | 452 s | 186 s | 59 % | 49 % (range 10-72 %) |
| Warm (OCR from cache) | 212 s | 111 s | 47 % | |

Per input (cold, mean of two rounds):

| Input | Before | After |
|---|---|---|
| 3.pdf p5 | 51.8 s | 25.0 s |
| 3.pdf p4 | 67.3 s | 35.7 s |
| 22.pdf p6 | 105.5 s (HEAD round 2 outlier; 88.1 s round 1) | 39.6 s |
| 24.pdf | 70.3 s | 30.2 s |
| 29.pdf | 43.4 s | 22.3 s |
| 26.jpg | 40.2 s | 18.3 s |
| 9.jpg | 27.6 s | 17.0 s |
| 1.png | 12.2 s | 6.7 s |
| 20.jpg | 7.4 s | 6.3 s |
| 27.pdf | 5.8 s | 4.8 s |
| 7.png | 3.0 s | 2.1 s |
