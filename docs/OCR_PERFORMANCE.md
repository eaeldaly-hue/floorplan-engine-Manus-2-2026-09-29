# OCR performance (concurrent reads, indexed aggregation)

The analysis did the same work as before, one step at a time on one core: 32 full-page Tesseract
passes in a row, a quadratic aggregation of their readings, the dimension re-reads one by one,
and the structural pass after OCR. The results are unchanged; the work is now scheduled better.

| Change | Where | Why the output cannot change |
|---|---|---|
| The 32 passes (4 variants × 4 rotations × 2 PSMs) run concurrently on a shared pool of Tesseract processes (default 8, `FLOORPLAN_OCR_WORKERS`; `1` = sequential) | `engine/analysis/ocr.py` (`run_ocr_tasks`), `ocr_aggregation.extract_aggregated` | Each pass is an independent process on the same pixels; results are consumed in the fixed pass order, so the readings list is identical |
| Aggregation: a grid index finds the regions a reading can belong to, instead of comparing it with every region | `ocr_aggregation.group_regions`, `_merge_regions` | The index only skips pairs that provably cannot match (boxes that do not touch); every candidate is tested with the unchanged rule and the first match in the original order wins |
| Dimension re-reads near labels run concurrently | `analyzer._build_room_records` | Each re-read depends only on its own label; results are applied in room order |
| The structural pass (walls, gaps, spaces) runs while OCR reads the text | `FloorPlanAnalyzer.analyze` | It needs only the image; its result (or exception) is taken where it was computed before |

The Tesseract installed here is single-threaded (no OpenMP): one process uses one core, which is
why processes, not threads inside Tesseract, are the unit of parallelism. Rotated copies are made
inside each task, so at most `workers` copies exist at a time.

## Measurements (2026-10-06, Apple M4 10 cores, 16 GB, on battery, Low Power Mode on)

Same inputs, same machine state, one process per input; baseline = frozen copy of the code before
this change. Times vary ±30% with machine load (the same configuration measured 98 s and 141 s on
one page), so read ranges, not single digits.

| | 22.pdf p5–8 (4 pages) | 20 development images |
|---|---|---|
| Total time | 1768 s → 513 s (**3.4×**; per page 2.9–5.1×) | 475 s → 201 s (**2.4×**; per image 1.2–4.4×) |
| OCR passes (elapsed) | 1473 s → 444 s | 351 s → 125 s |
| Aggregation | 132.8 s → 3.6 s (37×) | 2.9 s → 1.0 s |
| Dimension re-reads (elapsed) | 45.9 s → 14.7 s | 42.2 s → 16.3 s |
| Structural pass | 77.5 s → 248.6 s, now overlapped with OCR | 50.1 s → 135.0 s, overlapped |
| Average cores in use | 1.05 → 4.6 | 1.17 → 4.0 |
| CPU time | 1854 → 2360 CPU-s (+27%) | 558 → 802 CPU-s (+44%) |
| Peak memory (process + Tesseracts) | 881 → 1507 MB | 651 → 1213 MB |
| Tesseract processes | 376 → 376 | 940 → 940 |
| Output identical | 4/4 | 20/20 |

"Output identical": every OCR reading of every pass, the pass summaries, every region with its
observations and decision, the accepted boxes, the merged text (PDF text layer + OCR) and the final
API result including overlay images. 9.jpg is rejected by the engine (> 25 MP) in both versions.

Worker count (22.pdf p7 / 1.png / 19.jpg): 4 → 173 / 16 / 22 s, 6 → 149 / 14 / 30 s,
8 → 98–141 / 15 / 25 s, 10 → 88 / 14 / 25 s. 8 is the default: 10 is within noise of 8 with
more memory and no core left for the structural pass. Lower Tesseract priority (nice 10) gave no
measurable gain and was not kept. All 18 tuning runs were output-identical to the baseline.

CPU time rises because each Tesseract runs slower when 8 run at once (efficiency cores, power
limit in Low Power Mode): the summed pass time grows ~2.3–2.7× while the elapsed time falls ~3×.
On AC power the speed-up should be larger; it was not measured here.
