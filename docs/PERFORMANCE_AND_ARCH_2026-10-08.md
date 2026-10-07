# Analyze speed and typed openings (2026-10-08)

Canonical regression source: `../Test Cases` (benchmark.real_plans for the scored images,
benchmark.pdf_plans for the PDF pages with ground truth, benchmark.perf_suite for timing of every file).

## Where the time went (benchmark/perf_suite.py, real OCR, battery power)

| Inputs | Time | Dominant cost |
|---|---|---|
| small images (0.2-1 MP) | 1-3 s | OCR |
| large images (6-12 MP) | 4-7 s | structure + OCR (overlapped) |
| PDF sheets (17-24 MP) | 41-51 s | OCR 29-39 s (32 Tesseract passes, ~3 s each), structure 10-14 s, Plan Model 4-14 s *after* OCR |
| PDF sheets with cleaning (app) | 79-84 s | cleaning (~30 s) *before* analysis + the above |

Not the bottleneck: PNG hand-off to Tesseract (0.07 s), OCR preprocessing (1.4 s on 24 MP), PDF render (0.4 s).

## Changes

- **One structural job overlapping OCR**: [cleaner ->] legacy structure -> Plan Model reconstruction
  run while OCR reads; labels are attached after OCR (`engine.plan.adapter.prepare/finish`).
- **OCR memo** (`engine.analysis.ocr._memo`, `FLOORPLAN_OCR_MEMO`, default 4 pages): the same page
  is not re-read (another page tab, "Analyze again with cleaning").
- **Large images** are reduced to the 24 MP working resolution instead of being rejected.
- **Rejected**: OCR pass pruning. Measured offline on all test cases: every pass contributes unique
  accepted readings (dropping only the upside-down rotation loses 7.4% of room/dimension readings).

| Case | Before | After |
|---|---|---|
| 22.pdf p6 | 47.5 s | 38.5 s |
| 22.pdf p6, cleaning | 79 s (app) | 42.7 s |
| 3.pdf p4, cleaning | 84 s (app) | 42.3 s |
| re-analysis of the same page | full OCR again | OCR from memo |

Accuracy: development benchmark summary identical; PDF ground truth identical
(22.pdf p6 66/74, 3.pdf p4 18/19 with cleaning).

## Typed openings (engine/arch)

With cleaning, openings were re-detected from pixels on the sealed plan (3.pdf: 1 door, 1 window
reported). They now come from the cleaner's typed elements (`engine.arch.openings.typed_openings`),
related to the final spaces; the role follows topology: glazing on an exterior wall = window,
glazing-style panels between two rooms = sliding / bypass door, glass inside one space = partition
(unclassified), gaps < half the plan's door width = wall joints (not reported).
3.pdf p4: 19 doors / 12 windows reported vs 14 / 11 tags on the drawing (was 1 / 1). Known excess:
bifold halves counted twice; two wall pieces the poché missed reported as windows.

Investigated and not built: label-driven sealing of rejected wall gaps - on the development set only
3 label-conflicted spaces remain and none is separable by a rejected gap.
