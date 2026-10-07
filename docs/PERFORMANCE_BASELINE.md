# Performance baseline — `POST /api/analyze`

Measured 2026-10-05 with `scripts/diagnostics/profile_analysis.py` (runtime
`perf_counter` wrappers on the real functions; no production code changed).

**Machine state matters:** Apple M4 (10 cores), macOS 26.5.2, **on battery with
Low Power Mode on**. Earlier ad-hoc runs on the same code took ~19–20 s, so
expect roughly half these times on AC power. Always record power state when
comparing against this baseline.

Re-run: `.venv/bin/python -m scripts.diagnostics.profile_analysis --http --cprofile`

## Baseline

| | test_floorplan.png (2490×2420) | synthetic/generated/floorplan_0001.png (1200×900) |
|---|---|---|
| Runs | 37.49 / 37.63 / 38.58 s | 10.60 / 10.61 / 10.62 s |
| Average (engine, in-process) | **37.90 s** | **10.61 s** |
| Real HTTP round trip (avg) | 37.84 s (overhead 0.01 s) | — |
| Shipped desktop app, via UI (avg) | 38.07 s | — |
| Desktop server startup (one-time) | 0.41 s | — |

## Stage breakdown — test_floorplan.png (mean of 3)

```
POST /api/analyze handler                         37.90 s
├─ decode upload (cv2.imdecode)                    0.03
├─ FloorPlanAnalyzer.analyze                      37.87
│  ├─ OCR (extract_ocr)                           19.74
│  │  ├─ preprocessing (CLAHE, NL-means, thresh.)  0.80   (fastNlMeansDenoising 0.75)
│  │  ├─ 32 Tesseract search passes               18.42   (4 variants × 4 rotations × 2 PSMs)
│  │  ├─ re-run of the selected pass               0.50   (duplicate)
│  │  └─ rotation, mapping, dedup                  0.02
│  ├─ WallMaskBuilder.build (analyzer)             6.46   (bridging loop 6.44)
│  ├─ detect_openings                              7.27
│  │  ├─ WallMaskBuilder.build (again)             6.04   (bridging loop 6.03)
│  │  ├─ WallSegmentExtractor.detect (+Hough)      0.55
│  │  ├─ classification (clearance etc.)           0.58
│  │  └─ candidates, dedup, Canny, hull            0.10
│  ├─ room records                                 4.21
│  │  ├─ OCR retry crops (17 × psm 6+7)            4.20   (34 Tesseract runs)
│  │  └─ association, scale, snapping              0.01
│  ├─ SpaceSegmenter.detect_spaces                 0.11
│  └─ overlays (rooms + openings PNG)              0.08
└─ JSON + result storage                          <0.01
```

Tesseract: 67 runs per request, 21.89 s in the subprocess plus 1.18 s of
pytesseract temp-image encoding and TSV parsing, which is 61% of the request. The wall-mask
bridging loop runs its maximum of 8 iterations (14 → 6 components) twice:
12.5 s (33%).

## After Phase 3 (structure + door/window engine) — 2026-10-05

Same machine and state (battery, Low Power Mode on), same command, 3 runs:
**26.93 / 26.95 / 26.89 s (avg 26.92 s), down from 37.90 s.**

| Stage | Before | After |
|---|---|---|
| Wall mask (analyzer) with bridging loop | 6.46 s | – |
| `detect_openings` (second wall mask + extractor + classification) | 7.27 s | – |
| `SpaceSegmenter` | 0.11 s | – |
| `analyze_structure` (deskew 0.74 s, one structure pass 1.87 s) | – | 2.60 s |
| `classify_openings` (31 candidates) | – | 0.11 s |
| OCR + room-label OCR retries (unchanged) | 19.74 + 4.21 s | 19.9 + 4.2 s |

The duplicated wall-mask work is gone; OCR is now about 90 % of a request and
is the remaining optimisation target (not changed in Phase 3).
