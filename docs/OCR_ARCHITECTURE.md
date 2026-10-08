# OCR architecture (2026-10-08)

Measured on the canonical `../Test Cases` (17 scored images + PDF pages with ground truth).

## Findings

1. **Cleaning before OCR does not help.** A text-only rendering (glyph-sized marks only; walls,
   long lines, large outlines removed - `engine.analysis.text_lines.text_image`) gives about the
   same Tesseract time (22.pdf p6 33 -> 39 s, 3.pdf p4 46 -> 44 s, images +-10 %) and loses words
   that touch lines or hatch (3.pdf: 19 -> 17 room words). Tesseract's cost is image area x number
   of passes, not drawing clutter. The architectural cleaned image contains no text by design.
   Rejected; `text_image` is kept as a representation (text as negative evidence for geometry).
2. **Text size, not clutter, limits recall.** Page passes read at one scale; small-plan text
   (4-6 px) is unreadable. Fix: scale-normalised text lines (`engine.analysis.text_lines`).
3. **Most page passes are redundant once lines are read individually.** Offline (all passes and
   text-line readings recorded once per case, re-aggregated per subset), printed room names found
   of 382:

| Configuration | Names | OCR CPU |
|---|---|---|
| 32 page passes (before) | 277 | 1242 s |
| 32 page passes + text lines | 304 | 1350 s |
| **8 upright page passes + text lines (adopted)** | **299** | **387 s** |
| + rotated grey passes | 300 | 470-557 s |
| text lines only | 128 | 108 s |

## Pipeline now

page -> [4 preprocessing variants x upright x psm 11/6] + [text lines found geometrically, each
cropped, turned upright, rescaled to ~30 px glyphs, packed into mosaics, psm 11/6] -> all in one
parallel Tesseract pool -> region aggregation (text seen only by the line reader must be domain
text) -> PDF text layer merged where present -> per-process OCR memo.

Switches: `FLOORPLAN_TEXT_LINES=off` (old 32-pass reading), `FLOORPLAN_OCR_ROTATIONS=all`.

## End-to-end (same machine, mains power)

| | Before | After |
|---|---|---|
| Analyze, 26 test-case inputs | 406 s | 230 s |
| OCR stage total | 332 s | 136 s |
| Dev names correct (of 154 printed) | 106 (32 passes) / 112 (32 + lines) | 113 |
| Dev geometry (separation) | 0.867 | 0.867 |
| PDF ground truth, cleaning (22.pdf p6 / p7, 3.pdf p4 names) | 40 / 40 / 12 | 40 / 40 / 12 |
