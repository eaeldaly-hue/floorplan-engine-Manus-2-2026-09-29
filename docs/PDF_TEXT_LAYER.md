# PDF text layer

PDF pages exported from CAD carry room names, dimensions and scale notes as real text: exact
strings at exact positions. For a PDF page, the Workbench and API read that text layer and merge
it with OCR of the same page. Images and scans are analysed exactly as before.

```
PDF page ─► render (ingest/pdf.py)                 ─► image ─┐
         └► text layer (ingest/pdf_text.py)                  │
              quality gate ─► OCR-style boxes (pixels) ──────┼► FloorPlanAnalyzer.analyze(image, text_evidence=boxes)
                                                             │     OCR of the unchanged page
                                                             │     merge_document_text: PDF text wins where both read,
                                                             │     OCR keeps the rest
```

## Why a merge and not a replacement (evidence)

Production engine on full PDF pages. "Text only" replaces OCR with the text layer. "Union" (the
shipped design) adds it to OCR. Named rooms / analysis time:

| Page | OCR (before) | Text only | Union (shipped) |
|---|---|---|---|
| 05.pdf p9 (independent) | 0 / 29 s | 21 / 4.8 s | 21 / 32 s |
| Richmond A1 (independent) | 10 / 54 s | 19 / 3.4 s | 21 / 56 s |
| 22.pdf p5 (dev) | 33 / 192 s | 16 / 7.1 s | 34 / 194 s |
| 23.pdf p1 (dev) | 11 / 57 s | 11 / 8.4 s | 11 / 57 s |

* On 22.pdf, bedrooms, closets and the den are not in the text layer. CAD programs export some
  fonts (e.g. AutoCAD SHX) as drawn strokes, which only OCR can read. Text-only lost 17 names.
* A faster hybrid that blanked the text-layer words before OCR lost 5 bedrooms on 22.pdf. The
  multi-pass OCR voting depends on the whole page, so it was rejected.
* Union never lost a valid OCR name on these pages. The one OCR name it dropped on Richmond,
  "Terrace London", was a title-block address. Names from the text layer are exact (no misreads).

Speed is therefore not gained yet: OCR still runs. Skipping OCR safely needs a reliable signal
that a page has no stroked text; that is an open item.

## Quality gate (ingest/pdf_text.py)

The text layer is not used when the page has no words, when fewer than 80% of its words are made
of ordinary drawing characters (broken font encodings), or when a non-vector page carries an
invisible OCR layer added by another program (text render mode 3, or the GlyphLessFont of
Tesseract/OCRmyPDF). Across 79 pages with text in 7 PDFs, the median share of ordinary words was
0.99 and the lowest 0.89.

Dimensions that CAD exports as three words (`12'-0"`, `x`, `7'-0"`) are joined into one, the way
OCR reads the line.

## What changes in results

* PDF pages: `source.text_layer` = {words, quality, used, reason, boxes, seconds}. Rooms named from
  PDF text have `label_source: "pdf-text"`. One info line in `warnings`.
* Images, scans, `/api/analyze`: nothing. `analyze()` without `text_evidence` runs the
  unchanged code path (tests/test_pdf_text_layer.py, plus a before/after comparison of every
  development image).
