# PDF input

PDF support is an input layer in front of the unchanged recognition engine:

```
PDF upload ─► inspect (pages, sizes, rotation, vector / scanned) ─► page thumbnails
          ─► the user selects page(s) ─► each page rendered at the analysis resolution
          ─► existing engine (FloorPlanAnalyzer) ─► existing API result + source page ─► Workbench
```

The user chooses the floor-plan pages. Nothing classifies pages, merges floors or relates pages
to each other: every selected page is an independent analysis that remembers its source page.

## Code

* `ingest/pdf.py` - inspection (`inspect_pdf`), thumbnails (`render_thumbnail`), the resolution
  policy (`analysis_dpi`) and page rendering (`render_page`); typed, user-readable errors
  (`PdfInputError` with a `code`). Uses pypdf (structure) and poppler (`pdftoppm` via pdf2image for
  rendering, `pdfimages -list` for embedded-image metadata) - the dependencies the project
  already had.
* `app.py` - `POST /api/pdf` (store + inspect), `GET /api/pdf/<id>/pages/<n>/thumbnail.png`,
  `POST /api/pdf/<id>/pages/<n>/analyze`, `GET /api/results/<id>/page.png` (the exact image that
  was analysed, which the Workbench draws over). PDFs are kept in `instance/pdfs/` (last 20).
* `static/js/main.js`, `api.js`, `templates/index.html`, `static/css/app.css` - page picker and
  page tabs in the Workbench.
* The original `POST /api/analyze` keeps its behaviour for PDFs (first page, 150 DPI) for API
  clients; the Workbench uses the page-selection flow.

## Rendering resolution

* Default **150 DPI**. At 150 DPI, 2.5 mm drawing text is about 15 px tall (readable by the OCR) and
  a 100 mm wall at 1:50 about 12 px thick, the range the engine was built and validated for.
  Rendering is cheap (0.3-1.2 s per sheet); analysis time grows with pixels (an ARCH D sheet at
  150 DPI, 19 MP, takes ~2.5 min).
* **Large sheets**: the DPI is lowered so the page stays within 24 MP (the engine limit is 25 MP):
  ARCH E1 (30 x 42 in) -> 138 DPI, A0 -> ~123 DPI. Below 36 DPI the page is refused as too large.
* **Scanned pages** (images cover the page): never rendered above the scan's own resolution (no
  upsampling); a 100 ppi scan is analysed at 100 DPI, i.e. at its native pixels.
* **Vector pages** are rasterized by poppler with anti-aliasing at the chosen DPI; **mixed** pages
  are treated like vector pages.
* Poppler applies the page `/Rotate` and renders the CropBox (what a PDF viewer shows).
* Thumbnails are rendered separately at 360 px, on demand per page (~0.14 s each), never at
  analysis resolution.

## Errors

not a PDF (`not_pdf`), damaged (`corrupt`), password-protected (`encrypted`; PDFs with an empty
user password open normally), no pages (`no_pages`), unreadable page (`bad_page`), page too large
(`page_too_large`), rendering failure / timeout / out of memory (`render_failed`, `render_timeout`,
`out_of_memory`), page out of range (`bad_selection`). Uploads up to 100 MB.

## Limitations (recognition, not ingestion)

Engineering sheets are analysed as whole sheets, including title blocks, schedules and notes, and
CAD drawings often draw walls as hatched or hollow double lines, which the engine does not yet
treat as walls. Result quality on such sheets is limited by the engine, not by the PDF layer.
