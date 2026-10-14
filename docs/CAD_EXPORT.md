# CAD export: DXF and DWG

The Workbench offers the reconstructed plan as a CAD drawing (`engine/export/`). The exporter only
serialises the analysis: walls, openings and rooms come from the structured plan
(`result["building"]`, engine/arch/building.py) and the room records; nothing is detected,
corrected or filled in during export. A wrong wall in the drawing is a detection problem, a
displaced one a coordinate problem (see "Coordinate frames"), and only a malformed entity an
export problem.

## Access

| Where | What |
|---|---|
| Workbench, *Raw API response* | **CAD DXF** link on every result with a model; **CAD DWG** only when a DWG converter is installed |
| `GET /api/results/<id>/export.dxf` | DXF (R2010, `application/dxf`) |
| `GET /api/results/<id>/export.dwg` | DWG (R2000, `application/acad`); `503 dwg_unavailable` without a converter, `500 dwg_conversion_failed` when the converted file fails its check. Never a DXF under a `.dwg` name. |
| `GET /api/export/formats` | `{"dxf": {...}, "dwg": {"available", "backend", "verified", "reason"}}` |
| Python | `engine.export.dxf.build_dxf(result)` / `dxf_bytes` / `save_dxf`; `engine.export.dwg.dwg_bytes(result)` |

Each result directory keeps `export.json` (source name, image size, scale, source, building, rooms,
openings): the input of both exporters. Responses carry `X-Export-Units` and `X-Export-Skipped`.

## Layers

| Layer | Content | Source |
|---|---|---|
| `WALLS` | each wall piece as a closed outline: centreline ± half its **measured** thickness | `building.walls` (`p0`, `p1`, `thickness_px`) |
| `WALL_CENTERLINES` | the wall pieces' axes | same |
| `DOORS` | door span (threshold); leaf + 90° swing arc from each hinge **only** where the hinge and the swing side were read from the drawing | `building.openings[].door.hinges`, `openings[].evidence.swing_side` |
| `WINDOWS` | frame across the measured wall thickness + glazing axis | `building.openings`, `evidence.wall_thickness_px` |
| `OPENINGS_UNCLASSIFIED` | openings the classifier left untyped | `type == "opening"` |
| `ROOM_BOUNDARIES` | named rooms bounded by walls | room `boundary.method == "wall-region"` |
| `ROOM_BOUNDARIES_INFERRED` | named rooms with an inferred extent (open-plan zones, dimension estimates) | every other boundary method |
| `SPACE_BOUNDARIES` | structural spaces not already drawn as one wall-bounded room (unnamed spaces, open-plan spaces) | `building.spaces` |
| `ROOM_LABELS` | room name (+ area when it is in real units) at the label position | room `name`, `label_center`, `area` |
| `NOTES` | source, units, scale source, what was not applied, how many elements were not drawn | |

Every entity carries XDATA (application `FLOORPLAN_ENGINE`): `id=<model id>`, `type=<kind>` and the
confidence where the model has one. Furniture, fixtures and functional zones are not exported
(their detection is not reliable enough to hand to CAD as geometry; see FUNCTIONAL_ZONES.md).

## Units and coordinates

The model is in pixels of the analysed image (origin top-left, y down). The drawing uses

    X = x · s,   Y = (H − y) · s        H = image height

so the plan is not mirrored. `s` is millimetres per pixel **only** when the plan's printed room
dimensions calibrated it (`result["pixel_scale"]`); the drawing is then in millimetres
(`$INSUNITS = 4`). Otherwise `s = 1`, the drawing is unitless (`$INSUNITS = 0`) and one drawing unit
is one source pixel: scale it in CAD with one known dimension. The door-width estimate
(`scale_estimate`, doors taken as 82 cm) and a PDF's render DPI are written to `NOTES`, never
applied. Single-storey: one result is one drawing.

## Invalid geometry

Non-finite coordinates, zero-length segments and degenerate polygons are skipped and listed in the
export report (`ExportReport.skipped`, `NOTES`); a wall without a measured thickness keeps its
centreline only. Nothing is repaired.

## DWG

No Python library writes DWG under a permissive licence. `engine/export/dwg.py` converts the DXF
with a converter installed on the same computer (no upload):

| Backend | Install | Licence | Status |
|---|---|---|---|
| `libredwg` (default when found) | `brew install libredwg` (dxf2dwg / dwg2dxf 0.14) | GPL-3.0-or-later; run as a separate program, never linked or bundled | used and tested here; R2000 only |
| `oda` | ODA File Converter (opendesign.com), used through `ezdxf.addons.odafc` | free of charge, proprietary; its terms are accepted by whoever installs it | **not verified** (not installed in this environment) |

`FLOORPLAN_DWG_BACKEND=libredwg|oda|off` selects one; `FLOORPLAN_DXF2DWG=/path/to/dxf2dwg` points at a
LibreDWG build. The desktop app looks in `/opt/homebrew/bin` and `/usr/local/bin` too (apps started
from Finder do not inherit the shell's PATH).

Every DWG is checked before it is served: the file header must be `AC1015` (R2000), and the DWG read
back by the converter must hold the same layers and the same number of entities of each type per
layer as the DXF it came from; otherwise the request fails. That check uses the converter's own
reader. **It is not an independent DWG validation, and no CAD application (AutoCAD, BricsCAD,
DWG TrueView, …) has opened these files in this project yet.**

Known LibreDWG limits (measured on this project's exports):

- R2004 output cannot be read back completely by LibreDWG itself, so only R2000 is offered.
- The DXF's custom header notes (`FPE_*`) do not survive; units (`$INSUNITS`), layers, geometry
  (byte-identical line and polyline coordinates) and XDATA do.
- Reading the DWG back reports a duplicated block-end record (`ENDBLK`, handle 76) of the paper-space
  block; model-space geometry is unaffected. Whether the DWG or only the read-back has it is not
  known without an independent reader.

To verify DWG for production: open the R2000 files in DWG TrueView or the ODA viewer, or install
the ODA File Converter and set `FLOORPLAN_DWG_BACKEND=oda`.

## Coordinate frames (fixed with the exporter)

Everything in a result is in the original image's pixels. Two exceptions existed and are fixed:

1. **Walls of a deskewed plan.** `analyze_structure` analyses a rotated copy when the walls are
   skewed by ≥ 0.35° and maps candidates, spaces and masks back, but returned the wall bands of the
   rotated copy. The building model's walls were therefore off the walls (rotated 2.jpg / 8.png by
   3–5°: 4–23 % of wall samples on the wall mask, opening ends 13–37 px from the nearest wall; after
   the fix 100 % and 1 px). Bands are now mapped back like everything else; the working frame keeps
   its own (`structure.work.bands`).
2. **Door hinges and wall-run lengths of a plan analysed at a working scale.** `engine/rescale.py`
   did not rescale `hinges`, `solid_length_px`, `open_length_px` (7.png, 10.png, 15.png: all 97
   hinge points 3× off). They are now in its key list.

Tests: tests/test_coordinate_frames.py, tests/test_export_dxf.py, tests/test_export_dwg.py.
