# Migration notes

## What was here before

The project had **two unfinished, disconnected detection systems**:

- `engine/geometry/*` (Hough lines → pairing → wall reconstruction) —
  the only one actually wired into `pipeline.py`. It crashed on any real
  image (`LineDetector` unpacked `cv2.HoughLinesP` output one dimension
  short of what it needed) and its room detection
  (`rooms/segmentation.py`) tried every combination of 4 walls and
  called anything rectangular a "room," producing many false positives.
- `engine/structural/*` (~2000 lines, 9 files: region/band detection,
  segment extraction, centerlines, a wall network, a snapper, a graph
  builder) — a more promising, region-based rewrite that was **never
  imported by `pipeline.py`**. It was only exercised by 14 standalone
  `test_wall_*.py` scripts at the repo root, one per module.

`engine/rooms/space_segmentation.py` was a *third*, independent
room-detection approach (flood fill on raw merged lines), also run from
`pipeline.py` in parallel with the rectangle-combination approach, with
neither result feeding the other.

There was also no `requirements.txt`, no `pyproject.toml`, no `.git`,
relative imports that only worked when run from inside `engine/`, and a
full `.venv/` accidentally included in the shared zip.

## What's here now

One pipeline, three stages, each tested against the real
`test_floorplan.png` in this repo rather than assumed correct:

`engine/walls/mask.py` → `engine/walls/segments.py` →
`engine/walls/room_mask.py` + `engine/rooms/detect.py`

- Wall thickness is **measured**, not guessed: `WallSegmentExtractor`
  clusters row/column runs into bands, so a wall's thickness falls out
  of how many rows/columns the band spans. This also means it survives
  T-junctions and L-junctions, which broke the old connected-components
  approach in `structural/wall_geometry.py`.
- Room detection is flood fill on a mask with door/window gaps bridged,
  replacing the O(H²V²) brute-force rectangle search. The bridging
  threshold is derived from the wall thickness actually measured in the
  drawing, not a hardcoded pixel count.
- Most geometric thresholds are derived from measured wall thickness or
  image size, but resolution invariance is not yet fully verified:
  `mask.py` and `room_mask.py` still use fixed 5x5 morphology kernels,
  and `mask.py` retains a 50-pixel minimum component-area floor. Test
  resized versions of the same plan before claiming equivalent behavior.

## Known limitation, found while testing on the bundled image

`test_floorplan.png`'s exterior wall has one ~741px stretch drawn with
a much fainter/thinner line than the rest of the wall (visible if you
zoom into the top wall around x=448–1189 in the original image) rather
than a normal double-line window or a solid stroke. Any gap-bridging
kernel large enough to safely paper over that specific gap is also
large enough to fuse unrelated text and nearby interior walls together
elsewhere, which silently corrupts the whole mask - so `room_mask.py`
deliberately does **not** try to force-close it. `RoomMaskBuilder`
reports it (`last_unsealed_gap`) instead of hiding it, and
`pipeline.py` prints a warning when it happens. On this test image, the
practical effect is that Living Room and Kitchen (which are already
open-concept, no wall, in the source drawing) flood-fill out to
"outside" through that specific gap and aren't reported as separate
enclosed rooms.

If your real floor plans consistently have this kind of faint boundary
line, the fix is to detect it specifically (e.g. a lower secondary
threshold applied only to pixels that fall between two wall segments
that share the same centerline) rather than raising the general
bridging kernel.

## Files removed

- `engine/geometry/` (dead-end Hough pipeline, had the crash bug)
- `engine/structural/` (unfinished, never-wired experiment)
- `engine/rooms/segmentation.py`, `engine/rooms/space_segmentation.py`
  (superseded by `engine/rooms/detect.py`)
- `engine/preprocessing/` (grayscale/threshold/text-masking logic is
  now inline in `WallMaskBuilder`, which achieves the same text
  removal via thickness filtering rather than a separate
  connected-components pass)
- root-level `test_wall_*.py` scripts (each tested a module that no
  longer exists)
- the bundled `.venv/` (add dependencies via `requirements.txt` instead)
