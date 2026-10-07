"""Plan Model in the production analyzer (transition period).

The legacy structure stays authoritative wherever it works. The Plan Model runs:

  off       never
  fallback  (default) only when the legacy structure found no wall or no enclosed space - the
            plans the legacy engine cannot read (hollow, hatched or thin-line walls, CAD sheets).
            Plans the legacy engine reads are untouched: same output, same time.
  shadow    on every plan, for comparison (costs time; diagnostics and benchmarks)

Its result is added to the response as `plan_model`. In fallback, when it finds spaces, they also
replace the (empty) legacy spaces, so rooms, unlabeled spaces and overlays are built from them.
Set with FLOORPLAN_PLAN_MODEL.
"""

from __future__ import annotations

import os

import numpy as np

MODES = ("off", "fallback", "shadow")


def plan_model_mode() -> str:
    mode = os.environ.get("FLOORPLAN_PLAN_MODEL", "fallback").strip().lower()
    return mode if mode in MODES else "fallback"


def legacy_failed(structure) -> bool:
    """The legacy structure produced nothing to build rooms on."""
    mask = getattr(structure, "wall_mask", None)
    no_walls = mask is None or not np.any(mask)
    return no_walls or not getattr(structure, "spaces", None)


def _pt(p) -> list:
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def payload(model, role: str) -> dict:
    """JSON-safe, compact view of a PlanModel."""
    return {
        "engine": "plan-model",
        "role": role,
        "status": model.status,
        "summary": {"walls": len(model.walls), "openings": len(model.openings), "spaces": len(model.spaces)},
        "wall_classes_px": [round(float(c), 1) for c in model.wall_classes],
        "walls": [{"id": w.id, "p0": _pt(w.p0), "p1": _pt(w.p1), "thickness_px": round(float(w.thickness), 1),
                   "style": w.style, "confidence": round(float(w.confidence), 2),
                   "support": None if w.support is None else round(float(w.support), 2),
                   "thickness_estimated": bool(w.thickness_estimated)} for w in model.walls],
        "openings": [{"id": o.id, "p0": _pt(o.p0), "p1": _pt(o.p1), "kind": o.kind, "wall_ids": list(o.wall_ids),
                      "width_px": round(float(o.width), 1), "confidence": round(float(o.confidence), 2),
                      "connects": list(o.connects), "evidence": list(o.provenance)} for o in model.openings],
        "spaces": [{"id": s.id, "polygon": [[int(x), int(y)] for x, y in s.polygon], "area_px": int(s.area_px),
                    "centroid": _pt(s.centroid), "labels": list(s.labels), "confidence": round(float(s.confidence), 2)}
                   for s in model.spaces],
        "adjacency": [list(a) for a in model.adjacency],
        "warnings": list(model.warnings),
    }


def as_structure(structure, model, keep_openings: bool = True):
    """The legacy StructureResult with its walls and spaces replaced by the Plan Model's, in the
    legacy format: the analyzer's label association, room records, overlays and topology then
    work unchanged. Openings (candidates) stay the legacy engine's."""
    import dataclasses

    from engine.structure import wall_adjacency

    # Paper outside every space stays 0, not EXTERIOR: where an opening was not closed, a room
    # leaks into the outside, and its label must not be discarded as text outside the building.
    labels = model._labels.astype(np.int32).copy()
    wall_mask = model._wall_mask
    spaces = []
    for k, sp in enumerate(model.spaces, 1):
        xs = [p[0] for p in sp.polygon]
        ys = [p[1] for p in sp.polygon]
        spaces.append({
            "id": f"space_{k}",
            "bbox": {"x": int(min(xs)), "y": int(min(ys)), "width": int(max(xs) - min(xs) + 1), "height": int(max(ys) - min(ys) + 1)},
            "area_pixels": int(sp.area_px),
            "center": (int(round(sp.centroid[0])), int(round(sp.centroid[1]))),
            "polygon": [(int(x), int(y)) for x, y in sp.polygon],
            "source": "plan-model",
        })
    thickness = float(np.median([w.thickness for w in model.walls])) if model.walls else structure.wall_thickness
    extra = {}
    if not keep_openings:
        # the legacy gaps refer to the legacy spaces it found: not valid for these spaces
        extra = {"candidates": [], "cracks": [], "rejected_candidates": [], "work": None, "to_original": None}
    return dataclasses.replace(structure, wall_mask=wall_mask, space_labels=labels, spaces=spaces,
                               wall_thickness=thickness,
                               wall_adjacency=wall_adjacency(labels, spaces, max(w.thickness for w in model.walls)),
                               **extra)


def run(image, room_lines, structure, evidence_image=None, prefer: bool = False):
    """(plan_model payload or None, warnings, structure) for the analyzer, per the configured mode.

    In fallback, when the Plan Model finds spaces, the returned structure carries them (see
    `as_structure`), so rooms and overlays are built from them; otherwise the legacy structure is
    returned unchanged. `prefer` (structural-layer input): the Plan Model is used whenever it finds
    spaces, even if the legacy structure found some; `evidence_image` is the full drawing whose
    symbols explain the openings (engine.plan.reconstruct)."""
    mode = plan_model_mode()
    if mode == "off" and not prefer:
        return None, [], structure
    failed = legacy_failed(structure) or prefer
    if mode == "fallback" and not failed:
        return None, [], structure
    from .reconstruct import reconstruct

    labels = [(line.text, line.center) for line in room_lines]
    try:
        model = reconstruct(image, room_labels=labels, evidence_image=evidence_image)
    except Exception as exc:                       # never break the analysis
        return None, [f"Plan Model reconstruction failed: {type(exc).__name__}"], structure
    out = payload(model, "fallback" if failed else "shadow")
    notes = []
    if failed and model.spaces:
        structure = as_structure(structure, model, keep_openings=not (prefer and not legacy_failed(structure)))
        why = ("The drawing's structural layer was reconstructed by" if prefer else
               "The standard wall detection found no enclosed space; room regions come from")
        notes.append(
            f"{why} the experimental Plan Model ({len(model.walls)} walls, {len(model.openings)} openings, "
            f"{len(model.spaces)} spaces). Review them against the drawing.")
    elif failed:
        notes.append("The standard wall detection found no enclosed space, and the experimental Plan Model could "
                     "not reconstruct one either.")
    return out, notes, structure
