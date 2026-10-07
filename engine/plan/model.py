"""The Architectural Plan Model: what building the drawing represents.

Walls are structural elements (centerline + thickness), not pixels; openings are intervals on
walls; spaces are the regions enclosed by the wall network; topology says which spaces connect
through which openings. Every element keeps where it came from (provenance), how sure the
engine is (confidence) and, for walls, how well the drawing confirms it (ink support from the
re-drawing check). Uncertain or unsupported results are reported, never silently emitted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

Point = tuple[float, float]


@dataclass
class Stroke:
    """A drawn straight line of the plan with its width (px). One wall may be one stroke
    (filled band, thin wall) or two (hollow wall)."""

    p0: Point
    p1: Point
    width: float
    darkness: float = 0.0          # mean ink coverage along the stroke (0..1)
    source: str = "raster"          # raster | vector
    roles: dict = field(default_factory=dict)   # role -> score (negative evidence)

    @property
    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    @property
    def angle(self) -> float:
        """Undirected angle in degrees, [0, 180)."""
        return math.degrees(math.atan2(self.p1[1] - self.p0[1], self.p1[0] - self.p0[0])) % 180.0

    @property
    def direction(self) -> tuple[float, float]:
        L = self.length or 1.0
        return ((self.p1[0] - self.p0[0]) / L, (self.p1[1] - self.p0[1]) / L)


@dataclass
class Wall:
    id: str
    p0: Point                       # centerline
    p1: Point
    thickness: float                # px; structural thickness
    style: str                      # filled | hollow | thin | hatched | vector
    confidence: float = 0.0
    support: float | None = None    # ink support of the re-drawing (0..1)
    exterior: bool = False
    thickness_estimated: bool = False   # thin walls: drawn as one line, thickness assumed
    provenance: list = field(default_factory=list)
    strokes: list = field(default_factory=list)  # indices of the strokes it was built from

    @property
    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.p1[1] - self.p0[1], self.p1[0] - self.p0[0])) % 180.0


@dataclass
class Opening:
    id: str
    p0: Point                       # on the wall axis
    p1: Point
    kind: str                       # door | window | passage | opening
    wall_ids: tuple = ()
    thickness: float = 0.0
    confidence: float = 0.0
    connects: tuple = ()            # (space id | "exterior", space id | "exterior")
    provenance: list = field(default_factory=list)

    @property
    def width(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])


@dataclass
class Space:
    id: str
    polygon: list                   # [(x, y), ...] clear floor outline
    area_px: int
    centroid: Point
    kind: str = "room"              # room | circulation | outdoor | uncertain
    confidence: float = 0.0
    labels: list = field(default_factory=list)       # room labels found inside
    wall_ids: list = field(default_factory=list)      # bounding walls
    provenance: list = field(default_factory=list)
    issues: list = field(default_factory=list)


@dataclass
class PlanModel:
    width: int
    height: int
    walls: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    spaces: list = field(default_factory=list)
    adjacency: list = field(default_factory=list)     # [(space a, space b | "exterior", opening id)]
    annotations: dict = field(default_factory=dict)   # role -> count of strokes
    wall_classes: list = field(default_factory=list)  # dominant thicknesses (px)
    status: str = "ok"              # ok | partial | no_structure
    warnings: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = {"walls": len(self.walls), "openings": len(self.openings), "spaces": len(self.spaces),
                        "status": self.status}
        return d
