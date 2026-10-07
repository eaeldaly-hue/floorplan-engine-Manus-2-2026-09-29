from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from engine.geometry_v2.groups import WallGroupBuilder
from engine.geometry_v2.segments import WallSegment


@dataclass
class GraphNode:
    id: int
    x: float
    y: float


@dataclass
class GraphEdge:
    id: int
    a: int
    b: int
    virtual: bool = False


@dataclass
class RoomFace:
    id: int
    node_ids: list[int]
    polygon: list[tuple[float, float]]
    area: float
    bbox: tuple[int, int, int, int]
    rectangularity: float
    perimeter: float


@dataclass
class ReconstructionResult:
    wall_segments: int
    wall_groups: int
    virtual_bridges: int
    graph_nodes: int
    graph_edges: int
    raw_faces: int
    valid_rooms: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    rooms: list[RoomFace]


class _NodeRegistry:
    def __init__(self, tolerance: float = 10.0):
        self.tolerance = float(tolerance)
        self.nodes: list[GraphNode] = []

    def add(self, point: tuple[float, float]) -> int:
        x, y = float(point[0]), float(point[1])

        best_id = None
        best_dist = float("inf")

        for node in self.nodes:
            d = math.hypot(node.x - x, node.y - y)
            if d <= self.tolerance and d < best_dist:
                best_id = node.id
                best_dist = d

        if best_id is not None:
            node = self.nodes[best_id]
            # Slowly stabilize the representative point.
            node.x = (node.x + x) / 2.0
            node.y = (node.y + y) / 2.0
            return best_id

        node_id = len(self.nodes)
        self.nodes.append(GraphNode(node_id, x, y))
        return node_id


class RoomReconstructionEngine:
    """
    Geometry-first room reconstruction.

    Pipeline:
        wall segments
            -> wall groups
            -> virtual bridges across group gaps
            -> intersection splitting
            -> planar graph
            -> bounded faces
            -> room candidates

    This engine intentionally does NOT modify FloorPlanAnalyzer.
    """

    def __init__(
        self,
        snap_tolerance: float = 10.0,
        min_room_area: float = 7000.0,
        max_room_area_ratio: float = 0.85,
        min_rectangularity: float = 0.15,
    ):
        self.snap_tolerance = float(snap_tolerance)
        self.min_room_area = float(min_room_area)
        self.max_room_area_ratio = float(max_room_area_ratio)
        self.min_rectangularity = float(min_rectangularity)

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _point_on_segment(
        p: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
        tol: float = 1e-6,
    ) -> bool:
        px, py = p
        ax, ay = a
        bx, by = b

        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        if abs(cross) > tol:
            return False

        return (
            min(ax, bx) - tol <= px <= max(ax, bx) + tol
            and min(ay, by) - tol <= py <= max(ay, by) + tol
        )

    @staticmethod
    def _hv_intersection(
        h: tuple[float, float, float, float],
        v: tuple[float, float, float, float],
        tol: float,
    ):
        hx1, hy1, hx2, hy2 = h
        vx1, vy1, vx2, vy2 = v

        hy = (hy1 + hy2) / 2.0
        vx = (vx1 + vx2) / 2.0

        if (
            min(hx1, hx2) - tol <= vx <= max(hx1, hx2) + tol
            and min(vy1, vy2) - tol <= hy <= max(vy1, vy2) + tol
        ):
            return vx, hy

        return None

    @staticmethod
    def _orientation(segment):
        value = getattr(segment, "orientation", None)

        if hasattr(value, "value"):
            value = value.value

        value = str(value).lower()

        if "horizontal" in value:
            return "horizontal"
        if "vertical" in value:
            return "vertical"

        return value

    @staticmethod
    def _segment_points(segment):
        return (
            (float(segment.start[0]), float(segment.start[1])),
            (float(segment.end[0]), float(segment.end[1])),
        )

    @staticmethod
    def _length(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    # ------------------------------------------------------------------
    # Topology construction
    # ------------------------------------------------------------------

    def _build_topology_segments(self, segments):
        """
        Keep original wall segments and add virtual bridges for every
        WallGroup gap.

        Bridges exist only for topology reconstruction.
        """

        # WallSegmentExtractor currently returns dictionaries,
        # while WallGroupBuilder expects WallSegment objects.
        normalized_segments = []

        for segment in segments:
            if isinstance(segment, WallSegment):
                normalized_segments.append(segment)
                continue

            start = tuple(float(v) for v in segment["start"])
            end = tuple(float(v) for v in segment["end"])

            orientation = segment.get("orientation", "horizontal")

            # Reuse the existing enum when available.
            try:
                from engine.geometry_v2.orientation import Orientation

                if hasattr(Orientation, str(orientation).upper()):
                    orientation_value = getattr(
                        Orientation,
                        str(orientation).upper(),
                    )
                else:
                    orientation_value = orientation
            except Exception:
                orientation_value = orientation

            normalized_segments.append(
                WallSegment(
                    start=start,
                    end=end,
                    orientation=orientation_value,
                    length=float(
                        segment.get(
                            "length",
                            self._length(start, end),
                        )
                    ),
                    thickness=float(segment.get("thickness", 1.0)),
                    confidence=float(segment.get("confidence", 1.0)),
                )
            )

        builder = WallGroupBuilder()
        groups = builder.build(normalized_segments)

        topology = []
        virtual_count = 0

        for segment in normalized_segments:
            topology.append(
                {
                    "start": self._segment_points(segment)[0],
                    "end": self._segment_points(segment)[1],
                    "virtual": False,
                }
            )

        for group in groups:
            for gap in group.gaps:
                start = (float(gap.start[0]), float(gap.start[1]))
                end = (float(gap.end[0]), float(gap.end[1]))

                if self._length(start, end) < 2:
                    continue

                topology.append(
                    {
                        "start": start,
                        "end": end,
                        "virtual": True,
                    }
                )
                virtual_count += 1

        return groups, topology, virtual_count

    def _split_segments(self, topology):
        """
        Split every segment at:
          - its endpoints
          - H/V intersections
          - endpoints of overlapping collinear segments

        Returns small atomic segments.
        """

        split_points = [
            [seg["start"], seg["end"]]
            for seg in topology
        ]

        for i, a in enumerate(topology):
            a0, a1 = a["start"], a["end"]
            ao = self._orientation_from_points(a0, a1)

            for j in range(i + 1, len(topology)):
                b = topology[j]
                b0, b1 = b["start"], b["end"]
                bo = self._orientation_from_points(b0, b1)

                if ao == "horizontal" and bo == "vertical":
                    p = self._hv_intersection(
                        (a0[0], a0[1], a1[0], a1[1]),
                        (b0[0], b0[1], b1[0], b1[1]),
                        self.snap_tolerance,
                    )
                    if p is not None:
                        split_points[i].append(p)
                        split_points[j].append(p)

                elif ao == "vertical" and bo == "horizontal":
                    p = self._hv_intersection(
                        (b0[0], b0[1], b1[0], b1[1]),
                        (a0[0], a0[1], a1[0], a1[1]),
                        self.snap_tolerance,
                    )
                    if p is not None:
                        split_points[i].append(p)
                        split_points[j].append(p)

                elif ao == bo and ao in {"horizontal", "vertical"}:
                    # Collinear overlap: endpoints become split points.
                    for p in (a0, a1):
                        if self._point_on_segment(
                            p,
                            b0,
                            b1,
                            self.snap_tolerance,
                        ):
                            split_points[j].append(p)

                    for p in (b0, b1):
                        if self._point_on_segment(
                            p,
                            a0,
                            a1,
                            self.snap_tolerance,
                        ):
                            split_points[i].append(p)

        atomic = []

        for seg, points in zip(topology, split_points):
            a, b = seg["start"], seg["end"]
            orientation = self._orientation_from_points(a, b)

            unique = []

            for p in points:
                if not any(
                    math.hypot(p[0] - q[0], p[1] - q[1])
                    <= self.snap_tolerance / 2
                    for q in unique
                ):
                    unique.append(p)

            if orientation == "horizontal":
                unique.sort(key=lambda p: p[0])
            else:
                unique.sort(key=lambda p: p[1])

            for p1, p2 in zip(unique, unique[1:]):
                if self._length(p1, p2) < 2:
                    continue

                atomic.append(
                    {
                        "start": p1,
                        "end": p2,
                        "virtual": bool(seg["virtual"]),
                    }
                )

        return atomic

    @staticmethod
    def _orientation_from_points(a, b):
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])

        if dx >= dy:
            return "horizontal"
        return "vertical"

    def _build_graph(self, atomic):
        registry = _NodeRegistry(self.snap_tolerance)

        edge_map = {}
        edges = []

        for seg in atomic:
            a = registry.add(seg["start"])
            b = registry.add(seg["end"])

            if a == b:
                continue

            key = (min(a, b), max(a, b))

            if key in edge_map:
                # Prefer a real wall edge over a virtual duplicate.
                existing_id = edge_map[key]

                if edges[existing_id].virtual and not seg["virtual"]:
                    edges[existing_id].virtual = False

                continue

            edge_id = len(edges)

            edges.append(
                GraphEdge(
                    id=edge_id,
                    a=a,
                    b=b,
                    virtual=bool(seg["virtual"]),
                )
            )

            edge_map[key] = edge_id

        return registry.nodes, edges

    # ------------------------------------------------------------------
    # Planar faces
    # ------------------------------------------------------------------

    def _extract_faces(self, nodes, edges):
        adjacency = {node.id: [] for node in nodes}

        for edge in edges:
            a = nodes[edge.a]
            b = nodes[edge.b]

            angle_ab = math.atan2(b.y - a.y, b.x - a.x)
            angle_ba = math.atan2(a.y - b.y, a.x - b.x)

            adjacency[edge.a].append((edge.id, edge.b, angle_ab))
            adjacency[edge.b].append((edge.id, edge.a, angle_ba))

        for node_id in adjacency:
            adjacency[node_id].sort(key=lambda item: item[2])

        # Directed half-edge IDs:
        # edge_id*2     = a -> b
        # edge_id*2 + 1 = b -> a
        visited = set()
        cycles = []

        def halfedge_data(hid):
            edge_id = hid // 2
            reverse = hid % 2 == 1
            edge = edges[edge_id]

            if not reverse:
                return edge.a, edge.b
            return edge.b, edge.a

        def next_halfedge(hid):
            edge_id = hid // 2
            reverse = hid % 2 == 1

            edge = edges[edge_id]

            if not reverse:
                current = edge.a
                target = edge.b
            else:
                current = edge.b
                target = edge.a

            outgoing = adjacency[target]

            # Locate the reverse direction target -> current.
            reverse_index = None

            for idx, (_, neighbor, _) in enumerate(outgoing):
                if neighbor == current:
                    reverse_index = idx
                    break

            if reverse_index is None:
                return None

            # Clockwise predecessor gives the face on the left.
            return outgoing[(reverse_index - 1) % len(outgoing)][0] * 2 + (
                1
                if outgoing[(reverse_index - 1) % len(outgoing)][1]
                == edges[outgoing[(reverse_index - 1) % len(outgoing)][0]].a
                else 0
            )

        # The helper above is intentionally replaced by a simpler directed
        # lookup to avoid ambiguity around half-edge orientation.
        outgoing_halfedges = {node.id: [] for node in nodes}

        for edge in edges:
            outgoing_halfedges[edge.a].append(
                (edge.b, edge.id * 2)
            )
            outgoing_halfedges[edge.b].append(
                (edge.a, edge.id * 2 + 1)
            )

        for node_id in outgoing_halfedges:
            outgoing_halfedges[node_id].sort(
                key=lambda item: math.atan2(
                    nodes[item[0]].y - nodes[node_id].y,
                    nodes[item[0]].x - nodes[node_id].x,
                )
            )

        next_map = {}

        for node_id, outgoing in outgoing_halfedges.items():
            for idx, (neighbor, hid) in enumerate(outgoing):
                # At the destination, find the reverse half-edge.
                dest_outgoing = outgoing_halfedges[neighbor]

                reverse_idx = None
                for j, (dest_neighbor, dest_hid) in enumerate(dest_outgoing):
                    if dest_neighbor == node_id:
                        reverse_idx = j
                        break

                if reverse_idx is None:
                    continue

                # Clockwise predecessor.
                next_map[hid] = dest_outgoing[
                    (reverse_idx - 1) % len(dest_outgoing)
                ][1]

        for start_hid in sorted(next_map):
            if start_hid in visited:
                continue

            current = start_hid
            cycle_nodes = []
            local_seen = set()

            while current is not None and current not in local_seen:
                local_seen.add(current)
                visited.add(current)

                src, _ = halfedge_data(current)
                cycle_nodes.append(src)

                current = next_map.get(current)

                if current == start_hid:
                    break

            if current != start_hid or len(cycle_nodes) < 3:
                continue

            polygon = [
                (nodes[nid].x, nodes[nid].y)
                for nid in cycle_nodes
            ]

            area_signed = self._polygon_signed_area(polygon)

            if abs(area_signed) < 100:
                continue

            cycles.append(
                {
                    "node_ids": cycle_nodes,
                    "polygon": polygon,
                    "signed_area": area_signed,
                }
            )

        return cycles

    @staticmethod
    def _polygon_signed_area(polygon):
        area = 0.0

        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            area += x1 * y2 - x2 * y1

        return area / 2.0

    @staticmethod
    def _polygon_perimeter(polygon):
        total = 0.0

        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            total += math.hypot(
                b[0] - a[0],
                b[1] - a[1],
            )

        return total

    @staticmethod
    def _bbox(polygon):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        x1 = int(round(min(xs)))
        y1 = int(round(min(ys)))
        x2 = int(round(max(xs)))
        y2 = int(round(max(ys)))

        return (
            x1,
            y1,
            max(0, x2 - x1),
            max(0, y2 - y1),
        )

    def _validate_faces(self, cycles, image_shape):
        height, width = image_shape[:2]
        image_area = float(width * height)

        # Remove duplicate cycles by normalized node set.
        unique = {}
        for cycle in cycles:
            key = tuple(sorted(cycle["node_ids"]))

            existing = unique.get(key)

            if existing is None:
                unique[key] = cycle
            elif abs(cycle["signed_area"]) < abs(existing["signed_area"]):
                unique[key] = cycle

        candidates = []

        for cycle in unique.values():
            polygon = cycle["polygon"]
            area = abs(cycle["signed_area"])

            if area < self.min_room_area:
                continue

            if area > image_area * self.max_room_area_ratio:
                continue

            bbox = self._bbox(polygon)
            bbox_area = max(1, bbox[2] * bbox[3])

            rectangularity = area / bbox_area

            if rectangularity < self.min_rectangularity:
                continue

            candidates.append(
                RoomFace(
                    id=len(candidates) + 1,
                    node_ids=list(cycle["node_ids"]),
                    polygon=polygon,
                    area=area,
                    bbox=bbox,
                    rectangularity=rectangularity,
                    perimeter=self._polygon_perimeter(polygon),
                )
            )

        candidates.sort(key=lambda room: room.area, reverse=True)

        # Re-number after sorting.
        for idx, room in enumerate(candidates, start=1):
            room.id = idx

        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(self, segments, image_shape):
        groups, topology, virtual_count = self._build_topology_segments(
            segments
        )

        atomic = self._split_segments(topology)

        nodes, edges = self._build_graph(atomic)

        cycles = self._extract_faces(nodes, edges)

        rooms = self._validate_faces(
            cycles,
            image_shape,
        )

        return ReconstructionResult(
            wall_segments=len(segments),
            wall_groups=len(groups),
            virtual_bridges=virtual_count,
            graph_nodes=len(nodes),
            graph_edges=len(edges),
            raw_faces=len(cycles),
            valid_rooms=len(rooms),
            nodes=nodes,
            edges=edges,
            rooms=rooms,
        )


def draw_reconstruction(
    image,
    result: ReconstructionResult,
    output_path: str | Path,
):
    """
    Debug visualization:
      - original image
      - wall graph
      - virtual bridges
      - reconstructed room faces
      - node points
    """

    canvas = image.copy()

    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    overlay = canvas.copy()

    # Room faces.
    for room in result.rooms:
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in room.polygon],
            dtype=np.int32,
        )

        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], (70, 180, 255))
            cv2.polylines(
                canvas,
                [pts],
                True,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

            cx = int(round(sum(x for x, _ in room.polygon) / len(room.polygon)))
            cy = int(round(sum(y for _, y in room.polygon) / len(room.polygon)))

            cv2.putText(
                canvas,
                f"R{room.id}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                3,
                cv2.LINE_AA,
            )

    canvas = cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0)

    # Graph edges.
    for edge in result.edges:
        a = result.nodes[edge.a]
        b = result.nodes[edge.b]

        p1 = (int(round(a.x)), int(round(a.y)))
        p2 = (int(round(b.x)), int(round(b.y)))

        thickness = 2 if not edge.virtual else 4

        cv2.line(
            canvas,
            p1,
            p2,
            (0, 180, 255) if edge.virtual else (255, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # Graph nodes.
    for node in result.nodes:
        cv2.circle(
            canvas,
            (int(round(node.x)), int(round(node.y))),
            4,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

    return output_path
