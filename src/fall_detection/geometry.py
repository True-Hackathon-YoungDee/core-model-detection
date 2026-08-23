"""Pure numpy 2D geometry: convex hulls and point/polygon distance.

Base-of-support polygons are ~6 points, so a hand-rolled hull is cheaper than
a scipy dependency.
"""

from __future__ import annotations

import numpy as np


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. points: (N,2) -> CCW hull vertices (M,2), M<=N.

    Degenerate inputs (fewer than 3 distinct points, or all collinear) are
    returned as-is rather than raising.
    """
    unique_points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(unique_points) <= 2:
        return unique_points

    sorted_points = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in sorted_points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        return sorted_points
    return np.array(hull)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting test. polygon need not be convex; point on an edge is undefined."""
    x, y = point[0], point[1]
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


def point_to_hull_signed_distance(point: np.ndarray, hull: np.ndarray) -> float:
    """Distance from point to the nearest edge of hull.

    Positive = outside the hull (unstable), negative = inside.
    """
    point = np.asarray(point, dtype=float)
    hull = np.asarray(hull, dtype=float)
    n = len(hull)

    if n == 1:
        return float(np.linalg.norm(point - hull[0]))
    if n == 2:
        distance = _point_to_segment_distance(point, hull[0], hull[1])
        return distance

    min_distance = min(
        _point_to_segment_distance(point, hull[i], hull[(i + 1) % n]) for i in range(n)
    )
    return -min_distance if point_in_polygon(point, hull) else min_distance


def _point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, segment)) / length_sq
    t = min(max(t, 0.0), 1.0)
    projection = a + t * segment
    return float(np.linalg.norm(point - projection))
