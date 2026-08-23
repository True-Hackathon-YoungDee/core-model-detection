import numpy as np
import pytest

from fall_detection.geometry import convex_hull_2d, point_in_polygon, point_to_hull_signed_distance


def test_convex_hull_of_square_with_interior_point_returns_four_corners():
    points = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [2, 2]], dtype=float)
    hull = convex_hull_2d(points)
    assert len(hull) == 4
    hull_set = {tuple(p) for p in hull}
    assert hull_set == {(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)}


def test_convex_hull_of_single_point_returns_that_point():
    points = np.array([[1.0, 2.0]])
    hull = convex_hull_2d(points)
    assert len(hull) == 1
    assert tuple(hull[0]) == (1.0, 2.0)


def test_convex_hull_of_two_points_returns_both():
    points = np.array([[0.0, 0.0], [3.0, 3.0]])
    hull = convex_hull_2d(points)
    assert len(hull) == 2


def test_point_in_polygon_inside_square():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    assert point_in_polygon(np.array([2.0, 2.0]), square) is True


def test_point_in_polygon_outside_square():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    assert point_in_polygon(np.array([5.0, 2.0]), square) is False


def test_point_to_hull_signed_distance_inside_is_negative():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    distance = point_to_hull_signed_distance(np.array([2.0, 2.0]), square)
    assert distance == pytest.approx(-2.0)


def test_point_to_hull_signed_distance_outside_is_positive():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    distance = point_to_hull_signed_distance(np.array([5.0, 2.0]), square)
    assert distance == pytest.approx(1.0)
