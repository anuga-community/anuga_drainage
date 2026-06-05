"""Unit tests for the pure geometry helper n_sided_inlet.

These need no ANUGA/SWMM install (the heavy imports in inlet_initialization
are deferred into initialize_inlets).
"""
import math

import pytest

from anuga_drainage.inlet_initialization import n_sided_inlet


def _polygon_area(vertices):
    """Shoelace area of a polygon given as a list of [x, y]."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _centroid(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


@pytest.mark.parametrize("n_sides", [3, 4, 6, 8])
def test_polygon_has_requested_area(n_sides):
    area = 2.5
    center = (10.0, -3.0)
    vertices, side_length = n_sided_inlet(n_sides, area, center, rotation=0.0)

    assert len(vertices) == n_sides
    # The regular polygon should enclose exactly the requested area.
    assert _polygon_area(vertices) == pytest.approx(area, rel=1e-9)
    assert side_length > 0


@pytest.mark.parametrize("n_sides", [3, 4, 6, 8])
def test_polygon_centred_on_inlet_coordinate(n_sides):
    center = (305_700.0, 6_188_000.0)
    vertices, _ = n_sided_inlet(n_sides, 1.0, center, rotation=0.0)
    cx, cy = _centroid(vertices)
    assert cx == pytest.approx(center[0], abs=1e-6)
    assert cy == pytest.approx(center[1], abs=1e-6)


def test_square_side_length():
    # For a square (n=4) of area A the side length is sqrt(A).
    area = 9.0
    _, side_length = n_sided_inlet(4, area, (0.0, 0.0), rotation=0.0)
    assert side_length == pytest.approx(math.sqrt(area), rel=1e-9)


def test_rejects_degenerate_polygon():
    with pytest.raises(RuntimeError):
        n_sided_inlet(2, 1.0, (0.0, 0.0), rotation=0.0)
