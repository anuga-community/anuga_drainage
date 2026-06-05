"""Tests for the weir/orifice coupling flux calculate_Q.

calculate_Q pulls gravity from `anuga.g`, so these tests are skipped when
ANUGA is not importable (e.g. a lightweight CI without the conda env). They
run in the ANUGA conda environment.
"""
import numpy as np
import pytest

pytest.importorskip("anuga")

from anuga import g  # noqa: E402
from anuga_drainage import calculate_Q  # noqa: E402

CW = 0.67
CO = 0.67


def test_weir_branch_when_pipe_head_below_bed():
    # head1D < bed2D  ->  free weir inflow (positive: surface -> pipe).
    head1D = np.array([-1.0])
    depth2D = np.array([1.0])
    bed2D = np.array([0.0])
    length_weir = np.array([2.0])
    area_manhole = np.array([1.0])

    Q = calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole, cw=CW, co=CO)

    expected = CW * length_weir * depth2D * np.sqrt(2 * g * depth2D)
    assert Q[0] == pytest.approx(expected[0], rel=1e-12)
    assert Q[0] > 0


def test_orifice_branch_when_pipe_head_between_bed_and_surface():
    # bed2D <= head1D < depth2D + bed2D  ->  orifice inflow (positive).
    head1D = np.array([0.5])
    depth2D = np.array([1.0])
    bed2D = np.array([0.0])
    length_weir = np.array([2.0])
    area_manhole = np.array([1.0])

    Q = calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole, cw=CW, co=CO)

    expected = CO * area_manhole * np.sqrt(2 * g * (depth2D + bed2D - head1D))
    assert Q[0] == pytest.approx(expected[0], rel=1e-12)
    assert Q[0] > 0


def test_surcharge_branch_is_negative_when_pipe_head_above_surface():
    # head1D > depth2D + bed2D  ->  surcharge back onto surface (negative).
    head1D = np.array([5.0])
    depth2D = np.array([1.0])
    bed2D = np.array([0.0])
    length_weir = np.array([2.0])
    area_manhole = np.array([1.0])

    Q = calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole, cw=CW, co=CO)

    expected = -CO * area_manhole * np.sqrt(2 * g * (head1D - depth2D - bed2D))
    assert Q[0] == pytest.approx(expected[0], rel=1e-12)
    assert Q[0] < 0


def test_vectorised_over_multiple_inlets():
    # One inlet per branch, in a single call.
    head1D = np.array([-1.0, 0.5, 5.0])
    depth2D = np.array([1.0, 1.0, 1.0])
    bed2D = np.array([0.0, 0.0, 0.0])
    length_weir = np.array([2.0, 2.0, 2.0])
    area_manhole = np.array([1.0, 1.0, 1.0])

    Q = calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole)

    assert Q.shape == (3,)
    assert Q[0] > 0 and Q[1] > 0 and Q[2] < 0
