"""Tests for the SWMM .inp parser and pipedream converter (no ANUGA/pyswmm)."""
import numpy as np
import pytest

from anuga_drainage.inp import (
    read_inp, inp_to_pipedream, InpNetwork, _shape_geometry,
)

_INP = """\
[TITLE]
Test network

[JUNCTIONS]
;;Name   Elev   MaxD   InitD   Sur   Apond
J1       10.0   2.0    0.5     0     0
J2        9.0   2.0    0       0     0

[OUTFALLS]
;;Name   Elev   Type   Stage   Gated
O1        8.0   FREE           NO

[CONDUITS]
;;Name   From   To    Len    Rough   InOff   OutOff   InitFlow   MaxFlow
C1       J1     J2    20.0   0.013   0       0        0          0
C2       J2     O1    10.0   0.013   0       0        0          0

[XSECTIONS]
;;Link   Shape       Geom1   Geom2   Geom3   Geom4   Barrels
C1       CIRCULAR    0.5     0       0       0       1
C2       RECT_OPEN   1.0     2.0     0       0       1

[COORDINATES]
;;Node   X      Y
J1       0.0    0.0
J2       20.0   0.0
O1       30.0   5.0
"""


@pytest.fixture
def inp_path(tmp_path):
    p = tmp_path / "net.inp"
    p.write_text(_INP)
    return str(p)


def test_read_inp_sections(inp_path):
    inp = read_inp(inp_path)
    assert list(inp.junctions["name"]) == ["J1", "J2"]
    assert list(inp.outfalls["name"]) == ["O1"]
    assert list(inp.conduits["name"]) == ["C1", "C2"]
    assert list(inp.xsections["link"]) == ["C1", "C2"]
    assert len(inp.coordinates) == 3
    # numeric coercion
    assert inp.junctions.loc[0, "elevation"] == 10.0
    assert inp.conduits.loc[0, "length"] == 20.0


def test_inp_to_pipedream_superjunctions(inp_path):
    sj, sl = inp_to_pipedream(read_inp(inp_path), manhole_area=1.5, h_0=1e-4)
    assert list(sj["name"]) == ["J1", "J2", "O1"]          # junctions then outfalls
    assert list(sj["id"]) == [0, 1, 2]
    assert list(sj["z_inv"]) == [10.0, 9.0, 8.0]
    assert list(sj["bc"]) == [False, False, True]          # outfall is a boundary
    assert (sj["c"] == 1.5).all()                          # manhole_area -> storage c
    assert sj.loc[0, "h_0"] == 0.5                         # junction init depth honoured
    assert sj.loc[1, "h_0"] == pytest.approx(1e-4)         # default where none given
    assert list(sj.loc[2, ["map_x", "map_y"]]) == [30.0, 5.0]


def test_inp_to_pipedream_superlinks(inp_path):
    sj, sl = inp_to_pipedream(read_inp(inp_path), pit_area=1.2)
    assert list(sl["name"]) == ["C1", "C2"]
    assert list(sl["sj_0"]) == [0, 1]                      # J1->J2, J2->O1 by id
    assert list(sl["sj_1"]) == [1, 2]
    assert list(sl["dx"]) == [20.0, 10.0]
    assert list(sl["shape"]) == ["circular", "rect_open"]
    assert sl.loc[0, "g1"] == 0.5                          # circular diameter
    assert list(sl.loc[1, ["g1", "g2"]]) == [1.0, 2.0]     # rect height, width
    assert (sl["A_s"] == 1.2).all()                        # pit_area


def test_shape_geometry_conversions():
    # direct (height/width) shapes
    assert _shape_geometry("CIRCULAR", 0.5, 0, 0, 0) == (0.5, 0.0, 0.0, 0.0)
    assert _shape_geometry("RECT_OPEN", 1.0, 2.0, 0, 0) == (1.0, 2.0, 0.0, 0.0)
    assert _shape_geometry("HORIZ_ELLIPSE", 1.2, 1.8, 0, 0) == (1.2, 1.8, 0.0, 0.0)
    # triangular: SWMM top width 4 at height 2 -> pipedream slope m = 4/(2*2) = 1
    assert _shape_geometry("TRIANGULAR", 2.0, 4.0, 0, 0) == (2.0, 1.0, 0.0, 0.0)
    # trapezoidal: two SWMM bank slopes -> single mean slope
    assert _shape_geometry("TRAPEZOIDAL", 2.0, 3.0, 1.0, 3.0) == (2.0, 3.0, 2.0, 0.0)
    # force_main: diameter kept; SWMM Geom2 (roughness) is dropped for the slot default
    g1, g2, g3, g4 = _shape_geometry("FORCE_MAIN", 0.6, 130.0, 0, 0)
    assert (g1, g3, g4) == (0.6, 0.0, 0.0) and g2 == pytest.approx(0.01)


def test_triangular_area_round_trips_through_pipedream():
    # SWMM triangle height=2, top width=4 -> full area = 0.5*4*2 = 4 m^2.
    g = pytest.importorskip("pipedream_solver.geometry")
    import numpy as np
    g1, m, _, _ = _shape_geometry("TRIANGULAR", 2.0, 4.0, 0, 0)
    A = g.Triangular().A_ik(np.array([2.0]), np.array([2.0]),
                            np.array([g1]), np.array([m]))
    assert float(A[0]) == pytest.approx(4.0)


def test_irregular_shape_not_yet_supported(inp_path):
    inp = read_inp(inp_path)
    inp.xsections.loc[0, "shape"] = "IRREGULAR"
    with pytest.raises(NotImplementedError, match="TRANSECTS"):
        inp_to_pipedream(inp)


def test_unsupported_shape_raises(inp_path):
    inp = read_inp(inp_path)
    inp.xsections.loc[0, "shape"] = "EGG"                  # no pipedream equivalent
    with pytest.raises(ValueError, match="no pipedream equivalent"):
        inp_to_pipedream(inp)


def test_conduit_without_xsection_raises(inp_path):
    inp = read_inp(inp_path)
    inp.xsections = inp.xsections.iloc[0:0]                # drop all xsections
    with pytest.raises(ValueError, match="no \\[XSECTIONS\\] entry"):
        inp_to_pipedream(inp)
