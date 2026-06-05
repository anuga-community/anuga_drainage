"""Unit tests for the SWMM .inp [COORDINATES] parser."""
from anuga_drainage.inlet_initialization import read_inp_coordinates

INP = """\
[JUNCTIONS]
;;Name  Elev
J1      0.0

[COORDINATES]
;;Node           X-Coord            Y-Coord
;;-------------- ------------------ ------------------
Inlet_1          305698.510         6188004.630
Inlet_2          305703.390         6187999.000
Outfall_1        305736.680         6188026.650

[VERTICES]
;;Link  X  Y
C1      1.0  2.0
"""


def _write(tmp_path, text):
    p = tmp_path / "model.inp"
    p.write_text(text)
    return str(p)


def test_parses_only_coordinates_section(tmp_path):
    coords = read_inp_coordinates(_write(tmp_path, INP))
    # Three coordinate rows; the JUNCTIONS/VERTICES rows must be ignored.
    assert list(coords.index) == ["Inlet_1", "Inlet_2", "Outfall_1"]
    assert list(coords.columns) == ["X_Coord", "Y_Coord"]


def test_coordinate_values(tmp_path):
    coords = read_inp_coordinates(_write(tmp_path, INP))
    assert coords.loc["Inlet_1"].X_Coord == 305698.510
    assert coords.loc["Inlet_1"].Y_Coord == 6188004.630
    assert coords.loc["Outfall_1"].X_Coord == 305736.680


def test_empty_when_no_coordinates_section(tmp_path):
    coords = read_inp_coordinates(_write(tmp_path, "[JUNCTIONS]\nJ1 0.0\n"))
    assert len(coords) == 0
