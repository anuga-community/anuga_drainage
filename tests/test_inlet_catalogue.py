"""Tests for the inlet asset catalogue (InletSpec / INLET_LIBRARY / loader).

Pure data + helpers, so this needs neither ANUGA nor SWMM.
"""
import pytest

from anuga_drainage import InletSpec, INLET_LIBRARY, load_inlet_library


# --- InletSpec blockage derating ------------------------------------------- #

def test_no_blockage_uses_full_geometry():
    spec = InletSpec("S", clear_area=0.5, effective_perimeter=3.0)
    assert spec.operational_area == pytest.approx(0.5)
    assert spec.operational_perimeter == pytest.approx(3.0)


def test_blockage_derates_area_and_perimeter():
    spec = InletSpec("S", 0.5, 3.0, blockage=0.25)
    assert spec.operational_area == pytest.approx(0.5 * 0.75)
    assert spec.operational_perimeter == pytest.approx(3.0 * 0.75)


def test_full_blockage_zeroes_geometry():
    spec = InletSpec("S", 0.5, 3.0, blockage=1.0)
    assert spec.operational_area == pytest.approx(0.0)
    assert spec.operational_perimeter == pytest.approx(0.0)


# --- INLET_LIBRARY catalogue ----------------------------------------------- #

@pytest.mark.parametrize("key", ["Grate_600x600", "Grate_900x900", "Lintel_1.2m",
                                 "Lintel_2.4m", "Combo_1.2m_G600", "Combo_2.4m_G900"])
def test_expected_spec_present(key):
    assert key in INLET_LIBRARY


@pytest.mark.parametrize("key", list(INLET_LIBRARY))
def test_specs_have_positive_geometry(key):
    spec = INLET_LIBRARY[key]
    assert spec.clear_area > 0.0
    assert spec.effective_perimeter > 0.0


# --- TOML loader ------------------------------------------------------------ #

def test_load_inlet_library_roundtrip(tmp_path):
    p = tmp_path / "lib.toml"
    p.write_text(
        "[inlets.MyGrate]\n"
        "clear_area = 0.5\n"
        "effective_perimeter = 3.0\n"
    )
    lib = load_inlet_library(str(p))
    assert set(lib) == {"MyGrate"}
    spec = lib["MyGrate"]
    assert isinstance(spec, InletSpec)
    assert spec.clear_area == pytest.approx(0.5)
    assert spec.effective_perimeter == pytest.approx(3.0)
    assert spec.blockage == pytest.approx(0.0)


def test_load_inlet_library_optional_blockage(tmp_path):
    p = tmp_path / "lib.toml"
    p.write_text(
        '[inlets."Lintel_1.2m"]\n'   # dotted name must be quoted
        "clear_area = 0.18\n"
        "effective_perimeter = 1.20\n"
        "blockage = 0.4\n"
    )
    spec = load_inlet_library(str(p))["Lintel_1.2m"]
    assert spec.blockage == pytest.approx(0.4)
    assert spec.operational_area == pytest.approx(0.18 * 0.6)


def test_load_inlet_library_missing_key_raises(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[inlets.Broken]\nclear_area = 0.5\n")   # no effective_perimeter
    with pytest.raises(ValueError):
        load_inlet_library(str(p))


def test_load_inlet_library_empty_raises(tmp_path):
    p = tmp_path / "empty.toml"
    p.write_text("# nothing here\n")
    with pytest.raises(ValueError):
        load_inlet_library(str(p))
