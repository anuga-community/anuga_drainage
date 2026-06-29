"""Tests for the hydrograph viewer's pure data helper and entry points.

The GUI itself isn't exercised (no display in CI); we test combine_hydrographs
and that the module/entry points import. Skips when the [viewer] extra
(matplotlib) or tkinter are unavailable.
"""
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("tkinter")

from anuga_drainage.viewer import combine_hydrographs, HydrographViewerApp, main


def _write(path, time, captured, bypass):
    pd.DataFrame({
        "Time_s": time,
        "Depth_m": [0.0] * len(time),          # superset columns are tolerated
        "Captured_Q_cms": captured,
        "Bypass_Q_cms": bypass,
    }).to_csv(path, index=False)


def test_combine_sums_and_integrates(tmp_path):
    _write(tmp_path / "a.csv", [0.0, 1.0, 2.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2])
    _write(tmp_path / "b.csv", [0.0, 1.0, 2.0], [0.3, 0.3, 0.3], [0.0, 0.0, 0.0])

    combined, skipped = combine_hydrographs(
        [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])

    assert skipped == []
    assert combined["Captured_total_cms"].tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert combined["Combined_total_cms"].tolist() == pytest.approx([0.6, 0.6, 0.6])
    # dt = [0, 1, 1] s -> cumulative captured = [0, 0.4, 0.8] m^3
    assert combined["Captured_cum_m3"].tolist() == pytest.approx([0.0, 0.4, 0.8])


def test_combine_skips_files_missing_columns(tmp_path):
    _write(tmp_path / "good.csv", [0.0, 1.0], [0.1, 0.1], [0.0, 0.0])
    pd.DataFrame({"Time_s": [0, 1], "Foo": [1, 2]}).to_csv(tmp_path / "bad.csv", index=False)

    combined, skipped = combine_hydrographs(
        [str(tmp_path / "good.csv"), str(tmp_path / "bad.csv")])

    assert skipped == ["bad.csv"]
    assert not combined.empty


def test_combine_empty_when_none_valid(tmp_path):
    pd.DataFrame({"Time_s": [0], "Foo": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    combined, skipped = combine_hydrographs([str(tmp_path / "bad.csv")])
    assert combined.empty
    assert skipped == ["bad.csv"]


def test_entry_points_exist():
    assert callable(main)
    assert isinstance(HydrographViewerApp, type)
