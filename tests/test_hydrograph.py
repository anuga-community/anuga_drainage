"""Tests for the per-inlet HydrographLogger.

Pure: record() takes plain arrays, so this needs neither ANUGA nor a backend.
"""
import pandas as pd
import pytest

from anuga_drainage import HydrographLogger
from anuga_drainage.hydrograph import COLUMNS

# The columns the Simple_SW_Inlets viewer requires; the logger schema is a superset.
VIEWER_REQUIRED = ["Time_s", "Depth_m", "Approach_Q_cms", "Captured_Q_cms",
                   "Bypass_Q_cms", "Cum_Captured_m3", "Cum_Bypassed_m3"]


def test_record_splits_capture_and_surcharge():
    log = HydrographLogger(["J1", "J2"])
    # J1 captures (+), J2 surcharges (-).
    log.record(time=10.0, dt=1.0, depths=[0.3, 0.1], heads=[-1.0, 0.5],
               approach_Q=[0.5, 0.0], exchange_Q=[0.2, -0.3])
    r1 = log.to_dataframe("J1").iloc[0]
    r2 = log.to_dataframe("J2").iloc[0]
    assert r1["Captured_Q_cms"] == pytest.approx(0.2)
    assert r1["Surcharge_Q_cms"] == pytest.approx(0.0)
    assert r2["Captured_Q_cms"] == pytest.approx(0.0)
    assert r2["Surcharge_Q_cms"] == pytest.approx(0.3)


def test_bypass_is_approach_minus_capture():
    log = HydrographLogger(["J1"])
    log.record(0.0, 1.0, depths=[0.3], heads=[-1.0],
               approach_Q=[0.5], exchange_Q=[0.2])
    row = log.to_dataframe("J1").iloc[0]
    assert row["Bypass_Q_cms"] == pytest.approx(0.3)   # 0.5 - 0.2


def test_cumulative_volumes_integrate_over_steps():
    log = HydrographLogger(["J1"])
    for k in range(3):
        log.record(time=k * 2.0, dt=2.0, depths=[0.3], heads=[-1.0],
                   approach_Q=[1.0], exchange_Q=[0.5])
    last = log.to_dataframe("J1").iloc[-1]
    # 3 steps x 2 s: captured 0.5*2*3 = 3.0, inflow 1.0*2*3 = 6.0, bypass 0.5*2*3=3.0
    assert last["Cum_Captured_m3"] == pytest.approx(3.0)
    assert last["Cum_Inflow_m3"] == pytest.approx(6.0)
    assert last["Cum_Bypassed_m3"] == pytest.approx(3.0)


def test_dataframe_schema_is_viewer_superset():
    log = HydrographLogger(["J1"])
    log.record(0.0, 1.0, [0.3], [-1.0], [0.5], [0.2])
    cols = list(log.to_dataframe("J1").columns)
    assert cols == ["Asset_ID"] + COLUMNS
    assert set(VIEWER_REQUIRED).issubset(cols)


def test_empty_dataframe_has_columns():
    log = HydrographLogger(["J1"])
    df = log.to_dataframe("J1")
    assert df.empty
    assert "Asset_ID" in df.columns


def test_write_csv_roundtrip(tmp_path):
    log = HydrographLogger(["J1", "J2"])
    log.record(0.0, 1.0, [0.3, 0.2], [-1.0, -1.0], [0.5, 0.4], [0.2, 0.1])
    paths = log.write_csv(directory=str(tmp_path))
    assert len(paths) == 2
    df = pd.read_csv(tmp_path / "hydrograph_J1.csv")
    assert set(VIEWER_REQUIRED).issubset(df.columns)
    assert df.iloc[0]["Asset_ID"] == "J1"
