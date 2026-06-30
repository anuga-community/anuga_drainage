# Hydrograph logging & viewer

The coupling can record a **per-inlet hydrograph** each step and write it to CSV,
and a bundled Tkinter **viewer** plots those CSVs.

## Logging during a run

Pass `log_hydrographs=True` to `couple_from_inp` (or hand a `HydrographLogger`
to a hand-built `Coupler` via `logger=`). The logger is reachable at
`coupling.coupler.logger`:

```python
from anuga_drainage import couple_from_inp

coupling = couple_from_inp(domain, "network.inp", backend="pipedream",
                           manhole_area=1.0, log_hydrographs=True)

for t in domain.evolve(yieldstep=1.0, finaltime=400.0):
    coupling.step(1.0)

logger = coupling.coupler.logger
logger.write_csv("hydrographs/")        # one hydrograph_<junction>.csv per inlet
df = logger.to_dataframe("J1")          # or get a single inlet as a DataFrame
```

Each step records, per inlet: surface depth, the 1D pipe head, the exchange
discharge split into **capture** (surface→pipe) and **surcharge** (pipe→surface),
an approach-flow estimate (region-averaged momentum × √area) and the resulting
bypass, plus running cumulative volumes.

### CSV schema

The columns are a **superset** of the Simple_SW_Inlets viewer schema (so that
viewer, and the one below, open them):

```
Asset_ID, Time_s, Depth_m, Head1D_m,
Approach_Q_cms, Captured_Q_cms, Surcharge_Q_cms, Bypass_Q_cms,
Cum_Inflow_m3, Cum_Captured_m3, Cum_Surcharged_m3, Cum_Bypassed_m3
```

Flows are in m³/s; `Head1D_m` and the `Surcharge`/`Cum_Surcharged` columns are
the coupling-specific extras (a one-way surface capture model has no pipe head or
surcharge).

```{admonition} Pure helper
:class: note
`HydrographLogger.record(time, dt, depths, heads, approach_Q, exchange_Q)` takes
plain arrays (no ANUGA), so the logging logic is unit-tested standalone; the
`Coupler` simply supplies the per-step samples.
```

## The viewer

A Tkinter dashboard for inspecting those CSVs. It is an optional extra (it pulls
in matplotlib; Tkinter is in the standard library) and is **not** imported by the
package core, so a headless install stays GUI-free:

```bash
pip install -e .[viewer]
anuga-drainage-viewer            # or: python -m anuga_drainage.viewer
```

Pick a folder of `hydrograph_*.csv` in the sidebar. The **View** menu offers:

- **Pit Hydrograph** — four diagnostic plots for the selected inlet (approach vs
  captured, accumulated volumes, flows + depth on a twin axis, and a
  time-coloured depth–discharge hysteresis loop).
- **Combined Hydrograph** — folder totals: captured / bypass summed across every
  CSV onto a common time axis, with cumulative volumes.

It is HiDPI-aware (reads `Xft.dpi`) with live UI / plot font-size sliders.

### Reusing the combine logic

The combined view's maths is a pure, importable, tested helper:

```python
from anuga_drainage.viewer import combine_hydrographs

combined, skipped = combine_hydrographs(["hydrographs/hydrograph_J1.csv",
                                         "hydrographs/hydrograph_J2.csv"])
combined["Combined_total_cms"]   # per-step total exchange across the inlets
combined["Captured_cum_m3"]      # cumulative captured volume
```
