# anuga_drainage

`anuga_drainage` couples the 2D hydrodynamic shallow-water model
[**ANUGA**](https://github.com/anuga-community/anuga_core) (overland / surface
flow) with a 1D stormwater / sewer-network solver — either **SWMM** (via
[`pyswmm`](https://pypi.org/project/pyswmm/)) or
[**pipedream**](https://github.com/mdbartos/pipedream). The 2D model handles the
surface; the 1D model handles the underground pipe network; the two exchange
water at inlets / manholes every timestep.

The installable package is intentionally small — it provides the shared physics
of the 2D↔1D exchange and the tooling around it:

- **`calculate_Q`** — the weir/orifice exchange flux (Leandro & Martins, 2016).
- **`Coupler`** — the per-step exchange driver, with `SwmmBackend` and
  `PipedreamBackend` behind a common interface.
- **`VolumeBalance`** — a mass-balance audit that localises where water is
  lost or gained (ANUGA, the pipe solver, or the coupling).
- **`couple_from_inp`** — build the entire sewer **and** the ANUGA coupling from
  a single SWMM `.inp` file, for either backend.

```{admonition} The headline
:class: tip
Write one SWMM `.inp`, pick a backend, run the evolve loop. Both backends then
model the *same* sewer, and the volume audit confirms conservation.
```

## Contents

```{toctree}
:maxdepth: 2

installation
quickstart
coupling
diagnostics
inp_format
api
```

## A taste

```python
import anuga
from anuga_drainage import couple_from_inp

domain = ...                      # your ANUGA domain (mesh, elevation, boundaries)

coupling = couple_from_inp(domain, 'network.inp', backend='pipedream',
                           manhole_area=1.0, time_average=10.0, clamp=True)
coupling.add_volume_balance(inflow_operators=[my_inflow_op])   # optional audit

for t in domain.evolve(yieldstep=1.0, finaltime=400.0):
    coupling.step(1.0)            # exchange + (if attached) volume audit

print(coupling.volume_balance.summary())   # ANUGA / pipe / coupling residuals
coupling.close()                  # closes the SWMM sim; no-op for pipedream
```
