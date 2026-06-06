# anuga_drainage — coupling ANUGA to SWMM and pipedream

Couples the 2D hydrodynamic shallow-water model **ANUGA** (overland / surface
flow) with a 1D stormwater / sewer network solver — **SWMM** (via `pyswmm`) or
**pipedream** (via `pipedream_solver`). The 2D model handles the surface, the 1D
model the underground pipe network, and the two exchange water at
inlets/manholes every timestep using weir/orifice equations.

## Installation

ANUGA is heavy and is expected to come from an existing **conda environment**
where it is already built — it is *not* pip-installed here. On top of that env:

```bash
pip install -e .              # the anuga_drainage package (numpy, pandas)
pip install -e .[swmm]        # + SWMM backend (standard pyswmm release)
pip install -e .[pipedream]   # + pipedream backend (from git; see note below)
```

> The PyPI release of `pipedream-solver` (0.2.2) uses `np.bool8`, removed in
> numpy 2.x, so the `[pipedream]` extra installs it from git master. See
> [`CLAUDE.md`](CLAUDE.md) for this and other environment constraints (notably
> that stock pyswmm 2.1 couples in whole-second steps).

## Running an example

Each example runs from its own directory (scripts use relative paths):

```bash
cd examples/simple_culvert_example
python run_swmm_short.py     # ANUGA + SWMM
python run_pipedream.py      # ANUGA + pipedream
python run_boyd.py           # ANUGA-only reference (Boyd culvert operator)
```

ANUGA writes `<name>.sww`; scripts print a running mass-balance `loss` and may
save `Figure*.png`.

## The package

- `calculate_Q(...)` — the weir/orifice exchange-flux physics
  (Leandro & Martins, 2016). Positive Q = surface → pipe; negative = surcharge.
- `Coupler` with `SwmmBackend` / `PipedreamBackend` — drives the per-step
  exchange (read depths → `calculate_Q` → smooth → optional clamp → step the 1D
  model → feed the realised flow back to ANUGA). All the coupled `run_*` example
  scripts use it.
- `initialize_inlets(...)` / `read_inp_coordinates(...)` — build ANUGA inlet
  operators from a SWMM `.inp` file.

## Tests

```bash
pip install -e .[test]
pytest
```

The pure logic (flux physics, smoothing, geometry, `.inp` parsing) is tested
without requiring ANUGA; tests that need ANUGA's gravity constant skip when it
is absent.

## Examples

- `simple_culvert_example/` — canonical minimal channel + culvert; the clearest
  reference for the coupling pattern (SWMM, pipedream and Boyd variants).
- `culvert_example/`, `yshape/`, `real_example/` — further SWMM/pipedream cases.
- `pyswmm_with_openings/` — **legacy/reference only**: built against a patched
  pyswmm fork that is not in the standard release; not part of the active path.

See [`CLAUDE.md`](CLAUDE.md) for the architecture and coupling-loop details.
