# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`anuga_drainage` couples the 2D hydrodynamic shallow-water model **ANUGA** (overland/surface flow) with a 1D stormwater/sewer network solver (**SWMM** via `pyswmm`, or **pipedream** via `pipedream_solver`). The 2D model handles the surface; the 1D model handles the underground pipe network; the two exchange water at inlets/manholes each timestep.

The installable package is intentionally tiny — almost all the real logic lives in the per-example `run_*.py` scripts. The package only provides the shared physics of the 2D↔1D exchange.

## Install & environment

Assume work happens inside an existing conda environment that already has a current **ANUGA** built and importable. On top of that env:

```bash
pip install -e .                  # installs the anuga_drainage package
pip install pyswmm                 # SWMM backend (standard PyPI release, 2.1.0)
pip install git+https://github.com/mdbartos/pipedream.git   # pipedream backend — NOT the PyPI release
```

**pipedream must come from git master, not PyPI.** The released `pipedream-solver 0.2.2` uses `np.bool8`, removed in numpy 2.x (which the ANUGA env requires), so it crashes on `SuperLink(...)` construction. Git master replaced those with `np.bool_`. Caveat: even git master still has `np.bool8` in two optional code paths — `nsuperlink.py` `reposition_junctions()` and the `nquality.py` water-quality solver. None of the examples use these (all have `#superlink.reposition_junctions()` commented out), but uncommenting `reposition_junctions()` will hit the bug. Fix submitted upstream: https://github.com/mdbartos/pipedream/pull/73 — once merged (and ideally released to PyPI), this caveat and the git-master requirement can be revisited.

Do **not** `pip install anuga` into the conda env — ANUGA is provided by the env, not installed from `requirements.txt`. For a from-scratch setup, `environment.yml` builds a conda env with ANUGA 3.3.6 from conda-forge (Python 3.12) plus this package and both backends (`conda env create -f environment.yml`). The maintainer's dev box instead runs ANUGA from source (3.3.6.dev) on Python 3.14, which conda-forge ANUGA does not yet build for.

The package's own logic is tested under `tests/` (pytest). Note the `test_*.py` files under `examples/` are unrelated — they are standalone runnable scripts, not part of the pytest suite.

### CI and platform support
Three GitHub Actions workflows (`.github/workflows/`):
- `tests.yml` — fast lane, no ANUGA: installs `-e .[test]` on Python 3.10/3.12 and runs the pure-logic tests (geometry, `.inp` parser, `calculate_Q` with explicit `g`). ANUGA-dependent tests skip.
- `conda-env.yml` — validates `environment.yml` end to end (conda-forge ANUGA + both backends via micromamba) and runs the **full** suite on **ubuntu-latest + windows-latest**.
- `pip.yml` — installs ANUGA + this package from **PyPI wheels** and runs the full suite on **ubuntu-latest, windows-latest and macos-latest** (Apple Silicon). This is the lane that exercises the published wheels, so it catches wheel-packaging issues — it found the macOS arm64 missing-libomp bug, fixed in ANUGA 3.3.7 (pinned here). **Windows is an allowed-failure** (`continue-on-error`) for a *dependency* bug, not an ANUGA one: meshpy's MinGW-built Windows wheel doesn't bundle its runtime DLLs (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`), so `import meshpy._internals` fails on a clean pip install (tracking: `inducer/meshpy#150`). Remove the flag once meshpy ships a self-contained Windows wheel.

macOS distribution is split and asymmetric, which is why it needs its own lane:
- **conda-forge** ANUGA built for `osx-64` (Intel) **only** — no `osx-arm64`. So `macos-latest` (arm64) couldn't use the conda `environment.yml`; Intel `macos-13` runners exist but are scarce, so conda-env omits macOS. (osx-arm64 is being added to the feedstock **natively** — `provider: {osx_arm64: default}` on Azure's native arm64 agents, conda-forge/anuga-feedstock#17; the cross-compile route was abandoned because this recipe's meson build can't resolve numpy via pkg-config when cross-compiling.)
- **PyPI** ANUGA wheels are `macosx_*_arm64` (Apple Silicon) **only** — no Intel macOS wheel. So Apple Silicon is covered by `pip install anuga`. (Through 3.3.6 these wheels were not self-contained — the macOS arm64 wheel needed `libomp` and the Windows wheel had unbundled DLLs; ANUGA 3.3.7 fixed this by running `repairwheel` on all platforms, so plain `pip install anuga` now works.) Note `macos-latest` being arm64 means the `pip` lane already tests ANUGA on Apple Silicon (via the arm64 wheel).

### Pending CI follow-ups
- **Done:** `environment.yml` pins conda-forge `anuga=3.3.7` (published 2026-06-07).
- Once conda-forge ships **osx-arm64** (feedstock#17, native build), add a **`macos-latest` (arm64) lane to `conda-env.yml`** — `macos-latest` is arm64, so this tests the conda arm64 package natively. (Less critical than it was: the native feedstock build already runs the test suite on arm64.)
- Drop the Windows `continue-on-error` in `pip.yml` once meshpy ships a self-contained Windows wheel (`inducer/meshpy#150` / PR #151).

### Project decision: standard pyswmm + our own `calculate_Q`
We target the **standard pyswmm release** (currently 2.1.0 from PyPI) and compute the 2D↔1D exchange flux ourselves in Python via `anuga_drainage.calculate_Q`. We do **not** use the patched pyswmm fork (the `2D-coupling` branch with `node.create_opening(...)` / `sim.coupling_step(...)` used by `examples/pyswmm_with_openings/`); that API does not exist in the standard release. Treat `pyswmm_with_openings/` as legacy/reference, not the active path.

### pyswmm 2.1.0 stepping constraints (important)
Stock pyswmm 2.1.0 is **whole-second resolution**:
- `sim.next()` was removed — step the iterator with `next(sim)`.
- `sim.step_advance(...)` feeds `swmm_stride`, whose binding requires an **`int`** (whole seconds). Even `1.0` (a float) raises `TypeError`. Always pass `int(dt)`, and keep `dt` a whole number of seconds.
- `sim.current_time` also only resolves to whole seconds.

Consequence: **SWMM coupling exchanges at 1-second granularity.** The SWMM examples use `dt = 1.0` (yield/coupling step) and rely on `time_average` smoothing for stability. Sub-second coupling is *only* available on the **pipedream** path (`superlink.step(dt=...)` is pure Python — `run_pipedream.py` uses `dt=0.05`). Note ANUGA still sub-steps internally at its CFL timestep regardless of the yieldstep, so a 1 s yieldstep does not hurt ANUGA's own mass conservation — only the coupling-exchange frequency.

## Running examples

Each example is run by executing a script from inside its own directory (the scripts use relative paths like `./swmm_input_short.inp`):

```bash
cd examples/simple_culvert_example
python run_swmm_short.py      # ANUGA + SWMM (pyswmm)
python run_pipedream.py       # ANUGA + pipedream
python run_boyd.py            # ANUGA-only reference using a Boyd culvert operator
```

Naming convention: `run_<backend>.py` selects the 1D backend (`swmm`, `pipedream`, `boyd`). The `boyd` variants use ANUGA's built-in culvert operator with no external 1D solver — they are the physical reference the coupled runs are validated against.

Outputs: ANUGA writes `<outname>.sww` (set via `domain.set_name(...)`); SWMM reads `*.inp` and writes `*.rpt`/`*.out`. Many scripts also save `Figure*.png` and, when `visualise`/plotting is enabled, pop up matplotlib windows (and may `input('Enter key ...')` at the end — they block waiting for a keypress).

## The package (`src/anuga_drainage/`)

- `coupling.py` — `calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole, cw, co, eps, g=None)`: the core exchange-flux physics. Implements weir/orifice equations from Leandro & Martins (2016). Sign convention: **positive Q = water leaving the 2D surface and entering the 1D network**; negative Q = surcharge back up onto the surface. Vectorised over numpy arrays of inlets. `g` (gravity) defaults to `None`, which lazily imports ANUGA's value; pass `g` explicitly to use the function (and `Coupler`, which forwards it) without ANUGA, e.g. in tests. Exported at top level: `from anuga_drainage import calculate_Q`.
- `inlet_initialization.py` — `initialize_inlets(...)` builds ANUGA `Inlet_operator`s automatically from a SWMM `.inp` file: reads junction nodes (via `pyswmm`) and constructs a regular n-sided polygon region of the given manhole area at each node's coordinates. `n_sided_inlet(...)` is the geometry helper. Node map coordinates come from `read_inp_coordinates(inp_path)` in this module — a small zero-dependency parser of the `.inp` `[COORDINATES]` section (returns a DataFrame with `X_Coord`/`Y_Coord` indexed by node id). This replaced the former `hymo` dependency, which is unmaintained and not on PyPI; do **not** reintroduce it. `initialize_inlets` takes that coordinates DataFrame as its 3rd argument. (`pyswmm`/`anuga` are imported lazily inside `initialize_inlets` so the pure helpers stay importable for testing without ANUGA.)
- `coupler.py` — the per-step coupling driver, factored out of the duplicated example loops. `Coupler(inlets, beds, weir_lengths, manhole_areas, backend, time_average, clamp=...)` owns the read-depths → `calculate_Q` → `smooth_Q` → optional `limit_outflow` → backend step → feed realised flux back via `set_Q` sequence; `coupler.step(dt)` returns `CouplingStep(Q_in, anuga_flux)`. The 1D-backend differences live behind `SwmmBackend(sim)` (heads from junction nodes, `int(dt)` stride, realised flow from node statistics) and `PipedreamBackend(superlink)` (heads = `H_j`, feeds back `-Q_in`). `smooth_Q`/`limit_outflow` are exported as standalone pure functions. Volume-balance/logging stays in the run scripts.

## The coupling loop (the key architecture to understand)

All six coupled run scripts now drive the loop through `anuga_drainage.Coupler` (above): per step they call `coupler.step(dt)` and keep their own volume-balance/logging. The four SWMM/pipedream "mature" scripts were verified bit-identical to their pre-refactor mass balance. The two formerly-experimental pipedream scripts (`culvert_example/run_pipedream_culvert.py`, `real_example/run_pipedream.py`) were also converted — the culvert one preserved its historical `cw=co=1.0`; `real_example/run_pipedream.py` additionally had real bugs fixed (line ~116 reassigned `inlet4_anuga_inlet_op` to the outlet region, orphaning inlet4 and duplicating the outlet into both junction slots) so its 5th superjunction now correctly couples to its own outlet ANUGA region, and `calculate_Q` + smoothing replace the old inline weir formula. That last change altered numerics by design (final mass-balance loss ≈ 0.077 m³ on 12.5 m³ input).

The underlying structure each script follows (and what `Coupler` encapsulates) — preserve this when editing or adding examples:

1. **Build the ANUGA domain** — mesh, `topography` elevation function, friction, boundary conditions, and an upstream inflow `Inlet_operator`.
2. **Define coupling regions** — for each shared inlet/outlet, an `anuga.Region` polygon and a zero-Q `anuga.Inlet_operator`. Capture each region's bed elevation (`inlet.get_average_elevation()`) into an `anuga_beds` array, plus parallel `anuga_length_weirs` / `anuga_area_manholes` arrays. Index order of these arrays must line up with the 1D nodes.
3. **Build the 1D model** — SWMM: `Simulation('./*.inp'); sim.start()` then grab `Nodes`/`Links` by name. Pipedream: construct `superjunctions`/`superlinks` DataFrames and a `SuperLink`.
4. **`for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):`** — each yield:
   - read 2D state at each inlet: `get_average_depth()`, `get_average_stage()`, water volume;
   - read 1D heads (`node.head` / `superlink.H_j`);
   - `Q_in = calculate_Q(heads, anuga_depths, anuga_beds, weirs, manholes)`;
   - **temporal smoothing** (critical for stability): `Q_in = ((time_average - dt)*Q_in_old + dt*Q_in)/time_average`. Oscillation control relies on this plus a small routing step in the `.inp`;
   - clamp so you never remove more water than the 2D cell holds (`Q_limit = anuga_volumes/dt`);
   - advance the 1D model by `dt` (SWMM: `node.generated_inflow(Q); sim.step_advance(int(dt)); next(sim)` — see the pyswmm 2.1.0 constraints above. Pipedream: `superlink.step(Q_in=Q_in, dt=dt)`);
   - feed the **actual** realised 1D flow back into ANUGA via `inlet_op.set_Q(...)`. For SWMM this is derived from node statistics (`lateral_infow_vol`, `flooding_volume`); sign is flipped relative to `calculate_Q`'s convention.

5. **Volume-balance check** — scripts continuously compute `loss = total_volume_real - total_volume_correct`, where real = ANUGA water volume + sewer volume and correct = cumulative inflow + boundary flux (+ initial pipe volume). This conservation diagnostic is the primary signal of whether a coupling change is correct; watch it when modifying flux logic.

### Gotchas
- SWMM/pyswmm sign and bookkeeping differs from pipedream — `Q_in` sign passed to `set_Q` is **not** the same across backends (compare `run_swmm_short.py` line ~321 vs `run_pipedream.py` line ~235). Don't blindly copy between backends.
- `calculate_Q` returns positive for surface→pipe; some scripts negate it when setting the ANUGA operator and others use SWMM's realised flow instead. Trace the specific script.
- Mesh refinement (`rf`) must be fine enough that inlet polygons don't overlap walls/each other — comments in the scripts flag this.
- **ANUGA API drift:** some examples were written against older ANUGA and need small fixes to run on a current build. Known: `create_domain_from_regions(...)` no longer accepts `mesh_filename` (the mesh is built in-memory — drop the kwarg). Expect similar minor signature changes when reviving older scripts.

## Examples directory map

- `simple_culvert_example/` — canonical minimal channel + culvert; the best reference for the coupling pattern. Has `boyd`, `swmm` (short/long inp), and `pipedream` variants. A local copy of `coupling.py` lives here (the scripts now import from the installed package instead — see git history).
- `culvert_example/` — culvert on a meshed terrain (`terrain.csv`/`.msh`), boyd vs pipedream.
- `yshape/` — Y-shaped pipe topology; `pyswmm` coupling. Paired tutorial: `examples/pyswmm_with_openings/how_to_ANUGA_SWMM_coupling.md`.
- `pyswmm_with_openings/` — **legacy/reference.** Experiments against a patched `2D-coupling` pyswmm fork (`create_opening`/`coupling_step`/`overland_depth` API) that is *not* in the standard release. Not the active path — see "Project decision" above.
- `real_example/` — a real catchment: terrain/kerb/wall CSVs, a DRAINS-exported network, and `run_{no_pipes,swmm,pipedream}.py`.
