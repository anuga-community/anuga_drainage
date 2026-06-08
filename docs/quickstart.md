# Quickstart: coupling from a `.inp`

The simplest way to set up a coupled model is {func}`~anuga_drainage.couple_from_inp`.
SWMM's `.inp` is the de-facto standard sewer-network format; pipedream has no
file format of its own. `couple_from_inp` reads a `.inp` and builds **both** the
1D backend and the ANUGA coupling inlets, so a single file drives either backend
— and both then model the *same* sewer.

## One call

```python
import anuga
from anuga_drainage import couple_from_inp

domain = ...   # an ANUGA domain: mesh, elevation, friction, boundaries

coupling = couple_from_inp(
    domain, 'network.inp',
    backend='swmm',            # or 'pipedream'
    manhole_area=1.0,          # surface footprint of each inlet region
    time_average=10.0,         # exchange-flux smoothing (stability)
    clamp=True,                # never draw more than the 2D cell holds
)
```

`couple_from_inp` returns a {class}`~anuga_drainage.Coupling`:

```python
coupling.step(dt)            # run one coupled exchange step
coupling.close()             # release the backend (closes the SWMM sim; no-op otherwise)
coupling.add_volume_balance(...)   # attach a mass-balance audit (see Diagnostics)

coupling.coupler     # the underlying Coupler
coupling.inlets      # {junction name -> ANUGA Inlet_operator}
coupling.backend     # SwmmBackend / PipedreamBackend
coupling.handle      # the pyswmm Simulation or pipedream SuperLink
coupling.inp         # the parsed InpNetwork
coupling.domain      # the ANUGA domain
```

## The evolve loop

```python
dt = 1.0
for t in domain.evolve(yieldstep=dt, finaltime=400.0):
    coupling.step(dt)        # read 2D state -> exchange flux -> step 1D -> feed back

coupling.close()             # closes the SWMM simulation; no-op for pipedream
```

That's the whole coupled model: build the ANUGA domain, call `couple_from_inp`,
and `coupling.step(dt)` inside `domain.evolve`.

## What it does under the hood

1. Parses the `.inp` ({func}`~anuga_drainage.read_inp`).
2. Creates a regular-polygon ANUGA `Inlet_operator` at each `[JUNCTIONS]` node.
3. Builds the 1D backend — a pyswmm `Simulation`, or a pipedream `SuperLink`
   converted from the `.inp` ({func}`~anuga_drainage.inp_to_pipedream`).
4. Wires them into a {class}`~anuga_drainage.Coupler`.

**Junctions are coupled to the surface; outfalls are boundaries** (free
drainage for pipedream; SWMM handles its own). The upstream inflow, the ANUGA
boundary conditions, and the run loop stay in your script.

## Living examples

Two runnable examples drive the whole coupling from one `.inp`:

```bash
cd examples/simple_culvert_example && python run_from_inp.py swmm
cd examples/real_example          && python run_from_inp.py pipedream
```

See {doc}`inp_format` for how the `.inp` maps onto pipedream, and
{doc}`diagnostics` for the volume audit these examples print.
