# How the coupling works

If you need finer control than {func}`~anuga_drainage.couple_from_inp` gives,
you can build the coupling yourself. This page explains the moving parts.

## The exchange flux: `calculate_Q`

At each shared inlet/manhole the surface (2D) and the pipe (1D) exchange water
through a **weir/orifice** relationship ({func}`~anuga_drainage.calculate_Q`,
after Leandro & Martins, 2016):

```python
from anuga_drainage import calculate_Q

Q = calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole,
                cw=0.67, co=0.67)
```

Sign convention: **positive `Q` = water leaving the 2D surface and entering the
1D network**; negative `Q` = surcharge back up onto the surface. The function is
vectorised over numpy arrays of inlets. `g` (gravity) defaults to ANUGA's value;
pass it explicitly to use the function without ANUGA (e.g. in tests).

```{admonition} The `min_head` deadband
:class: note
`calculate_Q(..., min_head=1.0e-3)` is a deadband on the driving head: no
exchange is computed when the relevant head difference is below `min_head` (m).
This suppresses spurious exchange from sub-millimetre, numerically-noisy head
differences — for example when a 1D solver initialises a junction head a hair
(~1e-5 m) above the bed. Without it, that phantom head surcharges onto a *dry*
surface and sets off a growing capture/surcharge oscillation before any real
water arrives (seen with the pipedream backend, whose superjunctions start just
above their invert; SWMM reports head == invert exactly and was unaffected).
Real exchange (head differences ≫ `min_head`) is unchanged. Lower it toward 0 to
recover the old behaviour, or raise it to ignore larger head differences.
```

## The driver: `Coupler`

{class}`~anuga_drainage.Coupler` encapsulates the per-step sequence:

1. read 2D state at each inlet (depth, stage, water volume);
2. read 1D heads from the backend;
3. compute `Q_in = calculate_Q(...)`;
4. **temporally smooth** `Q_in` (critical for stability) —
   {func}`~anuga_drainage.smooth_Q`;
5. optionally **clamp** so you never remove more water than the 2D cell holds —
   {func}`~anuga_drainage.limit_outflow`;
6. advance the 1D model by `dt`;
7. feed the **realised** 1D flow back into ANUGA via the inlet operators.

```python
from anuga_drainage import Coupler, SwmmBackend

coupler = Coupler(
    inlets=[...],            # ANUGA Inlet_operators, ordered to match the backend
    beds=anuga_beds,         # inlet bed elevations
    weir_lengths=...,        # parallel arrays for calculate_Q
    manhole_areas=...,
    backend=SwmmBackend(sim),
    time_average=10.0, clamp=True, cw=0.67, co=0.67,
)

step = coupler.step(dt)      # returns CouplingStep(Q_in, anuga_flux)
```

The index order of `inlets`, `beds`, `weir_lengths` and `manhole_areas` must
line up with the 1D nodes (the backend's head order).

## Backends

The 1D-solver differences live behind a small interface:

`SwmmBackend(sim)`
: heads from junction nodes; advances SWMM with whole-second strides
  (`int(dt)`); realised flow comes from node statistics.

`PipedreamBackend(superlink, coupled_indices=None, H_bc=None, outfall_indices=None)`
: heads are the superjunction heads `H_j`; the requested flux is taken as
  realised. `coupled_indices` selects which superjunctions exchange with ANUGA
  (default all); `H_bc` holds boundary (outfall) superjunctions at a fixed head;
  `outfall_indices` enables outfall-outflow tracking. The defaults reproduce the
  earlier all-coupled / no-boundary behaviour.

```{admonition} Backend sign/bookkeeping differs
:class: note
The `Q_in` sign passed back to the ANUGA inlet operators is **not** the same
across backends, and SWMM uses its realised node-statistics flow while pipedream
takes the requested flux as realised. The `Coupler` handles this; if you drive a
backend directly, trace the specific convention.
```

## The loop you write

```python
for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):
    step = coupler.step(dt)
    # ... your logging / diagnostics ...
```

ANUGA still sub-steps internally at its CFL timestep regardless of the
yieldstep, so a 1-second yieldstep does not hurt ANUGA's own mass conservation —
only the coupling-exchange frequency. Sub-second coupling is available on the
pipedream path (`superlink.step(dt=...)` is pure Python).
