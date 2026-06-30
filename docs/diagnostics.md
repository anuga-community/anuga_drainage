# Volume balance & diagnostics

A coupled run should conserve water. {class}`~anuga_drainage.VolumeBalance` is a
per-step mass-balance audit that doesn't just report *that* water is lost — it
**localises where**, by measuring each subsystem from its own bookkeeping and
comparing.

## The three residuals

All volumes are signed (`+` = added to that subsystem). The audit forms three
independent budgets:

```
ANUGA :  V_anuga(t) - V_anuga(0)  =  inflow + boundary + inlets_anuga
pipe  :  V_pipe(t)  - V_pipe(0)   =  inlets_pipe - outfall
couple:  inlets_anuga + inlets_pipe - outfall   (~0 when the handoff conserves)
```

and reports the residual `R = LHS - RHS` of each:

`R_anuga`
: ANUGA closes — should be ~machine precision (it's a finite-volume method).

`R_pipe`
: the pipe closes — ~0 for finite-volume pipedream, and it isolates SWMM's
  finite-difference loss.

`R_couple`
: the surface↔pipe handoff is consistent — ~0 when the coupling conserves; this
  is the one that catches *coupling* bugs.

And the single overall loss splits exactly:

```
loss = R_anuga + R_pipe + R_couple
```

## Usage

The easiest path, if you built the model with
{func}`~anuga_drainage.couple_from_inp`, is to attach the audit to the coupling
— `coupling.step()` then records it for you with the correct timing:

```python
coupling.add_volume_balance(inflow_operators=[my_inflow_op])

for t in domain.evolve(yieldstep=dt, finaltime=ft):
    coupling.step(dt)             # exchange + audit

vb = coupling.volume_balance
print(vb.summary())               # text report of the final-step budget
df = vb.to_dataframe()            # the full time series
vb.plot('volume_balance.png')     # components + residuals vs time
```

If you drive the `Coupler` yourself, use {class}`~anuga_drainage.VolumeBalance`
directly — but mind the timing:

```python
from anuga_drainage import VolumeBalance

vb = VolumeBalance(domain, coupling.inlets.values(), coupling.backend,
                   inflow_operators=[my_inflow_op])
prev = None
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    vb.step(t, dt, prev)          # call at the TOP of the loop, previous step
    prev = coupling.coupler.step(dt)
```

```{admonition} Timing matters
:class: warning
`vb.step()` must be called at the **top** of the loop, passing the **previous**
step's `CouplingStep` — which is exactly what `coupling.step()` does internally.
Reading at the bottom misaligns ANUGA's applied volume (applied on the next
evolve) from the SWMM statistics by one step.
```

## Per-inlet breakdown

Passing the `CouplingStep` (as above) also records, per inlet:

- `requested` — `Q_in · dt` asked of the sewer;
- `accepted` — what the sewer actually took;
- `removed` — what ANUGA actually exchanged.

From these, `reject = requested - accepted` (sewer capacity) and
`drying = accepted + removed` (the inlet over-drawn) are localised per inlet —
e.g. it shows that turning the `clamp` off lets a requested draw dry an inlet,
making `R_couple` nonzero.

For outfalls that **return** water to the surface at a specific inlet (as the
`simple_culvert` SWMM example does), pass `outfall_inlet=<index>` so that return
is subtracted from the inlet's `removed` (and shown in its own column). See
[where outfall water goes](coupling.md#where-outfall-water-goes) for the two
fates of outfall water and how each lands in this audit.

## What the audit has shown

- **ANUGA conserves** and **the coupling conserves** to machine precision across
  SWMM and pipedream, simple and real catchments.
- The only real losses are the **1D solvers' own** numerics: SWMM's
  finite-difference scheme, and pipedream's smaller (~0.1–1%) linearised
  continuity / Preissmann-slot residual on surcharged networks. The audit
  attributes these to `R_pipe`, never to the coupling.
