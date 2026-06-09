"""ANUGA <-> stormwater coupling driven entirely by a single SWMM ``.inp``.

This is the "living example" of ``anuga_drainage.couple_from_inp``: the sewer
network *and* the ANUGA coupling inlets are built from ``swmm_input_short.inp``
for whichever backend you pick, so the only model-specific code here is the
ANUGA domain, the upstream inflow, and the evolve loop. Contrast with
``run_swmm_short.py`` / ``run_pipedream.py``, which hand-build the inlets (and,
for pipedream, the whole superjunction/superlink network).

    python run_from_inp.py [swmm|pipedream]      # default: swmm

Note: couple_from_inp couples the [JUNCTIONS] (Inlet, Outlet) and treats the
Outfall as a free-drainage boundary, so here water *leaves* at the outfall
(rather than being returned to the surface as in run_swmm_short.py).

The VolumeBalance audit is the point: for *both* backends R_anuga stays ~1e-13
(ANUGA conserves) and R_couple is tiny (the coupling conserves) — the loss is
isolated to R_pipe: SWMM's finite-difference loss (~0.08 m^3) and pipedream's
own linearisation/surcharge residual (~3 m^3 at a 0.01 s inner step; shrinks
toward a ~3 m^3 floor as the inner step tightens). Three things make the
pipedream comparison work: the culvert is RECT_CLOSED (a box culvert surcharges
via pipedream's Preissmann slot, conserving); inp_to_pipedream leaves the
superjunctions uncapped (max_depth=inf) so surcharge is pushed back to the 2D
surface by the coupling rather than silently lost at the node cap (pipedream has
no flooding model, so honouring the .inp MaxDepth would drop the surcharge); and
pipedream is sub-stepped internally (pipedream_max_step=0.01) — its semi-implicit
solver is unstable at the 1 s coupling step, so the exchange stays at 1 s while
the 1D integration is refined (the more internal_links, the smaller this must be).
"""
import sys

import numpy as np
import anuga

from anuga_drainage import couple_from_inp

backend = sys.argv[1] if len(sys.argv) > 1 else 'swmm'

#------------------------------------------------------------------------------
# ANUGA domain (the only model-specific 2D setup)
#------------------------------------------------------------------------------
rf = 20
domain = anuga.rectangular_cross_domain(3 * rf, rf, len1=60, len2=20)
domain.set_minimum_storable_height(0.0001)
domain.set_name(f'run_from_inp_{backend}')


def topography(x, y):
    z = 5 * np.ones_like(x)
    channel = np.logical_and(y > 5, y < 15)
    z = np.where(np.logical_and(channel, x < 10), x / 300, z)
    z = np.where(np.logical_and(channel, x > 20), x / 300, z)
    return z


domain.set_quantity('elevation', topography, location='centroids')
domain.set_quantity('friction', 0.035)

# Upstream inflow into the 2D domain.
input_Q = 1.0
inflow_op = anuga.Inlet_operator(domain, [[59.0, 5.0], [59.0, 15.0]], input_Q)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([-1.0, 0, 0])
domain.set_boundary({'left': Bd, 'bottom': Br, 'top': Br, 'right': Br})

#------------------------------------------------------------------------------
# Sewer network + coupling inlets, straight from the .inp
#------------------------------------------------------------------------------
dt = 1.0
# The .inp only carries a point per junction, so the auto inlet polygon can be
# too narrow for a channel — flow then overtops the embankment instead of being
# captured into the culvert. Give the inlet/outlet a footprint that spans the
# channel (y 6..14), matching the hand-coded run_swmm_short.py regions.
# pipedream's semi-implicit solver is only stable at a small internal step, so
# keep the 1 s coupling/yield step but subdivide pipedream into 0.01 s sub-steps
# (cf. the hand-built run_pipedream.py: internal_links=6 @ dt=0.05; a smaller
# step shrinks pipedream's own linearisation R_pipe ~linearly). These two kwargs
# are ignored by the swmm backend, which routes at its own .inp ROUTING_STEP.
coupling = couple_from_inp(domain, './swmm_input_short.inp', backend=backend,
                           inlet_polygons={
                               'Inlet':  [[20, 6], [22, 6], [22, 14], [20, 14]],
                               'Outlet': [[8, 6], [10, 6], [10, 14], [8, 14]],
                           },
                           time_average=10.0, clamp=True,
                           internal_links=6, pipedream_max_step=0.01)
print(f'Coupled {len(coupling.inlets)} junctions from the .inp: '
      f'{list(coupling.inlets)}  (backend={backend})')

coupling.add_volume_balance(inflow_operators=[inflow_op])

#------------------------------------------------------------------------------
# Evolve loop — coupling.step() runs the exchange and the volume audit
#------------------------------------------------------------------------------
for t in domain.evolve(yieldstep=dt, outputstep=10.0, finaltime=400.0):
    coupling.step(dt)
    if domain.yieldstep_counter % 50 == 0:
        domain.print_timestepping_statistics()

print()
print(coupling.volume_balance.summary())
coupling.volume_balance.plot('volume_balance.png')
coupling.close()
