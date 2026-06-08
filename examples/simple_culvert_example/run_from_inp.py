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

The VolumeBalance audit is the point: for *both* backends R_anuga and R_couple
stay ~1e-13 (ANUGA conserves; the coupling conserves) and the loss is isolated
to R_pipe — SWMM's finite-difference loss (~0.01 m^3) and pipedream's ~0.3%
surcharge residual (~0.6 m^3). Two things make the pipedream comparison clean:
the culvert is RECT_CLOSED (a box culvert surcharges via pipedream's Preissmann
slot, conserving), and inp_to_pipedream leaves the superjunctions uncapped
(max_depth=inf) so surcharge is pushed back to the 2D surface by the coupling
rather than silently lost at the node cap (pipedream has no flooding model, so
honouring the .inp MaxDepth would drop the surcharge).
"""
import sys

import numpy as np
import anuga

from anuga_drainage import couple_from_inp, VolumeBalance

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
coupling = couple_from_inp(domain, './swmm_input_short.inp', backend=backend,
                           manhole_area=16.0, time_average=10.0, clamp=True)
print(f'Coupled {len(coupling.inlets)} junctions from the .inp: '
      f'{list(coupling.inlets)}  (backend={backend})')

vb = VolumeBalance(domain, coupling_inlets=list(coupling.inlets.values()),
                   backend=coupling.backend, inflow_operators=[inflow_op])

#------------------------------------------------------------------------------
# Evolve loop — audit at the top with the previous step, then exchange
#------------------------------------------------------------------------------
prev_step = None
for t in domain.evolve(yieldstep=dt, outputstep=10.0, finaltime=400.0):
    vb.step(t, dt, prev_step)
    prev_step = coupling.coupler.step(dt)
    if domain.yieldstep_counter % 50 == 0:
        domain.print_timestepping_statistics()

print()
print(vb.summary())
vb.plot('volume_balance.png')

if backend == 'swmm':
    coupling.handle.close()
