"""Y-shaped pipe network: ANUGA <-> pipedream coupling.

Two inlet pits at the same elevation feed, via a Y-junction, a single lower
outlet pit. Water is injected onto the 2D surface near one inlet and finds its
way through the surface and the buried pipes to the outlet. The 1D sewer is a
hand-built pipedream ``SuperLink`` (three superjunctions, two circular
conduits) and the per-step exchange is driven by ``anuga_drainage.Coupler``
(calculate_Q + smoothing) — the same pattern as the other examples.

    python run_pipedream.py
"""
import anuga
from anuga import Inlet_operator, Region, rectangular_cross_domain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipedream_solver.hydraulics import SuperLink

from anuga_drainage import Coupler, PipedreamBackend, VolumeBalance

# ---- parameters --------------------------------------------------------------
dt = 1.0         # yield/coupling step (s)
ft = 100.0       # final time (s)
inflow_Q = 1.0   # upstream 2D inflow (m^3/s)
radius = 1.0     # pit radius (m): weir crest = perimeter, manhole area = pit area
visualise = False     # pop up head/loss plots at the end

# ---- 1. ANUGA domain ---------------------------------------------------------
length, width, dx = 20., 6., 0.2
domain = rectangular_cross_domain(int(length / dx), int(width / dx),
                                  len1=length, len2=width)
domain.set_name('Y_shape_pipedream')


def topography(x, y):
    """Two upstream pools split by a wall, draining to a lower outlet pool."""
    z = 0 * x - 5
    z[x < 10] = -3                                 # upstream pools
    z[(10 < x) & (x < 15)] = 3                     # wall between pools and outlet
    z[(x < 10) & (2.5 < y) & (y < 3.5)] = 3        # wall splitting the two inlets
    return z


domain.set_quantity('elevation', topography, location='centroids')
domain.set_quantity('friction', 0.01)
domain.set_quantity('stage', expression='elevation', location='centroids')  # dry

Br = anuga.Reflective_boundary(domain)
domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

# Upstream inflow onto the 2D surface (only the first inlet pool is fed).
inflow_op = Inlet_operator(domain, Region(domain, radius=1.0, center=(2., 1.)),
                           Q=inflow_Q)

# ---- 2. Coupling inlets + pipedream network ----------------------------------
# Two inlet pits and one outlet pit, each a circular surface region.
centres = [(7., 1.), (7., 5.), (17., 3.)]   # inlet 1, inlet 2, outlet
inlets = [Inlet_operator(domain, Region(domain, radius=radius, center=c), Q=0.0)
          for c in centres]
beds = np.array([op.inlet.get_average_elevation() for op in inlets])

# pipedream network: 3 superjunctions (two inlets + lower outlet) in a Y.
superjunctions = pd.DataFrame({
    'name': [0, 1, 2], 'id': [0, 1, 2], 'z_inv': [-4., -4., -6.], 'h_0': 3 * [1e-5],
    'bc': 3 * [False], 'storage': 3 * ['functional'],
    'a': 3 * [0.], 'b': 3 * [0.], 'c': 3 * [1.], 'max_depth': 3 * [np.inf],
    'map_x': 3 * [0], 'map_y': 3 * [0]})
superlinks = pd.DataFrame({
    'name': [0, 1], 'id': [0, 1], 'sj_0': [0, 1], 'sj_1': [1, 2],
    'in_offset': 2 * [0.], 'out_offset': 2 * [0.], 'dx': [4., 11.], 'n': 2 * [0.01],
    'shape': 2 * ['circular'], 'g1': [0.5, 0.25], 'g2': 2 * [0.], 'g3': 2 * [0.],
    'g4': 2 * [0.], 'Q_0': 2 * [0.], 'h_0': 2 * [1e-5], 'ctrl': 2 * [False],
    'A_s': 2 * [0.25], 'A_c': 2 * [0.], 'C': 2 * [0.]})
superlink = SuperLink(superlinks, superjunctions, internal_links=10)

# pipedream's semi-implicit solver is unstable if stepped once at the 1 s
# coupling dt (the full-perimeter weir drives a large surcharge), so refine its
# internal step to 0.05 s while keeping the exchange/yield at 1 s.
coupler = Coupler(inlets=inlets, beds=beds,
                  weir_lengths=np.full(3, 2 * np.pi * radius),
                  manhole_areas=np.full(3, np.pi * radius ** 2),
                  backend=PipedreamBackend(superlink, max_step=0.05),
                  time_average=10.0, clamp=True)

# ---- 3. Per-component volume-balance audit -----------------------------------
vb = VolumeBalance(domain, coupling_inlets=inlets, backend=coupler.backend,
                   inflow_operators=[inflow_op])

# ---- 4. Evolve loop ----------------------------------------------------------
times, heads = [], []
prev_step = None   # previous CouplingStep, for the aligned (top-of-loop) audit
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    vb.step(t, dt, prev_step)        # audit at the top, with the previous step
    prev_step = coupler.step(dt)     # exchange + pipedream advance + feedback
    times.append(t)
    heads.append(coupler.backend.get_heads().copy())
    if domain.yieldstep_counter % domain.output_frequency == 0:
        domain.print_timestepping_statistics()

# ---- 5. Report ---------------------------------------------------------------
print()
print(vb.summary())
vb.plot('volume_balance.png')

if visualise:
    times, heads = np.array(times), np.vstack(heads)
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(['Inlet 1', 'Inlet 2', 'Outlet']):
        plt.plot(times, heads[:, i], label=name)
    plt.xlabel('time (s)'); plt.ylabel('head (m)')
    plt.legend(); plt.title('Head at junctions')
    plt.tight_layout(); plt.savefig('Figure_heads.png'); plt.show()
