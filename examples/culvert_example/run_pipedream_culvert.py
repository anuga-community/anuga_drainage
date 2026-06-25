"""Culvert on meshed terrain: ANUGA <-> pipedream coupling, hand-built.

A single culvert (one pipedream conduit between two pits) on a real meshed
terrain loaded from ``terrain.csv``. The per-step exchange is driven by
``anuga_drainage.Coupler``; the 1D sewer is a hand-built two-superjunction
``SuperLink``. This example historically uses weir/orifice coefficients
cw = co = 1.0 (vs the package default 0.67), preserved here.

    python run_pipedream_culvert.py
"""
import anuga
from anuga import create_domain_from_regions, Inlet_operator, Region
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipedream_solver.hydraulics import SuperLink

from anuga_drainage import Coupler, PipedreamBackend, VolumeBalance

# ---- parameters --------------------------------------------------------------
dt = 0.1         # yield/coupling step (s)
out_dt = 1.0     # sww output step
ft = 800         # final time (s)
radius = 0.5     # pit radius (m): weir crest = perimeter, manhole area = pit area
visualise = False     # pop up head/loss plots at the end

# Terrain extent (model coordinates).
W, E, S, N = 296600., 296730., 6179960., 6180070.

# ---- 1. ANUGA domain ---------------------------------------------------------
domain = create_domain_from_regions(
    [[W, S], [E, S], [E, N], [W, N]],
    boundary_tags={'south': [0], 'east': [1], 'north': [2], 'west': [3]},
    maximum_triangle_area=1.0, use_cache=False, verbose=False)
domain.set_minimum_storable_height(0.0001)
domain.set_name('domain_pipedream_culvert')

domain.set_quantity('friction', 0.035)
domain.set_quantity('elevation', filename='terrain.csv', use_cache=True,
                    verbose=False, alpha=0.1)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([0, 0, 0])
domain.set_boundary({'west': Br, 'south': Br, 'north': Bd, 'east': Br})

# Upstream inflow into the 2D domain.
inflow_op = Inlet_operator(domain, [[296669.258, 6179974.191],
                                    [296677.321, 6179976.449]], 1.0)

# ---- 2. Coupling inlets + pipedream network ----------------------------------
inlet_region = Region(domain, radius=radius, center=(296660.390, 6180017.186))
outlet_region = Region(domain, radius=radius, center=(296649.976, 6180038.872))
inlet_op = Inlet_operator(domain, inlet_region, Q=0.0, zero_velocity=True)
outlet_op = Inlet_operator(domain, outlet_region, Q=0.0, zero_velocity=False)
beds = np.array([inlet_op.inlet.get_average_elevation(),
                 outlet_op.inlet.get_average_elevation()])

# pipedream network: two superjunctions joined by one circular culvert.
superjunctions = pd.DataFrame({
    'name': [0, 1], 'id': [0, 1], 'z_inv': [12.2, 12.2], 'h_0': 2 * [1e-5],
    'bc': 2 * [False], 'storage': 2 * ['functional'],
    'a': 2 * [0.], 'b': 2 * [1.], 'c': 2 * [10.], 'max_depth': 2 * [np.inf],
    'map_x': 2 * [0], 'map_y': 2 * [0]})
superlinks = pd.DataFrame({
    'name': [0], 'id': [0], 'sj_0': [0], 'sj_1': [1],
    'in_offset': [0.], 'out_offset': [0.], 'dx': [24], 'n': [0.013],
    'shape': ['circular'], 'g1': [0.5], 'g2': [0.], 'g3': [0.], 'g4': [0.],
    'Q_0': [0.], 'h_0': [1e-5], 'ctrl': [False], 'A_s': [0.], 'A_c': [0.], 'C': [0.]})
superlink = SuperLink(superlinks, superjunctions, internal_links=20)

# This example historically used cw = co = 1.0 (vs the package default 0.67).
coupler = Coupler(inlets=[inlet_op, outlet_op], beds=beds,
                  weir_lengths=np.full(2, 2 * np.pi * radius),
                  manhole_areas=np.full(2, np.pi * radius ** 2),
                  backend=PipedreamBackend(superlink), cw=1.0, co=1.0)

# ---- 3. Per-component volume-balance audit -----------------------------------
vb = VolumeBalance(domain, coupling_inlets=[inlet_op, outlet_op],
                   backend=coupler.backend, inflow_operators=[inflow_op])

# ---- 4. Evolve loop ----------------------------------------------------------
times, heads = [], []
prev_step = None   # previous CouplingStep, for the aligned (top-of-loop) audit
for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):
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
    plt.plot(times, heads[:, 0], label='Inlet')
    plt.plot(times, heads[:, 1], label='Outlet')
    plt.xlabel('time (s)'); plt.ylabel('head (m)')
    plt.legend(); plt.title('Head at junctions')
    plt.tight_layout(); plt.savefig('Figure_heads.png'); plt.show()
