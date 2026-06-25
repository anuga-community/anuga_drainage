"""Real catchment: ANUGA <-> pipedream coupling, hand-built.

The hand-built pipedream twin of ``run_swmm.py`` / ``run_from_inp.py`` on the
real (DRAINS-exported) terrain. The 1D sewer is a hand-coded pipedream
``SuperLink`` of five superjunctions (four inlets plus the downstream outlet)
joined by four circular conduits; the per-step exchange is driven by
``anuga_drainage.Coupler`` (calculate_Q + smoothing). pipedream is
finite-volume, so the pipe budget closes (R_pipe ~0); the loss is reported by
the VolumeBalance summary.

Compare with ``run_from_inp.py``, which builds the pipedream network from the
``.inp`` instead of the hand-coded table below.

    python run_pipedream.py
"""
import glob

import anuga
from anuga import create_domain_from_regions, Inlet_operator, Region
import anuga.utilities.spatialInputUtil as su
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipedream_solver.hydraulics import SuperLink

from anuga_drainage import Coupler, PipedreamBackend, VolumeBalance

# ---- parameters --------------------------------------------------------------
dt = 1.0              # yield/coupling step (s)
ft = 250.0            # final time (s)
input_rate = 0.05     # upstream 2D inflow (m^3/s)
time_average = 10.0   # s; smooths the coupling flux (matches the run_swmm.py twin)
radius = 0.25         # inlet pit radius (m): weir crest = perimeter, area = pit area
visualise = False     # pop up head/loss plots at the end

# ---- 1. ANUGA terrain domain -------------------------------------------------
riverWalls, _ = su.readListOfRiverWalls(glob.glob('model/wall/*.csv'))
CatchmentDictionary = {'model/kerb/kerb1.csv': 0.01, 'model/kerb/kerb2.csv': 0.01}
bounding_polygon = anuga.read_polygon('model/domain.csv')
interior_regions = anuga.read_polygon_dir(CatchmentDictionary, 'model/kerb')

domain = create_domain_from_regions(
    bounding_polygon,
    boundary_tags={'inflow': [12], 'bottom': [0, 1, 2, 3, 4, 5],
                   'top': [7, 8, 9, 10, 11], 'outflow': [6]},
    maximum_triangle_area=0.1, breaklines=riverWalls.values(),
    interior_regions=interior_regions, use_cache=False, verbose=False)
domain.set_minimum_storable_height(0.0)
domain.riverwallData.create_riverwalls(riverWalls, verbose=False)
domain.set_name('domain_pipedream')

domain.set_quantity('friction', 0.025)
domain.set_quantity('stage', 0)
domain.set_quantity('elevation', filename='model/terrain.csv', alpha=0.99)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([0, 0, 0])
domain.set_boundary({'inflow': Br, 'bottom': Br, 'outflow': Bd, 'top': Br})

# Upstream inflow into the 2D domain.
input_op = Inlet_operator(domain, Region(domain, radius=1.0,
                          center=(305694.91, 6188013.94)), Q=input_rate)

# ---- 2. Coupling inlets + pipedream network ----------------------------------
# Four inlet pits plus the downstream outlet, each a circular surface region.
centres = [(305698.51, 6188004.63), (305703.39, 6187999.00),
           (305713.18, 6188002.02), (305727.24, 6188004.61),
           (305736.68, 6188026.65)]   # last is the outlet (Outfall coords)
inlets = [Inlet_operator(domain, Region(domain, radius=radius, center=c),
                         Q=0.0, zero_velocity=True) for c in centres]
beds = np.array([op.inlet.get_average_elevation() for op in inlets])

# pipedream network: 5 superjunctions (the inlets/outlet above) chained by 4
# circular conduits. A_s = pit area (1 m^2) + 1.2 m lintel (0.1 x 1.2 m).
superjunctions = pd.DataFrame({
    'name': [0, 1, 2, 3, 4], 'id': [0, 1, 2, 3, 4],
    'z_inv': [37.5, 36.4, 34.5, 32.0, 32.0], 'h_0': 5 * [1e-5],
    'bc': 5 * [False], 'storage': 5 * ['functional'],
    'a': 5 * [0.], 'b': 5 * [0.], 'c': 5 * [1.], 'max_depth': 5 * [np.inf],
    'map_x': 5 * [0], 'map_y': 5 * [0]})
superlinks = pd.DataFrame({
    'name': [0, 1, 2, 3], 'id': [0, 1, 2, 3],
    'sj_0': [0, 1, 2, 3], 'sj_1': [1, 2, 3, 4],
    'in_offset': 4 * [0.], 'out_offset': 4 * [0.], 'dx': [7.4, 10.3, 14.3, 24.0],
    'n': 4 * [0.013], 'shape': 4 * ['circular'],
    'g1': [0.375, 0.375, 0.375, 0.45], 'g2': 4 * [0.], 'g3': 4 * [0.], 'g4': 4 * [0.],
    'Q_0': 4 * [0.], 'h_0': 4 * [1e-5], 'ctrl': 4 * [False],
    'A_s': 4 * [1.12], 'A_c': 4 * [0.], 'C': 4 * [0.]})
superlink = SuperLink(superlinks, superjunctions, internal_links=20)

coupler = Coupler(inlets=inlets, beds=beds,
                  weir_lengths=np.full(5, 2 * np.pi * radius),
                  manhole_areas=np.full(5, np.pi * radius ** 2),
                  backend=PipedreamBackend(superlink),
                  time_average=time_average, clamp=True)

# ---- 3. Per-component volume-balance audit -----------------------------------
vb = VolumeBalance(domain, coupling_inlets=inlets, backend=coupler.backend,
                   inflow_operators=[input_op])

# ---- 4. Evolve loop ----------------------------------------------------------
domain.output_frequency = 100
times, heads, losses = [], [], []
prev_step = None   # previous CouplingStep, for the aligned (top-of-loop) audit
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    vb.step(t, dt, prev_step)        # audit at the top, with the previous step
    prev_step = coupler.step(dt)     # exchange + pipedream advance + feedback
    times.append(t)
    heads.append(coupler.backend.get_heads().copy())
    losses.append(vb.records[-1].loss)
    if domain.yieldstep_counter % domain.output_frequency == 0:
        domain.print_timestepping_statistics()

# ---- 5. Report ---------------------------------------------------------------
print()
print(vb.summary())
vb.plot('volume_balance.png')

if visualise:
    times, heads = np.array(times), np.vstack(heads)
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(['Inlet 1', 'Inlet 2', 'Inlet 3', 'Inlet 4', 'Outlet']):
        plt.plot(times, heads[:, i], label=name)
    plt.xlabel('time (s)'); plt.ylabel('head (m)')
    plt.legend(); plt.title('Head at junctions')
    plt.tight_layout(); plt.savefig('Figure_heads.png'); plt.show()
