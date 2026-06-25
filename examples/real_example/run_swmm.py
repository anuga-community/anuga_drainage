"""Real catchment: ANUGA <-> SWMM coupling, hand-built from a DRAINS network.

The hand-built SWMM twin of ``run_from_inp.py`` on the real (DRAINS-exported)
terrain. The 2D domain is built from the terrain/wall/kerb CSVs; the coupling
inlets are created at the SWMM junction coordinates with
``anuga_drainage.initialize_inlets`` (regular polygons of a fixed manhole area),
and the per-step exchange is driven by ``anuga_drainage.Coupler``. Water leaves
the system at the outfall (no outfall return), so that volume is a genuine sink
counted in the loss.

Compare with ``run_from_inp.py``, which builds the inlets and the backend in one
``couple_from_inp`` call.

    python run_swmm.py
"""
import glob
import time

import anuga
from anuga import create_domain_from_regions, Inlet_operator, Region
import anuga.utilities.spatialInputUtil as su
import numpy as np
from pyswmm import Simulation, Nodes, Links

from anuga_drainage.inlet_initialization import initialize_inlets, read_inp_coordinates
from anuga_drainage import Coupler, SwmmBackend, VolumeBalance

# ---- parameters --------------------------------------------------------------
dt = 1.0              # yield/coupling step (whole seconds for pyswmm 2.1.0)
ft = 100              # final time (s)
input_rate = 0.102    # upstream 2D inflow (m^3/s)
time_average = 10.0   # s; smooths the coupling flux
manhole_area = 1.167  # surface area of each inlet coupling region (m^2)
n_sides = 6           # regular-polygon inlet geometry
inp_name = 'real_example.inp'

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
domain.set_name('domain_swmm')

domain.set_quantity('friction', 0.025)
domain.set_quantity('stage', 0)
domain.set_quantity('elevation', filename='model/terrain.csv', use_cache=False,
                    verbose=False, alpha=0.99)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([0, 0, 0])
domain.set_boundary({'inflow': Br, 'bottom': Br, 'outflow': Bd, 'top': Br})

# Upstream inflow into the 2D domain.
input_op = Inlet_operator(domain, Region(domain, radius=1.0,
                          center=(305694.91, 6188013.94)), Q=input_rate)

# ---- 2. Coupling inlets (from the .inp) + SWMM network -----------------------
sim = Simulation(inp_name)
node_coordinates = read_inp_coordinates(inp_name)
in_node_ids = [node.nodeid for node in Nodes(sim) if node.is_junction()]
n_in = len(in_node_ids)

# Build a regular-polygon Inlet_operator at each junction node.
inlet_area = np.full(n_in, manhole_area)
inlet_operators, inlet_elevation, _, _ = initialize_inlets(
    domain, sim, node_coordinates, n_sides, inlet_area, n_in * [0.0], rotation=0)
inlet_weir_length = 2 * np.sqrt(np.pi * inlet_area)
sim.start()

# Inlets ordered to match the SWMM junction order (the backend's head order).
inlets = [inlet_operators[node_id] for node_id in in_node_ids]
coupler = Coupler(inlets=inlets, beds=inlet_elevation,
                  weir_lengths=inlet_weir_length, manhole_areas=inlet_area,
                  backend=SwmmBackend(sim), time_average=time_average, clamp=True)

# ---- 3. Per-component volume-balance audit -----------------------------------
# No outfall return here (water leaving at the outfall exits the system), so
# outfall_inlet is unset and that volume is counted in the loss.
vb = VolumeBalance(domain, coupling_inlets=inlets, backend=coupler.backend,
                   inflow_operators=[input_op])

# ---- 4. Evolve loop ----------------------------------------------------------
wall_clock_start = time.perf_counter()
times, losses = [], []
prev_step = None   # previous CouplingStep, for the aligned (top-of-loop) audit
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    vb.step(t, dt, prev_step)        # audit at the top, with the previous step
    prev_step = coupler.step(dt)     # exchange + SWMM advance + feedback
    times.append(t)
    losses.append(vb.records[-1].loss)
    if domain.yieldstep_counter % 10 == 0:
        domain.print_timestepping_statistics()

# ---- 5. Report ---------------------------------------------------------------
sim.report()
sim.close()
print(f'\nComputation time: {time.perf_counter() - wall_clock_start:.2f} s')
print()
print(vb.summary())
vb.plot('volume_balance.png')
