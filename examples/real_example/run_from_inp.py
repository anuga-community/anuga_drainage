"""Real catchment: ANUGA <-> stormwater coupling from a single SWMM ``.inp``.

The living example of ``couple_from_inp`` on the real (DRAINS-exported) network.
Compare with ``run_swmm.py`` / ``run_pipedream.py``: the sewer network and the
coupling inlets are built straight from ``real_example.inp`` (for either
backend) instead of being hand-assembled (``initialize_inlets`` + a hand-coded
pipedream superjunction/superlink table). The only model-specific code is the
ANUGA terrain domain, the upstream inflow, and the evolve loop.

    python run_from_inp.py [swmm|pipedream]      # default: swmm
"""
import glob
import sys

import numpy as np
import anuga
from anuga import create_domain_from_regions, Inlet_operator, Region
import anuga.utilities.spatialInputUtil as su

from anuga_drainage import couple_from_inp

backend = sys.argv[1] if len(sys.argv) > 1 else 'swmm'
dt, ft, input_rate = 1.0, 100.0, 0.102
inp_name = 'real_example.inp'

#------------------------------------------------------------------------------
# ANUGA terrain domain (unchanged from run_swmm.py)
#------------------------------------------------------------------------------
riverWall_csv_files = glob.glob('model/wall/*.csv')
riverWalls, _ = su.readListOfRiverWalls(riverWall_csv_files)
CatchmentDictionary = {'model/kerb/kerb1.csv': 0.01, 'model/kerb/kerb2.csv': 0.01}
bounding_polygon = anuga.read_polygon('model/domain.csv')
interior_regions = anuga.read_polygon_dir(CatchmentDictionary, 'model/kerb')

domain = create_domain_from_regions(
    bounding_polygon,
    boundary_tags={'inflow': [12], 'bottom': [0, 1, 2, 3, 4, 5],
                   'top': [7, 8, 9, 10, 11], 'outflow': [6]},
    maximum_triangle_area=0.1,
    breaklines=riverWalls.values(),
    interior_regions=interior_regions,
    use_cache=False, verbose=False)
domain.set_minimum_storable_height(0.0)
domain.riverwallData.create_riverwalls(riverWalls, verbose=False)
domain.set_name(f'run_from_inp_{backend}')

domain.set_quantity('friction', 0.025)
domain.set_quantity('stage', 0)
domain.set_quantity('elevation', filename='model/terrain.csv', use_cache=False,
                    verbose=False, alpha=0.99)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([0, 0, 0])
domain.set_boundary({'inflow': Br, 'bottom': Br, 'outflow': Bd, 'top': Br})

# Upstream inflow into the 2D domain.
input1_op = Inlet_operator(domain, Region(domain, radius=1.0,
                           center=(305694.91, 6188013.94)), Q=input_rate)

#------------------------------------------------------------------------------
# Sewer network + coupling inlets, straight from the .inp
#------------------------------------------------------------------------------
coupling = couple_from_inp(domain, inp_name, backend=backend,
                           manhole_area=1.167, n_sides=6,
                           time_average=10.0, clamp=True)
print(f'Coupled {len(coupling.inlets)} junctions from {inp_name}: '
      f'{list(coupling.inlets)}  (backend={backend})')

coupling.add_volume_balance(inflow_operators=[input1_op])

#------------------------------------------------------------------------------
# Evolve loop — coupling.step() runs the exchange and the volume audit
#------------------------------------------------------------------------------
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    coupling.step(dt)
    if domain.yieldstep_counter % 10 == 0:
        domain.print_timestepping_statistics()

print()
print(coupling.volume_balance.summary())
coupling.close()
