import anuga, numpy, time, os, glob
from anuga import file_function, Polygon_function, read_polygon, create_domain_from_regions, Domain, Inlet_operator
from anuga import distribute, myid, numprocs, finalize, barrier
from anuga import Inlet_operator, Boyd_box_operator, Boyd_pipe_operator
from anuga import Rate_operator
from anuga import Region

import anuga.utilities.spatialInputUtil as su

import numpy as np
import pickle
from pyswmm import Simulation, Nodes, Links
import matplotlib.pyplot as plt

import time
from anuga_drainage.inlet_initialization import initialize_inlets, read_inp_coordinates
from anuga_drainage import Coupler, SwmmBackend, VolumeBalance

time_average = 10 # sec
dt           = 1.0     # yield step
ft           = 100 # final timestep
input_rate   = 0.102


from pyswmm import SystemStats


output_frequency = 1
do_print         = True
do_data_save     = False

basename = 'model/terrain'
inp_name = 'real_example.inp'
outname  = 'domain_swmm'
meshname = 'model/terrain.tsh'

#------------------------------------------------------------------------------
# SETUP COMPUTATIONAL DOMAIN
#------------------------------------------------------------------------------

riverWall_csv_files = glob.glob('model/wall/*.csv') # Make a list of the csv files in BREAKLINES

(riverWalls, riverWall_parameters) = su.readListOfRiverWalls(riverWall_csv_files)

CatchmentDictionary = {'model/kerb/kerb1.csv':0.01, 'model/kerb/kerb2.csv':0.01}
    
bounding_polygon = anuga.read_polygon('model/domain.csv')
interior_regions = anuga.read_polygon_dir(CatchmentDictionary, 'model/kerb')


domain = create_domain_from_regions(bounding_polygon,
    boundary_tags={'inflow': [12], 'bottom': [0,1,2,3,4,5], 'top': [7,8,9,10,11], 'outflow': [6]},
    #boundary_tags=None,
    maximum_triangle_area = 0.1,
    breaklines = riverWalls.values(),
    interior_regions = interior_regions,
    use_cache = False,
    verbose = False)

domain.set_minimum_storable_height(0.0)
domain.riverwallData.create_riverwalls(riverWalls,verbose = False) 
domain.set_name(outname) 
# 
#------------------------------------------------------------------------------
# APPLY MANNING'S ROUGHNESSES
#------------------------------------------------------------------------------

domain.set_quantity('friction', 0.025)
domain.set_quantity('stage', 0)
domain.set_quantity('elevation', filename=basename+'.csv', use_cache=False, verbose=False, alpha=0.99)

#------------------------------------------------------------------------------
# SETUP BOUNDARY CONDITIONS
#------------------------------------------------------------------------------
Br = anuga.Reflective_boundary(domain)  
Bd = anuga.Dirichlet_boundary([0,0,0])

domain.set_boundary({'inflow': Br, 'bottom': Br, 'outflow': Bd, 'top': Br})
 
# ------------------------------------------------------------------------------
# Setup inject water
# ------------------------------------------------------------------------------

input1_anuga_region   = Region(domain, radius=1.0, center=(305694.91,6188013.94))
input1_anuga_inlet_op = Inlet_operator(domain, input1_anuga_region, Q=input_rate) 

sim = Simulation(inp_name)
node_coordinates = read_inp_coordinates(inp_name)


link_volume_0 = 0
for link in Links(sim):
    link_volume_0 += link.volume

node_ids      = [node.nodeid for node in Nodes(sim)]
in_node_ids   = [node.nodeid for node in Nodes(sim) if node.is_junction()]
n_in_nodes    = len(in_node_ids)

### Initialize inlet operators
inlet_area = np.full((n_in_nodes),1.167)
Q_in_0     = n_in_nodes*[0.0]
n_sides    = 6
inlet_operators,inlet_elevation,_,_ = initialize_inlets(domain,sim,node_coordinates,n_sides,inlet_area,Q_in_0,rotation = 0)

inlet_weir_length = 2*np.sqrt(np.pi*inlet_area)


outfall_vol    = 0

times          = []
losses         = []

if do_data_save:
    Q_ins          = []
    conduit_depths = []
    node_heads     = []
    cumulative_inlet_flooding = np.array(n_in_nodes*[0.0])
    cumulative_inlet_flow     = np.array(n_in_nodes*[0.0])

system_routing = SystemStats(sim)


wall_clock_start = time.perf_counter()
sim.start()

print('')
print('')
print('Start ANUGA evolve')

old_inlet_vol = [- node.statistics['lateral_infow_vol'] + node.statistics['flooding_volume'] for node in Nodes(sim) if node.is_junction()]
node_volume    = sum(old_inlet_vol)

# Coupler drives the per-step exchange; inlets ordered to match the SWMM
# junction order (same order as in_node_ids / the backend's heads).
inlets  = [inlet_operators[node_id] for node_id in in_node_ids]
coupler = Coupler(inlets=inlets, beds=inlet_elevation,
                  weir_lengths=inlet_weir_length, manhole_areas=inlet_area,
                  backend=SwmmBackend(sim), time_average=time_average)

# Per-component + per-inlet water-volume audit. No outfall return here (water
# leaving at the outfall exits the system), so outfall_inlet is unset.
vb = VolumeBalance(domain, coupling_inlets=inlets, backend=coupler.backend,
                   inflow_operators=[input1_anuga_inlet_op])

prev_step = None   # previous CouplingStep, for the aligned audit
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    anuga_depths = np.array([inlet_operators[in_id].inlet.get_average_depth() for in_id in in_node_ids])
    vb.step(t, dt, prev_step)   # audit at the top of the loop (aligned reads)

    if domain.yieldstep_counter%output_frequency == 0 and do_print:
        print(f'External flow: {system_routing.routing_stats["external_inflow"]}')
        print('t = ',t)

    # Reset link volume at every iteration and sum volumes
    link_volume = 0
    for link in Links(sim):
        link_volume += link.volume

    if do_data_save:
        conduit_depths.append(np.array([link.depth for link in Links(sim)]))

    inlet_head_swmm   = np.array([node.head for node in Nodes(sim) if node.is_junction()])

    # Compute the coupling flux (smoothed), step SWMM, and feed the realised
    # SWMM flow back to ANUGA (see anuga_drainage.Coupler). inlet_head_swmm is
    # read above only for the saved diagnostics.
    step       = coupler.step(dt)
    Q_in       = step.Q_in
    inlet_flow = step.anuga_flux
    prev_step  = step

    if do_data_save:
        Q_ins.append(Q_in.copy())
        node_heads.append(inlet_head_swmm)

    if domain.yieldstep_counter%output_frequency == 0 and do_print:
        print(f'Q_in = {Q_in}')

    outfall_vol += Links(sim)['Conduit_4'].flow*dt

    sewer_volume         = link_volume + node_volume
    domain_volume        = domain.get_water_volume()
    sewer_volume         = link_volume + node_volume
    boundary_flow        = domain.get_boundary_flux_integral()
    total_volume_correct = t*input_rate + boundary_flow + link_volume_0
    total_volume_real    = domain_volume + sewer_volume + outfall_vol

    loss = total_volume_real - total_volume_correct
    losses.append(loss)

    if do_data_save:
        cumulative_inlet_flooding += np.array([node.flooding for node in Nodes(sim) if node.is_junction()])
        cumulative_inlet_flow     += np.array(inlet_flow)*dt
    times.append(t)


sim.report()
sim.close()
wall_clock_end = time.perf_counter()
# print('\n')
print(f'\nComputation time: {wall_clock_end - wall_clock_start:.2f} seconds')
print(f'Loss = {loss:.2f}m^3 of total {t*input_rate}m^3')

print()
print(vb.summary())
vb.plot('volume_balance.png')


if do_data_save:

    pick = outname + '.dat'

    data = {'times' : times, 'conduit_depths' : conduit_depths, 'node_heads' : node_heads}

    with open(pick, "wb") as f:
        pickle.dump(data, f)
