"""
Simple example of using swmm to model a culvert

We are coupling via weir and orifice equation 
(checkout coupling module)

To control oscilations we seem to need small routing_step
set in swmm_input.inp and smoothing of the calculated
Q. At present step to 10secs"""


#------------------------------------------------------------------------------
print('ABOUT to Start Simulation: IMPORT NECESSARY MODULES')
#------------------------------------------------------------------------------

import anuga
import numpy as np

#------------------------------------------------------------------------------
print('SETUP FILENAMES, MODEL DOMAIN and VARIABLES')
#------------------------------------------------------------------------------

basename = 'simple_culvert'
outname =  'anuga_swmm_short_simple_culvert'

rf = 20  # refinement factor for domain, if too coarse the inlets will overlap the wall

dt = 0.2     # yield step  (note that this can be larger than routing step set in the .inp file)
out_dt = 1.0 # output step
ft = 400     # final timestep

# slow the response of the coupling calculation
time_average = 10.0 # sec

verbose   = False
visualise = False

#------------------------------------------------------------------------------
print('SETUP COMPUTATIONAL DOMAIN')
#------------------------------------------------------------------------------

domain = anuga.rectangular_cross_domain(3*rf, rf, len1=60, len2=20)

domain.set_minimum_storable_height(0.0001) 
domain.set_name(outname) 
print (domain.statistics())


#------------------------------------------------------------------------------
print('SETUP ELEVATION FUNCTION')
#------------------------------------------------------------------------------
def topography(x,y):

    z = 5*np.ones_like(x)

    channel = np.logical_and(y>5,y<15)

    z = np.where(np.logical_and(channel,x<10), x/300, z)
    z = np.where(np.logical_and(channel,x>20), x/300, z)

    return z

domain.set_quantity('elevation', topography, location='centroids')

#------------------------------------------------------------------------------
print("APPLY MANNING'S ROUGHNESSES")
#------------------------------------------------------------------------------

domain.set_quantity('friction', 0.035)

#------------------------------------------------------------------------------
print('SETUP ANUGA INFLOW Inlet_operator')
#------------------------------------------------------------------------------

input_Q = 1.0
line=[[59.0, 5.0],[59.0, 15.0]]
inflow_anuga_inlet_op = anuga.Inlet_operator(domain, line, input_Q)

#------------------------------------------------------------------------------
print('SETUP BOUNDARY CONDITIONS')
#------------------------------------------------------------------------------

print ('Available boundary tags', domain.get_boundary_tags())

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([-1.0,0,0])

domain.set_boundary({'left': Bd, 'bottom': Br, 'top': Br, 'right': Br})

#------------------------------------------------------------------------------
print('SETUP ANUGA Inlet_operators to support coupling with stormwater system')
#------------------------------------------------------------------------------


# Make sure to set the culvert width in the swmm_simple_short_culvert.inp. For full channel set
# cw = 8
# For small inlets in centre of channel set
# cw = 2.0
# In the future all this info will be extracted from the appropriate swmm input file. 
# QUESTION SR: Does the swmm inp file have enough info to create inlet regions?


cw = 8

inlet_polygon = [[20.0, 10-cw/2], [22.0, 10-cw/2], [22.0, 10+cw/2], [20.0, 10+cw/2]]
inlet1_anuga_region = anuga.Region(domain, polygon=inlet_polygon)

outlet_polygon = [[8.0, 10-cw/2], [10.0, 10-cw/2], [10.0, 10+cw/2], [8.0, 10+cw/2]]
outlet_anuga_region = anuga.Region(domain, polygon=outlet_polygon)

outfall_polygon = [[1.0, 10-cw/2], [2.0, 10-cw/2], [2.0, 10+cw/2], [1.0, 10+cw/2]]
outfall_anuga_region = anuga.Region(domain, polygon=outfall_polygon)

anuga_length_weirs = np.array([2*cw, 2*cw])
anuga_area_manholes = np.array([cw*2, cw*2])


# now setup anuga Inlet_operators to remove or add water from anuga domain.
inlet1_anuga_inlet_op = anuga.Inlet_operator(domain, inlet1_anuga_region, Q=0.0, zero_velocity=False)
outlet_anuga_inlet_op = anuga.Inlet_operator(domain, outlet_anuga_region, Q=0.0, zero_velocity=False)
#outfall_anuga_inlet_op = anuga.Inlet_operator(domain, outfall_anuga_region, Q=0.0, zero_velocity=False)

anuga_beds = np.array([inlet1_anuga_inlet_op.inlet.get_average_elevation(),
                       outlet_anuga_inlet_op.inlet.get_average_elevation()])

print(anuga_beds)


#------------------------------------------------------------------------------
print('SETUP SWMM')
#------------------------------------------------------------------------------

print('Setup swmm simulation')
from pyswmm import Simulation, Nodes, Links
import matplotlib.pyplot as plt
import pandas as pd

sim = Simulation('./swmm_input_short_culvert.inp')
sim.start()

swmm_inlet = Nodes(sim)['Inlet']
swmm_outlet = Nodes(sim)['Outlet']
swmm_outfall = Nodes(sim)['Outfall']
swmm_culvert = Links(sim)['Culvert']
swmm_outpipe = Links(sim)['Outpipe']

nodes = [swmm_inlet, swmm_outlet, swmm_outfall]
links = [swmm_culvert, swmm_outpipe]

swmm_inlet.statistics

link_volume_0 = swmm_culvert.volume + swmm_outpipe.volume

#--------------------------------------------------------------------------
print('Setup storage for output')
#--------------------------------------------------------------------------
H_js = []
losses = []

Q_iks =[]
Q_uks =[]
Q_dks =[]
time_series = []
anuga_ws = []
Q_ins = []


from coupling import calculate_Q

#---------------------------------------------------------------------------
print('Average Q calculation')
#---------------------------------------------------------------------------

Q_in_old = np.array([0.0, 0.0])

cumulative_inlet_flooding = 0.0
cumulative_outlet_flooding = 0.0
cumulative_inlet_flow = 0.0
cumulative_outlet_flow = 0.0

old_outlet_vol = 0.0
old_inlet_vol = 0.0

#---------------------------------------------------------------------------
print('Start Evolve')
#---------------------------------------------------------------------------

for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):
    #print('\n')
    print_out = domain.yieldstep_counter%domain.output_frequency == 0
    #print_out = True

    if print_out:
        domain.print_timestepping_statistics()

    anuga_depths = np.array([inlet1_anuga_inlet_op.inlet.get_average_depth(),
                             outlet_anuga_inlet_op.inlet.get_average_depth()])
    
    anuga_stages = np.array([inlet1_anuga_inlet_op.inlet.get_average_stage(),
                             outlet_anuga_inlet_op.inlet.get_average_stage()])

    anuga_volumes = np.array([inlet1_anuga_inlet_op.inlet.get_total_water_volume(),
                             outlet_anuga_inlet_op.inlet.get_total_water_volume()])


    # Compute the water volumes in the swmm model
    link_volume = swmm_culvert.volume + swmm_outpipe.volume
    node_volume = 0.0
    sewer_volume = link_volume + node_volume

    # Compute anuga water volumes and boundary fluxes
    boundary_flux = domain.get_boundary_flux_integral()
    domain_volume = domain.get_water_volume()

    # Calculate correct and real volumes
    total_volume_correct = t*input_Q + boundary_flux  + link_volume_0
    total_volume_real = domain_volume + sewer_volume

    loss = total_volume_real - total_volume_correct


    # Append data for later plots
    time_series.append(t)
    losses.append(loss)
    anuga_ws.append(anuga_stages.copy())
    

    # setup some aliases
    inlet_head  = swmm_inlet.head
    outlet_head = swmm_outlet.head
    outfall_head = swmm_outfall.head

    inlet_invert = swmm_inlet.invert_elevation
    outlet_invert = swmm_outlet.invert_elevation
    outfall_invert = swmm_outfall.invert_elevation


    if print_out:    
        print('    swmm/anuga time   :', sim.current_time, '/', t)
        print('    Loss              :', loss)
        print('    TV correct        :', total_volume_correct)
        print('    domain volume     :', domain.get_water_volume())
        print('    boundary flux     :', boundary_flux)
        print('    node_volume       :', node_volume)
        print('    sewer_volume      :', sewer_volume)
        print('    anuga_depths      :', anuga_depths)
        print('    anuga_beds        :', anuga_beds)
        print('    anuga_stages      :', anuga_stages)
        print('    anuga_volumes     :', anuga_volumes)        

        for node in nodes:
            print('   ', node.nodeid,' head         :', node.head)
            print('   ', node.nodeid,' invert elev  :', node.invert_elevation)
            print('   ', node.nodeid,' flooding     :', node.flooding)
            print('   ', node.nodeid,' depth        :', node.depth)
            print('   ', node.nodeid,' volume       :', node.volume)
            print('   ', node.nodeid,' surcharge    :', node.surcharge_depth)
            print('   ', node.nodeid,' lat inflow   :', node.lateral_inflow)
            print('   ', node.nodeid,' tot inflow   :', node.total_inflow)
            print('   ', node.nodeid,' tot outflow  :', node.total_outflow)
            print('   ', node.nodeid,' losses       :', node.losses)
            
        for link in links:
            print('   ', link.linkid,' flow         :', link.flow)
            print('   ', link.linkid,' volume       :', link.volume)



    cumulative_inlet_flooding += swmm_inlet.flooding*dt 
    cumulative_outlet_flooding += swmm_outlet.flooding*dt

    # Calculate the coupling flux and smooth to response
    node_heads = np.array([inlet_head, outlet_head])
    
    Q_in = calculate_Q(node_heads, anuga_depths, anuga_beds, anuga_length_weirs, anuga_area_manholes)

    Q_in = ((time_average - dt)*Q_in_old + dt*Q_in)/time_average


    # if removing water from anuga, try to may sure there is enough water in the anuga model to allow this. 
    safety_factor = 1.0
    Q_limit = safety_factor*anuga_volumes/dt

    Q_in = np.where(Q_in > 0, np.minimum(Q_in, Q_limit), Q_in)

    Q_in_old = Q_in

    Q_ins.append(Q_in.copy())

    if print_out:
        print('    Calculated Q     ', Q_in[0], Q_in[1]) 

    # Run SWMM for a time of dt sewer using the calculated coupling fluxes
    swmm_inlet.generated_inflow(Q_in[0])
    swmm_outlet.generated_inflow(Q_in[1])
    sim.step_advance(dt)
    sim.next()

    # Determine how much actually flowed into 1D model
    inlet_vol = - swmm_inlet.statistics['lateral_infow_vol'] + swmm_inlet.statistics['flooding_volume'] 
    inlet_flow = (inlet_vol - old_inlet_vol)/dt
    old_inlet_vol = inlet_vol

    if print_out:
        print('    inlet vol   :', inlet_vol)
        print('    inlet flow  :', inlet_flow)

    outlet_vol = - swmm_outlet.statistics['lateral_infow_vol'] + swmm_outlet.statistics['flooding_volume'] 
    outlet_flow = (outlet_vol - old_outlet_vol)/dt
    old_outlet_vol = outlet_vol

    if print_out:
        print('    outlet vol   :', outlet_vol)
        print('    outlet flow  :', outlet_flow)

    cumulative_inlet_flow += inlet_flow*dt
    cumulative_outlet_flow += outlet_flow*dt



    # And consequently set anuga coupling Inlet_operators with actual SWMM fluxes
    inlet1_anuga_inlet_op.set_Q(inlet_flow)
    outlet_anuga_inlet_op.set_Q(outlet_flow + swmm_outfall.total_inflow)


#print(swmm_inlet.statistics)
#print(swmm_outlet.statistics)
#print(swmm_outfall.statistics)



print('Cumulative inlet flooding', cumulative_inlet_flooding)
print('Cumulative outlet flooding', cumulative_outlet_flooding)


print('anuga inlet applied volume ',inlet1_anuga_inlet_op.get_total_applied_volume())
print('anuga inlet cumulative flow', cumulative_inlet_flow)
print('swmm inlet lateral flow', swmm_inlet.statistics['lateral_infow_vol'])
print('swmm inlet flooding vol', swmm_inlet.statistics['flooding_volume'])
print('swmm inlet vol', swmm_inlet.statistics['lateral_infow_vol'] - swmm_inlet.statistics['flooding_volume'])


print('anuga outlet applied volume ',outlet_anuga_inlet_op.get_total_applied_volume())
print('anuga outlet Cumulative flow', cumulative_outlet_flow)
print('swmm outlet lateral flow', swmm_outlet.statistics['lateral_infow_vol'])
print('swmm outlet flooding vol', swmm_outlet.statistics['flooding_volume'])
print('swmm outlet vol', swmm_outlet.statistics['lateral_infow_vol'] - swmm_outlet.statistics['flooding_volume'])

print('anuga inflow applied volume ',inflow_anuga_inlet_op.get_total_applied_volume())



sim.close()

if visualise:
    H_j = np.vstack(H_js)
    anuga_j = np.vstack(anuga_ws)
    Q_ins = np.vstack(Q_ins)

    plt.ion()

    plt.figure(1)
    plt.plot(time_series, H_j[:,0], label='Pipe Inlet 0')
    plt.plot(time_series, H_j[:,1], label='Pipe Inlet 1')
    plt.plot(time_series, anuga_j[:,0], label='Anuga Inlet 0')
    plt.plot(time_series, anuga_j[:,1], label='Anuga Inlet 1')
    plt.legend()
    plt.title('Head at junctions')
    plt.xlabel('Time (s)')
    plt.ylabel('Head (m)')
    plt.savefig('Figure1.png')
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.plot(time_series, losses)
    plt.title('Losses')
    plt.savefig('Figure2.png')
    plt.show()

    plt.figure(3)
    plt.clf()
    plt.plot(time_series, Q_dks)
    plt.title('Q_dks')
    plt.savefig('Figure3.png')
    plt.show()

    plt.figure(4)
    plt.clf()
    plt.plot(time_series, Q_ins[:,0], label='Inlet 0')
    plt.plot(time_series, Q_ins[:,1], label='Inlet 1')
    plt.legend()
    plt.title('Q_in')
    plt.savefig('Figure4.png')
    plt.show()

    input('Enter key ...')
