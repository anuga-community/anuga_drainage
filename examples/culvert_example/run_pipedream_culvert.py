#------------------------------------------------------------------------------
# IMPORT NECESSARY MODULES
#------------------------------------------------------------------------------
print (' ABOUT to Start Simulation:- Importing Modules')


import anuga, numpy, time, os, glob
from anuga import create_domain_from_regions, Domain, Inlet_operator
import anuga.utilities.spatialInputUtil as su
from anuga_drainage import Coupler, PipedreamBackend

from anuga import Region

import numpy as np
#------------------------------------------------------------------------------
# FILENAMES, MODEL DOMAIN and VARIABLES
#------------------------------------------------------------------------------

basename = 'terrain'
outname =  'domain_pipedream_culvert'
meshname = 'terrain.msh'

dt = 0.1      # yield step
out_dt = 1.0  # output step
ft = 800      # final timestep

verbose = False

W=296600.
N=6180070.

E=296730.
S=6179960.

#------------------------------------------------------------------------------
# CREATING MESH
#------------------------------------------------------------------------------

bounding_polygon = [[W, S], [E, S], [E, N], [W, N]]


domain = anuga.create_domain_from_regions(bounding_polygon,
    boundary_tags={'south': [0], 'east': [1], 'north': [2], 'west': [3]},
    maximum_triangle_area=1.0,
    use_cache=False,
    verbose=verbose)

#------------------------------------------------------------------------------
# SETUP COMPUTATIONAL DOMAIN
#------------------------------------------------------------------------------
domain.set_minimum_storable_height(0.0001) 
domain.set_name(outname) 
print (domain.statistics())

#------------------------------------------------------------------------------
# APPLY MANNING'S ROUGHNESSES
#------------------------------------------------------------------------------

domain.set_quantity('friction', 0.035)
domain.set_quantity('elevation', filename=basename+'.csv', use_cache=True, verbose=verbose, alpha=0.1)

#------------------------------------------------------------------------------
# SETUP BOUNDARY CONDITIONS
#------------------------------------------------------------------------------

print ('Available boundary tags', domain.get_boundary_tags())

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([0,0,0])

domain.set_boundary({'west': Br, 'south': Br, 'north': Bd, 'east': Br})


radius_inlet = 0.5
radius_outlet = 0.5


inlet1_anuga_region = Region(domain, radius=radius_inlet, center=(296660.390,6180017.186))
outlet_anuga_region = Region(domain, radius=radius_outlet, center=(296649.976,6180038.872))

anuga_Lweirs = np.array([2*np.pi*radius_inlet, 2*np.pi*radius_outlet])
anuga_Amanholes = np.array([np.pi*radius_inlet**2, np.pi*radius_outlet**2])

inlet1_anuga_inlet_op = Inlet_operator(domain, inlet1_anuga_region, Q=0.0, zero_velocity=True)
outlet_anuga_inlet_op = Inlet_operator(domain, outlet_anuga_region, Q=0.0, zero_velocity=False)

anuga_beds = np.array([inlet1_anuga_inlet_op.inlet.get_average_elevation(),
                        outlet_anuga_inlet_op.inlet.get_average_elevation()])

print(anuga_beds)

line=[[296669.258,6179974.191],[296677.321,6179976.449]]
Inlet_operator(domain, line, 1.0)

#------------------------------------------------------------------------------
# PIPEDREAM
#------------------------------------------------------------------------------

print('Setup pipedream structures')
from pipedream_solver.hydraulics import SuperLink
import matplotlib.pyplot as plt
import pandas as pd


superjunctions = pd.DataFrame({'name': [0, 1],
                               'id': [0, 1],
                               'z_inv': [12.2, 12.2],
                               'h_0': 2*[1e-5],
                               'bc': 2*[False],
                               'storage': 2*['functional'],
                               'a': 2*[0.],
                               'b': 2*[1.],
                               'c': 2*[10.],
                               'max_depth': 2*[np.inf],
                               'map_x': 2*[0],
                               'map_y': 2*[0]})

superlinks = pd.DataFrame({'name': [0],
                           'id': [0],
                           'sj_0': [0],
                           'sj_1': [1],
                           'in_offset': 1*[0.],
                           'out_offset': 1*[0.],
                           'dx': [24],
                           'n': 1*[0.013],
                           'shape': 1*['circular'],
                           'g1': [0.5],
                           'g2': 1*[0.],
                           'g3': 1*[0.],
                           'g4': 1*[0.],
                           'Q_0': 1*[0.],
                           'h_0': 1*[1e-5],
                           'ctrl': 1*[False],
                           'A_s': 1*[0.],
                           'A_c': 1*[0.],
                           'C': 1*[0.]})

superlink = SuperLink(superlinks, superjunctions, internal_links=20)

surface_elevs = np.array([12.2, 12.4]) 

input_velocity = 1


H_js = []
losses = []

Q_iks =[]
Q_uks =[]
Q_dks =[]
time_series = []

print('Start Evolve')

# This example historically used cw = co = 1.0 (vs the package default 0.67).
coupler = Coupler(inlets=[inlet1_anuga_inlet_op, outlet_anuga_inlet_op],
                  beds=anuga_beds,
                  weir_lengths=anuga_Lweirs,
                  manhole_areas=anuga_Amanholes,
                  backend=PipedreamBackend(superlink),
                  cw=1.0, co=1.0)


for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):
    #print('\n')
    if domain.yieldstep_counter%domain.output_frequency == 0:
        domain.print_timestepping_statistics()

    anuga_depths = np.array([inlet1_anuga_inlet_op.inlet.get_average_depth(),
                             outlet_anuga_inlet_op.inlet.get_average_depth()])
    
    anuga_stages = np.array([inlet1_anuga_inlet_op.inlet.get_average_stage(),
                             outlet_anuga_inlet_op.inlet.get_average_stage()])


    # Compute volumes
    link_volume = ((superlink._A_ik * superlink._dx_ik).sum() +
                   (superlink._A_SIk * superlink._h_Ik).sum())
    node_volume = (superlink._A_sj * (superlink.H_j - superlink._z_inv_j)).sum()
    sewer_volume = link_volume + node_volume

    boundary_flux = domain.get_boundary_flux_integral()
    total_volume_correct = t * input_velocity + boundary_flux 
    
    total_volume_real = domain.get_water_volume() + sewer_volume
    loss = total_volume_real - total_volume_correct

    if domain.yieldstep_counter%domain.output_frequency == 0:
        print('    Loss         ', loss)
        print('    TV correct   ', total_volume_correct)
        print('    domain volume', domain.get_water_volume())
        print('    node_volume  ', node_volume)
        print('    sewer_volume ', sewer_volume)
        print('    anuga_depths ', anuga_depths)
        print('    anuga_beds   ', anuga_beds)
        print('    MOInvert     ', superlink._z_inv_j)
        print('    Head         ', superlink.H_j)
        print('    anuga_stages ', anuga_stages)

    # Append data
    time_series.append(t)
    losses.append(loss)
    H_js.append(superlink.H_j.copy())

    # record flow time series in each pipe
    Q_iks.append(superlink.Q_ik.copy())
    Q_uks.append(superlink.Q_uk.copy())
    Q_dks.append(superlink.Q_dk.copy())

        
    # Calculate the exchange flux, step the sewer and feed the realised flow
    # back to ANUGA (see anuga_drainage.Coupler).
    Q_in = coupler.step(dt).Q_in

    if domain.yieldstep_counter%domain.output_frequency == 0:
        print('    Q            ', Q_in)


H_j = np.vstack(H_js)

plt.ion()

plt.figure(1)
plt.plot(time_series, H_j[:,0], label='Inlet 0')
plt.plot(time_series, H_j[:,1], label='Inlet 1')
plt.legend()
plt.title('Head at junctions')
plt.xlabel('Time (s)')
plt.ylabel('Head (m)')
plt.show()

plt.figure(2)
plt.clf()
plt.plot(time_series, losses)
plt.title('Losses')
plt.show()

plt.figure(3)
plt.clf()
plt.plot(time_series, Q_dks)
plt.title('Q_dks')
plt.show()

input('Enter key ...')
