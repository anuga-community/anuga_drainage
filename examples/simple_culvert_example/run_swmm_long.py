"""Long channel + culvert: ANUGA <-> SWMM coupling, hand-built.

Same coupling pattern as ``run_swmm_short.py`` (read it first) but on a longer
channel: the inlet sits at x~50 and the channel falls away both upstream and
downstream of the culvert, so the culvert spans a longer reach. Drives the
per-step exchange with ``anuga_drainage.Coupler`` and returns outfall water to
the surface at the outlet inlet.

    python run_swmm_long.py
"""
import anuga
import numpy as np
import matplotlib.pyplot as plt

from anuga_drainage import Coupler, SwmmBackend, VolumeBalance

# ---- parameters --------------------------------------------------------------
rf = 20          # domain refinement; too coarse and inlets overlap the wall
dt = 1.0         # yield/coupling step (whole seconds: pyswmm 2.1.0 strides in int s)
out_dt = 1.0     # sww output step
ft = 400         # final time (s)
time_average = 10.0   # s; smooths the coupling flux to damp oscillations
visualise = False     # pop up head/loss/flux plots at the end

# ---- 1. ANUGA domain ---------------------------------------------------------
domain = anuga.rectangular_cross_domain(3 * rf, rf, len1=60, len2=20)
domain.set_minimum_storable_height(0.0001)
domain.set_name('domain_swmm_long')


def topography(x, y):
    z = 5 * np.ones_like(x)
    channel = np.logical_and(y > 5, y < 15)
    z = np.where(np.logical_and(channel, x < 10), x / 300, z)
    z = np.where(np.logical_and(channel, x > 50), x / 300, z)
    return z


domain.set_quantity('elevation', topography, location='centroids')
domain.set_quantity('friction', 0.035)

# Upstream inflow into the 2D domain.
inflow_Q = 1.0
inflow_op = anuga.Inlet_operator(domain, [[59.0, 5.0], [59.0, 15.0]], inflow_Q)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([-1.0, 0, 0])
domain.set_boundary({'left': Bd, 'bottom': Br, 'top': Br, 'right': Br})

# ---- 2. Coupling inlets + SWMM network ---------------------------------------
# Channel-spanning inlet/outlet footprints (width cw). weir_length ~ perimeter,
# manhole_area ~ area.
cw = 8
inlet_region = anuga.Region(domain, polygon=[[50, 10 - cw / 2], [52, 10 - cw / 2],
                                             [52, 10 + cw / 2], [50, 10 + cw / 2]])
outlet_region = anuga.Region(domain, polygon=[[8, 10 - cw / 2], [10, 10 - cw / 2],
                                              [10, 10 + cw / 2], [8, 10 + cw / 2]])
inlet_op = anuga.Inlet_operator(domain, inlet_region, Q=0.0, zero_velocity=False)
outlet_op = anuga.Inlet_operator(domain, outlet_region, Q=0.0, zero_velocity=False)

weir_lengths = np.array([2 * cw, 2 * cw])
manhole_areas = np.array([2 * cw, 2 * cw])
beds = np.array([inlet_op.inlet.get_average_elevation(),
                 outlet_op.inlet.get_average_elevation()])

from pyswmm import Simulation, Nodes
sim = Simulation('./swmm_input_long.inp')
sim.start()
swmm_inlet, swmm_outlet = Nodes(sim)['Inlet'], Nodes(sim)['Outlet']
swmm_outfall = Nodes(sim)['Outfall']

coupler = Coupler(inlets=[inlet_op, outlet_op], beds=beds,
                  weir_lengths=weir_lengths, manhole_areas=manhole_areas,
                  backend=SwmmBackend(sim, junctions=[swmm_inlet, swmm_outlet]),
                  time_average=time_average, clamp=True)

# ---- 3. Per-component volume-balance audit -----------------------------------
vb = VolumeBalance(domain, coupling_inlets=[inlet_op, outlet_op],
                   backend=coupler.backend, inflow_operators=[inflow_op],
                   outfall_inlet=1)   # outlet inlet receives the outfall return

# ---- 4. Evolve loop ----------------------------------------------------------
times, heads, stages, Q_ins = [], [], [], []
prev_step = None   # previous CouplingStep, for the aligned (top-of-loop) audit
for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):
    vb.step(t, dt, prev_step)        # audit at the top, with the previous step
    step = coupler.step(dt)          # exchange + SWMM advance + feedback
    # Also return water leaving the system at the outfall to the outlet inlet.
    outlet_op.set_Q(step.anuga_flux[1] + swmm_outfall.total_inflow)
    prev_step = step

    times.append(t)
    heads.append(coupler.backend.get_heads().copy())
    stages.append(np.array([inlet_op.inlet.get_average_stage(),
                            outlet_op.inlet.get_average_stage()]))
    Q_ins.append(step.Q_in.copy())
    if domain.yieldstep_counter % domain.output_frequency == 0:
        domain.print_timestepping_statistics()

# ---- 5. Report ---------------------------------------------------------------
print()
print(vb.summary())
vb.plot('volume_balance.png')
sim.close()

if visualise:
    times = np.array(times)
    heads, stages, Q_ins = np.vstack(heads), np.vstack(stages), np.vstack(Q_ins)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    for i, name in enumerate(['Inlet', 'Outlet']):
        ax1.plot(times, heads[:, i], label=f'pipe head {name}')
        ax1.plot(times, stages[:, i], '--', label=f'ANUGA stage {name}')
        ax2.plot(times, Q_ins[:, i], label=f'Q_in {name}')
    ax1.set_ylabel('head (m)'); ax1.legend(fontsize=8); ax1.set_title('Heads')
    ax2.set_xlabel('time (s)'); ax2.set_ylabel('Q (m^3/s)')
    ax2.legend(fontsize=8); ax2.set_title('Coupling flux (surface -> pipe +ve)')
    fig.tight_layout(); fig.savefig('Figure_heads.png'); plt.show()
