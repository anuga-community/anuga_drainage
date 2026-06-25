"""Simple channel + culvert: ANUGA <-> pipedream coupling, hand-built.

The pipedream twin of ``run_swmm_short.py``: same 2D domain and inlets, but the
1D sewer is a hand-built pipedream ``SuperLink`` (two superjunctions, one box
culvert) instead of a SWMM ``.inp``. The per-step exchange is driven by
``anuga_drainage.Coupler`` exactly as in the SWMM case; only the backend
differs. Compare with ``run_from_inp.py``, which builds the pipedream network
automatically from the ``.inp``.

pipedream is finite-volume, so the pipe budget closes (R_pipe ~0) far more
tightly than SWMM's finite-difference loss; the loss is reported by the
VolumeBalance summary at the end.

    python run_pipedream.py
"""
import anuga
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipedream_solver.hydraulics import SuperLink

from anuga_drainage import Coupler, PipedreamBackend, VolumeBalance

# ---- parameters --------------------------------------------------------------
rf = 20          # domain refinement; too coarse and inlets overlap the wall
dt = 0.05        # yield/coupling step. pipedream's semi-implicit solver is only
                 # stable at a small step here (cf. couple_from_inp's internal
                 # sub-stepping, which lets the exchange stay coarse).
out_dt = 1.0     # sww output step
ft = 400         # final time (s)
time_average = 1.0    # s; smooths the coupling flux
visualise = False     # pop up head/loss plots at the end

# ---- 1. ANUGA domain ---------------------------------------------------------
domain = anuga.rectangular_cross_domain(3 * rf, rf, len1=60, len2=20)
domain.set_minimum_storable_height(0.0001)
domain.set_name('domain_pipedream')


def topography(x, y):
    z = 5 * np.ones_like(x)
    channel = np.logical_and(y > 5, y < 15)
    z = np.where(np.logical_and(channel, x < 10), x / 300, z)
    z = np.where(np.logical_and(channel, x > 20), x / 300, z)
    return z


domain.set_quantity('elevation', topography, location='centroids')
domain.set_quantity('friction', 0.035)

# Upstream inflow into the 2D domain.
inflow_Q = 1.0
inflow_op = anuga.Inlet_operator(domain, [[59.0, 5.0], [59.0, 15.0]], inflow_Q)

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([-1.0, 0, 0])
domain.set_boundary({'left': Bd, 'bottom': Br, 'top': Br, 'right': Br})

# ---- 2. Coupling inlets + pipedream network ----------------------------------
inlet_region = anuga.Region(domain, polygon=[[20, 5], [22, 5], [22, 15], [20, 15]])
outlet_region = anuga.Region(domain, polygon=[[8, 5], [10, 5], [10, 15], [8, 15]])
inlet_op = anuga.Inlet_operator(domain, inlet_region, Q=0.0, zero_velocity=True)
outlet_op = anuga.Inlet_operator(domain, outlet_region, Q=0.0, zero_velocity=False)

weir_lengths = np.array([20.0, 20.0])
manhole_areas = np.array([20.0, 20.0])
beds = np.array([inlet_op.inlet.get_average_elevation(),
                 outlet_op.inlet.get_average_elevation()])

# pipedream network: two superjunctions joined by one rect_closed (box) culvert.
# Geometry reference: https://mattbartos.com/pipedream/geometry-reference.html
superjunctions = pd.DataFrame({
    'name': [0, 1], 'id': [0, 1], 'z_inv': [0.04, 0.00], 'h_0': 2 * [0],
    'bc': 2 * [False], 'storage': 2 * ['functional'],
    'a': 2 * [0.], 'b': 2 * [1.], 'c': 2 * [10.], 'max_depth': 2 * [np.inf],
    'map_x': 2 * [0], 'map_y': 2 * [0]})
superlinks = pd.DataFrame({
    'name': [0], 'id': [0], 'sj_0': [0], 'sj_1': [1],
    'in_offset': [0.], 'out_offset': [0.], 'dx': [10], 'n': [0.013],
    'shape': ['rect_closed'], 'g1': [1.0], 'g2': [10.0], 'g3': [0.1], 'g4': [0.],
    'Q_0': [0.], 'h_0': [1e-5], 'ctrl': [False], 'A_s': [0.], 'A_c': [0.], 'C': [0.]})
superlink = SuperLink(superlinks, superjunctions, internal_links=6)

coupler = Coupler(inlets=[inlet_op, outlet_op], beds=beds,
                  weir_lengths=weir_lengths, manhole_areas=manhole_areas,
                  backend=PipedreamBackend(superlink), time_average=time_average)

# ---- 3. Per-component volume-balance audit -----------------------------------
vb = VolumeBalance(domain, coupling_inlets=[inlet_op, outlet_op],
                   backend=coupler.backend, inflow_operators=[inflow_op])

# ---- 4. Evolve loop ----------------------------------------------------------
times, heads, stages = [], [], []
prev_step = None   # previous CouplingStep, for the aligned (top-of-loop) audit
for t in domain.evolve(yieldstep=dt, outputstep=out_dt, finaltime=ft):
    vb.step(t, dt, prev_step)        # audit at the top, with the previous step
    prev_step = coupler.step(dt)     # exchange + pipedream advance + feedback

    times.append(t)
    heads.append(coupler.backend.get_heads().copy())
    stages.append(np.array([inlet_op.inlet.get_average_stage(),
                            outlet_op.inlet.get_average_stage()]))
    if domain.yieldstep_counter % domain.output_frequency == 0:
        domain.print_timestepping_statistics()

# ---- 5. Report ---------------------------------------------------------------
print()
print(vb.summary())
vb.plot('volume_balance.png')

if visualise:
    times = np.array(times)
    heads, stages = np.vstack(heads), np.vstack(stages)
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(['Inlet', 'Outlet']):
        plt.plot(times, heads[:, i], label=f'pipe head {name}')
        plt.plot(times, stages[:, i], '--', label=f'ANUGA stage {name}')
    plt.xlabel('time (s)'); plt.ylabel('head (m)')
    plt.legend(); plt.title('Heads at junctions')
    plt.tight_layout(); plt.savefig('Figure_heads.png'); plt.show()
