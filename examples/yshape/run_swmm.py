"""Y-shaped pipe network: ANUGA <-> SWMM coupling.

The SWMM twin of ``run_pipedream.py`` on the same Y-shaped domain, but the 1D
sewer is read from ``2inlets_short.inp`` (junctions J1, J2 draining through a
Y-junction to the outfall Out1). The whole model — coupling inlets at the two
junctions plus the SWMM backend — is built in one ``couple_from_inp`` call, and
the per-step exchange is driven by the returned coupling. The outfall is a SWMM
boundary, so water leaving there exits the system (counted in the loss).

This replaces an earlier script that used a patched pyswmm fork
(``create_opening`` / ``coupling_step``); we target the standard pyswmm release
and compute the exchange ourselves via ``anuga_drainage.calculate_Q`` — see
CLAUDE.md ("Project decision: standard pyswmm + our own calculate_Q").

    python run_swmm.py
"""
import anuga
from anuga import Inlet_operator, Region, rectangular_cross_domain
import numpy as np
import matplotlib.pyplot as plt

from anuga_drainage import couple_from_inp

# ---- parameters --------------------------------------------------------------
dt = 1.0         # yield/coupling step (whole seconds for pyswmm 2.1.0)
ft = 100.0       # final time (s)
inflow_Q = 1.0   # upstream 2D inflow (m^3/s)
visualise = False     # pop up head/loss plots at the end

# ---- 1. ANUGA domain ---------------------------------------------------------
length, width, dx = 20., 6., 0.2
domain = rectangular_cross_domain(int(length / dx), int(width / dx),
                                  len1=length, len2=width)
domain.set_name('Y_shape_swmm')


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

# Upstream inflow onto the 2D surface (fed near junction J2 at (7, 1)).
inflow_op = Inlet_operator(domain, Region(domain, radius=1.0, center=(2., 1.)),
                           Q=inflow_Q)

# ---- 2. Sewer network + coupling inlets, straight from the .inp --------------
# Couples the [JUNCTIONS] J1, J2 (circular pits of the given manhole_area at
# their .inp coordinates); the outfall Out1 is left as a SWMM boundary.
coupling = couple_from_inp(domain, './2inlets_short.inp', backend='swmm',
                           manhole_area=np.pi, n_sides=12,
                           time_average=10.0, clamp=True)
print(f'Coupled junctions: {list(coupling.inlets)}')

# ---- 3. Per-component volume-balance audit -----------------------------------
coupling.add_volume_balance(inflow_operators=[inflow_op])

# ---- 4. Evolve loop ----------------------------------------------------------
times, heads = [], []
for t in domain.evolve(yieldstep=dt, finaltime=ft):
    coupling.step(dt)                # exchange + SWMM advance + audit
    times.append(t)
    heads.append(coupling.backend.get_heads().copy())
    if domain.yieldstep_counter % 10 == 0:
        domain.print_timestepping_statistics()

# ---- 5. Report ---------------------------------------------------------------
print()
print(coupling.volume_balance.summary())
coupling.volume_balance.plot('volume_balance.png')
coupling.close()

if visualise:
    times, heads = np.array(times), np.vstack(heads)
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(coupling.inlets):
        plt.plot(times, heads[:, i], label=f'pipe head {name}')
    plt.xlabel('time (s)'); plt.ylabel('head (m)')
    plt.legend(); plt.title('Head at junctions')
    plt.tight_layout(); plt.savefig('Figure_heads.png'); plt.show()
