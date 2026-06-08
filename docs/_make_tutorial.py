"""One-off generator for docs/tutorial.ipynb (then executed with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# Tutorial: a coupled culvert from one `.inp`

This walkthrough builds a small **coupled** model — 2D overland flow (ANUGA)
over a channel with a culvert under a road, plus a 1D sewer — driven entirely
from a single SWMM `.inp` via {func}`~anuga_drainage.couple_from_inp`. We run it
on **both** backends (pipedream and SWMM) and compare the exchange flows and the
node water levels.

```{note}
This notebook is executed locally when the docs are built (it needs ANUGA,
pyswmm and pipedream); on Read the Docs it renders the stored outputs.
```
"""))

cells.append(code("""%matplotlib inline
import os, tempfile, textwrap
import numpy as np
import matplotlib.pyplot as plt
import anuga
from anuga_drainage import couple_from_inp"""))

cells.append(md(r"""## The sewer network

A tiny culvert: water enters at **Inlet**, flows through a box **Culvert** under
the road to **Outlet**, then down an **Outpipe** to the free **Outfall**. We
write it as a SWMM `.inp` — the standard format both backends understand. The
`[OPTIONS]` block (simulation dates, units, routing step) is what SWMM needs;
pipedream ignores it and reads only the network sections."""))

cells.append(code("""INP = textwrap.dedent('''\\
    [TITLE]
    Tutorial culvert

    [OPTIONS]
    FLOW_UNITS           CMS
    INFILTRATION         HORTON
    FLOW_ROUTING         DYNWAVE
    LINK_OFFSETS         DEPTH
    START_DATE           01/01/2024
    START_TIME           00:00:00
    END_DATE             01/01/2024
    END_TIME             00:30:00
    REPORT_START_DATE    01/01/2024
    REPORT_START_TIME    00:00:00
    REPORT_STEP          00:00:01
    ROUTING_STEP         0.1
    IGNORE_RAINFALL      YES

    [JUNCTIONS]
    ;;Name  Elev  MaxDepth  InitDepth  Sur  Aponded
    Inlet   0.07  2.0       0          0    0
    Outlet  0.03  2.0       0          0    0

    [OUTFALLS]
    Outfall 0.00  FREE                 NO

    [CONDUITS]
    ;;Name   From    To       Length  Rough  InOff  OutOff  Init  Max
    Culvert  Inlet   Outlet   11.0    0.013  0      0       0     0
    Outpipe  Outlet  Outfall  5.0     0.013  0      0       0     0

    [XSECTIONS]
    Culvert  RECT_CLOSED  1.0   4.0   0   0   1
    Outpipe  CIRCULAR     0.6   0     0   0   1

    [COORDINATES]
    Inlet    21.0   10.0
    Outlet    9.0   10.0
    Outfall   3.0   10.0
    ''')

inp_path = os.path.join(tempfile.mkdtemp(), 'culvert.inp')
with open(inp_path, 'w') as f:
    f.write(INP)
print(INP)"""))

cells.append(md(r"""## The 2D domain

A 60 m x 20 m channel with a raised road across the middle (x = 10-20 m); the
culvert carries water under it. Water is injected at the right-hand end and
drains through a Dirichlet boundary on the left. We wrap it in a function so we
can build a fresh domain for each backend (a domain can't be re-run after
`evolve`). The mesh is kept coarse so the notebook runs quickly."""))

cells.append(code("""def build_domain():
    rf = 10
    domain = anuga.rectangular_cross_domain(3 * rf, rf, len1=60, len2=20)
    domain.set_name('tutorial')

    def topography(x, y):
        z = 5 * np.ones_like(x)
        channel = (y > 5) & (y < 15)
        z = np.where(channel & (x < 10), x / 300, z)   # low channel downstream
        z = np.where(channel & (x > 20), x / 300, z)   # low channel upstream
        return z                                       # raised road for 10 < x < 20

    domain.set_quantity('elevation', topography, location='centroids')
    domain.set_quantity('friction', 0.035)

    inflow = anuga.Inlet_operator(domain, [[59.0, 5.0], [59.0, 15.0]], 1.0)
    Br = anuga.Reflective_boundary(domain)
    Bd = anuga.Dirichlet_boundary([-1.0, 0, 0])
    domain.set_boundary({'left': Bd, 'bottom': Br, 'top': Br, 'right': Br})
    return domain, inflow"""))

cells.append(md(r"""## Run on each backend

`run(backend)` builds a fresh domain, couples it from the **same** `.inp`,
attaches a volume-balance audit, and records the exchange flux `Q` (surface →
pipe) and the pipe head at each coupled node every step. `coupling.step(dt)`
drives the exchange and the audit together."""))

cells.append(code("""def run(backend):
    domain, inflow = build_domain()
    coupling = couple_from_inp(domain, inp_path, backend=backend,
                               manhole_area=8.0, time_average=5.0, clamp=True)
    coupling.add_volume_balance(inflow_operators=[inflow])

    t, Q, head = [], [], []
    for time in domain.evolve(yieldstep=1.0, finaltime=120.0):
        step = coupling.step(1.0)
        t.append(time)
        Q.append(np.asarray(step.Q_in).copy())            # exchange flux per node
        head.append(coupling.backend.get_heads().copy())  # pipe head per node
    coupling.close()
    return dict(names=list(coupling.inlets), t=np.array(t),
                Q=np.array(Q), head=np.array(head), vb=coupling.volume_balance)

res = {backend: run(backend) for backend in ('pipedream', 'swmm')}
names = res['pipedream']['names']
print('coupled nodes:', names)"""))

cells.append(md(r"""## Compare the exchange flows

The exchange flux `Q` at each node — positive is water leaving the 2D surface
and entering the pipe. The two solvers track each other closely; differences
come from how each routes flow through the 1D network."""))

cells.append(code("""fig, axes = plt.subplots(1, len(names), figsize=(11, 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, i, name in zip(axes, range(len(names)), names):
    for backend, style in [('pipedream', '-'), ('swmm', '--')]:
        ax.plot(res[backend]['t'], res[backend]['Q'][:, i], style, lw=2, label=backend)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title(f'exchange flux at {name}')
    ax.set_xlabel('time (s)')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Q  surface -> pipe  (m$^3$/s)')
axes[0].legend()
fig.tight_layout()"""))

cells.append(md(r"""## Compare the node water levels

The pipe head at each node over time, for both backends — you can see water
filling the pipe and the system settling."""))

cells.append(code("""fig, axes = plt.subplots(1, len(names), figsize=(11, 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, i, name in zip(axes, range(len(names)), names):
    for backend, style in [('pipedream', '-'), ('swmm', '--')]:
        ax.plot(res[backend]['t'], res[backend]['head'][:, i], style, lw=2, label=backend)
    ax.set_title(f'pipe head at {name}')
    ax.set_xlabel('time (s)')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('pipe head (m)')
axes[0].legend()
fig.tight_layout()"""))

cells.append(md(r"""## Did they conserve?

The audit splits the mass balance into ANUGA, the pipe solver, and the coupling
handoff. For **both** backends `R_anuga` and `R_couple` are ~machine precision
(ANUGA conserves; the coupling conserves); the difference lives in `R_pipe` —
SWMM's finite-difference loss vs pipedream's small surcharge residual."""))

cells.append(code("""for backend in ('pipedream', 'swmm'):
    print(f'================  {backend}  ================')
    summary = res[backend]['vb'].summary()
    print(summary.split('--- per inlet')[0].rstrip())
    print()"""))

cells.append(md(r"""## Recap

From **one** `.inp` we built the sewer and the coupling, ran it on both
pipedream and SWMM, and compared the exchange flows and node water levels — with
the volume audit confirming conservation on each. Same file, same loop, swap one
argument.
"""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                "name": "python3"}
nbf.write(nb, "docs/tutorial.ipynb")
print("wrote docs/tutorial.ipynb with", len(cells), "cells")
