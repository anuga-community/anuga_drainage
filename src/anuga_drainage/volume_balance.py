"""Per-step water-volume audit of the coupled 2D (ANUGA) <-> 1D (SWMM/pipedream)
system, to localise where a mass-balance discrepancy comes from.

Each subsystem's water budget is measured from *its own* bookkeeping and the
two are then compared, so the consistency check is independent (not circular).
All volumes are signed; ``+`` means water added to that subsystem.

    ANUGA :  V_anuga(t) - V_anuga(0)  =  inflow + boundary + inlets_anuga
    pipe  :  V_pipe(t)  - V_pipe(0)   =  inlets_pipe - outfall
    couple:  inlets_anuga + inlets_pipe - outfall  (~0 when the handoff conserves)

with residuals R = LHS - RHS:

    R_anuga   should be ~machine precision (ANUGA is finite volume).
    R_pipe    ~0 for pipedream (finite volume); exposes SWMM's finite-difference
              non-conservation in isolation.
    R_couple  catches *coupling* errors (sign/double-count, SWMM realised vs
              requested, the outfall-return term).

and the usual single loss splits exactly:  loss = R_anuga + R_pipe + R_couple.

Quantities are read from authoritative accessors: ANUGA's
``Inlet_operator.get_total_applied_volume()`` (captures every ``set_Q``,
including the outfall-return override), ``domain.get_water_volume()`` /
``get_boundary_flux_integral()``, and the backend's independent pipe-side
volumes (``pipe_volume`` / ``coupling_inflow_volume`` / ``outfall_volume``).
"""
from collections import namedtuple

VolumeRecord = namedtuple("VolumeRecord", [
    "t", "V_anuga", "V_pipe", "inflow", "boundary",
    "inlets_anuga", "inlets_pipe", "outfall",
    "R_anuga", "R_pipe", "R_couple", "loss",
])


class VolumeBalance:
    """Records the coupled-system water budget each step.

    Parameters
    ----------
    domain : the ANUGA domain.
    coupling_inlets : the ANUGA Inlet_operators that exchange with the 1D network.
    backend : a Coupler backend (SwmmBackend/PipedreamBackend) exposing
        pipe_volume(), coupling_inflow_volume() and outfall_volume().
    inflow_operators : the upstream-source Inlet_operators feeding the domain.
    """

    def __init__(self, domain, coupling_inlets, backend, inflow_operators=()):
        self.domain = domain
        self.coupling_inlets = list(coupling_inlets)
        self.backend = backend
        self.inflow_operators = list(inflow_operators)
        # Baselines are captured on the first step() call, not here: before the
        # evolve loop the domain can be in an unphysical initial state (stage
        # below the bed), so domain.get_water_volume() is meaningless until ANUGA
        # has stepped. All budgets are then measured relative to that first step.
        self._base = None
        self.V_anuga0 = None
        self.V_pipe0 = None
        self.records = []

    @staticmethod
    def _applied(ops):
        return sum(op.get_total_applied_volume() for op in ops)

    def step(self, t):
        """Record the budget at time ``t`` and return the VolumeRecord."""
        V_a = self.domain.get_water_volume()
        V_p = self.backend.pipe_volume()
        inflow = self._applied(self.inflow_operators)
        boundary = self.domain.get_boundary_flux_integral()
        inlets_a = self._applied(self.coupling_inlets)
        inlets_p = self.backend.coupling_inflow_volume()
        outfall = self.backend.outfall_volume()

        if self._base is None:
            self._base = (V_a, V_p, inflow, boundary, inlets_a, inlets_p, outfall)
            self.V_anuga0, self.V_pipe0 = V_a, V_p
        bV_a, bV_p, bI, bB, bA, bP, bO = self._base

        dV_a, dV_p = V_a - bV_a, V_p - bV_p           # changes since the first step
        dI, dB = inflow - bI, boundary - bB
        dA, dP, dO = inlets_a - bA, inlets_p - bP, outfall - bO

        R_anuga = dV_a - (dI + dB + dA)
        R_pipe = dV_p - (dP - dO)
        R_couple = dA + dP - dO
        loss = (dV_a + dV_p) - (dI + dB)

        rec = VolumeRecord(t, V_a, V_p, inflow, boundary, inlets_a, inlets_p,
                           outfall, R_anuga, R_pipe, R_couple, loss)
        self.records.append(rec)
        return rec

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records, columns=VolumeRecord._fields)

    def summary(self):
        """Return a short multi-line report of the final-step budget/residuals."""
        if not self.records:
            return "VolumeBalance: no records"
        r = self.records[-1]
        return "\n".join([
            f"Volume balance at t = {r.t:g} s",
            f"  ANUGA water    V_anuga = {r.V_anuga:12.6f}  (start {self.V_anuga0:.6f})",
            f"  pipe water     V_pipe  = {r.V_pipe:12.6f}  (start {self.V_pipe0:.6f})",
            f"  inflow source          = {r.inflow:12.6f}",
            f"  boundary flux          = {r.boundary:12.6f}",
            f"  inlets -> ANUGA        = {r.inlets_anuga:12.6f}",
            f"  inlets -> pipe         = {r.inlets_pipe:12.6f}",
            f"  outfall <- pipe        = {r.outfall:12.6f}",
            f"  --- residuals (should be ~0) ---",
            f"  R_anuga  (ANUGA closes)     = {r.R_anuga: .3e}",
            f"  R_pipe   (pipe closes)      = {r.R_pipe: .3e}",
            f"  R_couple (handoff consistent)= {r.R_couple: .3e}",
            f"  total loss = R_anuga+R_pipe+R_couple = {r.loss: .3e}",
        ])

    def plot(self, filename=None, show=False):
        """Plot the component volumes and the three residuals vs time."""
        import matplotlib.pyplot as plt
        df = self.to_dataframe()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for col in ["V_anuga", "V_pipe", "inflow", "boundary",
                    "inlets_anuga", "inlets_pipe", "outfall"]:
            ax1.plot(df["t"], df[col], label=col)
        ax1.set_ylabel("volume (m^3)")
        ax1.legend(fontsize=8, ncol=2)
        ax1.set_title("Coupled-system water volumes")
        for col in ["R_anuga", "R_pipe", "R_couple", "loss"]:
            ax2.plot(df["t"], df[col], label=col)
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("residual (m^3)")
        ax2.legend(fontsize=8)
        ax2.set_title("Mass-balance residuals")
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
        if show:
            plt.show()
        return fig
