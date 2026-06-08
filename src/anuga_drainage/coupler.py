"""Driver for the 2D (ANUGA) <-> 1D (SWMM / pipedream) coupling loop.

The per-step coupling sequence is the same across the example run scripts:

    1. read the 2D water depth at each inlet,
    2. read the 1D head at each inlet/junction,
    3. compute the exchange flux with calculate_Q (weir/orifice),
    4. smooth it in time to damp oscillations,
    5. optionally clamp the outflow to the water actually available in the 2D cell,
    6. advance the 1D model by dt using that flux,
    7. feed the *realised* 1D flow back into the 2D model via Inlet_operator.set_Q.

This module captures that sequence in `Coupler`, with the 1D-backend-specific
parts (head access, stepping, realised-flow sign/bookkeeping) behind a small
backend interface. The two backends differ in step 7: SWMM reports the flow it
actually accepted via node statistics, whereas pipedream takes the requested
flux as realised and feeds back -Q_in.

Volume-balance / logging / plotting stays in the run scripts (it varies per
example); the backends expose the underlying sim/superlink handles for that.

See CLAUDE.md for the coupling-loop background and the pyswmm 2.1.0 / numpy 2.x
environment constraints.
"""
from collections import namedtuple

import numpy as np

from .coupling import calculate_Q

CouplingStep = namedtuple("CouplingStep", ["Q_in", "anuga_flux"])


def smooth_Q(Q_new, Q_old, dt, time_average):
    """Time-average the coupling flux to damp oscillations.

    Q = ((time_average - dt) * Q_old + dt * Q_new) / time_average

    `time_average <= 0` disables smoothing (returns Q_new unchanged).
    """
    if time_average <= 0:
        return Q_new
    return ((time_average - dt) * Q_old + dt * Q_new) / time_average


def limit_outflow(Q_in, available_volume, dt, safety_factor=1.0):
    """Clamp positive (surface -> pipe) flux so a step cannot remove more water
    than is present in the 2D cell. Negative (surcharge) flux is untouched.
    """
    Q_limit = safety_factor * np.asarray(available_volume) / dt
    return np.where(Q_in > 0, np.minimum(Q_in, Q_limit), Q_in)


class SwmmBackend:
    """Coupling backend for the standard pyswmm release (>= 2.1).

    Heads come from the junction nodes; the model is advanced with whole-second
    strides (pyswmm 2.1.0 requires an int); the flow fed back to ANUGA is the
    flow SWMM actually accepted, derived from node statistics.
    """

    def __init__(self, sim, junctions=None, links=None, outfalls=None):
        from pyswmm import Nodes, Links

        self.sim = sim
        if junctions is None:
            junctions = [n for n in Nodes(sim) if n.is_junction()]
        self.junctions = list(junctions)
        self.links = list(links) if links is not None else list(Links(sim))
        if outfalls is None:
            outfalls = [n for n in Nodes(sim) if n.is_outfall()]
        self.outfalls = list(outfalls)
        self._old_vol = np.array([self._inlet_vol(n) for n in self.junctions])
        self._outfall_vol = 0.0   # cumulative volume that left the network at outfalls

    @staticmethod
    def _inlet_vol(node):
        # Net volume that has left the 2D surface at this node: the generated
        # (lateral) inflow that entered the pipe, minus what flooded back out.
        s = node.statistics
        return -s["lateral_infow_vol"] + s["flooding_volume"]

    def get_heads(self):
        return np.array([n.head for n in self.junctions])

    def step(self, Q_in, dt):
        for node, q in zip(self.junctions, Q_in):
            node.generated_inflow(q)
        self.sim.step_advance(int(dt))  # swmm_stride requires an int (whole seconds)
        next(self.sim)
        # Accumulate the volume leaving at outfalls (read post-step, matching the
        # outfall-return term the scripts add back to ANUGA).
        self._outfall_vol += sum(o.total_inflow for o in self.outfalls) * dt

    def anuga_flux(self, Q_in, dt):
        new = np.array([self._inlet_vol(n) for n in self.junctions])
        flux = (new - self._old_vol) / dt
        self._old_vol = new
        return flux

    def link_volume(self):
        return sum(link.volume for link in self.links)

    # --- independent pipe-side volume accounting (for VolumeBalance) ---
    def pipe_volume(self):
        """Water currently held in the network: conduits + junction storage."""
        return self.link_volume() + sum(n.volume for n in self.junctions)

    def coupling_inflow_volumes(self):
        """Per-junction cumulative net volume the surface injected (lateral
        inflow accepted minus what flooded back out), from SWMM's statistics."""
        return [n.statistics["lateral_infow_vol"] - n.statistics["flooding_volume"]
                for n in self.junctions]

    def coupling_inflow_volume(self):
        """Cumulative net volume the surface injected at the coupling junctions,
        measured independently of the ANUGA side."""
        return sum(self.coupling_inflow_volumes())

    def outfall_volume(self):
        """Cumulative volume that has left the network at outfalls."""
        return self._outfall_vol


class PipedreamBackend:
    """Coupling backend for pipedream's SuperLink.

    Heads are the superjunction heads H_j; the requested flux is taken as
    realised, so the flow fed back to ANUGA is -Q_in.
    """

    def __init__(self, superlink, coupled_indices=None, H_bc=None, outfall_indices=None):
        # coupled_indices: which superjunctions exchange with ANUGA (default all,
        #   matching the hand-built examples). couple_from_inp couples only the
        #   junctions and lists the outfalls as boundary (bc) superjunctions.
        # H_bc: fixed boundary heads for bc superjunctions (e.g. outfall inverts
        #   for free drainage); None keeps the previous no-bc behaviour exactly.
        # outfall_indices: bc superjunctions that shed water out of the system,
        #   for outfall_volume tracking.
        self.superlink = superlink
        n = len(superlink.H_j)
        self.coupled = (np.arange(n) if coupled_indices is None
                        else np.asarray(coupled_indices, dtype=int))
        self.H_bc = None if H_bc is None else np.asarray(H_bc, dtype=float)
        self._injected = None   # per-coupled-junction cumulative injected volume
        # Superlink ends discharging at an outfall: Q_dk is the flow into a
        # superlink's downstream superjunction, Q_uk the flow out of its upstream
        # one — so outfall outflow = sum(Q_dk at outfall d/s ends) - sum(Q_uk at
        # outfall u/s ends), accumulated over time.
        outs = set() if outfall_indices is None else {int(i) for i in outfall_indices}
        self._outfall_dk = [k for k in range(len(superlink._J_dk))
                            if int(superlink._J_dk[k]) in outs]
        self._outfall_uk = [k for k in range(len(superlink._J_uk))
                            if int(superlink._J_uk[k]) in outs]
        self._outfall_vol = 0.0

    def get_heads(self):
        return self.superlink.H_j[self.coupled]

    def step(self, Q_in, dt):
        q = np.asarray(Q_in, dtype=float)
        full = np.zeros(len(self.superlink.H_j))
        full[self.coupled] = q
        self._injected = q * dt if self._injected is None else self._injected + q * dt
        if self.H_bc is None:
            self.superlink.step(Q_in=full, dt=dt)
        else:
            self.superlink.step(Q_in=full, H_bc=self.H_bc, dt=dt)
        if self._outfall_dk or self._outfall_uk:
            s = self.superlink
            out = (sum(float(s.Q_dk[k]) for k in self._outfall_dk)
                   - sum(float(s.Q_uk[k]) for k in self._outfall_uk))
            self._outfall_vol += out * dt

    def anuga_flux(self, Q_in, dt):
        return -np.asarray(Q_in)

    def link_volume(self):
        s = self.superlink
        return (s._A_ik * s._dx_ik).sum() + (s._A_SIk * s._h_Ik).sum()

    def node_volume(self):
        # Use pipedream's own storage-curve volume: it honours the functional/
        # tabular storage curves and clamps depth at min_depth, matching how
        # pipedream conserves mass. The linear A_sj*(H_j - z_inv_j) goes negative
        # when a superjunction's head drops below its invert (common upstream),
        # which otherwise shows up as a spurious R_pipe.
        return float(self.superlink.compute_storage_volumes().sum())

    def sewer_volume(self):
        return self.link_volume() + self.node_volume()

    # --- independent pipe-side volume accounting (for VolumeBalance) ---
    def pipe_volume(self):
        # Physical water volume (flow-area conduit + pit + superjunction). This
        # is exact, but pipedream's semi-implicit linearised continuity + the
        # Preissmann slot don't conserve *physical* volume to machine precision
        # during redistribution/surcharge, so VolumeBalance's R_pipe shows
        # pipedream's own ~0.1-1% numerical non-conservation (cf. SWMM's larger
        # finite-difference loss). That is the solver's behaviour, not a coupling
        # error or a measurement bug to "correct" away. See CLAUDE.md.
        return self.sewer_volume()

    def coupling_inflow_volumes(self):
        """Per-junction cumulative injected volume (pipedream takes the requested
        flux as realised, so this is the integral of Q_in per junction)."""
        return [] if self._injected is None else list(self._injected)

    def coupling_inflow_volume(self):
        """Cumulative volume injected at the coupling junctions."""
        return 0.0 if self._injected is None else float(self._injected.sum())

    def outfall_volume(self):
        """Cumulative volume shed at outfall (bc) superjunctions (0 if none)."""
        return self._outfall_vol


class Coupler:
    """Drives the per-step 2D<->1D exchange for a set of inlets and a backend.

    `inlets` are ANUGA Inlet_operators, ordered to match the backend's heads
    (the SWMM junction order / pipedream superjunction order). `beds`,
    `weir_lengths` and `manhole_areas` are parallel arrays for calculate_Q.
    """

    def __init__(self, inlets, beds, weir_lengths, manhole_areas, backend,
                 time_average=0.0, clamp=False, safety_factor=1.0,
                 cw=0.67, co=0.67, g=None):
        self.inlets = list(inlets)
        self.beds = np.asarray(beds, dtype=float)
        self.weir_lengths = np.asarray(weir_lengths, dtype=float)
        self.manhole_areas = np.asarray(manhole_areas, dtype=float)
        self.backend = backend
        self.time_average = time_average
        self.clamp = clamp
        self.safety_factor = safety_factor
        self.cw = cw
        self.co = co
        self.g = g  # gravity for calculate_Q; None -> ANUGA's value (see calculate_Q)
        self.Q_in = np.zeros(len(self.inlets))

    def depths(self):
        return np.array([op.inlet.get_average_depth() for op in self.inlets])

    def volumes(self):
        return np.array([op.inlet.get_total_water_volume() for op in self.inlets])

    def step(self, dt):
        """Advance the coupling by dt and return the (Q_in, anuga_flux) used."""
        depths = self.depths()
        heads = self.backend.get_heads()

        Q = calculate_Q(heads, depths, self.beds, self.weir_lengths,
                        self.manhole_areas, cw=self.cw, co=self.co, g=self.g)
        Q = smooth_Q(Q, self.Q_in, dt, self.time_average)
        if self.clamp:
            Q = limit_outflow(Q, self.volumes(), dt, self.safety_factor)
        self.Q_in = Q

        self.backend.step(Q, dt)

        flux = self.backend.anuga_flux(Q, dt)
        for op, f in zip(self.inlets, flux):
            op.set_Q(f)

        return CouplingStep(Q_in=Q, anuga_flux=flux)
