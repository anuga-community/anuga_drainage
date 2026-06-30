import numpy as np


def calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole,
                cw=0.67, co=0.67, min_head=1.0e-3, g=None):
    """Coupling discharge between the 2D (surface) and 1D (pipe) models.

    Based on the weir and orifice equations of:

        A methodology for linking 2D overland flow models with the sewer network
        model SWMM 5.1 based on dynamic link libraries.
        Leandro & Martins, Water Science and Technology, 2016, 73, 3017-3026.

    Sign convention: positive Q = water leaving the 2D surface and entering the
    1D network; negative Q = surcharge back up onto the surface. Vectorised over
    numpy arrays of inlets.

    Parameters
    ----------
    cw, co : weir / orifice discharge coefficients.
    length_weir : weir crest width [m].
    area_manhole : manhole area [m^2].
    min_head : deadband on the driving head [m]. No exchange is computed when the
        relevant head difference is below this. It suppresses spurious exchange
        from sub-millimetre, numerically-noisy head differences -- e.g. when a 1D
        solver initialises a junction head a hair (~1e-5 m) above the bed, which
        would otherwise surcharge onto a dry surface and set off a growing
        capture/surcharge oscillation before any real water arrives.
    g : gravitational acceleration [m/s^2]. Defaults to ANUGA's value (imported
        lazily) so the function can also be used standalone by passing g.
    """
    if g is None:
        from anuga import g

    with np.errstate(invalid='ignore'):
        Q = np.zeros_like(head1D)

        # head1D < bed2D: free weir inflow (Reference Eq. 10). The depth2D factor
        # already vanishes on a dry surface; the deadband ignores a negligible film.
        Q = np.where(np.logical_and(head1D < bed2D, depth2D > min_head),
                     cw * length_weir * depth2D * np.sqrt(2 * g * depth2D), Q)

        # bed2D <= head1D < depth2D + bed2D: orifice inflow (Eq. 11). Only fires
        # when the surface water level sits at least min_head above the pipe head.
        Q = np.where(np.logical_and(bed2D <= head1D, head1D < depth2D + bed2D - min_head),
                     co * area_manhole * np.sqrt(2 * g * (depth2D + bed2D - head1D)), Q)

        # head1D > depth2D + bed2D: orifice surcharge back onto the surface (Eq. 11).
        Q = np.where(head1D > depth2D + bed2D + min_head,
                     -co * area_manhole * np.sqrt(2 * g * (head1D - depth2D - bed2D)), Q)

    return Q
