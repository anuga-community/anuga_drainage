import numpy as np


def calculate_Q(head1D, depth2D, bed2D, length_weir, area_manhole,
                cw=0.67, co=0.67, eps=1e-14, g=None):
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
    g : gravitational acceleration [m/s^2]. Defaults to ANUGA's value (imported
        lazily) so the function can also be used standalone by passing g.
    """
    if g is None:
        from anuga import g

    with np.errstate(invalid='ignore'):
        Q = np.zeros_like(head1D)

        # head1D < bed2D: free weir inflow (Reference Eq. 10).
        Q = np.where(head1D < bed2D,
                     cw * length_weir * depth2D * np.sqrt(2 * g * depth2D), Q)

        # bed2D <= head1D < depth2D + bed2D: orifice inflow (Eq. 11).
        Q = np.where(np.logical_and(bed2D <= head1D, head1D < depth2D + bed2D),
                     co * area_manhole * np.sqrt(2 * g * (depth2D + bed2D - head1D)), Q)

        # head1D > depth2D + bed2D: orifice surcharge back onto the surface (Eq. 11).
        Q = np.where(head1D > depth2D + bed2D + eps,
                     -co * area_manhole * np.sqrt(2 * g * (head1D - depth2D - bed2D)), Q)

    return Q
