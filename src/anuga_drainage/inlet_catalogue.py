"""Named stormwater inlet asset catalogue.

Maps standard inlet types (grates, lintels, combos) to their hydraulic geometry
-- the clear opening area and the effective weir perimeter -- with an optional
blockage that derates both. Pure data + helpers (no ANUGA, no SWMM), so this is
importable and unit-testable standalone.

It is intended to supply per-junction area/perimeter to the weir/orifice coupling
(``calculate_Q`` / ``couple_from_inp``): a spec's ``operational_area`` feeds the
orifice ``area_manhole`` and its ``operational_perimeter`` feeds ``length_weir``.

Geometry values are *representative* standard-inlet figures; see the source
project (Simple_SW_Inlets) docs for how they are derived.
"""

try:
    import tomllib            # Python 3.11+
except ModuleNotFoundError:   # pragma: no cover - fallback for older Pythons
    import tomli as tomllib


class InletSpec:
    """Geometric parameters of a standard inlet asset.

    Parameters
    ----------
    name : str
        Catalogue key / label.
    clear_area : float
        Clear opening area [m^2] (the orifice area).
    effective_perimeter : float
        Effective weir perimeter / crest length [m].
    blockage : float, optional
        Clogging fraction, 0.0 (clear) .. 1.0 (fully blocked); derates both the
        area and the perimeter. Default 0.0.
    """

    def __init__(self, name, clear_area, effective_perimeter, blockage=0.0):
        self.name = name
        self.clear_area = clear_area
        self.effective_perimeter = effective_perimeter
        self.blockage = blockage

    @property
    def operational_area(self):
        """Clear area derated by blockage [m^2]."""
        return self.clear_area * (1.0 - self.blockage)

    @property
    def operational_perimeter(self):
        """Effective perimeter derated by blockage [m]."""
        return self.effective_perimeter * (1.0 - self.blockage)

    def __repr__(self):
        return (f"InletSpec({self.name!r}, clear_area={self.clear_area}, "
                f"effective_perimeter={self.effective_perimeter}, "
                f"blockage={self.blockage})")


# Catalogue of representative standard inlets, keyed by name.
INLET_LIBRARY = {
    "Grate_600x600":   InletSpec("Grate_600x600", 0.21, 2.40),
    "Grate_900x900":   InletSpec("Grate_900x900", 0.48, 3.60),
    "Lintel_1.2m":     InletSpec("Lintel_1.2m",   0.18, 1.20),
    "Lintel_2.4m":     InletSpec("Lintel_2.4m",   0.36, 2.40),
    "Combo_1.2m_G600": InletSpec("Combo_1.2m_G600", 0.39, 3.00),
    "Combo_2.4m_G900": InletSpec("Combo_2.4m_G900", 0.84, 5.10),
}


def load_inlet_library(path):
    """Load an inlet asset catalogue from a TOML file.

    Expected layout (one table per named inlet)::

        [inlets.Grate_600x600]
        clear_area = 0.21
        effective_perimeter = 2.40

    Inlet names containing a ``.`` must be quoted, e.g. ``[inlets."Lintel_1.2m"]``,
    otherwise TOML reads them as nested tables.

    Returns
    -------
    dict
        ``{name: InletSpec}``.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    inlets = data.get("inlets", {})
    if not inlets:
        raise ValueError(f"No [inlets.*] tables found in {path}")

    library = {}
    for name, props in inlets.items():
        try:
            library[name] = InletSpec(
                name, props["clear_area"], props["effective_perimeter"],
                blockage=props.get("blockage", 0.0))
        except KeyError as e:
            raise ValueError(
                f"Inlet '{name}' in {path} is missing required key {e}") from e
    return library


def resolve_inlet_spec(spec_ref, library=None, blockage=0.0):
    """Resolve a spec reference to an InletSpec, applying a blockage.

    ``spec_ref`` is either a catalogue key (looked up in ``library``, default
    INLET_LIBRARY) or an InletSpec instance. The returned spec carries the given
    ``blockage`` (overriding any on the source), so its ``operational_area`` /
    ``operational_perimeter`` are the values to feed the weir/orifice coupling.
    """
    if library is None:
        library = INLET_LIBRARY
    if isinstance(spec_ref, InletSpec):
        base = spec_ref
    elif spec_ref in library:
        base = library[spec_ref]
    else:
        raise KeyError(
            f"Inlet spec {spec_ref!r} not found in library "
            f"(known: {sorted(library)})")
    return InletSpec(base.name, base.clear_area, base.effective_perimeter, blockage)
