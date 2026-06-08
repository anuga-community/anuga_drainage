"""Parse SWMM ``.inp`` network files and convert them to pipedream DataFrames.

SWMM's ``.inp`` is the de-facto standard sewer-network format; pipedream has no
file format of its own (its networks are pandas DataFrames). This module reads
the network sections of a ``.inp`` and maps them onto pipedream's
``superjunctions`` / ``superlinks`` tables, so a single ``.inp`` can drive
either the SWMM or the pipedream coupling backend (and both then model the
*same* sewer).

Pure parsing/mapping — no ANUGA or pyswmm needed, so it is unit-testable
standalone, like ``read_inp_coordinates``.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# SWMM 5 ``.inp`` section -> column names. Rows may omit trailing optional
# fields (padded with None); only the columns we map are named.
_SECTION_COLUMNS = {
    "JUNCTIONS":   ["name", "elevation", "max_depth", "init_depth", "sur_depth", "aponded"],
    "OUTFALLS":    ["name", "elevation", "type", "stage_data", "gated", "route_to"],
    "CONDUITS":    ["name", "from_node", "to_node", "length", "roughness",
                    "in_offset", "out_offset", "init_flow", "max_flow"],
    "XSECTIONS":   ["link", "shape", "geom1", "geom2", "geom3", "geom4", "barrels", "culvert"],
    "COORDINATES": ["node", "x", "y"],
}
_NUMERIC = {
    "JUNCTIONS":   ["elevation", "max_depth", "init_depth", "sur_depth", "aponded"],
    "OUTFALLS":    ["elevation"],
    "CONDUITS":    ["length", "roughness", "in_offset", "out_offset", "init_flow", "max_flow"],
    "XSECTIONS":   ["geom1", "geom2", "geom3", "geom4", "barrels"],
    "COORDINATES": ["x", "y"],
}

# SWMM XSECTION shape -> pipedream geometry name. Common shapes map 1:1; SWMM
# shapes with no pipedream equivalent (EGG, HORSESHOE, ...) raise on conversion.
SHAPE_MAP = {
    "CIRCULAR":     "circular",
    "FORCE_MAIN":   "force_main",
    "RECT_CLOSED":  "rect_closed",
    "RECT_OPEN":    "rect_open",
    "TRAPEZOIDAL":  "trapezoidal",
    "TRIANGULAR":   "triangular",
    "PARABOLIC":    "parabolic",
    "ELLIPTICAL":   "elliptical",
    "IRREGULAR":    "irregular",
}


@dataclass
class InpNetwork:
    """Parsed SWMM ``.inp`` network sections (each a DataFrame)."""
    junctions: pd.DataFrame
    outfalls: pd.DataFrame
    conduits: pd.DataFrame
    xsections: pd.DataFrame
    coordinates: pd.DataFrame


def _read_sections(inp_path):
    """Return ``{SECTION: [row_tokens, ...]}`` for every ``[SECTION]``."""
    sections, current = {}, None
    with open(inp_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            if s.startswith("["):
                current = s.strip("[]").upper()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(s.split())
    return sections


def _section_df(rows, columns, numeric):
    padded = [list(r[:len(columns)]) + [None] * (len(columns) - len(r)) for r in rows]
    df = pd.DataFrame(padded, columns=columns)
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_inp(inp_path):
    """Parse the network sections of a SWMM ``.inp`` into an :class:`InpNetwork`."""
    sec = _read_sections(inp_path)
    return InpNetwork(**{
        attr: _section_df(sec.get(name, []), _SECTION_COLUMNS[name], _NUMERIC[name])
        for attr, name in [("junctions", "JUNCTIONS"), ("outfalls", "OUTFALLS"),
                           ("conduits", "CONDUITS"), ("xsections", "XSECTIONS"),
                           ("coordinates", "COORDINATES")]
    })


def _f(x, default=0.0):
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def inp_to_pipedream(inp, manhole_area=1.0, pit_area=1.0, h_0=1e-5):
    """Build pipedream ``(superjunctions, superlinks)`` DataFrames from a ``.inp``.

    A ``.inp`` doesn't carry a few pipedream-only parameters, so these are
    sensible, overridable defaults:

    - ``manhole_area`` — superjunction functional-storage surface area (curve ``c``);
    - ``pit_area``     — superlink internal-junction storage area ``A_s``;
    - ``h_0``          — initial depth where the ``.inp`` gives none.

    Outfalls become **boundary** superjunctions (``bc=True``). ``internal_links``
    is a ``SuperLink()`` constructor kwarg (not a column), so pass it there.
    """
    # Node table: junctions first, then outfalls; each gets a sequential id.
    nodes = []  # (name, z_inv, h_0, bc, max_depth)
    for _, r in inp.junctions.iterrows():
        d = _f(r["init_depth"], h_0) or h_0
        md = r["max_depth"]
        md = float(md) if md and not np.isnan(md) and md > 0 else np.inf
        nodes.append((r["name"], _f(r["elevation"]), d, False, md))
    for _, r in inp.outfalls.iterrows():
        nodes.append((r["name"], _f(r["elevation"]), h_0, True, np.inf))
    name_to_id = {n[0]: i for i, n in enumerate(nodes)}

    coords = inp.coordinates.set_index("node") if len(inp.coordinates) else None

    def xy(name):
        if coords is not None and name in coords.index:
            return _f(coords.loc[name, "x"]), _f(coords.loc[name, "y"])
        return 0.0, 0.0

    superjunctions = pd.DataFrame({
        "name": [n[0] for n in nodes],
        "id": list(range(len(nodes))),
        "z_inv": [n[1] for n in nodes],
        "h_0": [n[2] for n in nodes],
        "bc": [n[3] for n in nodes],
        "storage": ["functional"] * len(nodes),
        "a": [0.0] * len(nodes),
        "b": [0.0] * len(nodes),
        "c": [float(manhole_area)] * len(nodes),
        "max_depth": [n[4] for n in nodes],
        "map_x": [xy(n[0])[0] for n in nodes],
        "map_y": [xy(n[0])[1] for n in nodes],
    })

    xs = inp.xsections.set_index("link") if len(inp.xsections) else None
    rows = []
    for i, (_, c) in enumerate(inp.conduits.iterrows()):
        link = c["name"]
        if xs is None or link not in xs.index:
            raise ValueError(f"conduit {link!r} has no [XSECTIONS] entry")
        shape_raw = str(xs.loc[link, "shape"]).upper()
        if shape_raw not in SHAPE_MAP:
            raise ValueError(
                f"conduit {link!r}: SWMM shape {shape_raw!r} has no pipedream equivalent")
        for end in ("from_node", "to_node"):
            if c[end] not in name_to_id:
                raise ValueError(f"conduit {link!r}: node {c[end]!r} is not a junction/outfall")
        rows.append({
            "name": link, "id": i,
            "sj_0": name_to_id[c["from_node"]], "sj_1": name_to_id[c["to_node"]],
            "in_offset": _f(c["in_offset"]), "out_offset": _f(c["out_offset"]),
            "dx": _f(c["length"]), "n": _f(c["roughness"]),
            "shape": SHAPE_MAP[shape_raw],
            "g1": _f(xs.loc[link, "geom1"]), "g2": _f(xs.loc[link, "geom2"]),
            "g3": _f(xs.loc[link, "geom3"]), "g4": _f(xs.loc[link, "geom4"]),
            "Q_0": _f(c["init_flow"]), "h_0": h_0,
            "ctrl": False, "A_s": float(pit_area), "A_c": 0.0, "C": 0.0,
        })
    superlinks = pd.DataFrame(rows)
    return superjunctions, superlinks
