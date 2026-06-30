# Inlet asset catalogue

By default `couple_from_inp` sizes every coupling inlet from one number,
`manhole_area` (the surface footprint, which also sets the orifice area and weir
length in `calculate_Q`). The **inlet asset catalogue** lets you instead drive a
junction's exchange flux from a *named, real* inlet type — a grate or lintel with
its own clear opening area and weir perimeter — and to model **clogging** with a
blockage fraction.

```{admonition} Key idea
:class: tip
A catalogued inlet sets the *hydraulics* (`area_manhole` / `length_weir` in
`calculate_Q`) **independently of the surface footprint**. The ANUGA coupling
region and the pipedream storage area still come from `manhole_area` /
`inlet_polygons`, so a small grate opening (~0.2 m²) drives the flux without
shrinking the footprint it sits in.
```

## `InletSpec` and the built-in library

An `InletSpec` holds the geometry of a standard inlet; `operational_area` and
`operational_perimeter` apply the blockage:

```python
from anuga_drainage import InletSpec

spec = InletSpec("Grate_600x600", clear_area=0.21, effective_perimeter=2.40)
spec.operational_area          # 0.21
spec.operational_perimeter     # 2.40

blocked = InletSpec("Grate_600x600", 0.21, 2.40, blockage=0.5)
blocked.operational_area       # 0.105  (50% clogged)
blocked.operational_perimeter  # 1.20
```

`INLET_LIBRARY` is a catalogue of representative standard inlets:

| Key | clear_area (m²) | effective_perimeter (m) |
|-----|-----------------|--------------------------|
| `Grate_600x600`   | 0.21 | 2.40 |
| `Grate_900x900`   | 0.48 | 3.60 |
| `Lintel_1.2m`     | 0.18 | 1.20 |
| `Lintel_2.4m`     | 0.36 | 2.40 |
| `Combo_1.2m_G600` | 0.39 | 3.00 |
| `Combo_2.4m_G900` | 0.84 | 5.10 |

```python
from anuga_drainage import INLET_LIBRARY
INLET_LIBRARY["Grate_900x900"].operational_area    # 0.48
```

## Assigning inlets per junction

Pass `inlet_specs` to `couple_from_inp` to give specific junctions a catalogue
inlet; `blockage` may be a scalar (all spec'd junctions) or a per-junction dict:

```python
from anuga_drainage import couple_from_inp

coupling = couple_from_inp(
    domain, "network.inp", backend="pipedream",
    manhole_area=1.0,                       # surface footprint (unchanged role)
    inlet_specs={                           # which junctions use a catalogue inlet
        "J1": "Grate_600x600",
        "J2": "Combo_2.4m_G900",
    },
    blockage={"J1": 0.4},                   # J1 40% clogged; others clear
)
```

For junction `J1` above, `calculate_Q` now uses `area_manhole = 0.21·0.6` and
`length_weir = 2.40·0.6`, while its ANUGA region (and pipedream storage) stay
sized to `manhole_area = 1.0`. Junctions **without** an `inlet_specs` entry are
unchanged — they keep the footprint-derived geometry.

You can also pass an `InletSpec` directly (handy for one-off geometry), and an
inline value of `blockage` baked into the spec is honoured:

```python
from anuga_drainage import InletSpec
coupling = couple_from_inp(domain, "network.inp",
                           inlet_specs={"J3": InletSpec("Custom", 0.30, 2.0)})
```

## Loading a catalogue from TOML

Keep your own catalogue in a TOML file and load it with `load_inlet_library`.
Each inlet is an `[inlets.<name>]` table (`blockage` optional):

```toml
# inlets.toml
[inlets.Grate_600x600]
clear_area = 0.21
effective_perimeter = 2.40

[inlets."Lintel_1.2m"]      # quote names containing a "." or TOML nests them
clear_area = 0.18
effective_perimeter = 1.20
```

```python
from anuga_drainage import load_inlet_library, couple_from_inp

library = load_inlet_library("inlets.toml")       # {name: InletSpec}
coupling = couple_from_inp(domain, "network.inp",
                           inlet_specs={"J1": "Grate_600x600"},
                           library=library)
```

```{admonition} TOML gotcha
:class: warning
Inlet names with a dot (`Lintel_1.2m`, `Combo_1.2m_G600`) **must be quoted** in
the table header — `[inlets."Lintel_1.2m"]` — otherwise TOML reads them as
nested tables and the loader raises a `ValueError`.
```

## Helper

`resolve_inlet_spec(spec_ref, library=None, blockage=0.0)` resolves a name *or*
an `InletSpec` against a library and applies a blockage, returning the derated
`InletSpec`. `couple_from_inp` uses it internally; it is exported for direct use.

```{admonition} Coefficient note
:class: note
The catalogue's areas/perimeters feed the existing `calculate_Q` (default
`cw = co = 0.67`, Leandro & Martins). Those clear-area / weir-perimeter figures
follow the HEC-22 grate-inlet convention; if you port results from a HEC-22
capture model note its weir coefficient differs — see the source project's
hydraulics notes.
```
