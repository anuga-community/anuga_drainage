# The SWMM `.inp` as the standard input

SWMM's `.inp` is a documented, GUI-editable, tool-exported (e.g. DRAINS) format.
pipedream has none — its networks are pandas DataFrames. `anuga_drainage` bridges
the two so **one `.inp` drives either backend** and both model the same sewer.

## Parsing: `read_inp`

{func}`~anuga_drainage.read_inp` is a zero-dependency parser of the network
sections into an {class}`~anuga_drainage.InpNetwork` of DataFrames:

```python
from anuga_drainage import read_inp

inp = read_inp('network.inp')
inp.junctions      # [JUNCTIONS]
inp.outfalls       # [OUTFALLS]
inp.conduits       # [CONDUITS]
inp.xsections      # [XSECTIONS]
inp.coordinates    # [COORDINATES]
```

## Conversion: `inp_to_pipedream`

{func}`~anuga_drainage.inp_to_pipedream` maps an `InpNetwork` onto pipedream's
`(superjunctions, superlinks)` DataFrames:

| `.inp` section | → pipedream |
| --- | --- |
| `[JUNCTIONS]` | `superjunctions` (`z_inv`, `h_0`) |
| `[OUTFALLS]` | boundary `superjunctions` (`bc=True`) |
| `[CONDUITS]` | `superlinks` (`sj_0`/`sj_1`, `dx`, `n`) |
| `[XSECTIONS]` | `superlinks` (`shape`, `g1`–`g4`) |
| `[COORDINATES]` | ANUGA inlet locations |

```python
from anuga_drainage import inp_to_pipedream

superjunctions, superlinks = inp_to_pipedream(inp, manhole_area=1.0)
```

### Cross-section geometry

Shape mapping is **per-shape**, not a positional copy, because pipedream
parametrises some channels differently from SWMM:

- circular / rectangular / parabolic / ellipse — height/width copied directly;
- **triangular** — SWMM's top width → pipedream's side slope `m = Geom2/(2·Geom1)`;
- **trapezoidal** — SWMM's two bank slopes → pipedream's single mean slope;
- **force-main** — diameter kept; `g2` is a Preissmann-slot ratio (SWMM's
  `Geom2` there is roughness, not geometry).

Unsupported shapes raise a clear error (`IRREGULAR` needs `[TRANSECTS]`; shapes
pipedream lacks such as `EGG`/`HORSESHOE` are rejected).

### Parameters the `.inp` doesn't carry

A few pipedream-only quantities default (and are overridable): `manhole_area`
(superjunction storage), `pit_area` (internal-junction storage), and
`internal_links` (a `SuperLink()` kwarg).

```{admonition} max_depth: pipedream has no flooding model
:class: important
SWMM *floods* above a node's `MaxDepth` (water returns to the surface — mass
conserved). pipedream has no flooding model and would instead **cap the head at
`max_depth` and silently lose the surcharge**. So `inp_to_pipedream` defaults
the superjunctions to **uncapped** (`max_depth=inf`, i.e. `cap_max_depth=False`)
and leaves surcharge to the *coupling* to push back onto the 2D surface
(`calculate_Q` reverses when the pipe head exceeds the surface). Set
`cap_max_depth=True` to honour the `.inp` `MaxDepth` (and accept the loss).
```

```{admonition} Closed conduits surcharge cleanly
:class: tip
For pipedream, prefer **closed** cross-sections (`RECT_CLOSED`, `CIRCULAR`, …)
for conduits that can surcharge — they pressurise via a Preissmann slot and
conserve. An open channel (`RECT_OPEN`) loses the overflow.
```
