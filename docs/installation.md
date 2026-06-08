# Installation

`anuga_drainage` assumes you already have a working **ANUGA** in your
environment — ANUGA is not installed by this package. On top of that:

```bash
pip install -e .            # the anuga_drainage package (needs numpy, pandas)
pip install pyswmm          # SWMM backend (standard PyPI release, >= 2.1)
pip install "pipedream-solver @ git+https://github.com/mdbartos/pipedream.git"
```

The package itself only depends on `numpy` and `pandas`; the two 1D backends are
optional extras, installed only for the backend(s) you use:

```bash
pip install -e .[swmm]            # pyswmm
pip install -e .[pipedream]       # pipedream (from git, see below)
pip install -e .[test]            # pytest
```

## Backend notes

```{admonition} pipedream must come from git, not PyPI
:class: warning
The released `pipedream-solver` (0.2.2) uses `np.bool8`, removed in numpy 2.x
(which a current ANUGA requires), so it crashes on `SuperLink(...)`
construction. Git master replaced those with `np.bool_`. Install it from git as
shown above.
```

### SWMM / pyswmm 2.1 stepping constraints

Stock pyswmm 2.1 is **whole-second resolution**: the coupling stride must be an
integer number of seconds (`int(dt)`), and SWMM coupling therefore exchanges at
1-second granularity. The `Coupler` handles this for you. Sub-second coupling is
only available on the **pipedream** path (its step is pure Python).

## From-scratch conda environment

For a reproducible setup, the repository ships an `environment.yml` that builds
a conda environment with ANUGA from conda-forge plus this package and both
backends:

```bash
conda env create -f environment.yml
```

## Running the tests

The package's own physics is tested with `pytest`:

```bash
pip install -e .[test]
pytest
```

The pure-logic tests (geometry, the `.inp` parser, `calculate_Q` with an
explicit gravity) run without ANUGA; ANUGA / pyswmm / pipedream-dependent tests
skip automatically when those aren't installed.
