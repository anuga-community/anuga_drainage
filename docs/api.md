# API reference

Everything below is importable from the top-level `anuga_drainage` package.

## Coupling from a `.inp`

```{eval-rst}
.. autofunction:: anuga_drainage.couple_from_inp

.. autoclass:: anuga_drainage.Coupling
   :members:
```

## The exchange flux

```{eval-rst}
.. autofunction:: anuga_drainage.calculate_Q
```

## The coupling driver

```{eval-rst}
.. autoclass:: anuga_drainage.Coupler
   :members:

.. autoclass:: anuga_drainage.SwmmBackend
   :members:

.. autoclass:: anuga_drainage.PipedreamBackend
   :members:

.. autofunction:: anuga_drainage.smooth_Q

.. autofunction:: anuga_drainage.limit_outflow
```

## Volume balance

```{eval-rst}
.. autoclass:: anuga_drainage.VolumeBalance
   :members:
```

## SWMM `.inp` parsing & conversion

```{eval-rst}
.. autofunction:: anuga_drainage.read_inp

.. autoclass:: anuga_drainage.InpNetwork
   :members:

.. autofunction:: anuga_drainage.inp_to_pipedream
```

## Inlet geometry helpers

```{eval-rst}
.. autofunction:: anuga_drainage.inlet_initialization.read_inp_coordinates

.. autofunction:: anuga_drainage.inlet_initialization.n_sided_inlet

.. autofunction:: anuga_drainage.inlet_initialization.initialize_inlets
```
