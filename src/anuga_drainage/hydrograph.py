"""Per-inlet hydrograph logging for the 2D<->1D coupling.

Records, per inlet and per coupling step, the surface depth, the 1D pipe head,
the exchange discharge (split into capture and surcharge), an approach-flow
estimate and the resulting bypass, plus running cumulative volumes. Writes one
CSV per inlet whose columns are a **superset** of the Simple_SW_Inlets hydrograph
schema, so the same viewer can open the files.

``HydrographLogger.record`` takes plain arrays (no ANUGA), so it is unit-testable
standalone; the :class:`~anuga_drainage.Coupler` supplies the per-step samples.
"""
import os

import numpy as np
import pandas as pd

# Per-row schema. The first seven names are exactly the Simple_SW_Inlets viewer's
# required headers (so it opens these CSVs); Head1D_m / Surcharge / Cum_Inflow /
# Cum_Surcharged are the coupled extras.
COLUMNS = [
    "Time_s", "Depth_m", "Head1D_m",
    "Approach_Q_cms", "Captured_Q_cms", "Surcharge_Q_cms", "Bypass_Q_cms",
    "Cum_Inflow_m3", "Cum_Captured_m3", "Cum_Surcharged_m3", "Cum_Bypassed_m3",
]


class HydrographLogger:
    """Accumulates per-inlet hydrograph rows over a coupled run.

    Parameters
    ----------
    names : sequence of str
        Inlet/junction ids, in the same order as the Coupler's inlets.
    """

    def __init__(self, names):
        self.names = list(names)
        self._logs = {n: [] for n in self.names}
        self._cum = {n: {"inflow": 0.0, "captured": 0.0,
                         "surcharged": 0.0, "bypassed": 0.0}
                     for n in self.names}

    def record(self, time, dt, depths, heads, approach_Q, exchange_Q):
        """Append one row per inlet for a coupling step.

        Parameters
        ----------
        time : float        -- simulation time at this step [s].
        dt : float          -- step length [s] (for cumulative volumes).
        depths : array      -- surface ponded depth per inlet [m].
        heads : array       -- 1D pipe hydraulic head per inlet [m].
        approach_Q : array  -- surface approach-flow estimate per inlet [m^3/s], >=0.
        exchange_Q : array  -- signed coupling discharge per inlet [m^3/s]:
                               positive = surface -> pipe (capture);
                               negative = pipe -> surface (surcharge).
        """
        depths = np.asarray(depths, dtype=float)
        heads = np.asarray(heads, dtype=float)
        approach_Q = np.asarray(approach_Q, dtype=float)
        exchange_Q = np.asarray(exchange_Q, dtype=float)

        for i, name in enumerate(self.names):
            q = float(exchange_Q[i])
            captured = max(0.0, q)
            surcharge = max(0.0, -q)
            approach = float(approach_Q[i])
            bypass = max(0.0, approach - captured)

            c = self._cum[name]
            c["inflow"] += approach * dt
            c["captured"] += captured * dt
            c["surcharged"] += surcharge * dt
            c["bypassed"] += bypass * dt

            self._logs[name].append({
                "Time_s": float(time),
                "Depth_m": float(depths[i]),
                "Head1D_m": float(heads[i]),
                "Approach_Q_cms": approach,
                "Captured_Q_cms": captured,
                "Surcharge_Q_cms": surcharge,
                "Bypass_Q_cms": bypass,
                "Cum_Inflow_m3": c["inflow"],
                "Cum_Captured_m3": c["captured"],
                "Cum_Surcharged_m3": c["surcharged"],
                "Cum_Bypassed_m3": c["bypassed"],
            })

    def to_dataframe(self, name):
        """Per-inlet log as a DataFrame with an Asset_ID column prepended."""
        rows = self._logs.get(name)
        if not rows:
            return pd.DataFrame(columns=["Asset_ID"] + COLUMNS)
        df = pd.DataFrame(rows)
        df.insert(0, "Asset_ID", name)
        return df

    def write_csv(self, directory=".", prefix="hydrograph_"):
        """Write one ``<prefix><name>.csv`` per inlet; returns the paths written."""
        paths = []
        for name in self.names:
            df = self.to_dataframe(name)
            if df.empty:
                continue
            path = os.path.join(directory, f"{prefix}{name}.csv")
            df.to_csv(path, index=False)
            paths.append(path)
        return paths
