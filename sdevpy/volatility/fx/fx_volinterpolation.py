import datetime as dt
import logging
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import dates as dts
from sdevpy.utilities.timegrids import model_time
from sdevpy.maths.interpolation import create_interpolation
log = logging.getLogger(__name__)


class FxVolInterpolation:
    """ Two-dimensional interpolation of the FX vol surface along expiry and delta directions.

        Delta direction: one Interpolation object per expiry pillar.
        Expiry direction: the two surrounding pillar smiles are read at the requested put delta,
                          giving vols s0 at t0 and s1 at t1. Those are then combined linearly in
                          variance, i.e.
                          w(t) = w0 + (t - t0) / (t1 - t0) * (w1 - w0),  w_i = s_i^2 * t_i,
                          and the result is returned as sqrt(w(t) / t).

        Args:
            expiries: pillar expiries. Dates, DATE_FILE_FORMAT strings or year fractions.
            delta_interps: one Interpolation per pillar, same order as expiries.
            time_extrap: 'flat' holds the first/last pillar vol outside the pillar range,
                         'linear' keeps extrapolating the total variance linearly.
            var_mode: 'total' interpolates s^2 * t, 'plain' interpolates s^2.
    """
    def __init__(self, valdate: dt.datetime, expiries, delta_interps,
                 time_extrap: str='flat', var_mode: str='total'):
        if len(expiries) != len(delta_interps):
            raise ValueError(f"Got {len(expiries)} expiries but {len(delta_interps)} interpolations")
        if len(expiries) == 0:
            raise ValueError("At least one expiry pillar is required")

        self.valdate = valdate
        self.time_extrap = time_extrap.lower()
        if self.time_extrap not in ('flat', 'linear'):
            raise ValueError(f"Unknown time extrapolation: {time_extrap}")
        self.var_mode = var_mode.lower()
        if self.var_mode not in ('total', 'plain'):
            raise ValueError(f"Unknown variance mode: {var_mode}")

        # Order pillars by increasing time
        times = self._to_times(expiries)
        if np.any(times <= 0.0):
            raise ValueError("Expiry pillars must be strictly after the valuation date")
        order = np.argsort(times)
        self.times = times[order]
        self.interps = [delta_interps[i] for i in order]
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("Duplicate expiry pillars")

    def vol(self, expiry, delta: npt.ArrayLike) -> npt.ArrayLike:
        """ Vol at (expiry, put delta). Arguments broadcast; the result has the broadcast shape """
        t = self._to_times(expiry)
        d = np.asarray(delta, dtype=float)
        t, d = np.broadcast_arrays(t, d)
        shape = t.shape
        tf, df = t.reshape(-1), d.reshape(-1)

        if tf.size and np.any(tf <= 0.0):
            raise ValueError("Requested expiries must be strictly after the valuation date")

        # Locate the surrounding pillars, one bracket per requested point
        i0, i1 = self._brackets(tf)
        t0, t1 = self.times[i0], self.times[i1]

        # Read both surrounding smiles at the requested deltas
        s0 = self._smile_values(i0, df)
        s1 = self._smile_values(i1, df)

        # Interpolate linearly in variance
        if np.all(i0 == i1): # single pillar surface: flat in time
            v = s0
        else:
            theta = np.where(t1 > t0, (tf - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
            if self.var_mode == 'total':
                w = s0 * s0 * t0 + theta * (s1 * s1 * t1 - s0 * s0 * t0)
                v = np.sqrt(np.maximum(w, 0.0) / tf)
            else:
                w = s0 * s0 + theta * (s1 * s1 - s0 * s0)
                v = np.sqrt(np.maximum(w, 0.0))

            if self.time_extrap == 'flat': # hold the end pillar vols flat
                v = np.where(tf < self.times[0], s0, v)
                v = np.where(tf > self.times[-1], s1, v)

        return v.reshape(shape)

    def vol_grid(self, expiries, deltas: npt.ArrayLike) -> npt.ArrayLike:
        """ Vols on the full (expiry, delta) grid, shape (n_expiries, n_deltas) """
        t = np.atleast_1d(self._to_times(expiries))
        d = np.atleast_1d(np.asarray(deltas, dtype=float))
        return self.vol(t[:, None], d[None, :])

    def total_variance(self, expiry, delta: npt.ArrayLike) -> npt.ArrayLike:
        """ Total variance sigma^2 * t at (expiry, put delta) """
        t = self._to_times(expiry)
        v = self.vol(t, delta)
        return v * v * t

    def calendar_check(self, deltas: npt.ArrayLike=(0.1, 0.25, 0.5, 0.75, 0.9)) -> bool:
        """ Total variance must increase with expiry at every delta, else the time interpolation
            can produce negative forward variance. Warns and returns False if violated. """
        w = self.vol_grid(self.times, deltas) ** 2 * np.asarray(self.times)[:, None]
        bad = np.diff(w, axis=0) <= 0.0
        if np.any(bad):
            for i, j in zip(*np.nonzero(bad)):
                log.warning(f"Calendar arbitrage between pillars {i} and {i + 1} "
                            f"at delta {np.asarray(deltas)[j]}")
        return not bool(np.any(bad))

    def _brackets(self, t: npt.ArrayLike) -> tuple:
        """ Indices of the pillars surrounding each t, clipped to the pillar range """
        n = len(self.times)
        if n == 1:
            z = np.zeros(np.shape(t), dtype=int)
            return z, z

        i1 = np.clip(np.searchsorted(self.times, t, side='left'), 1, n - 1)
        return i1 - 1, i1

    def _smile_values(self, idx: npt.ArrayLike, d: npt.ArrayLike) -> npt.ArrayLike:
        """ Vols at deltas d, each point read off the smile of its own pillar idx. Loops over the
            distinct pillars touched rather than over points, so the delta interpolation stays
            vectorized: one call per pillar on the sub-vector that needs it. """
        out = np.empty(d.shape, dtype=float)
        for k in np.unique(idx):
            mask = (idx == k)
            out[mask] = np.asarray(self.interps[k].value(d[mask]), dtype=float).reshape(-1)
        return out

    def _to_times(self, expiries) -> npt.ArrayLike:
        """ Year fractions from valdate. Accepts dates, DATE_FILE_FORMAT strings or year
            fractions, scalar or array-like, and preserves the input shape. """
        arr = np.asarray(expiries)
        if arr.dtype.kind in 'fiu':
            return arr.astype(float)

        flat = [self._to_time(e) for e in arr.reshape(-1).tolist()]
        return np.asarray(flat, dtype=float).reshape(arr.shape)

    def _to_time(self, expiry) -> float:
        """ Year fraction from valdate for a single expiry """
        if isinstance(expiry, str):
            expiry = dt.datetime.strptime(expiry, dts.DATE_FILE_FORMAT)
        if isinstance(expiry, dt.date):
            return float(model_time(self.valdate, expiry))
        return float(expiry)


def interpolation_from_fxvol_data(vol_data: dict, interp: str='pchip', l_extrap: str='builtin',
                                  r_extrap: str='builtin', **kwargs) -> FxVolInterpolation:
    """ Build the surface interpolation straight from the calibrated data as returned by
        CalibrationDataFileProvider.get_fxvol_data """
    valdate = dt.datetime.strptime(vol_data['date'], dts.DATE_FILE_FORMAT)
    expiries, interps = [], []
    for report in vol_data['tenor_reports']:
        interps.append(create_interpolation(interp=interp, l_extrap=l_extrap, r_extrap=r_extrap,
                                            x_grid=report['deltas'], y_grid=report['vols']))
        expiries.append(report['expiry'])

    return FxVolInterpolation(valdate, expiries, interps, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sdevpy.calibration.fileprovider import CalibrationDataFileProvider

    pair = "USDJPY"
    valdate = dt.datetime(2025, 12, 15)

    # Retrieve calibrated data and build the two-dimensional interpolation
    cal_prov = CalibrationDataFileProvider()
    vol_data = cal_prov.get_fxvol_data(pair, valdate)
    surface = interpolation_from_fxvol_data(vol_data)
    surface.calendar_check()

    print(f"Pillars (y): {np.round(surface.times, 4)}")

    # Single off-pillar expiry, vectorized in delta
    disp_deltas = np.linspace(0.05, 0.95, 100)
    print(f"6m 25d put vol: {surface.vol(0.5, 0.25):.6f}")

    # Vectorized in both directions at once
    disp_times = np.linspace(surface.times[0], surface.times[-1], 60)
    vols = surface.vol_grid(disp_times, disp_deltas)
    print(f"Grid shape: {vols.shape}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Smiles at pillars vs off-pillar
    for t in (surface.times[0], surface.times[1], 0.5 * (surface.times[0] + surface.times[1])):
        ax1.plot(disp_deltas, surface.vol(t, disp_deltas), label=f"t={t:.3f}")
    ax1.set_xlabel('Put delta')
    ax1.set_ylabel('Vol')
    ax1.set_title('Smiles, pillars and interpolated')
    ax1.legend()

    # Term structures at fixed deltas
    for d in (0.10, 0.25, 0.50, 0.75, 0.90):
        ax2.plot(disp_times, surface.vol(disp_times, d), label=f"d={d:.2f}")
    ax2.scatter(np.repeat(surface.times, 1), surface.vol(surface.times, 0.50), color='black',
                label='ATM pillars')
    ax2.set_xlabel('Expiry (y)')
    ax2.set_ylabel('Vol')
    ax2.set_title('Term structure at fixed delta')
    ax2.legend()

    plt.tight_layout()
    plt.show()
