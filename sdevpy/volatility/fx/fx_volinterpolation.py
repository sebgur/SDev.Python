import datetime as dt
import logging
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import dates as dts
from sdevpy.maths.interpolation import create_interpolation, Interpolation
from sdevpy.market.fx.fxconventions import fx_market_yearfraction
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
            expiries: pillar expiries (dates)
            time_interp: 'var' interpolates s^2 * t, 'vol2' interpolates s^2, 'vol' interpolates s.
            time_extrap: 'flat' holds the first/last pillar vol outside the pillar range,
                         'linear' extrapolates the variance linearly.
    """
    def __init__(self, valdate: dt.datetime, expiries: list[dt.datetime], delta_interps: list[Interpolation], **kwargs):
        time_interp = kwargs.get('time_interp', 'var')
        time_extrap = kwargs.get('time_extrap', 'flat')

        # Check grid consistency
        n_expiries, n_interps = len(expiries), len(delta_interps)
        if n_expiries != n_interps:
            raise ValueError(f"Incompatible sizes between expiries and interpolations: {n_expiries}/{n_interps}")

        if n_expiries == 0:
            raise ValueError("At least one expiry pillar is required")

        # Time interpolation
        self.time_interp = time_interp.lower()
        if self.time_interp not in ('var', 'vol2', 'vol'):
            raise ValueError(f"Unknown time interpolation: {time_interp}")

        self.time_extrap = time_extrap.lower()
        if self.time_extrap not in ('flat', 'linear'):
            raise ValueError(f"Unknown time extrapolation: {time_extrap}")

        # Order pillars by increasing expiry
        order = sorted(range(n_expiries), key=lambda i: expiries[i])
        self.expiries = [expiries[i] for i in order]
        self.interps = [delta_interps[i] for i in order]

        # Conversion to times
        self.valdate = valdate
        self.times = self._to_times(self.expiries)
        # self.times = [fx_market_yearfraction(self.valdate, expiry) for expiry in self.expiries]
        if np.any(self.times <= 0.0):
            raise ValueError("Expiry pillars must be strictly after the valuation date")

        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("Duplicate/unordered expiry pillars")

    def vol_at_delta(self, expiry: npt.ArrayLike, put_delta: npt.ArrayLike) -> npt.ArrayLike:
        """ Vol at (expiry, put delta) """
        t = self._to_times(expiry)
        d = np.asarray(put_delta, dtype=float)
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
            if self.time_interp == 'var':
                w = s0 * s0 * t0 + theta * (s1 * s1 * t1 - s0 * s0 * t0)
                v = np.sqrt(np.maximum(w, 0.0) / tf)
            elif self.time_interp == 'vol2':
                w = s0 * s0 + theta * (s1 * s1 - s0 * s0)
                v = np.sqrt(np.maximum(w, 0.0))
            else: # vol
                v = s0 + theta * (s1 - s0)

            # Time extrapolation. Before the first pillar, always extrapolate as flat.
            # After the last pillar, extrapolate as flat or follow on the chosen interpolation.
            v = np.where(tf < self.times[0], s0, v)
            if self.time_extrap == 'flat': # hold the end pillar vols flat
                v = np.where(tf > self.times[-1], s1, v)

        return v.reshape(shape)

    def var_at_delta(self, expiry: npt.ArrayLike, put_delta: npt.ArrayLike) -> npt.ArrayLike:
        """ Variance sigma^2 * t at (expiry, put delta) """
        t = self._to_times(expiry)
        v = self.vol_at_delta(t, put_delta)
        return v * v * t

    def vol_grid(self, expiries: npt.ArrayLike, deltas: npt.ArrayLike) -> npt.ArrayLike:
        """ Vols on the full (expiry, delta) grid, shape (n_expiries, n_deltas) """
        t = np.atleast_1d(self._to_times(expiries))
        d = np.atleast_1d(np.asarray(deltas, dtype=float))
        return self.vol_at_delta(t[:, None], d[None, :])

    def calendar_check(self, deltas: npt.ArrayLike=(0.1, 0.25, 0.5, 0.75, 0.9)) -> bool:
        """ Variance must increase with expiry at every delta. Warn and return False if violated. """
        w = self.vol_grid(self.times, deltas) ** 2 * np.asarray(self.times)[:, None]
        bad = np.diff(w, axis=0) <= 0.0
        if np.any(bad):
            for i, j in zip(*np.nonzero(bad)):
                log.warning(f"Calendar arbitrage between pillars {i} and {i + 1} at delta {np.asarray(deltas)[j]}")
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

    def _to_times(self, expiries: npt.ArrayLike) -> npt.ArrayLike:
        """ Year fractions from valdate """
        arr = np.asarray(expiries)
        if arr.dtype.kind in 'fiu':
            return arr.astype(float)

        flat = [fx_market_yearfraction(self.valdate, e) for e in arr.reshape(-1).tolist()]
        return np.asarray(flat, dtype=float).reshape(arr.shape)


def interpolation_from_fxvol_data(vol_data: dict, **kwargs) -> FxVolInterpolation:
    """ Build the surface interpolation straight from the calibrated data as returned by
        CalibrationDataFileProvider.get_fxvol_data """
    smile_interp = kwargs.get('smile_interp', 'pchip') # pchip, akima, cubicspline, linear
    smile_extrap = kwargs.get('smile_extrap', 'builtin') # builtin, flat, use for both left and right
    time_interp = kwargs.get('time_interp', 'var') # var, vol2, vol
    time_extrap = kwargs.get('time_extrap', 'flat') # flat, linear

    valdate = dt.datetime.strptime(vol_data['date'], dts.DATE_FILE_FORMAT)
    expiries, interps = [], []
    for report in vol_data['tenor_reports']:
        interps.append(create_interpolation(interp=smile_interp, l_extrap=smile_extrap, r_extrap=smile_extrap,
                                            x_grid=report['deltas'], y_grid=report['vols']))
        expiries.append(dt.datetime.strptime(report['expiry'], dts.DATE_FILE_FORMAT))

    return FxVolInterpolation(valdate, expiries, interps, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sdevpy.calibration.fileprovider import CalibrationDataFileProvider

    pair = "USDJPY"
    valdate = dt.datetime(2025, 12, 15)

    # Retrieve calibrated data and build the two-dimensional interpolation
    cal_prov = CalibrationDataFileProvider()
    vol_data = cal_prov.get_fxvol_data(pair, valdate)
    data_sections = vol_data['tenor_reports']
    smile_interp = 'linear' # pchip, akima, cubicspline, linear
    smile_extrap = 'flat' # builtin, flat
    time_interp = 'var' # var, vol2, vol
    time_extrap = 'flat' # flat, linear
    surface = interpolation_from_fxvol_data(vol_data, smile_interp=smile_interp, smile_extrap=smile_extrap,
                                            time_interp=time_interp, time_extrap=time_extrap)
    surface.calendar_check()

    print(f"Expiries: {surface.expiries}")

    #### Check smile direction ####
    # Smile interpolation/extrapolation at chosen expiry pillar (assuming reports are time-ordered for simplicity)
    disp_expiry_idx = 3
    expiry = surface.expiries[disp_expiry_idx]
    print(f"Viewing smile at expiry: {expiry}")
    data_section = data_sections[disp_expiry_idx]
    data_deltas = data_section['deltas']
    data_vols = data_section['vols']
    disp_deltas = np.linspace(0.01, 0.99, 200)
    plt.plot(disp_deltas, surface.vol_at_delta(expiry, disp_deltas), color='green', label='smile interpolation')
    plt.scatter(data_deltas, data_vols, color='black', label='data')
    plt.legend(loc='upper right')
    plt.show()

    #### Check time direction ####
    e0, e1, e2, en = surface.expiries[0], surface.expiries[2], surface.expiries[3], surface.expiries[-1]
    d = 0.25
    v0, v1, v2, vn = surface.vol_at_delta(e0, d), surface.vol_at_delta(e1, d), surface.vol_at_delta(e2, d), \
                     surface.vol_at_delta(en, d)

    print(f"{e0}/{v0}")
    print(f"{e1}/{v1}")
    print(f"{e2}/{v2}")
    print(f"{en}/{vn}")

    # Before first pillar, check flat case
    dm1 = dt.datetime(2025, 12, 20)
    print(f"Before first pillar: {dm1}/{surface.vol_at_delta(dm1, d)}")

    # Between two pillars, check variance case
    date_t = dt.datetime(2026, 1, 14)
    vol_t = surface.vol_at_delta(date_t, d)
    t1, t2 = fx_market_yearfraction(valdate, e1), fx_market_yearfraction(valdate, e2)
    t = fx_market_yearfraction(valdate, date_t)
    var_1, var_2 = v1**2 * t1, v2**2 * t2
    var_t = var_1 + (var_2 - var_1) / (t2- t1) * (t - t1)
    vol_t_check = np.sqrt(var_t / t)
    print(f"Interp/check: {vol_t}/{vol_t_check}")

    # After last pillar
    print(f"Far vol: {surface.vol_at_delta(dt.datetime(2050, 1, 1), d)}")
