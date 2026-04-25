import json
import datetime as dt
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator
from sdevpy.volatility.impliedvol.models.svi import SviSection
from sdevpy.utilities import algos, dates


class LocalVol(ABC):
    """ Base class for Local Vol """
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', dt.datetime.now())
        self.name = kwargs.get('name', 'MyIndex')
        self.snapdate = kwargs.get('snapdate', self.valdate)

    @abstractmethod
    def value(self, t: float, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Values of the LV along array of log-moneynesses at given time """
        pass

    @abstractmethod
    def section(self, t: float):
        """ Retrieve LV section at given time t, i.e. a function of the log-moneyness log_m """
        pass

    def dump(self, file: str, indent: int=2) -> None:
        """ Dump LV object into json file """
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)

    @abstractmethod
    def dump_data(self) -> dict:
        """ Dump LV object into dictionary """
        pass


class InterpolatedParamLocalVol(LocalVol):
    """ Local Vol subtype defined as a series spot-parametric functions along the time direction """
    def __init__(self, sections, **kwargs):
        super().__init__(**kwargs)

        # Sort by increasing date
        sections.sort(key=lambda x: x.time)

        self.sections = sections
        self.t_grid = [section.time for section in self.sections]

    def value(self, t: float, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Values of the LV along array of log-moneynesses at given time """
        t_idx = algos.upper_bound(self.t_grid, t)
        return self.sections[t_idx].value(t, logm)

    def section(self, t_idx: int):
        """ Retrieve local vol section at given time index """
        return self.sections[t_idx]

    def check_params(self, t_idx: int):
        """ Check validity of parameters at given time index """
        return self.sections[t_idx].check_params()

    def update_params(self, t_idx: int, new_params: npt.ArrayLike) -> None:
        """ Update parameters at given time index """
        self.sections[t_idx].update_params(new_params)

    def params(self, t_idx: int) -> npt.ArrayLike:
        """ Retrieve parameters at given time index """
        return self.sections[t_idx].params

    def dump_data(self) -> dict:
        """ Dump LV object into dictionary """
        sections = []
        for section in self.sections:
            sections.append(section.dump())

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'sections': sections}
        return data


class MatrixLocalVol(LocalVol):
    """ Local Vol subtype by interpolation of matrix along expiry and strike directions.
        Interpolation is in both dimensions (defaulting to Linear).
        Possible choices are: linear, nearest, cubic, quintic and pchip.
        More recommended choices: linear, pchip and cubic, in order of increasing smoothness/oscillations.
        Extrapolation is flat (clamping to nearest boundary value).
    """
    def __init__(self, t_grid: npt.ArrayLike, logm_grid: npt.ArrayLike, vol_matrix: npt.ArrayLike, **kwargs):
        super().__init__(**kwargs)
        self.t_grid = np.asarray(t_grid)
        self.logm_grid = np.asarray(logm_grid)
        self.vol_matrix = np.asarray(vol_matrix)
        self.method = kwargs.get('interpolation', 'linear')

        # Check sizes
        expected = (len(self.t_grid), len(self.logm_grid))
        vol_shape = self.vol_matrix.shape
        if vol_shape != expected:
            raise ValueError(f"Vol matrix size {vol_shape} inconsistent with time/strike size {expected}")

        # Set interpolator
        tx_axis = (self.t_grid, self.logm_grid)
        self.interp = RegularGridInterpolator(tx_axis, self.vol_matrix, method=self.method,
                                              bounds_error=False, fill_value=None)

    def value(self, t: float, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Interpolate local vol matrix at time t over array of log-moneynesses """
        scalar = np.ndim(logm) == 0
        logm_ = np.atleast_1d(np.asarray(logm, dtype=float))

        # Clamp to grid bounds for flat extrapolation
        t_c = float(np.clip(t, self.t_grid[0], self.t_grid[-1]))
        logm_c = np.clip(logm_, self.logm_grid[0], self.logm_grid[-1])

        pts = np.column_stack([np.full_like(logm_c, t_c), logm_c])
        result = self.interp(pts)
        return float(result[0]) if scalar else result

    def section(self, t: float):
        """ Retrieve LV section at given time t, i.e. a function of the log-moneyness log_m """
        return lambda logm: self.value(t, logm)

    def dump_data(self) -> dict:
        """ Dump object to dictionary """
        return {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT), 't_grid': self.t_grid.tolist(),
                'logm_grid': self.logm_grid.tolist(), 'vol_matrix': self.vol_matrix.tolist()}


if __name__ == "__main__":
    t_grid = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    alnv = 0.25
    a = alnv**2 * np.sqrt(1.0) # a > 0
    b = 0.2 # b > 0
    rho = 0.0 # -1 < rho < 1
    m = 0.5 # No constraints
    sigma = 0.25 # > 0
    params = [a, b, rho, m, sigma]

    section_grid = [SviSection(t_grid[i]) for i in range(t_grid.shape[0])]
    lv = InterpolatedParamLocalVol(section_grid)

    # Initialize parameters
    for t_idx in range(len(t_grid)):
        lv.update_params(t_idx, params)

    lv_t = 0.75
    lv_logm = 0.0
    lvol = lv.value(lv_t, lv_logm)
    print(f"T/LOG-M/LV: {lv_t}/{lv_logm}/{lvol}")
