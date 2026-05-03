import json
import datetime as dt
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator
from sdevpy.utilities import algos, dates


class LocalVolSection(ABC):
    """ Base class for Local Vol time sections, i.e. functions that return the local volatility at a certain
        point in time t, for a list of log-moneynesses x. These sections are used by numerical methods,
        i.e. PDE and MC, to speed up the calculation along the spot direction, effectively by caching the
        time-dependent part of the interpolation that is done only once per time section. """
    def __init__(self, time: float):
        self.time = time

    @abstractmethod
    def value(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """ In the base, we simply use the formula on the parameters. In inherited classes, we
            may have a more complex behaviour such as applying the formula to a transformed
            set of parameters. """
        pass


class LocalVol(ABC):
    """ Base class for Local Vol """
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', dt.datetime.now())
        self.name = kwargs.get('name', 'MyIndex')
        self.snapdate = kwargs.get('snapdate', self.valdate)

    def value(self, t: float, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ In the base, we assume that the value at time t and log-moneynesses logm is obtained by
            first retrieving a time section, then returning the values in that time section
            at the requested log-moneynesses. This behaviour can always be overriden in inherited
            classes if ever necessary. In practice, more likely than not, we will store the sections
            in a list before launching the numerical method (e.g. PDE/MC) """
        section = self.section(t)
        return section.value(logm)

    @abstractmethod
    def section(self, t: float) -> LocalVolSection:
        """ Retrieve LV section at given time t, i.e. a function of the log-moneyness log_m.
            How to retrieve an LV section is a modelling question and done differently in 
            inherited classes. """
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


class TimeInterpolatedLocalVol(LocalVol):
    """ Local Vol subtype whose sections are found by interpolation along the time direction. More specifically,
        the interpolation is piecewise constant. """
    def __init__(self, sections: list[LocalVolSection], **kwargs):
        super().__init__(**kwargs)

        # Sort by increasing date
        sections.sort(key=lambda x: x.time)

        self.sections = sections
        self.t_grid = [section.time for section in self.sections]

    def section(self, t: float) -> LocalVolSection:
        """ The section is obtained by locating the requested time t on the time axis and
            picking the section at the pillar index above (upper_bound) """
        t_idx = algos.upper_bound(self.t_grid, t)
        t_idx = min(t_idx, len(self.sections) - 1) # Flat extrapolation beyond last pillar
        print(f"Section requested at {t}, using pillar at time/time index: {self.t_grid}/{t_idx}")
        return self.section_at_index(t_idx)

    def section_at_index(self, t_idx: int):
        """ Retrieve local vol section at given time index """
        return self.sections[t_idx]


class ParamLocalVolSection(LocalVolSection):#(ABC):
    """ Wrapper class around formulas that return a local volatility at a certain
        point in time t, for a list of log-moneynesses x. The section has parameters
        that can be optimized on, and it is used by the LocalVol subtype InterpolatedParamLocalVol. """
    def __init__(self, time: float,
                 formula: Callable[[float, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]):
        super().__init__(time)
        self.params = None
        self.formula = formula
        self.model = None
        # self.time = time

    # def value(self, t: npt.ArrayLike, x: npt.ArrayLike) -> npt.ArrayLike:
    #     """ In the base, we simply use the formula on the parameters. In inherited classes, we
    #         may have a more complex behaviour such as applying the formula to a transformed
    #         set of parameters. """
    #     return self.formula(self.time, x, self.params)

    def value(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """ Use the member formula at the member time """
        return self.formula(self.time, x, self.params)

    def update_params(self, new_params: npt.ArrayLike) -> None:
        """ In the base, we only copy the new parameters in. Inherited classes may do more. """
        self.params = new_params.copy()

    def check_params(self):
        """ In the base, all parameters are allowed so we always answer True and penalty = 0.0.
            Inherited classes may have constraints and calculate non-trivial penalties. """
        return True, 0.0

    def constraints(self):
        return None

    @abstractmethod
    def dump_params(self):
        pass

    def dump(self):
        data = {'time': self.time, 'model': self.model, 'params': self.dump_params()}
        return data


class InterpolatedParamLocalVol(LocalVol):
    """ Local Vol subtype defined as a series spot-parametric functions along the time direction """
    def __init__(self, sections: list[ParamLocalVolSection], **kwargs):
        super().__init__(**kwargs)

        # Sort by increasing date
        sections.sort(key=lambda x: x.time)

        self.sections = sections
        self.t_grid = [section.time for section in self.sections]

    def value(self, t: float, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Values of the LV along array of log-moneynesses at given time """
        t_idx = algos.upper_bound(self.t_grid, t)
        t_idx = min(t_idx, len(self.sections) - 1) # Flat extrapolation beyond last pillar
        return self.sections[t_idx].value(logm)
        # return self.sections[t_idx].value(t, logm)

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
    #### LocalVol as matrix interpolation ####
    print("Hello")

    # #### LocalVol interpolated by sections ####
    # from sdevpy.volatility.impliedvol.models.svi import SviSection

    # t_grid = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    # alnv = 0.25
    # a = alnv**2 * np.sqrt(1.0) # a > 0
    # b = 0.2 # b > 0
    # rho = 0.0 # -1 < rho < 1
    # m = 0.5 # No constraints
    # sigma = 0.25 # > 0
    # params = [a, b, rho, m, sigma]

    # section_grid = [SviSection(t_grid[i]) for i in range(t_grid.shape[0])]
    # lv = InterpolatedParamLocalVol(section_grid)

    # # Initialize parameters
    # for t_idx in range(len(t_grid)):
    #     lv.update_params(t_idx, params)

    # lv_t = 0.75
    # lv_logm = 0.0
    # lvol = lv.value(lv_t, lv_logm)
    # print(f"T/LOG-M/LV: {lv_t}/{lv_logm}/{lvol}")
