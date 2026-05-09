import json
import datetime as dt
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from sdevpy.utilities import algos, dates
from sdevpy.maths.interpolation import create_interpolation


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

    @abstractmethod
    def dump(self) -> dict:
        """ Base method (abstract) for dumping to dictionary """
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
        the time interpolation is piecewise constant. """
    def __init__(self, sections: list[LocalVolSection], **kwargs):
        super().__init__(**kwargs)

        # Sort by increasing date
        sections.sort(key=lambda x: x.time)

        # Save sections and extract time grid
        self.sections = sections
        self.t_grid = [section.time for section in self.sections]

    def section(self, t: float) -> LocalVolSection:
        """ The section is obtained by locating the requested time t on the time axis and
            picking the section at the pillar index above (upper_bound) """
        t_idx = algos.upper_bound(self.t_grid, t, clamp=True)
        # t_idx = min(t_idx, len(self.sections) - 1) # Flat extrapolation beyond last pillar

        # Old version with upper_bound and manual clamping
        # t_idx = algos.upper_bound(self.t_grid, t)
        # t_idx = min(t_idx, len(self.sections) - 1) # Flat extrapolation beyond last pillar

        # print(f"Section requested at {t}, using pillar at time index/time: {t_idx}/{self.t_grid[t_idx]}")
        return self.section_at_index(t_idx)

    def section_at_index(self, t_idx: int):
        """ Retrieve local vol section at given time index """
        return self.sections[t_idx]

    def dump_data(self) -> dict:
        """ Dump LV object into dictionary """
        sections = []
        for section in self.sections:
            sections.append(section.dump())

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'sections': sections}
        return data


class ParamLocalVolSection(LocalVolSection):
    """ Wrapper class around formulas that return a local volatility at a certain
        point in time t, for a list of log-moneynesses x. The section has parameters
        that can be optimized on, and it is used by the LocalVol subtype InterpolatedParamLocalVol. """
    def __init__(self, time: float,
                 formula: Callable[[float, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]):
        super().__init__(time)
        self.params = None
        self.formula = formula
        self.model = None

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

    def dump(self) -> dict:
        """ Dump to dictionary """
        data = {'time': self.time, 'model': self.model, 'params': self.dump_params()}
        return data


class InterpolatedParamLocalVol(TimeInterpolatedLocalVol):
    """ Local Vol subtype defined as a series spot-parametric functions along the time direction """
    def __init__(self, sections: list[ParamLocalVolSection], **kwargs):
        super().__init__(sections, **kwargs)

    def check_params(self, t_idx: int):
        """ Check validity of parameters at given time index """
        return self.section_at_index(t_idx).check_params()

    def update_params(self, t_idx: int, new_params: npt.ArrayLike) -> None:
        """ Update parameters at given time index """
        self.section_at_index(t_idx).update_params(new_params)

    def params(self, t_idx: int) -> npt.ArrayLike:
        """ Retrieve parameters at given time index """
        return self.section_at_index(t_idx).params


class InterpolatedLocalVolSection(LocalVolSection):
    """ Wrapper class around formulas that return a local volatility at a certain
        point in time t, for a list of log-moneynesses x. The section has parameters
        that can be optimized on, and it is used by the LocalVol subtype InterpolatedParamLocalVol. """
    def __init__(self, time: float, logm_list: list[float], vol_list: list[float], **kwargs):
        super().__init__(time)
        self.logm_list = logm_list
        self.vol_list = vol_list
        interp_type = kwargs.get('interpolation', 'cubicspline')
        self.interp_type = interp_type
        self.interp = create_interpolation(interp=interp_type, l_extrap='flat', r_extrap='flat')
        self.interp.set_data(logm_list, vol_list)

    def value(self, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Use the interpolation method along the log-moneyness direction """
        return self.interp.value(logm)

    def dump(self) -> dict:
        """ Dump to dictionary """
        data = {'time': self.time, 'model': self.interp_type, 'logm': self.logm_list,
                'vol': self.vol_list}
        return data


class MatrixLocalVol(TimeInterpolatedLocalVol):
    """ Local Vol subtype by interpolation of matrix along expiry and strike directions.
        Interpolation is piecewise constant in time.
        The time section indexed at t_{i+1} is valid over (t_i, t_{i+1}].
        Possible choices in the strike direction are: linear, cubicspline.
        Extrapolation is flat (clamping to nearest boundary value).
    """
    def __init__(self, t_grid: list[float], logm_matrix: list[list[float]], vol_matrix: list[list[float]], **kwargs):
        # Check sizes
        n_times = len(t_grid)
        if len(logm_matrix) != n_times:
            raise ValueError("Incompatible sizes between time grid and log-moneyness matrix")

        if len(vol_matrix) != n_times:
            raise ValueError("Incompatible sizes between time grid and vol matrix")

        # Create sections
        sections = []
        for i in range(n_times):
            t, logm, vol = t_grid[i], logm_matrix[i], vol_matrix[i]
            sections.append(InterpolatedLocalVolSection(t, logm, vol, **kwargs))

        # Instantiate base
        super().__init__(sections, **kwargs)


class FlatLocalVolSection(LocalVolSection):
    """ Flat volatility across moneynesses (Black-Scholes) """
    def __init__(self, time: float, vol: float):
        super().__init__(time)
        self.vol = vol

    def value(self, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Return flat vol """
        return self.vol

    def dump(self) -> dict:
        """ Dump to dictionary """
        data = {'time': self.time, 'model': 'flat', 'vol': self.vol}
        return data


class VectorLocalVol(TimeInterpolatedLocalVol):
    """ Local Vol subtype returning one number per expiry section, with no spot dependence.
        Interpolation is piecewise constant in time.
        For Black-Scholes with time-dependent vol.
    """
    def __init__(self, t_grid: list[float], vol_grid: list[float], **kwargs):
        self.vol_grid = vol_grid
        # Check sizes
        n_times = len(t_grid)
        if len(vol_grid) != n_times:
            raise ValueError("Incompatible sizes between time grid and vol grid")

        # Create sections
        sections = []
        for i in range(n_times):
            t, vol = t_grid[i], vol_grid[i]
            sections.append(FlatLocalVolSection(t, vol))

        # Instantiate base
        super().__init__(sections, **kwargs)


class ConstantLocalVol(LocalVol):
    """ Local Vol subtype returning one single number for all expiries and strikes.
        For Black-Scholes model with constant vol.
    """
    def __init__(self, vol: float, **kwargs):
        super().__init__(**kwargs)
        self.vol = vol
        self.tmax = 100.0 # 100y

    def value(self, t: float, logm: npt.ArrayLike) -> npt.ArrayLike:
        """ Single flat number for all expiries and strikes """
        return np.full_like(logm, self.vol)

    def section(self, t: float) -> LocalVolSection:
        """ Retrieve flat LV section at max time 100y """
        return FlatLocalVolSection(self.tmax, self.vol)

    def dump_data(self) -> dict:
        """ Dump object to dictionary """
        return {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT), 'vol': self.vol}


if __name__ == "__main__":
    print("Hello")
