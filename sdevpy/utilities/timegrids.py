""" Generation of time grids for numerical methods such as Monte-Carlo or PDE. """
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import algos, tools
log = logging.getLogger(Path(__file__).stem)


def build_timegrid(t_start: float, t_end: float, config: dict) -> npt.NDArray[np.float64]:
    """ Very simple time grid for now, to be replaced by TimeGridBuilder """
    n_steps = config.n_timesteps
    return np.linspace(t_start, t_end, n_steps + 1)


class TimeGridBuilder(ABC):
    """ Base class for time grid builders """
    def __init__(self, include_t0: bool=False):
        self.atol_ = 1e-11
        self.time_grid_ = []
        self.valdate = None
        self.include_t0 = include_t0

    def reset(self):
        """ Resetting to fresh state """
        self.time_grid_ = []

    def add_dates(self, valdate: dt.datetime, dates: list[dt.datetime]) -> None:
        """ Add vector of dates respectively to valuation date """
        self.valdate = valdate
        times = []
        for d in dates:
            times.append(model_time(self.valdate, d))

        self.time_grid_.extend(times)

    def add_grid(self, times: npt.ArrayLike) -> None:
        """ Add vector of times. This function is risky when used in conjonction with dates,
            as the consistency of date conversion to times vs input times is the user's
            responsibility here. """
        self.time_grid_.extend(times.reshape(-1))

    def refine(self) -> None:
        """ Add a fine grid to the current grid """
        self.time_grid_.extend(self.fine_grid())

    def clean(self) -> None:
        """ Remove negative times and duplicates, then sort in ascending order """
        self.time_grid_ = [t for t in self.time_grid_ if t > self.atol_]

        if self.include_t0:
            self.time_grid_.insert(0, 0.0)

        self.time_grid_ = algos.unique_sorted(self.time_grid_, self.atol_)

    def complete_grid(self) -> npt.NDArray[np.float64]:
        """ Add a fine grid, clean and retrieve the final time grid """
        self.refine()
        self.clean()
        return self.get_grid()#self.time_grid_

    def max(self) -> float:
        """ Largest point on the grid """
        return np.max(self.time_grid_)

    def upper_bound(self, date):
        time = model_time(self.valdate, date)
        return algos.upper_bound(self.time_grid_, time, self.atol_)

    def get_grid(self) -> npt.NDArray[np.float64]:
        """ Retrieve the final time grid as numpy array """
        return np.asarray(self.time_grid_)

    @abstractmethod
    def fine_grid(self) -> npt.NDArray[np.float64]:
        """ Create a fine grid with different method for each inherited type """


class SimpleTimeGridBuilder(TimeGridBuilder):
    """ Specific TimeGridBuilder with a fine grid that is simply a homogeneous grid until last
        point """
    def __init__(self, include_t0: bool=False, points_per_year: int=1):
        super().__init__(include_t0)
        self.points_per_year_ = points_per_year

    def fine_grid(self) -> npt.NDArray[np.float64]:
        """ Generate a homogeneous grid until last point """
        tmax = self.max()
        dpoints = tmax * self.points_per_year_
        npoints = int(dpoints)
        if npoints < 1:
            log.debug("No points added in grid refinement")
            fine_grid = np.asarray([])
            # raise ValueError("Empty fine grid in simple time grid builder")
        else:
            fine_grid = np.linspace(0.0, tmax, npoints)
        return fine_grid


def model_time(date1: npt.ArrayLike, date2: npt.ArrayLike) -> npt.ArrayLike:
    """ Yearfraction (time) between two dates for models, using simply (date2 - date1) / 365."""
    spans = np.asarray(date2) - np.asarray(date1)
    if tools.isiterable(spans):
        span = [s.days / 365.0 for s in spans]
        return np.asarray(span)
    else:
        return spans.days / 365.0
    # span = date2 - date1
    # return span.days / 365.0


if __name__ == "__main__":
    base = dt.date(2023, 1, 24)
    fixing = dt.date(2022, 1, 24)
    monitor = dt.date(2024, 1, 24)
    expiry = dt.date(2025, 1, 24)
    settlement = dt.date(2026, 1, 24)
    builder = SimpleTimeGridBuilder(5)
    builder.add_dates(base, [fixing, settlement, expiry, monitor, settlement])
    builder.refine()
    builder.clean()

    # Test MC situation
    time_grid_builder = SimpleTimeGridBuilder(points_per_year=5)
    EXPIRIES = np.asarray([5.0, 1.0, 0.125, 0.250, 0.5]).reshape(-1, 1)
    print(EXPIRIES)
    time_grid_builder.add_grid(EXPIRIES)
    print(time_grid_builder.time_grid_)
    print("refine")
    time_grid_builder.refine()
    print(time_grid_builder.time_grid_)
    print("clean")
    time_grid_builder.clean()
    tg = time_grid_builder.time_grid_
    print(tg)

    # Test vectorization
    print("\n Vectorization")
    v = model_time(base, fixing)
    print(v)
    a = [fixing, monitor]
    b = base
    v = model_time(base, [fixing, monitor])
    print(v)

    spans = np.asarray(a) - np.asarray(b)
    print(spans)
    span = [s.days / 365.0 for s in spans]
    if len(span) == 1:
        x = span[0]
    else:
        x = span

    print(x)
