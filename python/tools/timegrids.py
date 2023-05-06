""" Generation of time grids for numerical methods such as Monte-Carlo or PDE. """
from abc import ABC, abstractmethod
from datetime import date
import numpy as np


class TimeGridBuilder(ABC):
    """ Base class for time grid builders """
    def __init__(self):
        self.epsilon_ = 1e-10
        self.time_grid_ = []

    def reset(self):
        """ Resetting to fresh state """
        self.time_grid_ = []

    def add_dates(self, val_date, dates):
        """ Add vector of dates respectively to valuation date """
        times = []
        for d in dates:
            times.append(model_time(val_date, d))

        self.time_grid_.extend(times)

    def add_grid(self, times):
        """ Add vector of times """
        self.time_grid_.extend(times)

    def refine(self):
        """ Add a fine grid to the current grid """
        self.time_grid_.extend(self.fine_grid())

    def clean(self):
        """ Remove negative times and duplicates, then sort in ascending order """
        self.time_grid_ = [t for t in self.time_grid_ if t > self.epsilon_]
        self.time_grid_ = np.unique(self.time_grid_)

    def complete_grid(self):
        """ Add a fine grid, clean and return the final grid """
        self.refine()
        self.clean()
        return self.time_grid_

    def max(self):
        """ Largest point on the grid """
        return np.max(self.time_grid_)

    @abstractmethod
    def fine_grid(self):
        """ Create a fine grid with different method for each inherited type """


class SimpleTimeGridBuilder(TimeGridBuilder):
    """ Specific TimeGridBuilder with a fine grid that is simply a homogeneous grid until last
        point"""
    def __init__(self, points_per_year=1):
        TimeGridBuilder.__init__(self)
        self.points_per_year_ = points_per_year

    def fine_grid(self):
        """ Generate a homogeneous grid until last point """
        tmax = self.max()
        dpoints = tmax * self.points_per_year_
        npoints = int(dpoints)
        if npoints < 1:
            raise ValueError("Empty fine grid in simple time grid builder")

        fine_grid = np.linspace(0.0, tmax, npoints)
        return fine_grid


def model_time(date1, date2):
    """ Yearfraction (time) between two dates for models, using simply (date2 - date1) / 365."""
    span = date2 - date1
    return span.days / 365.0

if __name__ == "__main__":
    base = date(2023, 1, 24)
    fixing = date(2022, 1, 24)
    monitor = date(2024, 1, 24)
    expiry = date(2025, 1, 24)
    settlement = date(2026, 1, 24)
    builder = SimpleTimeGridBuilder(5)
    builder.add_dates(base, [fixing, settlement, expiry, monitor, settlement])
    print(builder.time_grid_)
    builder.refine()
    print(builder.time_grid_)
    builder.clean()
    print(builder.time_grid_)
