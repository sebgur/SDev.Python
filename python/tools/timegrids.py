""" Generation of time grids for numerical methods such as Monte-Carlo or PDE. """
from abc import ABC, abstractmethod
from datetime import date
import numpy as np


class TimeGridBuilder(ABC):
    """ Base class for time grid builders """
    def __init__(self):
        self.epsilon_ = 1e-10
        self.time_grid_ = np.zeros((1, self.epsilon_))

    def reset(self):
        """ Resetting to fresh state """
        self.time_grid_ = np.zeros((1, self.epsilon_))

    # public void AddDates(Date valDate, Date[] dates)
    # {
    #     int nDates = dates.Length;
    #     List<double> list = new List<double>();
    #     for (int i = 0; i < nDates; i++)
    #         list.Add(dayCount.YearFraction(valDate, dates[i]));
    #     double[] t = list.ToArray();
    #     timeGrid = VectorAlgebra.MergeVectors(timeGrid, t);
    # }

    # public void AddGrid(double[] t)
    # {
    #     timeGrid = VectorAlgebra.MergeVectors(timeGrid, t);
    # }

    def extend(self):
        """ Add a fine grid to the current grid """
        # self.time_grid_ = VectorAlgebra.MergeVectors(self.time_grid_, fine_grid(self))
        return self.epsilon_

    def clean(self):
        """ Remove negative times and duplicates, then sort in ascending order """
        self.time_grid_ = 0 #TimeGridTools.RemoveNegative(timeGrid, timeEpsilon)
        # self.time_grid_ = timeGrid.RemoveDuplicates(timeEpsilon)
        # Array.Sort(self.time_grid_)

    def complete_grid(self):
        """ Add a fine grid, clean and return the final grid """
        self.extend()
        self.clean()
        return self.time_grid_

    def max(self):
        """ Largest point on the grid """
        return self.time_grid_.max()

    @abstractmethod
    def fine_grid(self):
        """ Create a fine grid with different method for each inherited type """


class SimpleTimeGridBuilder(TimeGridBuilder):
    """ Specific TimeGridBuilder with a fine grid that is simply a homogeneous grid until last
        point"""
    def __init__(self, steps_per_year=1):
        TimeGridBuilder.__init__(self)
        self.steps_per_year_ = steps_per_year

    def fine_grid(self):
        """ Generate a homogeneous grid until last point """
        tmax = self.max()
        dsteps = tmax * self.steps_per_year_
        nsteps = int(dsteps)
        if nsteps < 1:
            raise ValueError("Empty fine grid in simple time grid builder")

        fine_grid = np.linspace(0.0, tmax, nsteps)
        return fine_grid



def model_time(date1, date2):
    """ Yearfraction (time) between two dates for models, using simply (date2 - date1) / 365."""
    span = date2 - date1
    return span.days / 365.0

if __name__ == "__main__":
    expiry = date(2025, 1, 24)
    base = date(2023, 1, 24)
    print(expiry)
    print(base)
    print(model_time(base, expiry))
