import json
import datetime as dt
from abc import ABC, abstractmethod
from sdevpy.models.svi import *
from sdevpy.tools import algos, dates


class LocalVol(ABC):
    """ Base class for local vols """
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', dt.datetime.now())
        self.name = kwargs.get('name', 'MyIndex')
        self.snapdate = kwargs.get('snapdate', self.valdate)

    @abstractmethod
    def value(self, t, logm):
        pass

    @abstractmethod
    def section(self, t):
        """ Get a section at time t, i.e. a function of the log-moneyness log_m"""
        pass

    @abstractmethod
    def dump_data(self):
        pass


class InterpolatedParamLocalVol(LocalVol):
    def __init__(self, sections, **kwargs):
        super().__init__(**kwargs)

        # Sort by increasing date
        sections.sort(key=lambda x: x.time)

        self.sections = sections
        self.t_grid = [section.time for section in self.sections]

    def value(self, t, logm):
        t_idx = algos.upper_bound(self.t_grid, t)
        return self.sections[t_idx].value(t, logm)

    def section(self, t):
        return 0

    def check_params(self, t_idx):
        return self.sections[t_idx].check_params()

    def update_params(self, t_idx, new_params):
        self.sections[t_idx].update_params(new_params)

    def params(self, t_idx):
        return self.sections[t_idx].params

    def dump(self, file, indent=2):
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)

    def dump_data(self):
        sections = []
        for section in self.sections:
            sections.append(section.dump())

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'sections': sections}
        return data


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
