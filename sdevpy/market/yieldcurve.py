import json
from pathlib import Path
import datetime as dt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.maths import interpolation as itp


class YieldCurve(ABC):
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', None)
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')

    @abstractmethod
    def discount(self, date: dt.datetime) -> float:
        pass

    @abstractmethod
    def discount_float(self, t) -> float:
        pass

    @abstractmethod
    def dump_data(self):
        pass

    def dump(self, file: str, indent: int=2):
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)


class InterpolatedYieldCurve(YieldCurve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dates, self.dfs = None, None
        self.interp_var_str = kwargs.get('interp_var', 'zerorate')
        self.interp_type = kwargs.get('interp_type', 'cubicspline')
        self.interp_var, self.interp = None, None
        self.set_interpolation()

    def discount(self, date):
        df = self.discount_float(timegrids.model_time(self.valdate, date))
        return df

    def discount_float(self, t):
        y = self.interp.value(t)
        match self.interp_var:
            case YieldCurveVariable.ZERORATE:
                return np.exp(-y * t)
            case YieldCurveVariable.DISCOUNT:
                return y
            case YieldCurveVariable.LOG_DISCOUNT:
                return np.exp(y)
            case _:
                raise RuntimeError(f"Unsupported interpolation variable: {str(self.interp_var)}")

    def set_data(self, dates, dfs):
        if self.interp is None:
            raise RuntimeError("Interpolation not set")

        if dates[0] <= self.valdate: # Assuming the dates are sorted
            raise RuntimeError("Incorrect input contains dates <= valdate")

        # Sort by increasing date
        sorted_pillars = [{'expiry': d, 'df': df} for d, df in zip(dates, dfs, strict=True)]
        sorted_pillars.sort(key=lambda x: x['expiry'])

        # Store pillar information
        self.dates = [pillar['expiry'] for pillar in sorted_pillars]  # dates
        self.dfs = [pillar['df'] for pillar in sorted_pillars]  # dfs

        # Set data in interpolation
        if self.interp_var == YieldCurveVariable.ZERORATE:
            times = timegrids.model_time(self.valdate, self.dates)
            data_y = -np.log(self.dfs) / times
        elif self.interp_var in [YieldCurveVariable.DISCOUNT, YieldCurveVariable.LOG_DISCOUNT]:
            times = [0.0]
            times.extend(timegrids.model_time(self.valdate, self.dates))
            times = np.asarray(times)
            ext_dfs = [1.0]
            ext_dfs.extend(self.dfs)
            ext_dfs = np.asarray(ext_dfs)
            if self.interp_var == YieldCurveVariable.DISCOUNT:
                data_y = ext_dfs
            else:
                data_y = np.log(ext_dfs)

        self.interp.set_data(times, data_y)

    def set_interpolation(self):
        # Get interpolation variable
        match self.interp_var_str.lower():
            case 'zerorate':
                self.interp_var = YieldCurveVariable.ZERORATE
            case 'discount':
                self.interp_var = YieldCurveVariable.DISCOUNT
            case 'log_discount':
                self.interp_var = YieldCurveVariable.LOG_DISCOUNT
            case _:
                raise RuntimeError(f"Unknown interpolation variable: {self.interp_var_str}")

        # Set interpolation
        scheme = self.interp_type.lower()
        if self.interp_var == YieldCurveVariable.ZERORATE:
            match scheme:
                case 'linear':
                    self.interp = itp.create_interpolation(interp=scheme, l_extrap='flat', r_extrap='flat')
                case 'cubicspline':
                    self.interp = itp.create_interpolation(interp=scheme, l_extrap='flat', r_extrap='flat',
                                                           bc_type='clamped')
                case _:
                    raise RuntimeError(f"Unsupported scheme: {scheme}")
        elif self.interp_var in [YieldCurveVariable.DISCOUNT, YieldCurveVariable.LOG_DISCOUNT]:
            match scheme:
                case 'linear':
                    self.interp = itp.create_interpolation(interp=scheme, l_extrap='none', r_extrap='none')
                case 'cubicspline':
                    self.interp = itp.create_interpolation(interp=scheme, l_extrap='none', r_extrap='none')
                case _:
                    raise RuntimeError(f"Unsupported scheme: {scheme}")
        else:
            raise RuntimeError("Unknown interpolation variable(2)")

        if self.interp is None:
            raise RuntimeError("Failure to set curve interpolation")

    def dump_data(self):
        data = {'name': self.name, 'valdate': self.valdate.strftime(dts.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dts.DATETIME_FORMAT),
                'interp_var': self.interp_var_str, 'interp_type': self.interp_type}

        pillars = []
        for expiry, df in zip(self.dates, self.dfs, strict=True):
            pillar = {'expiry': expiry.strftime(dts.DATE_FORMAT), 'df': df}
            pillars.append(pillar)

        data['pillars'] = pillars
        return data


class YieldCurveVariable(Enum):
    ZERORATE = 0
    DISCOUNT = 1
    LOG_DISCOUNT = 2


def yieldcurve_from_file(file: str|Path) -> InterpolatedYieldCurve:
    """ Create yield curve object given file """
    data = jsm.deserialize(file)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = dt.datetime.strptime(data.get('snapdate'), dts.DATETIME_FORMAT)
    interp_var = data.get('interp_var')
    interp_type = data.get('interp_type')
    pillars = data.get('pillars')

    valdate = dt.datetime.strptime(valdate, dts.DATE_FORMAT)
    curve = InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=interp_type,
                                   name=name, snapdate=snapdate)

    # Read pillar data
    pillar_dates, pillar_dfs = [], []
    for pillar in pillars:
        date_str = pillar.get('expiry')
        date = dt.datetime.strptime(date_str, dts.DATE_FORMAT)
        df = pillar.get('df')
        pillar_dates.append(date)
        pillar_dfs.append(df)

    curve.set_data(pillar_dates, pillar_dfs)
    return curve


if __name__ == "__main__":
    valdate = dt.datetime(2026, 2, 15)
