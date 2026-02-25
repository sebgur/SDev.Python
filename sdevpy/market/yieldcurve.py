import os, json
from pathlib import Path
import datetime as dt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from sdevpy.tools import timegrids
from sdevpy.tools import dates
from sdevpy.maths import interpolation as itp



class YieldCurve(ABC):
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', None)
        self.snapdate = self.valdate  # For now, until we calibrate
        self.name = kwargs.get('name', '')

    @abstractmethod
    def discount(date):
        pass

    @abstractmethod
    def discount_float(t):
        pass

    @abstractmethod
    def dump_data(self):
        pass

    def dump(self, file, indent=2):
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)


class InterpolatedYieldCurve(YieldCurve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dates, self.dfs = None, None
        self.interp_var_str = kwargs.get('interp_var', 'zerorate')
        self.interp_type = kwargs.get('interp_type', 'cubispline')
        self.interp_var, self.interp = None, None
        self.set_interpolation()

    def discount(self, date):
        t = timegrids.model_time(self.valdate, date)
        df = self.discount_float(timegrids.model_time(self.valdate, date))
        return df

    def discount_float(self, t):
        y = self.interp.value(t)
        match self.interp_var:
            case YieldCurveVariable.ZERORATE: return np.exp(-y * t)
            case YieldCurveVariable.DISCOUNT: return y
            case YieldCurveVariable.LOG_DISCOUNT: return np.exp(y)
            case _: raise RuntimeError(f"Unsupported interpolation variable: {str(self.interp_var)}")

    def set_data(self, dates, dfs):
        if self.interp is None:
            raise RuntimeError("Interpolation not set")

        if dates[0] <= self.valdate: # Assuming the dates are sorted
            raise RuntimeError("Incorrect input contains dates <= valdate")

        # Sort by increasing date
        sorted_pillars = [{'expiry': d, 'df': df} for d, df in zip(dates, dfs)]
        sorted_pillars.sort(key=lambda x: x['expiry'])

        # Store pillar information
        self.dates = [pillar['expiry'] for pillar in sorted_pillars]  # dates
        self.dfs = [pillar['df'] for pillar in sorted_pillars]  # dfs

        # Set data in interpolation
        if self.interp_var == YieldCurveVariable.ZERORATE:
            times = timegrids.model_time(self.valdate, dates)
            data_y = -np.log(dfs) / times
        elif self.interp_var in [YieldCurveVariable.DISCOUNT, YieldCurveVariable.LOG_DISCOUNT]:
            times = [0.0]
            times = times.extend(timegrids.model_time(self.valdate, self.dates))
            times = np.asarray(times)
            ext_dfs = [1.0]
            ext_dfs = np.asarray(ext_dfs.extend(dfs))
            if self.interp_var == YieldCurveVariable.DISCOUNT:
                data_y = ext_dfs
            else:
                data_y = np.log(ext_dfs)

        self.interp.set_data(times, data_y)

    def set_interpolation(self):
        # Get interpolation variable
        match self.interp_var_str.lower():
            case 'zerorate': self.interp_var = YieldCurveVariable.ZERORATE
            case 'discount': self.interp_var = YieldCurveVariable.DISCOUNT
            case 'log_discount': self.interp_var = YieldCurveVariable.LOG_DISCOUNT
            case _: raise RuntimeError(f"Unknown interpolation variable: {self.interp_var_str}")

        # Set interpolation
        scheme = self.interp_type.lower()
        if self.interp_var == YieldCurveVariable.ZERORATE:
            match scheme:
                case 'linear': self.interp = itp.create_interpolation(interp=scheme, l_extrap='flat', r_extrap='flat')
                case 'cubicspline': self.interp = itp.create_interpolation(interp=scheme, l_extrap='flat', r_extrap='flat', bc_type='clamped')
                case _: raise RuntimeError(f"Unsupported scheme: {scheme}")
        elif self.interp_var in [YieldCurveVariable.DISCOUNT, YieldCurveVariable.LOG_DISCOUNT]:
            match scheme:
                case 'linear': self.interp = itp.create_interpolation(interp=scheme, l_extrap='none', r_extrap='none')
                case 'cubicspline': self.interp = itp.create_interpolation(interp=scheme, l_extrap='none', r_extrap='none')
                case _: raise RuntimeError(f"Unsupported scheme: {scheme}")
        else:
            raise RuntimeError("Unknown interpolation variable(2)")

        if self.interp is None:
            raise RuntimeError("Failure to set curve interpolation")

    def dump_data(self):
        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'interp_var': self.interp_var_str, 'interp_type': self.interp_type}

        pillars = []
        for expiry, df in zip(self.dates, self.dfs):
            pillar = {'expiry': expiry.strftime(dates.DATE_FORMAT), 'df': df}
            pillars.append(pillar)

        data['pillars'] = pillars
        return data


class YieldCurveVariable(Enum):
    ZERORATE = 0
    DISCOUNT = 1
    LOG_DISCOUNT = 2


def yieldcurve_from_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    interp_var = data.get('interp_var')
    interp_type = data.get('interp_type')
    pillars = data.get('pillars')

    valdate = dt.datetime.strptime(valdate, dates.DATE_FORMAT)
    curve = InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=interp_type)

    # Read pillar data
    pillar_dates, pillar_dfs = [], []
    for pillar in pillars:
        date_str = pillar.get('expiry')
        date = dt.datetime.strptime(date_str, dates.DATE_FORMAT)
        df = pillar.get('df')
        pillar_dates.append(date)
        pillar_dfs.append(df)

    curve.set_data(pillar_dates, pillar_dfs)
    return curve


def data_file(name, date, **kwargs):
    folder = kwargs.get('folder', test_data_folder())
    name_folder = os.path.join(folder, name)
    os.makedirs(name_folder, exist_ok=True)
    file = os.path.join(name_folder, date.strftime(dates.DATE_FILE_FORMAT) + ".json")
    return file


def test_data_folder():
    folder = Path(__file__).parent.parent.parent.resolve()
    dataset_folder = os.path.join(folder, "datasets")
    folder = os.path.join(os.path.join(dataset_folder, "marketdata"), "yieldcurves")
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sdevpy.tools import dates

    valdate = dt.datetime(2026, 2, 15)

    # Create curve
    name = 'USD.SOFR.1D'
    interp_var = 'zerorate'
    scheme = 'linear'
    curve = InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=scheme,
                                   name=name)

    # Create data
    zdates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15), dt.datetime(2056, 2, 15)]
    zrs = np.asarray([0.01, 0.015, 0.020, 0.022, 0.025, 0.030])
    ztimes = timegrids.model_time(valdate, zdates)
    dfs = np.exp(-zrs * ztimes)

    # Set data
    curve.set_data(zdates, dfs)

    # Interpolate and display
    test_dates = [dates.date_advance(valdate, months=1*n) for n in range(1, 600)]
    test_dfs = curve.discount(test_dates)
    test_times = timegrids.model_time(valdate, test_dates)
    test_zrs = -np.log(test_dfs) / test_times

    # Write to file to create first sample
    file = data_file(name, valdate)
    curve.dump(file)

    # Read from file
    curve2 = yieldcurve_from_file(file)
    curve2_dfs = curve2.discount(test_dates)
    curve2_zrs = -np.log(curve2_dfs) / test_times

    print(test_zrs[10:15])
    print(curve2_zrs[10:15])

    plt.plot(test_dates, test_zrs)
    plt.plot(test_dates, curve2_zrs, color='red')
    plt.scatter(zdates, zrs, color='black')
    plt.show()
