import datetime as dt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from sdevpy.tools import timegrids
from sdevpy.maths import interpolation as itp


########### TODO ########################################################################
# * Create data reading for both curve interp definitions
#   and dates/df reading per curve ID per date
# * Add unit testing

def create_yieldcurve(valdate, interp_var='zerorate', interp_scheme='spline'):
    curve = InterpolatedYieldCurve(valdate)

    # Get interpolation variable
    match interp_var.lower():
        case 'zerorate': variable = YieldCurveVariable.ZERORATE
        case 'discount': variable = YieldCurveVariable.DISCOUNT
        case 'log_discount': variable = YieldCurveVariable.LOG_DISCOUNT
        case _: raise RuntimeError(f"Unknown interpolation variable: {interp_var}")

    # Set interpolation
    interp = None
    scheme = interp_scheme.lower()
    if variable == YieldCurveVariable.ZERORATE:
        match scheme:
            case 'linear': interp = itp.create_interpolation(interp='linear', l_extrap='flat', r_extrap='flat')
            case 'spline': interp = itp.create_interpolation(interp='cubicspline', l_extrap='flat', r_extrap='flat', bc_type='clamped')
            case _: raise RuntimeError(f"Unsupported scheme: {scheme}")
    elif variable in [YieldCurveVariable.DISCOUNT, YieldCurveVariable.LOG_DISCOUNT]:
        match scheme:
            case 'linear': interp = itp.create_interpolation(interp='linear', l_extrap='none', r_extrap='none')
            case 'spline': interp = itp.create_interpolation(interp='cubicspline', l_extrap='none', r_extrap='none')
            case _: raise RuntimeError(f"Unsupported scheme: {scheme}")
    else:
        raise RuntimeError("Unknown interpolation variable(2)")

    if interp is None:
        raise RuntimeError("Failure to set curve interpolation")

    curve.set_interpolation(variable, interp)
    return curve


class YieldCurve(ABC):
    def __init__(self, valdate=None):
        self.valdate = valdate

    @abstractmethod
    def discount(date):
        pass

    @abstractmethod
    def discount_float(t):
        pass


class InterpolatedYieldCurve(YieldCurve):
    def __init__(self, valdate=None):
        super().__init__(valdate)
        self.dates, self.dfs = None, None
        self.variable, self.interpolation = None, None

    def discount(self, date):
        t = timegrids.model_time(self.valdate, date)
        df = self.discount_float(timegrids.model_time(self.valdate, date))
        return df

    def discount_float(self, t):
        y = self.interpolation.value(t)
        match self.interp_var:
            case YieldCurveVariable.ZERORATE: return np.exp(-y * t)
            case YieldCurveVariable.DISCOUNT: return y
            case YieldCurveVariable.LOG_DISCOUNT: return np.exp(y)
            case _: raise RuntimeError(f"Unsupported interpolation variable: {str(self.interp_var)}")

    def set_data(self, dates, dfs):
        if self.interpolation is None:
            raise RuntimeError("Interpolation not set")

        if dates[0] <= self.valdate: # Assuming the dates are sorted
            raise RuntimeError("Incorrect input contains dates <= valdate")

        # Store pillar information
        self.dates = dates
        self.dfs = dfs

        if self.interp_var == YieldCurveVariable.ZERORATE:
            times = timegrids.model_time(self.valdate, dates)
            data_y = -np.log(dfs) / times
        elif self.interp_var in [YieldCurveVariable.DISCOUNT, YieldCurveVariable.LOG_DISCOUNT]:
            times = [0.0]
            times = times.extend([timegrids.model_time(self.valdate, d) for d in self.dates])
            times = np.asarray(times)
            ext_dfs = [1.0]
            ext_dfs = np.asarray(ext_dfs.extend(dfs))
            if self.interp_var == YieldCurveVariable.DISCOUNT:
                data_y = ext_dfs
            else:
                data_y = np.log(ext_dfs)

        self.interpolation.set_data(times, data_y)

    def set_interpolation(self, interp_var, interpolation):
        self.interp_var = interp_var
        self.interpolation = interpolation


class YieldCurveVariable(Enum):
    ZERORATE = 0
    DISCOUNT = 1
    LOG_DISCOUNT = 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sdevpy.tools import dates
    valdate = dt.datetime(2026, 2, 15)

    # Create curve
    interp_var = 'zerorate'
    scheme = 'linear'
    curve = create_yieldcurve(valdate, interp_var, scheme)

    # Create data
    zdates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15), dt.datetime(2056, 2, 15)]
    zrs = np.asarray([0.01, 0.015, 0.020, 0.022, 0.025, 0.030])
    ztimes = timegrids.model_time(valdate, zdates)
    dfs = np.exp(-zrs * ztimes)

    # Set data
    curve.set_data(zdates, dfs)

    # Interpolate and display
    test_dates = [dates.date_advance(valdate, months=1*n) for n in range(600)]
    test_dfs = curve.discount(test_dates)
    test_times = timegrids.model_time(valdate, test_dates)
    test_zrs = -np.log(test_dfs) / test_times

    plt.plot(test_dates, test_zrs)
    plt.scatter(zdates, zrs, color='black')
    plt.show()
