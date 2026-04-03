from abc import ABC, abstractmethod
import numpy as np
from enum import Enum


class OptionQuotationType(Enum):
    LogNormalVol = 0
    NormalVol = 1
    ShiftedLogNormalVol = 2
    ForwardPremium = 3


class ZeroSurface(ABC):
    def __init__(self):
        # Set defaults
        self.modelledtype = OptionQuotationType.LogNormalVol
        self.shift = 0.0
        self.allow_negative_variables = False
        self.calculable_at_zero = True
        self.localvol_method = 0

    def volatility(self, t: float, x: float) -> float:
        """ Black volatility """
        return BlackVolatility(t, x, 1.0)

    def dvariance_dt(self, ts: float, te: float, x: float) -> float:
        """ Differential of variance against time """
        tmpe = self.volatility(te, x)
        tmps = self.volatility(ts, x)
        return (tmpe * tmpe * te - tmps * tmps * ts) / (te - ts)

    def dvolatility_dx(self, t: float, x: float) -> float:
        """ Differential of volatility against moneyness """
        h = 0.001
        if x - h < 0.0:
            raise ValueError("Negative strike in numerical 1st differential of implied volatility")

        tmp1 = Volatility(t, x + h)
        tmp2 = Volatility(t, x - h)
        return (tmp1 - tmp2) / (2.0 * h)

    def d2volatility_dx2(self, t: float, x: float) -> float:
        """ Second differential of the volatility against the moneyness """
        h = 0.001
        two_h = 2.0 * h
        if x - two_h < 0.0:
            raise ValueError("Negative strike in numerical 2nd differential of implied volatility")

        tmp = self.volatility(t, x)
        dxp = (self.volatility(t, x + two_h) - tmp) / two_h
        dxm = (tmp - self.volatility(t, x - two_h)) / two_h
        return (dxp - dxm) / two_h

    def differentiate(self, ts: float, te: float, x: float) -> float:
        """ Retrieve all the quantities needed for Dupire's formula """
        theta = self.volatility(ts, x)
        dVarDt = self.dvariance_dt(ts, te, x)
        dThetaDx = self.dvolatility_dx(ts, x)
        d2ThetaDx2 = self.d2volatility_dx2(ts, x)
        return theta, dvardt, dthetadx, d2thetad2x

    def density(self, t: float, fwd: float, strike: float) -> float:
        if np.abs(t) < self.time_epsilon:
            raise ValueError("Probability density cannot be calculated at t = 0")

        # Get from implied volatility
        x = strike / fwd
        sqrtT = Math.Sqrt(t)
        stDev = Volatility(t, x) * sqrtT
        xDThetaDx = x * DVolatilityDx(t, x)
        x2D2ThetaDx2 = x * x * D2VolatilityDx2(t, x)

        if np.abs(stDev) < self.stdev_epsilon:
            raise ValueError("Probability density cannot be calculated at standard deviation 0")

        if x < self.x_epsilon: # 0 or negative
            return 0.0

        dMinus = -np.log(x) / stDev - 0.5 * stDev
        dPlusSqrtT = (dMinus + stDev) * sqrtT
        deltaNMinus = np.exp(-0.5 * dMinus * dMinus) / Constant.C_SQRT2PI
        tmp = 1.0 + dPlusSqrtT * xDThetaDx
        main = x2D2ThetaDx2 - dPlusSqrtT * xDThetaDx * xDThetaDx + tmp * tmp / (stDev * sqrtT)
        return sqrtT * deltaNMinus * main / strike

    def cumulative(self, t: float, fwd: float, strike: float) -> float:
        x = strike / fwd
        theta = self.Volatility(t, x)
        sqrtT = np.sqrt(t)
        stDev = theta * sqrtT
        dm = -np.log(x) / stDev - 0.5 * stDev
        dtheta = self.dvolatility_dx(t, x)
        N = new NormalDistribution()
        return N.Density(dm) * x * sqrtT * dtheta - N.Cumulative(dm) + 1.0

    def cumulative_inverse(self, t: float, p: float) -> float:
        if np.abs(t) < self.time_epsilon:
            raise ValueError("Cumulative inverse at t = 0 is not defined for implied volatility")

        if np.abs(p) < 1e-10:
            return 0.0

        cumulativeFunction = new CumulativeFunction(this, t)
        solver = new ZBrent(1e-6, 100.0, 1000000, 0.000000001)
        return solver.Solve(cumulativeFunction.Value, p)

    def forward_price(self, t: float, k: float, f: float, is_call: bool) -> float:
        value = self.calculate(t, k, f, isCall)
        match self.ModelledType:
            case OptionQuotationType.ForwardPremium:
                 return value
            case OptionQuotationType.LogNormalVol:
                 return BlackFormula.Price(f, k, value * Math.Sqrt(t), isCall)
            case OptionQuotationType.NormalVol:
                 return BachelierFormula.Price(f, k, value * Math.Sqrt(t), isCall)
            case OptionQuotationType.ShiftedLogNormalVol:
                 return ShiftedBlackFormula.Price(f, k, value * Math.Sqrt(t), isCall, Shift)
            case _:
             raise TypeError(f"Invalid modelled type in zero-surface: " + ModelledType.ToString())

    def black_volatility(self, t: float, k: float, f: float) -> float:
        is_call = True
        value = self.calculate(t, k, f, is_call)
        if self.ModelledType == OptionQuotationType.LogNormalVol
            return value
        else:
            match self.ModelledType:
                case OptionQuotationType.ForwardPremium:
                    price = value
                case OptionQuotationType.NormalVol:
                    price = BachelierFormula.Price(f, k, value * Math.Sqrt(t), isCall)
                case OptionQuotationType.ShiftedLogNormalVol:
                    price = ShiftedBlackFormula.Price(f, k, value * Math.Sqrt(t), isCall, Shift)
                case _:
                    raise TypeError("Invalid modelled type in zero-surface: " + ModelledType.ToString())

            return BlackFormula.ImpliedVolatility(price, f, k, t, is_call)

    def bachelier_volatility(self, t: float, k: float, f: float):
        is_call = True
        value = self.calculate(t, k, f, is_call)
        if self.ModelledType == OptionQuotationType.NormalVol:
            return value
        else:
            match self.ModelledType:
                case OptionQuotationType.ForwardPremium:
                    price = value
                case OptionQuotationType.LogNormalVol:
                    price = BlackFormula.Price(f, k, value * Math.Sqrt(t), is_call)
                case OptionQuotationType.ShiftedLogNormalVol:
                    price = ShiftedBlackFormula.Price(f, k, value * Math.Sqrt(t), is_call, Shift)
                case _:
                    raise TypeError("Invalid modelled type in zero-surface: " + ModelledType.ToString())

            return BachelierFormula.ImpliedVolatility(price, f, k, t, is_call)

    public double ShiftedBlackVolatility(double t, double k, double f)
    {
        bool isCall = true;
        double value = Calculate(t, k, f, isCall);
        if (ModelledType == OptionQuotationType.ShiftedLogNormalVol)
            return value;
        else
        {
            double price;
            switch (ModelledType)
            {
                case OptionQuotationType.ForwardPremium: price = value; break;
                case OptionQuotationType.LogNormalVol: price = BlackFormula.Price(f, k, value * Math.Sqrt(t), isCall); break;
                case OptionQuotationType.NormalVol: price = BachelierFormula.Price(f, k, value * Math.Sqrt(t), isCall); break;
                default: throw new Exception("Invalid modelled type in zero-surface: " + ModelledType.ToString());
            }
            return ShiftedBlackFormula.ImpliedVolatility(price, f, k, t, isCall, Shift);
        }
    }
    #endregion

    #region Calibration
    //
    public void Calibrate(Date date, OptionSurface mktSurface)
    {
        baseDate = date;
        Date[] expiries;
        OptionTarget[][] inputOptions = mktSurface.CalibrationTargets(out expiries);
        Calibrate(baseDate, inputOptions);
    }
    //
    public void Calibrate(Date date, OptionTarget[][] options)
    {
        baseDate = date;
        // Check consistency of input data, convert to modelled type
        OptionTarget[][] targetOptions = CheckConsistency(options);

        // Get expiry times
        expiryTimes = targetOptions.Select(x => x[0].Expiry).ToArray();

        // Model-dependent calibration of inherited types
        CalibrateModelledType(date, targetOptions);
    }
    /// <summary>
    /// Take out negative rate options depending on model features, check consistency of expiries, forwards, etc...
    /// </summary>
    private OptionTarget[][] CheckConsistency(OptionTarget[][] options)
    {
        // Strip out negative rate options if needed
        OptionTarget[][] tOptions = (AllowNegativeVariables ? options : OptionTargetChecker.KeepPositive(options));

        // Check consistency of expiries and forwards
        OptionTargetChecker.CheckExpiriesAndForwards(tOptions);

        // Convert from quoted type to targetType required for model calibration.
        OptionTarget[][] cOptions = OptionTargetChecker.ConvertToTargetValues(tOptions, ModelledType, Shift);

        // Check degrees of freedom
        if (this is ParametricZeroSurface && checkDegreesOfFreedom)
        {
            int nParameters = ((ParametricZeroSurface)this).NumberParameters();
            OptionTargetChecker.CheckDegreesOfFreedom(cOptions, nParameters);
        }

        return cOptions;
    }
    #endregion

    #region Abstract Methods
    public abstract double Calculate(double t, double k, double f, bool isCall);
    public abstract void CalibrateModelledType(Date date, OptionTarget[][] options);
    #endregion

    #region Getters
    //
    public double[] ExpiryTimes()
    {
        return expiryTimes;
    }
    #endregion

    #region Fields
    protected DayCounter dayCount = new ModelDayCounter();
    protected double[] expiryTimes;
    protected const double epsilon = 100.0 * Constant.MACHINE_EPSILON;
    const double timeEpsilon = 0.000001;
    Date baseDate;
    protected bool checkDegreesOfFreedom = true;
    #endregion

    #region Properties
    public OptionQuotationType ModelledType { get; set; }
    /// <summary>
    /// In Math format, i.e. 0.01 for 1%
    /// </summary>
    public double Shift { get; set; }
    public bool AllowNegativeVariables { get; set; }
    public bool CalculableAtZero { get; set; }
    public int LocalVolMethod { get; set; }
    #endregion
}
