import numpy as np
from scipy.stats import norm
from scipy import optimize

# undiscounted black option price
def black_option_price(f, # forward, double
                       k, # strike, double
                       t, # time to maturity, double
                       v, # implied volatility, double
                       c_or_p # call (1) or put (-1), integer
                       ):

    d_1 = (np.log(f/k)+0.5*v*v*t)/(v*np.sqrt(t))
    d_2 = d_1 - v*np.sqrt(t)
    if c_or_p == 1 :
        return f * norm.cdf(d_1) - k * norm.cdf(d_2)
    elif c_or_p == -1 :
        return k * norm.cdf(-d_2) - f * norm.cdf(-d_1)
    else:
        raise ValueError('c_or_p is expected to be 1 for call or -1 for put.')

#undiscounted black option vega
def black_option_vega(f, # forward, double
                      k, # strike, double
                      t, # time to maturity, double
                      v  # implied volatility, double
                      ):

    d_1 = (np.log(f / k) + 0.5 * v * v * t) / (v * np.sqrt(t))
    return f * norm.pdf(d_1) * np.sqrt(t)

#compute black implied volatility
def black_implied_vol(p,  # option price, double
                      f,  # forward, double
                      k,  # strike, double
                      t,  # time to maturity, double
                      c_or_p,  # call (1) or put (-1), integer
                      init_guess = 0.2 # initial guess
                      ):

    f_ivol = lambda x: black_option_price(f, k, t, x, c_or_p) - p
    f_vega = lambda x: black_option_vega(f, k, t, x)
    black_implied_vol = optimize.newton(f_ivol, init_guess, f_vega)

    return black_implied_vol