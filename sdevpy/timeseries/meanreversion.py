import numpy as np
import pandas as pd
import statsmodels.api as sm


class MeanRevertingTimeSeries:
    def __init__(self, time_series): 
        self.time_series = time_series
        
        # Compute mean reversion statistics
        res = compute_mean_reversion_params(self.time_series)
        
        self.half_life_in_days = res['Half Life in days']
        self.mean_rev_rate_in_days = res['Mean Rev Rate in days']
        self.mean_rev_level =  res['Mean Rev Level']
        
        # Check accuracy of the OLS estimate. The smaller the value, the more accurate the result.
        self.const_pvalue = res['const p-value']
        self.Basket_pvalue = res['Basket p-value']   

        self.stdev = np.std(self.time_series)   
        
        # Compute z score
        self.z_score_ts = (self.time_series - self.mean_rev_level)/self.stdev
        self.z_score_ts = self.z_score_ts.rename('z score')   

    def get_half_life_in_days(self):
        return self.half_life_in_days
 
    def get_mean_rev_rate_in_days(self):
        return self.mean_rev_rate_in_days
        
    def get_mean_rev_level(self):
        return self.mean_rev_level
        
    def get_const_pvalue(self):
        return self.const_pvalue    
    
    def get_Basket_pvalue(self):
        return self.Basket_pvalue
    
    # get the value at date 
    def get_level_at_t(self, date):
        return self.time_series.loc[date]
    
    # get the latest value in the time series
    def get_current_level(self):
        return self.time_series.iloc[-1]

    def get_stdev(self):
        return self.stdev
        
    def get_zscores_time_series(self):
        return self.z_score_ts

    def get_current_zscore(self):
        return self.z_score_ts.iloc[-1]        


def compute_mean_reversion_params(s): 
    """ (1) half life
        (2) mean reversion level
        (3) mean reversion rate
        (4) p values of the OLS estimation (the smaller, the better) """

    # Compute the diff and the shift the position by -1 so that we have dS(t) vs S(t-1)
    ds = s.diff().shift(-1)     
    
    # Skip the last element which is NA    
    ds = ds.iloc[:-1]           
    
    # Skip the last element    
    s = s.iloc[:-1]                   

    # Perform regression: dS(t) = a + b * S(t-1)    
    s_const = sm.add_constant(s)    
    results = sm.OLS(ds, s_const).fit() 
    
    # If we assume dS(t) = lambda (S_bar - S(t-1))dt + \sigma dW(t), then
    # a = lambda * S_bar * dt
    # b = -lambda * dt
    a = results.params['const']
    b = results.params['Basket'] 
    
    # See Clewlow and Strickland's energy derivatives pricing and risk management p28, 29
    # this is the proper way to do it, not using np.mean(basket) to compute the mean
    mean_rev_level = -a / b
    
    # We expect this is a positive number. This is just a convention that quantopian use.
    if b > 0:
        print('The series is not mean reverting')
    
    # Solution of the equation: 1/2 = exp(b*T) -> T = -ln(2)/b
    half_life_in_days = -np.log(2) / b   
    
    # This is -lambda * dt, where dt depends on the data freq. If daily, dt = 1/365.
    # To use this later, all we need is to put the number of days rather than year fraction
    # and we don't need to put the minus sign
    # e.g. 5 days -> exp(mean_rev_rate_in_days * 5) NOT exp(mean_rev_rate_in_days * 5/365)
    mean_rev_rate_in_days = b
    
    return {'Half Life in days': half_life_in_days,
            'Mean Rev Rate in days': mean_rev_rate_in_days, 'Mean Rev Level': mean_rev_level,
            'const p-value': results.pvalues['const'], 'Basket p-value': results.pvalues['Basket']}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Generate a mean reverting time series i.e. a process S(t) defined by
    # dS(t) = \kappa (S_bar - S(t-1))dt + \sigma dW(t)
    # where S_bar is the long term mean
    n_days = 252
    s0 = 119
    sbar = 120
    starget = sbar * 0.99999
    ttarget = 1.0 # in years
    kappa = - np.log((starget - sbar) / (s0 - sbar)) / ttarget
    print(f"True mean reversion: {kappa:,.6f}")
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)
    sigma = 0.1 * s0
    t0 = 0.0
    svec = [s0]
    tvec = [t0]
    ltmean = [sbar] * (n_days + 1)
    rng = np.random.RandomState(42)
    gaussians = rng.normal(0.0, 1.0, n_days)
    se = s0
    te = t0
    for g in gaussians:
        ds = kappa * (sbar - se) * dt + sigma * sqrt_dt * g
        se = se + ds
        te = te + dt
        svec.append(se)
        tvec.append(te)

    # Plot
    # plt.plot(tvec, svec, color='blue')
    # plt.plot(tvec, ltmean, color='red')
    # plt.show()

    # Run the test and see if it finds its mean reversion
    df = pd.DataFrame({'T': tvec, 'S': svec})
    print(df.head())
    df = df[['S']]
    res = compute_mean_reversion_params(df)
    print(res)
