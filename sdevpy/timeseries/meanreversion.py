import numpy as np
import pandas as pd
import statsmodels.api as sm


class MeanRevertingTimeSeries:
    def __init__(self, time_series): 
        self.time_series = time_series
        
        # Compute mean reversion statistics
        res = compute_mean_reversion_params(self.time_series)
        
        self.half_life = res['Half Life']
        self.mr_rate = res['MR Rate']
        self.mr_level =  res['MR Level']
        
        # Check accuracy of the OLS estimate. The smaller the value, the more accurate the result.
        self.const_pvalue = res['Const p-value']
        self.series_pvalue = res['Series p-value']   

        self.stdev = np.std(self.time_series)   
        
        # Compute z score
        self.z_score_ts = (self.time_series - self.mr_level) / self.stdev
        self.z_score_ts = self.z_score_ts.rename('z score')   

    def get_half_life(self):
        return self.half_life
 
    def get_mr_rate(self):
        return self.mr_rate
        
    def get_mr_level(self):
        return self.mr_level
        
    def get_const_pvalue(self):
        return self.const_pvalue    
    
    def get_series_pvalue(self):
        return self.series_pvalue
    
    def get_level_at_t(self, date):
        return self.time_series.loc[date]
    
    def get_current_level(self):
        return self.time_series.iloc[-1]

    def get_stdev(self):
        return self.stdev
        
    def get_zscores_time_series(self):
        return self.z_score_ts

    def get_current_zscore(self):
        return self.z_score_ts.iloc[-1]        


def compute_mean_reversion_params(s): 
    """ Estimate mean reversion by assuming the process to be of the form
        ds = lambda x (sbar - s(t-1))dt + sigma x dW(t) """
    # Check consistency of input data and rename column
    cols = s.columns
    if len(cols) != 1:
        raise RuntimeError("Column number is unexpected: " + len(cols))
    
    s = s.rename(columns={cols[0]: 'Series'})

    # Compute the diff and the shift the position by -1 to have ds(t) facing s(t-1)
    ds = s.diff().shift(-1)
    
    # Skip the last element which is NA
    ds = ds.iloc[:-1]
    
    # Skip the last element of the original series as it's not used
    s = s.iloc[:-1]

    # Perform regression: dS(t) = a + b * S(t-1)    
    s_const = sm.add_constant(s)
    reg = sm.OLS(ds, s_const).fit()
    
    # If we assume ds(t) = lambda (sbar - s(t-1))dt + sigma dW(t), then
    # a = lambda * S_bar * dt
    # b = -lambda * dt
    a = reg.params['const']
    b = reg.params['Series']
    
    # See Clewlow and Strickland's energy derivatives pricing and risk management p28, 29
    # this is the proper way to do it, not using np.mean(basket) to compute the mean
    mr_level = -a / b
    
    # We expect this is a positive number. This is just a convention that quantopian use.
    if b > 0:
        print('The series is not mean reverting')
    
    # Modulo the Brownian noise, the proxe has the solution x(t) = x0 e^{-lambda t}
    # so the half-life is T1/2 = ln(2) / lambda. To obtain the half-life in number of days,
    # we need to do T1/2 / dt, which is -ln(2) / b.
    half_life = -np.log(2) / b # This is a number of days i.e. a number of dt  
    
    # This is -lambda * dt, where dt depends on the data freq. If daily, dt = 1/365.
    # To use this later, all we need is to put the number of days rather than year fraction
    # and we don't need to put the minus sign
    # e.g. 5 days -> exp(mean_rev_rate_in_days * 5) NOT exp(mean_rev_rate_in_days * 5/365)
    mr_rate = b
    
    return {'Half Life': half_life, 'MR Rate': mr_rate, 'MR Level': mr_level,
            'Const p-value': reg.pvalues['const'], 'Series p-value': reg.pvalues['Series']}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Generate a mean reverting time series i.e. a process s(t) defined by
    # ds(t) = lambda x (sbar - s(t-1))dt + sigma x dW(t)
    # where sbar is the long term mean and lambda is the mean reversion speed
    n_days = 252 * 10
    s0 = 119
    sbar = 120
    starget = sbar * 0.99999
    ttarget = 1.0 # in years
    kappa = - np.log((starget - sbar) / (s0 - sbar)) / ttarget
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)
    sigma = 0.1 * s0

    print(f"Simulating time series for {n_days} days")
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
    s = pd.DataFrame({'T': tvec, 'S': svec})
    s = s[['S']]

    print("Estimation mean reversion parameters")
    mr_res = compute_mean_reversion_params(s)
    mr_rate = mr_res['MR Rate']
    mr_level = mr_res['MR Level']

    print(f"True MR level: {sbar:,.6f}")
    print(f"Estimated MR level: {mr_level:,.6f}")
    print(f"True MR speed: {kappa:,.6f}")
    print(f"Estimated MR speed: {-mr_rate / dt:,.6f}")
 