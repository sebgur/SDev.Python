import numpy as np
from scipy.stats import norm
from sdevpy.pde.forwardpde import PdeConfig, build_spotgrid, lognormal_density, density_step, density
from sdevpy.utilities import timegrids
from sdevpy.analytics import black
from sdevpy.maths import metrics


def test_forward_pde():
    spot = 100.0
    r = 0.04
    q = 0.01
    atm_vol = 0.20
    maturities = np.array([0.1, 0.5, 1.0, 2.5, 5.0, 10.0])

    def my_lv(t, x_grid):
        """ As a function of log forward moneyness """
        return np.asarray([atm_vol for x in x_grid])

    #### Diagnostics #################################################################
    pde_config = PdeConfig(n_timesteps=50, n_meshes=250, mesh_vol=atm_vol, scheme='rannacher',
                           rescale_x=True, rescale_p=True)

    n_dev = 4 # Distribution display range in number of stdevs
    use_batches = True

    reports = []

    # Build spot grid (fixed throughout for now)
    if use_batches and pde_config.rescale_x:
        x, dx, spot_idx = build_spotgrid(maturities[0], pde_config)
    else:
        x, dx, spot_idx = build_spotgrid(maturities[-1], pde_config)

    # Start-up density
    start_time = 1.0 / 365.0 # Make sure no payoff is required before that
    p = lognormal_density(x, start_time, pde_config.mesh_vol)

    total_diff = 0.0
    for mty_idx in range(maturities.shape[0]):
        maturity = maturities[mty_idx]
        if use_batches:
            ts = start_time if mty_idx == 0 else maturities[mty_idx - 1]
            te = maturities[mty_idx]
            step_grid = timegrids.build_timegrid(ts, te, pde_config)
            x, dx, p = density_step(p, x, dx, step_grid, my_lv, pde_config)
        else:
            x, dx, p = density(maturity, my_lv, pde_config)

        ## Check density ##
        stdev = atm_vol * np.sqrt(maturity)
        x_max = stdev * n_dev # Display range

        # PDE
        pde_x = []
        pde_p = []
        for u, v in zip(x, p, strict=True):
            if np.abs(u) < x_max:
                pde_x.append(u)
                pde_p.append(v)

        # Closed-form (display)
        cf_x = np.linspace(-x_max, x_max, 100)
        cf_p = norm.pdf(cf_x, loc=-0.5 * stdev**2, scale=stdev)

        # Calculate diffs (ToDo: do on all points x, p)
        cf_all = norm.pdf(pde_x, loc=-0.5 * stdev**2, scale=stdev)
        diff = metrics.rmse(pde_p, cf_all)
        total_diff += diff

        report = {'rmse(dens)': diff, 'int(cf)': np.trapezoid(cf_p, cf_x), 'int(pde)': np.trapezoid(pde_p, pde_x),
                  'pde_x': pde_x, 'pde_p': pde_p, 'cf_x': cf_x, 'cf_p': cf_p}

        ## Check option prices ##
        strikes = np.linspace(0.50 * spot, 2.0 * spot, 16)
        is_call = True
        fwd = spot * np.exp((r - q) * maturity)
        cf_prices = black.price(maturity, strikes, is_call, fwd, atm_vol)
        it_prices = np.maximum(fwd - strikes, 0.0) # Intrinsic values

        s = fwd * np.exp(x)
        pde_prices = []
        for k in strikes:
            payoff = np.maximum(s - k, 0.0)
            weighted_payoff = payoff * p
            pde_prices.append(np.trapezoid(weighted_payoff, x))

        report['strikes'] = strikes
        report['pde_prices'] = pde_prices - it_prices
        report['cf_prices'] = cf_prices - it_prices

        reports.append(report)

    # Gather values
    pde_sums, cf_sums = [], []
    for r in reports:
        pde_sums.append(r['pde_prices'].sum())
        cf_sums.append(r['cf_prices'].sum())

    # for s in pde_sums:
    #     print(s)

    # for s in cf_sums:
    #     print(s)

    pde_ref = np.asarray([2.73414261826, 10.63198237231, 21.4763468108, 60.1939276268, 132.4162443424, 279.171161237])
    cf_ref = np.asarray([2.70369628483, 10.60016306545, 21.43931715705, 60.14359001452, 132.344239396, 279.0267305878])

    pde_ok = np.allclose(np.asarray(pde_sums), pde_ref, 1e-10)
    cf_ok = np.allclose(np.asarray(cf_sums), cf_ref, 1e-10)

    assert pde_ok and cf_ok


if __name__ == "__main__":
    test_forward_pde()
