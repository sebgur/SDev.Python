import numpy as np
from sdevpy.utilities.tools import isequal
from sdevpy.pde.forwardpde import PdeConfig, density, calculate_densities
from sdevpy.analytics import black


def check_forward_pde(rescale_x: bool, scheme: str='rannacher'):
    spot, r, q, atm_vol = 100.0, 0.04, 0.01, 0.20
    maturities = np.array([0.1, 0.5, 1.0, 2.5, 5.0, 10.0])

    class TestLocalVol:
        def value(self, t, x_grid):
            """ As a function of log forward moneyness """
            return np.asarray([atm_vol for x in x_grid])

    my_lv = TestLocalVol()
    # PDE config
    pde_config = PdeConfig(n_timesteps=50, n_meshes=250, mesh_vol=atm_vol, scheme=scheme,
                           rescale_x=rescale_x, rescale_p=True, shift_forward=True)

    # Run PDE
    density_reports = calculate_densities(maturities, my_lv, pde_config)

    # Diagnostics
    reports = []
    for density_report in density_reports:
        maturity = density_report['end_time']
        x = density_report['x_grid']
        p = density_report['p_grid']

        # Check option prices
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

        report = {'pde_prices': pde_prices - it_prices, 'cf_prices': cf_prices - it_prices}
        reports.append(report)

    return reports


def test_forward_pde():
    reports = check_forward_pde(rescale_x=True)

    # Gather values
    pde_sums, cf_sums = [], []
    for r in reports:
        pde_sums.append(r['pde_prices'].sum())
        cf_sums.append(r['cf_prices'].sum())

    # print(np.asarray(pde_sums))
    # print(np.asarray(cf_sums))
    pde_ref = np.asarray([2.70606853, 10.6012976, 21.44167511, 60.14815888, 132.34671851, 279.04775693])
    # pde_ref = np.asarray([2.73414261826, 10.63198237231, 21.4763468108, 60.1939276268, 132.4162443424, 279.171161237])
    cf_ref = np.asarray([2.70369628483, 10.60016306545, 21.43931715705, 60.14359001452, 132.344239396, 279.0267305878])

    pde_ok = np.allclose(np.asarray(pde_sums), pde_ref, 1e-10)
    cf_ok = np.allclose(np.asarray(cf_sums), cf_ref, 1e-10)

    assert pde_ok and cf_ok


def test_forward_pde_no_rescale_x():
    reports = check_forward_pde(rescale_x=False)

    # Gather values
    pde_sums, cf_sums = [], []
    for r in reports:
        pde_sums.append(r['pde_prices'].sum())
        cf_sums.append(r['cf_prices'].sum())

    # print(np.asarray(pde_sums))
    # print(np.asarray(cf_sums))
    pde_ref = np.asarray([2.67658397, 10.57688613, 21.43376434, 60.11816797, 132.3149149, 278.98433455])
    cf_ref = np.asarray([2.70369628483, 10.60016306545, 21.43931715705, 60.14359001452, 132.344239396, 279.0267305878])

    pde_ok = np.allclose(np.asarray(pde_sums), pde_ref, 1e-10)
    cf_ok = np.allclose(np.asarray(cf_sums), cf_ref, 1e-10)

    assert pde_ok and cf_ok


def test_forward_pde_implicit():
    reports = check_forward_pde(rescale_x=True, scheme='implicit')
    pde_sums = [r['pde_prices'].sum() for r in reports]
    # print(np.asarray(pde_sums))
    pde_ref = np.asarray([2.70813352, 10.60375539, 21.44415924, 60.13071881, 132.23445455, 278.74946682])
    assert np.allclose(np.asarray(pde_sums), pde_ref, 1e-10)


def test_forward_pde_explicit():
    spot, r, q, atm_vol, maturity = 100.0, 0.04, 0.01, 0.20, 1.5

    class TestLocalVol:
        def value(self, t, x_grid):
            """ As a function of log forward moneyness """
            return np.asarray([atm_vol for x in x_grid])

    my_lv = TestLocalVol()

    # PDE config
    pde_config = PdeConfig(n_timesteps=100, n_meshes=25, mesh_vol=atm_vol, scheme='explicit',
                           rescale_x=True, rescale_p=True)

    # Run PDE
    x, dx, p = density(maturity, my_lv, pde_config)

    # Check option prices
    strikes = np.linspace(0.50 * spot, 2.0 * spot, 16)
    fwd = spot * np.exp((r - q) * maturity)
    it_prices = np.maximum(fwd - strikes, 0.0) # Intrinsic values

    s = fwd * np.exp(x)
    pde_prices = []
    for k in strikes:
        payoff = np.maximum(s - k, 0.0)
        weighted_payoff = payoff * p
        pde_prices.append(np.trapezoid(weighted_payoff, x))

    pde_prices = pde_prices - it_prices

    # Gather values
    pde_sum = pde_prices.sum()
    # print(pde_sum)

    pde_ref = 33.51835236599183
    assert isequal(pde_sum, pde_ref, 1e-10)


def test_forward_pde_simple():
    spot, r, q, atm_vol, maturity = 100.0, 0.04, 0.01, 0.20, 1.5

    class TestLocalVol:
        def value(self, t, x_grid):
            """ As a function of log forward moneyness """
            return np.asarray([atm_vol for x in x_grid])

    my_lv = TestLocalVol()

    # PDE config
    pde_config = PdeConfig(n_timesteps=50, n_meshes=250, mesh_vol=atm_vol, scheme='rannacher',
                           rescale_x=True, rescale_p=True)

    # Run PDE
    x, dx, p = density(maturity, my_lv, pde_config)

    # Check option prices
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

    pde_prices = pde_prices - it_prices
    cf_prices = cf_prices - it_prices

    # Gather values
    pde_sum, cf_sum = pde_prices.sum(), cf_prices.sum()
    # print(pde_sum)
    # print(cf_sum)

    pde_ref = 33.4692251170
    cf_ref = 33.37108539482

    pde_ok = isequal(pde_sum, pde_ref, 1e-10)
    cf_ok = isequal(cf_sum, cf_ref, 1e-10)

    assert pde_ok and cf_ok


if __name__ == "__main__":
    test_forward_pde_explicit()
