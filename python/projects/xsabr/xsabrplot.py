""" Plot helpers for XSABR project """
import matplotlib.pyplot as plt
from analytics import bachelier
from analytics import black


def strike_ladder(expiry, spread_ladder, fwd, test_params, generator, model,
                  transform='ShiftedBlackScholes'):
    """ Plot volatilities along a ladder of strike spreads """
    is_call = generator.is_call

    # Calculate prices
    rf_prc, md_prc, strikes, sprds = generator.price_strike_ladder(model, expiry, spread_ladder,
                                                                   fwd, test_params)

    # Invert to normal vols
    rf_nvols = []
    md_nvols = []
    if transform is 'ShiftedBlackScholes':
        shift = 0.03
        for i, strike in enumerate(strikes):
            sstrike = strike + shift
            sfwd = fwd + shift
            rf_nvols.append(black.implied_vol(expiry, sstrike, is_call, sfwd, rf_prc[i]))
            md_nvols.append(black.implied_vol(expiry, sstrike, is_call, sfwd, md_prc[i]))
    elif transform is 'Bachelier':
        for i, strike in enumerate(strikes):
            rf_nvols.append(bachelier.implied_vol(expiry, strike, is_call, fwd, rf_prc[i]))
            md_nvols.append(bachelier.implied_vol(expiry, strike, is_call, fwd, md_prc[i]))
    else:
        raise ValueError("Unknown transform type: " + transform)

    lnvol = test_params['LnVol']
    beta = test_params['Beta']
    nu = test_params['Nu']
    rho = test_params['Rho']
    # Plot
    plt.title(f'T={expiry:.2f}, F={fwd * 100:.2f}, LnVol={lnvol * 100:.2f}, Beta={beta:.2f}' +
              f',\n Nu={nu*100:.2f}, Rho={rho * 100:.2f}')

    plt.xlabel('Spread')
    plt.ylabel('Volatility')
    plt.plot(sprds, rf_nvols, color='blue', label='Reference')
    plt.plot(sprds, md_nvols, color='red', label='Model')
    plt.legend(loc='upper right')
