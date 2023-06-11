""" Plot helpers for XSABR project """
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from analytics import bachelier
from analytics import black


def plot_transform_surface(expiries, strikes, is_call, fwd, ref_prices, mod_prices, title_,
                           transform='ShiftedBlackScholes'):
    """ Calculate quantities to display for the surface and display them in charts. Transformed
        quantities available are: Price, ShiftedBlackScholes (3%) and Bachelier (normal vols). """
    # Transform prices
    ref_disp = transform_surface(expiries, strikes, is_call, fwd, ref_prices, transform)
    mod_disp = transform_surface(expiries, strikes, is_call, fwd, mod_prices, transform)

    # Display transformed prices
    num_charts = expiries.shape[0]
    num_cols = 2
    num_rows = int(num_charts / num_cols)
    # print("num_rows: " + str(num_rows))
    ylabel = 'Price' if transform is 'Price' else 'Vol'

    fig, axs = plt.subplots(num_rows, num_cols, layout="constrained")
    fig.suptitle(title_, size='x-large', weight='bold')
    fig.set_size_inches(12, 8)
    for i in range(num_rows):
        for j in range(num_cols):
            k = num_cols * i + j
            axs[i, j].plot(strikes[k], ref_disp[k], color='blue', label='Reference')
            axs[i, j].plot(strikes[k], mod_disp[k], color='red', label='Model')
            axs[i, j].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
            axs[i, j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
            axs[i, j].set_xlabel('Strike')
            axs[i, j].set_ylabel(ylabel)
            axs[i, j].set_title(f"T={expiries[i, 0]}")
            axs[i, j].legend(loc='upper right')

    plt.show()


def transform_surface(expiries, strikes, is_call, fwd, prices, transform='ShiftedBlackScholes'):
    """ Tranform prices into: Price, ShiftedBlackScholes (3%) and Bachelier (normal vols). """
    # Transform prices
    trans_prices = []
    if transform is 'Price':
        trans_prices = prices
    elif transform is 'ShiftedBlackScholes':
        shift = 0.03
        sfwd = fwd + shift
        for i, expiry in enumerate(expiries):
            strikes_ = strikes[i]
            trans_prices_ = []
            for j, strike in enumerate(strikes_):
                sstrike = strike + shift
                trans_prices_.append(black.implied_vol(expiry, sstrike, is_call, sfwd, prices[i, j]))
            trans_prices.append(trans_prices_)
    elif transform is 'Bachelier':
        for i, expiry in enumerate(expiries):
            strikes_ = strikes[i]
            trans_prices_ = []
            for j, strike in enumerate(strikes_):
                trans_prices_.append(bachelier.implied_vol(expiry, strike, is_call, fwd, prices[i, j]))
            trans_prices.append(trans_prices_)
    else:
        raise ValueError("Unknown transform type: " + transform)
    
    return trans_prices


# def strike_ladder(expiry, spread_ladder, fwd, test_params, generator, model,
#                   transform='ShiftedBlackScholes'):
#     """ Plot volatilities along a ladder of strike spreads """
#     is_call = generator.is_call

#     # Calculate prices
#     rf_prc, md_prc, strikes, sprds = generator.price_strike_ladder(model, expiry, spread_ladder,
#                                                                    fwd, test_params)

#     # Invert to normal vols
#     rf_nvols = []
#     md_nvols = []
#     if transform is 'ShiftedBlackScholes':
#         shift = 0.03
#         for i, strike in enumerate(strikes):
#             sstrike = strike + shift
#             sfwd = fwd + shift
#             rf_nvols.append(black.implied_vol(expiry, sstrike, is_call, sfwd, rf_prc[i]))
#             md_nvols.append(black.implied_vol(expiry, sstrike, is_call, sfwd, md_prc[i]))
#     elif transform is 'Bachelier':
#         for i, strike in enumerate(strikes):
#             rf_nvols.append(bachelier.implied_vol(expiry, strike, is_call, fwd, rf_prc[i]))
#             md_nvols.append(bachelier.implied_vol(expiry, strike, is_call, fwd, md_prc[i]))
#     else:
#         raise ValueError("Unknown transform type: " + transform)

#     lnvol = test_params['LnVol']
#     beta = test_params['Beta']
#     nu = test_params['Nu']
#     rho = test_params['Rho']
#     # Plot
#     plt.title(f'T={expiry:.2f}, F={fwd * 100:.2f}, LnVol={lnvol * 100:.2f}, Beta={beta:.2f}' +
#               f',\n Nu={nu*100:.2f}, Rho={rho * 100:.2f}')

#     plt.xlabel('Spread')
#     plt.ylabel('Volatility')
#     plt.plot(sprds, rf_nvols, color='blue', label='Reference')
#     plt.plot(sprds, md_nvols, color='red', label='Model')
#     plt.legend(loc='upper right')
