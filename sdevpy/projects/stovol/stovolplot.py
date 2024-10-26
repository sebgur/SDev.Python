""" Plot helpers for XSABR project """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sdevpy.analytics import bachelier
from sdevpy.analytics import black
from sdevpy.tools import clipboard


def plot_transform_surface(expiries, strikes, are_calls, fwd, ref_prices, mod_prices, title_,
                           transform='ShiftedBlackScholes', ref_name='Reference',
                           mod_name='Model'):
    """ Calculate quantities to display for the surface and display them in charts. Transformed
        quantities available are: Price, ShiftedBlackScholes (3%) and Bachelier (normal vols). """
    # Transform prices
    ref_disp = transform_surface(expiries, strikes, are_calls, fwd, ref_prices, transform)
    mod_disp = transform_surface(expiries, strikes, are_calls, fwd, mod_prices, transform)

    # Display transformed prices
    num_charts = expiries.shape[0]
    num_cols = 2
    num_rows = int(num_charts / num_cols)
    ylabel = 'Price' if transform == 'Price' else 'Vol'

    fig, axs = plt.subplots(num_rows, num_cols, layout="constrained")
    fig.suptitle(title_, size='x-large', weight='bold')
    fig.set_size_inches(12, 8)
    for i in range(num_rows):
        for j in range(num_cols):
            k = num_cols * i + j
            axs[i, j].plot(strikes[k], ref_disp[k], color='blue', label=ref_name)
            axs[i, j].plot(strikes[k], mod_disp[k], color='red', label=mod_name)
            axs[i, j].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
            axs[i, j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
            axs[i, j].set_xlabel('Strike')
            axs[i, j].set_ylabel(ylabel)
            axs[i, j].set_title(f"T={expiries[k, 0]}")
            axs[i, j].legend(loc='upper right')

    plt.show()


def transform_surface(expiries, strikes, are_calls, fwd, prices, transform='ShiftedBlackScholes'):
    """ Tranform prices into: Price, ShiftedBlackScholes (3%) and Bachelier (normal vols). """
    # Transform prices
    # trans_prices = []
    num_expiries = expiries.shape[0]
    num_strikes = strikes.shape[1]
    if transform == 'Price':
        trans_prices = prices
    elif transform == 'ShiftedBlackScholes':
        trans_prices = np.ndarray(shape=(num_expiries, num_strikes))
        shift = 0.03
        sfwd = fwd + shift
        for i, expiry in enumerate(expiries):
            strikes_ = strikes[i]
            are_calls_ = are_calls[i]
            # trans_prices_ = []
            for j, strike in enumerate(strikes_):
                sstrike = strike + shift
                trans_prices[i, j] = black.implied_vol(expiry, sstrike, are_calls_[j], sfwd,
                                                       prices[i, j])
                # trans_prices_.append(black.implied_vol(expiry, sstrike, are_calls_[j], sfwd,
                #                                        prices[i, j]))
            # trans_prices.append(trans_prices_)
    elif transform == 'Bachelier':
        trans_prices = np.ndarray(shape=(num_expiries, num_strikes))
        for i, expiry in enumerate(expiries):
            strikes_ = strikes[i]
            are_calls_ = are_calls[i]
            # trans_prices_ = []
            for j, strike in enumerate(strikes_):
                trans_prices[i, j] = bachelier.implied_vol(expiry, strike, are_calls_[j], fwd,
                                                           prices[i, j])
            #     trans_prices_.append(bachelier.implied_vol(expiry, strike, are_calls_[j], fwd,
            #                                                prices[i, j]))
            # trans_prices.append(trans_prices_)
    else:
        raise ValueError("Unknown transform type: " + transform)

    return trans_prices
    # return np.asarray(trans_prices)
