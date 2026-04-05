import datetime as dt
import numpy as np
from enum import Enum
from sdevpy.montecarlo.payoffs.basic import Max, Abs, Terminal


class VanillaOptionType(Enum):
    CALL = 0
    PUT = 1
    STRADDLE = 2


def make_vanilla_option_payoff(payoff, strike, optiontype):
    optiontype_ = string_to_optiontype(optiontype)
    match optiontype_:
        case VanillaOptionType.CALL:
            option_payoff = Max([payoff - strike, 0.0])
        case VanillaOptionType.PUT:
            option_payoff = Max([strike - payoff, 0.0])
        case VanillaOptionType.STRADDLE:
            option_payoff = Abs(payoff - strike)
        case _:
            raise ValueError(f"Invalid option type: {optiontype}")

    return option_payoff


def make_vanilla_option(name, strike, optiontype, expiry):
    optiontype_ = string_to_optiontype(optiontype)
    match optiontype_:
        case VanillaOptionType.CALL:
            payoff = Max([Terminal(name, expiry) - strike, 0.0])
        case VanillaOptionType.PUT:
            payoff = Max([strike - Terminal(name, expiry), 0.0])
        case VanillaOptionType.STRADDLE:
            payoff = Abs(Terminal(name, expiry) - strike)
        case _:
            raise ValueError(f"Invalid option type: {optiontype}")

    return payoff


def vanilla_option(spot, strike, optiontype):
    match optiontype:
        case VanillaOptionType.CALL:
            payoff = np.maximum(spot - strike, 0.0)
        case VanillaOptionType.PUT:
            payoff = np.maximum(strike - spot, 0.0)
        case VanillaOptionType.STRADDLE:
            payoff = np.abs(spot - strike)
        case _:
            raise ValueError("Invalid option type")

    return payoff


def string_to_optiontype(s):
    match s.lower():
        case 'call':
            return VanillaOptionType.CALL
        case 'put':
            return VanillaOptionType.PUT
        case 'straddle':
            return VanillaOptionType.STRADDLE
        case _:
            raise ValueError(f"Invalid option type: {s}")


if __name__ == "__main__":
    expiry = dt.datetime(2026, 12, 15)
    payoff = make_vanilla_option('SPX', 100, 'call', expiry)
    print(payoff)
