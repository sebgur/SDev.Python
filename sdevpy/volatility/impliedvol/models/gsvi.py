""" Simple reparameterization of the original SVI model of J. Gatheral by incorporating the
    time dependence in the parameters, and using the original SVI formula to express the
    vol squared rather than the variance. This is the parameterization used in the
    Term-Structure SVI models of [Gurrieri2010] in
    See Gurrieri, 'A Class of Term Structures for SVI Implied Volatility', 2010
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1779463
    """
import numpy as np
import numpy.typing as npt


############## TODO ##########################################################
# * What about imposing the variance positivity constraint? Didn't we have it
#   in C#?


def gsvi_formula(x: npt.ArrayLike, params: list[float]) -> npt.ArrayLike:
    """ gSVI formula as in [Gurrieri2010] """
    # Retrieve parameters
    if len(params) != 5:
        raise ValueError(f"Incorrect parameter size in gSVI: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Calculate
    # print(f"x-shape: {x.shape}")
    # print(f"m-shape: {m.shape}")
    xm = x - m # x is the log-moneyness
    var = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(var < 0.0):
        raise ValueError("Negative variance in gSVI formula")

    vol = np.sqrt(var)
    return vol


def gsvi_check_params(params: list[float], check_butterfly: bool=False) -> None:
    """ Check consistency of gSVI parameters """
    return True, 0.0


if __name__ == "__main__":
    print("Hello")
