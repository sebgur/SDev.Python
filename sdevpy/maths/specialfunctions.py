import numpy as np
import numpy.typing as npt


def rebonato(v: npt.ArrayLike) -> npt.ArrayLike:
    """ Parametric funtion in Rebonato's book, convenient to fit yield and volatility curves """
    a0 = 0.5
    ainf = 2.0
    b = -0.01
    tau = 5.0
    return ainf + (b * v + a0 - ainf) * np.exp(-v / tau)
