import numpy as np
from abc import abstractmethod
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol


class ParametricImpliedVol(ImpliedVol):
    def __init__(self):
        super().__init__()
        self.n_params = None
        self.params = None

    def update_params(self, x: list[float]) -> None:
        """ Update the current parameters """
        self.params = np.asarray(x, dtype=float)

    def check_params(self) -> tuple[bool, float]:
        """ Check validity of the parameters, return is_ok state and penalty value (0.0 if is_ok) """
        return True, 0.0

    @abstractmethod
    def bounds(self, keep_feasible: bool=False):
        """ Recommended bounds for optimization on parameters """
        pass

    @abstractmethod
    def initial_point(self) -> list[float]:
        """ Recommended initial point for optimization on parameters """
        pass

    def _require_params(self) -> None:
        """ Make sure params have been set """
        if self.params is None:
            raise RuntimeError("Model has no parameters. Call update_params() first")
