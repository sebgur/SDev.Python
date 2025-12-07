from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from sdevpy.models import svi


class Section(ABC):
    @abstractmethod
    def value(self, t, x):
        pass


class ParamSection(Section):
    def __init__(self, formula):
        self.params = None
        self.formula = formula

    def value(self, t, x):
        return self.formula(t, x, self.params)

    def update_params(self, new_params):
        self.params = new_params.copy()

    def check_params(self):
        """ True if the parameters are not violating any constraints.
            Also returns a penalty number to estimate how bad the violation
            is, potentially usable by optimizations. """
        return True, 0.0


if __name__ == "__main__":
    alnv = 0.25 # a > 0
    b = 0.2 # b > 0
    rho = 0.0 # -1 < rho < 1
    m = 0.5 # No constraints
    sigma = 0.25 # > 0
    params = [alnv, b, rho, m, sigma]
    formula = svi.formula

    t = 1.5
    k = np.linspace(0.2, 3.0, 100)
    x = np.log(k)

    section = ParamSection(formula)
    section.update_params(params)
    vol = section.value(t, x)
    plt.plot(k, vol)
    plt.show()
