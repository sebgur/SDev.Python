""" Term-structure model for SVI. Give each SVI parameter a parametric formula along time and
    enforce no-arbitrage (approximately). This model has 11 parameters.
    ToDo: put reference to paper"""
import numpy as np
from sdevpy.models.surfaces.zerosurface import TermStructureParametricZeroSurface


class TsSvi1(TermStructureParametricZeroSurface):
    def __init__(self, **kwargs):
        super().__init__()
        self.set_calculator(svi_formula)
        self.n_params = 11
        self.calculable_at_zero = False
        self.tmax = kwargs('tmax', 42)

    def formula_parameters(self, t: float, x: list[float]) -> list[float]:
        """ Calculate parameters according to the TsSvi1 formulas """
        if t < self.eps:
            raise ValueError("TsSvi1 is not calculable at t = 0")

        # Get parameters
        v0, vinf, b_, tau, alpha, beta, r, x0star, lambda0, gamma, delta = self.GetParameters(x)
        if r < -1.0 or r > 1.0:
            raise ValueError("Correlation should be between -1 and 1 in TsSvi1")

        if delta + 1.0 < self.eps:
            raise ValueError("Delta should be stricktly higher than -1 in TsSvi1")

        # Calculate new variables
        one_minus_rho2 = 1.0 - r * r
        if one_minus_rho2 < 0.0:
            raise ValueError("Correlation should be between -1 and 1 in TsSvi1")

        pow_ = beta + delta
        vstar = alpha * gamma * one_minus_rho2 / (pow_ + 1.0) * np.power(t, pow_)
        f0 = v0 * v0
        finf = vinf * vinf
        vstar += -b_ * tau + finf + tau / t * (b_ * (t + tau) + f0 - finf) * (1.0 - np.exp(-t / tau))
        b = alpha * np.power(t, beta - 1.0)

        lambda_ = lambda0 + gamma / (delta + 1.0) * np.power(t, delta + 1.0)
        xstar = x0star - r * (lambda_ - lambda0)

        # Go back to standard parameters
        a = vstar - b * lambda_ * one_minus_rho2
        m = xstar + r * lambda_
        s = lambda_ * np.sqrt(one_minus_rho2)

        return [a, b, r, m, s]


if __name__ == "__main__":
    print("Hello")

