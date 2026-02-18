import numpy as np
from abc import ABC, abstractmethod
from sdevpy.maths import tridiag


def scheme(config, ts):
    scheme_type = config.scheme.upper()
    if scheme_type == 'IMPLICIT':
        return ImplicitScheme()
    elif scheme_type == 'CN':
        return ThetaScheme(0.5)
    elif scheme_type == 'THETA':
        return ThetaScheme(config['theta'])
    elif scheme_type == 'EXPLICIT':
        return ExplicitScheme()
    elif scheme_type == 'RANNACHER':
        eps = 1e-8
        if ts - eps <= config.rannacher_time:
            return ImplicitScheme()
        else:
            return ThetaScheme(0.5)
    else:
        raise TypeError(f"Unknown PDE scheme type: {scheme_type}")


class PdeScheme(ABC):
    def __init__(self):
        self.local_vol = None

    @abstractmethod
    def roll_forward(self, p, x, ts, te, dx):
        pass


class ThetaScheme(PdeScheme):
    """ Mixing scheme with particular sub-cases:
            Theta = 0.0: Implicit
            Theta = 0.5: Crank-Nicolson
            Theta = 1.0: Explicit """
    def __init__(self, theta):
        super().__init__()
        self.theta = theta
        self.one_m_theta = 1.0 - theta

    def roll_forward(self, p, x, ts, te, dx):
        n_x = x.shape[0]
        dt = te - ts
        a = 1.0 / dx**2 + 0.5 / dx
        b = 2.0 / dx**2
        c = 1.0 / dx**2 - 0.5 / dx

        # Calculate result vector using previous probabilities
        lv = self.local_vol(ts, x)
        one_m_theta_dt_2 = self.one_m_theta * dt / 2.0
        y = np.zeros(n_x)
        for j in range(n_x):
            p_tmp = (1.0 - one_m_theta_dt_2 * b * lv[j]**2) * p[j]

            if j < n_x - 1: # Beyond that the probability is 0
                p_tmp += one_m_theta_dt_2 * a * lv[j + 1]**2 * p[j + 1]

            if j > 0: # Before that the probability is 0
                p_tmp += one_m_theta_dt_2 * c * lv[j - 1]**2 * p[j - 1]

            y[j] = p_tmp

        # Calculate band vectors for tridiagonal system
        lv = self.local_vol(te, x)
        theta_dt_2 = self.theta * dt / 2.0
        upper = np.zeros(n_x - 1)
        main = np.zeros(n_x)
        lower = np.zeros(n_x - 1)
        for j in range(n_x):
            main[j] = (1.0 + theta_dt_2 * b * lv[j]**2)

            if j < n_x - 1:
                upper[j] = -theta_dt_2 * a * lv[j + 1]**2

            if j > 0:
                lower[j - 1] = -theta_dt_2 * c * lv[j - 1]**2

        # Solve tridiagonal system
        p_new = tridiag.solve(upper, main, lower, y)
        return p_new


class ImplicitScheme(PdeScheme):
    def __init__(self):
        super().__init__()

    def roll_forward(self, p, x, ts, te, dx):
        n_x = x.shape[0]
        dt = te - ts
        a = dt / 2.0 * (1.0 / dx**2 + 0.5 / dx)
        b = dt / dx**2
        c = dt / 2.0 * (1.0 / dx**2 - 0.5 / dx)

        lv = self.local_vol(te, x)
        upper = np.zeros(n_x - 1)
        main = np.zeros(n_x)
        lower = np.zeros(n_x - 1)
        for j in range(n_x):
            main[j] = (1.0 + b * lv[j]**2)

            if j < n_x - 1:
                upper[j] = -a * lv[j + 1]**2

            if j > 0:
                lower[j - 1] = -c * lv[j - 1]**2

        # Solve tridiagonal system
        p_new = tridiag.solve(upper, main, lower, p)
        return p_new


class ExplicitScheme(PdeScheme):
    def __init__(self):
        super().__init__()

    def roll_forward(self, p, x, ts, te, dx):
        n_x = x.shape[0]
        dt = te - ts
        a = dt / 2.0 * (1.0 / dx**2 + 0.5 / dx)
        b = dt / dx**2
        c = dt / 2.0 * (1.0 / dx**2 - 0.5 / dx)

        lv = self.local_vol(ts, x)
        y = np.zeros(n_x)
        for j in range(n_x):
            p_tmp = (1.0 - b * lv[j]**2) * p[j]

            if j < n_x - 1: # Beyond that the probability is 0
                p_tmp += a * lv[j + 1]**2 * p[j + 1]

            if j > 0: # Before that the probability is 0
                p_tmp += c * lv[j - 1]**2 * p[j - 1]

            y[j] = p_tmp

        return y
