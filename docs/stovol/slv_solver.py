"""
Stochastic Local Volatility (SLV) PDE Solver
=============================================

Model:
    dS_t = r S_t dt + L(S_t, t) sqrt(v_t) S_t dW^S
    dv_t = kappa(theta - v_t) dt + xi sqrt(v_t) dW^v
    d<W^S, W^v> = rho dt

Pricing PDE for U(S, v, t):
    U_t + 0.5 L^2 v S^2 U_SS + rho xi L v S U_Sv + 0.5 xi^2 v U_vv
        + r S U_S + kappa(theta - v) U_v - r U = 0

Solved backward in time using the Craig-Sneyd ADI scheme, which handles
the mixed-derivative term cleanly and is second-order accurate in time.
"""

import numpy as np
from scipy.sparse import diags, csc_matrix, identity#, eye#, kron
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_spot_grid(s0, s_max, n_s, concentration=0.1):
    """
    Non-uniform grid in S, concentrated around S0 using a sinh transform.
    Concentration controls how tightly points cluster near S0.
    """
    c = concentration * s0
    xi_min = np.arcsinh(-s0 / c)
    xi_max = np.arcsinh((s_max - s0) / c)
    xi = np.linspace(xi_min, xi_max, n_s)
    s = s0 + c * np.sinh(xi)
    s[0] = 0.0  # clamp lower bound
    return s


def build_spot_grid_with_barriers(s0, s_max, n_s, anchors, concentration=0.1):
    """
    Like build_spot_grid but ensures every value in `anchors` (e.g. barriers)
    appears exactly on the grid. Achieved by snapping the nearest grid point
    to each anchor. Returns a sorted, strictly-increasing array.
    """
    s = build_spot_grid(s0, s_max, n_s, concentration)
    for a in anchors:
        if a <= 0 or a >= s_max:
            continue
        # snap nearest-but-not-an-endpoint grid point to a
        idx = np.argmin(np.abs(s[1:-1] - a)) + 1
        s[idx] = a
    # ensure strictly increasing (snap can create duplicates if anchors close)
    s = np.unique(s)
    return s


def build_vol_grid(v_max, n_v, concentration=0.05):
    """Non-uniform grid in v, concentrated near v=0."""
    d = concentration
    xi_max = np.arcsinh(v_max / d)
    xi = np.linspace(0.0, xi_max, n_v)
    v = d * np.sinh(xi)
    v[0] = 0.0
    return v


# ---------------------------------------------------------------------------
# Finite-difference operators on a non-uniform grid
# ---------------------------------------------------------------------------
def fd_coeffs_first(x):
    """Centered first-derivative coefficients on a non-uniform grid.
    Returns (a_lower, a_diag, a_upper) of length N each, with
    boundaries handled with one-sided differences."""
    n = len(x)
    a_l = np.zeros(n)
    a_d = np.zeros(n)
    a_u = np.zeros(n)
    for i in range(1, n - 1):
        h_m = x[i] - x[i - 1]
        h_p = x[i + 1] - x[i]
        a_l[i] = -h_p / (h_m * (h_m + h_p))
        a_d[i] = (h_p - h_m) / (h_m * h_p)
        a_u[i] = h_m / (h_p * (h_m + h_p))
    # Forward difference at i=0
    h0 = x[1] - x[0]
    a_d[0] = -1.0 / h0
    a_u[0] = 1.0 / h0
    # Backward difference at i=N-1
    hn = x[-1] - x[-2]
    a_l[-1] = -1.0 / hn
    a_d[-1] = 1.0 / hn
    return a_l, a_d, a_u


def fd_coeffs_second(x):
    """Centered second-derivative coefficients on a non-uniform grid.
    Boundary rows are zero (we'll handle BCs separately)."""
    n = len(x)
    a_l = np.zeros(n)
    a_d = np.zeros(n)
    a_u = np.zeros(n)
    for i in range(1, n - 1):
        h_m = x[i] - x[i - 1]
        h_p = x[i + 1] - x[i]
        a_l[i] = 2.0 / (h_m * (h_m + h_p))
        a_d[i] = -2.0 / (h_m * h_p)
        a_u[i] = 2.0 / (h_p * (h_m + h_p))
    return a_l, a_d, a_u


def tri_to_sparse(a_l, a_d, a_u, n):
    """Build a sparse tridiagonal matrix from coefficient arrays of length N."""
    return diags([a_l[1:], a_d, a_u[:-1]], [-1, 0, 1], shape=(n, n), format='csc')


# ---------------------------------------------------------------------------
# Main SLV solver
# ---------------------------------------------------------------------------
class SLVPdeSolver:
    """
    2D PDE solver for the Stochastic Local Volatility model.

    Parameters
    ----------
    r        : risk-free rate
    kappa    : Heston mean-reversion speed
    theta    : Heston long-term variance
    xi       : vol-of-vol
    rho      : correlation between S and v
    L_func   : leverage function L(S, t). Default = 1.0 (pure Heston).
    """

    def __init__(self, r, kappa, theta, xi, rho, l_func=None):
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        # Leverage: L(S,t)=1 reduces the model to pure Heston.
        self.l_func = l_func if l_func is not None else (lambda s, t: np.ones_like(s))

    # ----- grid setup -------------------------------------------------------
    def setup_grid(self, s0, s_max, v_max, n_s, n_v, s_anchors=None):
        self.s0 = s0
        if s_anchors:
            self.s = build_spot_grid_with_barriers(s0, s_max, n_s, s_anchors)
        else:
            self.s = build_spot_grid(s0, s_max, n_s)
        self.v = build_vol_grid(v_max, n_v)
        self.n_s = len(self.s)   # may have shrunk after unique()
        self.n_v = n_v
        # FD coefficient stencils (1D)
        self._dS_l,  self._dS_d,  self._dS_u  = fd_coeffs_first(self.S)
        self._dSS_l, self._dSS_d, self._dSS_u = fd_coeffs_second(self.S)
        self._dv_l,  self._dv_d,  self._dv_u  = fd_coeffs_first(self.v)
        self._dvv_l, self._dvv_d, self._dvv_u = fd_coeffs_second(self.v)

    # ----- build the spatial operator A = A0 + A1 + A2 ----------------------
    def _build_operators(self, t):
        """
        Decompose A = A0 + A1 + A2 where
          A1 = pure-S terms (drift in S, diffusion in S)
          A2 = pure-v terms (drift in v, diffusion in v) plus the -r*U term
          A0 = mixed S-v cross derivative
        We split -rU equally? -- simpler to lump into A2 (any choice is fine
        as long as A0+A1+A2 = full operator).
        """
        n_s, n_v = self.n_s, self.n_v
        s, v = self.s, self.v

        # Leverage on the grid; size (N_S,)
        l_s = self.l_func(s, t)

        # ---- A1: operates along S for each fixed v --------------------------
        # 0.5 * L^2 * v * S^2 * U_SS  + r * S * U_S
        # We assemble per-row (for each v slice) into a big sparse matrix on
        # the flattened state of size N_S * N_v, with v as the slow index.
        # Index convention: idx(i, j) = j * N_S + i, where i indexes S, j indexes v.
        # i_v = eye(n_v, format='csc') # Seems to be unused

        # Operator on S-only (depends on v through coefficient): we build A1
        # as a block-diagonal sparse matrix where block j corresponds to v[j].
        rows, cols, data = [], [], []
        for j in range(n_v):
            coeff_diff = 0.5 * (l_s ** 2) * v[j] * (s ** 2)   # shape (N_S,)
            coeff_drift = self.r * s                          # shape (N_S,)
            for i in range(n_s):
                base = j * n_s + i
                if 0 < i < n_s - 1:
                    a_l = (coeff_diff[i] * self._dSS_l[i] +
                           coeff_drift[i] * self._dS_l[i])
                    a_d = (coeff_diff[i] * self._dSS_d[i] +
                           coeff_drift[i] * self._dS_d[i])
                    a_u = (coeff_diff[i] * self._dSS_u[i] +
                           coeff_drift[i] * self._dS_u[i])
                    rows += [base, base, base]
                    cols += [base - 1, base, base + 1]
                    data += [a_l, a_d, a_u]
                # boundaries: leave A1 zero -- BCs are imposed externally
        a1 = csc_matrix((data, (rows, cols)),
                        shape=(n_s * n_v, n_s * n_v))

        # ---- A2: operates along v for each fixed S -------------------------
        # 0.5 * xi^2 * v * U_vv + kappa*(theta - v) * U_v - r * U
        rows, cols, data = [], [], []
        for j in range(n_v):
            coeff_diff = 0.5 * self.xi ** 2 * v[j]
            coeff_drift = self.kappa * (self.theta - v[j])
            for i in range(n_s):
                base = j * n_s + i
                if 0 < j < n_v - 1:
                    a_l = (coeff_diff * self._dvv_l[j] +
                           coeff_drift * self._dv_l[j])
                    a_d = (coeff_diff * self._dvv_d[j] +
                           coeff_drift * self._dv_d[j] - self.r)
                    a_u = (coeff_diff * self._dvv_u[j] +
                           coeff_drift * self._dv_u[j])
                    rows += [base, base, base]
                    cols += [base - n_s, base, base + n_s]
                    data += [a_l, a_d, a_u]
                elif j == 0:
                    # v=0 boundary: degenerate diffusion. Use the PDE row:
                    #   U_t + r S U_S + kappa*theta*U_v - r U = 0
                    # so A2 contributes kappa*theta*U_v - r U here.
                    h0 = v[1] - v[0]
                    rows += [base, base]
                    cols += [base, base + n_s]
                    data += [-self.kappa * self.theta / h0 - self.r,
                             self.kappa * self.theta / h0]
                # j = N_v - 1: handled as Neumann (U_v = 0), implemented via BC step
        a2 = csc_matrix((data, (rows, cols)),
                        shape=(n_s * n_v, n_s * n_v))

        # ---- A0: mixed derivative rho*xi*L*v*S * U_Sv ----------------------
        rows, cols, data = [], [], []
        for j in range(1, n_v - 1):
            for i in range(1, n_s - 1):
                base = j * n_s + i
                coeff = self.rho * self.xi * l_s[i] * v[j] * s[i]
                # U_Sv ≈ (dU/dS at j+1 - dU/dS at j-1) / (v[j+1] - v[j-1])
                # Use centered FD in both directions: 4 corner points
                hsm = s[i] - s[i - 1]
                hsp = s[i + 1] - s[i]
                hvm = v[j] - v[j - 1]
                hvp = v[j + 1] - v[j]
                # Centered approximation (uniform-grid form, OK with mild non-unif.)
                fac = coeff / ((hsm + hsp) * (hvm + hvp))
                rows += [base] * 4
                cols += [(j + 1) * n_s + (i + 1),
                         (j + 1) * n_s + (i - 1),
                         (j - 1) * n_s + (i + 1),
                         (j - 1) * n_s + (i - 1)]
                data += [fac, -fac, -fac, fac]
        a0 = csc_matrix((data, (rows, cols)),
                        shape=(n_s * n_v, n_s * n_v))

        return a0, a1, a2

    # ----- apply boundary conditions to a flattened solution ----------------
    def _apply_bc(self, u, payoff_bc):
        """
        Impose BCs in place on U (shape (N_v, N_S)).
          - S=0: U(0,v,t) = payoff_bc['S0'](v,t)
          - S=Smax: U(Smax,v,t) = payoff_bc['Smax'](v,t)
          - v=vmax: U_v = 0  (Neumann -> copy from previous row)
          - v=0: handled via the PDE row inside A2
        """
        if 'S0' in payoff_bc and payoff_bc['S0'] is not None:
            u[:, 0] = payoff_bc['S0']
        if 'Smax' in payoff_bc and payoff_bc['Smax'] is not None:
            u[:, -1] = payoff_bc['Smax']
        # v=vmax: Neumann
        u[-1, :] = u[-2, :]
        return u

    # ----- one Craig-Sneyd ADI step ----------------------------------------
    def _cs_step(self, u_flat, a0, a1, a2, dt, theta_cs=0.5):
        """
        Craig-Sneyd ADI for backward step (we're stepping U(t+dt) -> U(t)):
            Y0 = U + dt * (A0+A1+A2) U
            (I - theta dt A1) Y1 = Y0 - theta dt A1 U
            (I - theta dt A2) Y2 = Y1 - theta dt A2 U
            Yhat0 = Y0 + 0.5 dt A0 (Y2 - U)
            (I - theta dt A1) Yhat1 = Yhat0 - theta dt A1 U
            (I - theta dt A2) U_new = Yhat1 - theta dt A2 U
        """
        i_full = identity(a0.shape[0], format='csc')
        m1 = (i_full - theta_cs * dt * a1).tocsc()
        m2 = (i_full - theta_cs * dt * a2).tocsc()
        lu1 = splu(m1)
        lu2 = splu(m2)

        a_full = a0 + a1 + a2

        # predictor
        y0 = u_flat + dt * (a_full @ u_flat)
        y1 = lu1.solve(y0 - theta_cs * dt * (a1 @ u_flat))
        y2 = lu2.solve(y1 - theta_cs * dt * (a2 @ u_flat))

        # corrector (Craig-Sneyd)
        yhat0 = y0 + 0.5 * dt * (a0 @ (y2 - u_flat))
        yhat1 = lu1.solve(yhat0 - theta_cs * dt * (a1 @ u_flat))
        u_new = lu2.solve(yhat1 - theta_cs * dt * (a2 @ u_flat))
        return u_new

    # ----- main solve routine ----------------------------------------------
    def solve(self, t, n_t, terminal_payoff, s_zero_bc=None, s_max_bc=None,
              barrier_check=None, rebate=0.0, l_time_independent=True):
        """
        Solve PDE backward in time from T to 0.

        Parameters
        ----------
        T               : maturity
        N_t             : number of time steps
        terminal_payoff : f(S_grid, v_grid) -> array shape (N_v, n_s)
        s_zero_bc       : f(v_grid, t) -> array shape (N_v,)  for S=0
        s_max_bc        : f(v_grid, t) -> array shape (N_v,)  for S=Smax
        barrier_check   : optional callable applied each step:
                          barrier_check(U_2d, t) -> U_2d   (knock-out etc.)
        L_time_independent : if True, build operators and LU-factor once.
        """
        n_s, n_v = self.n_s, self.N_v
        dt = t / n_t

        # terminal condition
        s_mesh, v_mesh = np.meshgrid(self.S, self.v)  # shape (N_v, n_s)
        u = terminal_payoff(s_mesh, v_mesh).astype(float)

        if barrier_check is not None:
            u = barrier_check(u, t)

        # Pre-build operators and LU factors if L is time-independent
        if l_time_independent:
            a0, a1, a2 = self._build_operators(0.0)
            i_full = identity(a0.shape[0], format='csc')
            theta_cs = 0.5
            m1 = (i_full - theta_cs * dt * a1).tocsc()
            m2 = (i_full - theta_cs * dt * a2).tocsc()
            lu1 = splu(m1)
            lu2 = splu(m2)
            cached = (a0, a1, a2, lu1, lu2, theta_cs)
        else:
            cached = None

        for n in range(n_t, 0, -1):
            t_old = n * dt
            t_new = (n - 1) * dt

            bc = {}
            bc['S0'] = s_zero_bc(self.v, t_old) if s_zero_bc else None
            bc['Smax'] = s_max_bc(self.v, t_old) if s_max_bc else None
            u = self._apply_bc(u, bc)

            u_flat = u.flatten()
            if cached is not None:
                u_flat_new = self._cs_step_cached(u_flat, dt, *cached)
            else:
                a0, a1, a2 = self._build_operators(0.5 * (t_old + t_new))
                u_flat_new = self._cs_step(u_flat, a0, a1, a2, dt)
            u = u_flat_new.reshape((n_v, n_s))

            bc['S0'] = s_zero_bc(self.v, t_new) if s_zero_bc else None
            bc['Smax'] = s_max_bc(self.v, t_new) if s_max_bc else None
            u = self._apply_bc(u, bc)

            if barrier_check is not None:
                u = barrier_check(u, t_new)

        return u

    def _cs_step_cached(self, u_flat, dt, a0, a1, a2, lu1, lu2, theta_cs):
        """Same as _cs_step but with pre-factorised LU."""
        a_full = a0 + a1 + a2
        y0 = u_flat + dt * (a_full @ u_flat)
        y1 = lu1.solve(y0 - theta_cs * dt * (a1 @ u_flat))
        y2 = lu2.solve(y1 - theta_cs * dt * (a2 @ u_flat))
        yhat0 = y0 + 0.5 * dt * (a0 @ (y2 - u_flat))
        yhat1 = lu1.solve(yhat0 - theta_cs * dt * (a1 @ u_flat))
        u_new = lu2.solve(yhat1 - theta_cs * dt * (a2 @ u_flat))
        return u_new

    # ----- price retrieval at (S0, v0) -------------------------------------
    def price_at(self, u, v0):
        """Bilinear interpolation of price at (S0, v0)."""
        # locate v0
        j = np.searchsorted(self.v, v0) - 1
        j = max(0, min(self.N_v - 2, j))
        wv = (v0 - self.v[j]) / (self.v[j + 1] - self.v[j])
        # locate S0
        i = np.searchsorted(self.S, self.S0) - 1
        i = max(0, min(self.N_S - 2, i))
        ws = (self.S0 - self.S[i]) / (self.S[i + 1] - self.S[i])
        # bilinear
        p = ((1 - wv) * (1 - ws) * u[j, i] +
             (1 - wv) * ws * u[j, i + 1] +
             wv * (1 - ws) * u[j + 1, i] +
             wv * ws * u[j + 1, i + 1])
        return p
