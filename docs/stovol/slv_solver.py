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
from scipy.sparse import diags, csc_matrix, identity, kron, eye
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_spot_grid(S0, S_max, N_S, concentration=0.1):
    """
    Non-uniform grid in S, concentrated around S0 using a sinh transform.
    Concentration controls how tightly points cluster near S0.
    """
    c = concentration * S0
    xi_min = np.arcsinh(-S0 / c)
    xi_max = np.arcsinh((S_max - S0) / c)
    xi = np.linspace(xi_min, xi_max, N_S)
    S = S0 + c * np.sinh(xi)
    S[0] = 0.0  # clamp lower bound
    return S


def build_spot_grid_with_barriers(S0, S_max, N_S, anchors, concentration=0.1):
    """
    Like build_spot_grid but ensures every value in `anchors` (e.g. barriers)
    appears exactly on the grid. Achieved by snapping the nearest grid point
    to each anchor. Returns a sorted, strictly-increasing array.
    """
    S = build_spot_grid(S0, S_max, N_S, concentration)
    for a in anchors:
        if a <= 0 or a >= S_max:
            continue
        # snap nearest-but-not-an-endpoint grid point to a
        idx = np.argmin(np.abs(S[1:-1] - a)) + 1
        S[idx] = a
    # ensure strictly increasing (snap can create duplicates if anchors close)
    S = np.unique(S)
    return S


def build_vol_grid(v_max, N_v, concentration=0.05):
    """Non-uniform grid in v, concentrated near v=0."""
    d = concentration
    xi_max = np.arcsinh(v_max / d)
    xi = np.linspace(0.0, xi_max, N_v)
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
    N = len(x)
    a_l = np.zeros(N); a_d = np.zeros(N); a_u = np.zeros(N)
    for i in range(1, N - 1):
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
    hN = x[-1] - x[-2]
    a_l[-1] = -1.0 / hN
    a_d[-1] = 1.0 / hN
    return a_l, a_d, a_u


def fd_coeffs_second(x):
    """Centered second-derivative coefficients on a non-uniform grid.
    Boundary rows are zero (we'll handle BCs separately)."""
    N = len(x)
    a_l = np.zeros(N); a_d = np.zeros(N); a_u = np.zeros(N)
    for i in range(1, N - 1):
        h_m = x[i] - x[i - 1]
        h_p = x[i + 1] - x[i]
        a_l[i] = 2.0 / (h_m * (h_m + h_p))
        a_d[i] = -2.0 / (h_m * h_p)
        a_u[i] = 2.0 / (h_p * (h_m + h_p))
    return a_l, a_d, a_u


def tri_to_sparse(a_l, a_d, a_u, N):
    """Build a sparse tridiagonal matrix from coefficient arrays of length N."""
    return diags([a_l[1:], a_d, a_u[:-1]], [-1, 0, 1], shape=(N, N), format='csc')


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

    def __init__(self, r, kappa, theta, xi, rho, L_func=None):
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        # Leverage: L(S,t)=1 reduces the model to pure Heston.
        self.L_func = L_func if L_func is not None else (lambda S, t: np.ones_like(S))

    # ----- grid setup -------------------------------------------------------
    def setup_grid(self, S0, S_max, v_max, N_S, N_v, S_anchors=None):
        self.S0 = S0
        if S_anchors:
            self.S = build_spot_grid_with_barriers(S0, S_max, N_S, S_anchors)
        else:
            self.S = build_spot_grid(S0, S_max, N_S)
        self.v = build_vol_grid(v_max, N_v)
        self.N_S = len(self.S)   # may have shrunk after unique()
        self.N_v = N_v
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
        N_S, N_v = self.N_S, self.N_v
        S, v = self.S, self.v

        # Leverage on the grid; size (N_S,)
        L_S = self.L_func(S, t)

        # ---- A1: operates along S for each fixed v --------------------------
        # 0.5 * L^2 * v * S^2 * U_SS  + r * S * U_S
        # We assemble per-row (for each v slice) into a big sparse matrix on
        # the flattened state of size N_S * N_v, with v as the slow index.
        # Index convention: idx(i, j) = j * N_S + i, where i indexes S, j indexes v.
        I_v = eye(N_v, format='csc')

        # Operator on S-only (depends on v through coefficient): we build A1
        # as a block-diagonal sparse matrix where block j corresponds to v[j].
        rows, cols, data = [], [], []
        for j in range(N_v):
            coeff_diff = 0.5 * (L_S ** 2) * v[j] * (S ** 2)   # shape (N_S,)
            coeff_drift = self.r * S                          # shape (N_S,)
            for i in range(N_S):
                base = j * N_S + i
                if 0 < i < N_S - 1:
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
        A1 = csc_matrix((data, (rows, cols)),
                        shape=(N_S * N_v, N_S * N_v))

        # ---- A2: operates along v for each fixed S -------------------------
        # 0.5 * xi^2 * v * U_vv + kappa*(theta - v) * U_v - r * U
        rows, cols, data = [], [], []
        for j in range(N_v):
            coeff_diff = 0.5 * self.xi ** 2 * v[j]
            coeff_drift = self.kappa * (self.theta - v[j])
            for i in range(N_S):
                base = j * N_S + i
                if 0 < j < N_v - 1:
                    a_l = (coeff_diff * self._dvv_l[j] +
                           coeff_drift * self._dv_l[j])
                    a_d = (coeff_diff * self._dvv_d[j] +
                           coeff_drift * self._dv_d[j] - self.r)
                    a_u = (coeff_diff * self._dvv_u[j] +
                           coeff_drift * self._dv_u[j])
                    rows += [base, base, base]
                    cols += [base - N_S, base, base + N_S]
                    data += [a_l, a_d, a_u]
                elif j == 0:
                    # v=0 boundary: degenerate diffusion. Use the PDE row:
                    #   U_t + r S U_S + kappa*theta*U_v - r U = 0
                    # so A2 contributes kappa*theta*U_v - r U here.
                    h0 = v[1] - v[0]
                    rows += [base, base]
                    cols += [base, base + N_S]
                    data += [-self.kappa * self.theta / h0 - self.r,
                             self.kappa * self.theta / h0]
                # j = N_v - 1: handled as Neumann (U_v = 0), implemented via BC step
        A2 = csc_matrix((data, (rows, cols)),
                        shape=(N_S * N_v, N_S * N_v))

        # ---- A0: mixed derivative rho*xi*L*v*S * U_Sv ----------------------
        rows, cols, data = [], [], []
        for j in range(1, N_v - 1):
            for i in range(1, N_S - 1):
                base = j * N_S + i
                coeff = self.rho * self.xi * L_S[i] * v[j] * S[i]
                # U_Sv ≈ (dU/dS at j+1 - dU/dS at j-1) / (v[j+1] - v[j-1])
                # Use centered FD in both directions: 4 corner points
                hSm = S[i] - S[i - 1]; hSp = S[i + 1] - S[i]
                hvm = v[j] - v[j - 1]; hvp = v[j + 1] - v[j]
                # Centered approximation (uniform-grid form, OK with mild non-unif.)
                fac = coeff / ((hSm + hSp) * (hvm + hvp))
                rows += [base] * 4
                cols += [(j + 1) * N_S + (i + 1),
                         (j + 1) * N_S + (i - 1),
                         (j - 1) * N_S + (i + 1),
                         (j - 1) * N_S + (i - 1)]
                data += [fac, -fac, -fac, fac]
        A0 = csc_matrix((data, (rows, cols)),
                        shape=(N_S * N_v, N_S * N_v))

        return A0, A1, A2

    # ----- apply boundary conditions to a flattened solution ----------------
    def _apply_bc(self, U, payoff_bc):
        """
        Impose BCs in place on U (shape (N_v, N_S)).
          - S=0: U(0,v,t) = payoff_bc['S0'](v,t)
          - S=Smax: U(Smax,v,t) = payoff_bc['Smax'](v,t)
          - v=vmax: U_v = 0  (Neumann -> copy from previous row)
          - v=0: handled via the PDE row inside A2
        """
        if 'S0' in payoff_bc and payoff_bc['S0'] is not None:
            U[:, 0] = payoff_bc['S0']
        if 'Smax' in payoff_bc and payoff_bc['Smax'] is not None:
            U[:, -1] = payoff_bc['Smax']
        # v=vmax: Neumann
        U[-1, :] = U[-2, :]
        return U

    # ----- one Craig-Sneyd ADI step ----------------------------------------
    def _cs_step(self, U_flat, A0, A1, A2, dt, theta_cs=0.5):
        """
        Craig-Sneyd ADI for backward step (we're stepping U(t+dt) -> U(t)):
            Y0 = U + dt * (A0+A1+A2) U
            (I - theta dt A1) Y1 = Y0 - theta dt A1 U
            (I - theta dt A2) Y2 = Y1 - theta dt A2 U
            Yhat0 = Y0 + 0.5 dt A0 (Y2 - U)
            (I - theta dt A1) Yhat1 = Yhat0 - theta dt A1 U
            (I - theta dt A2) U_new = Yhat1 - theta dt A2 U
        """
        I_full = identity(A0.shape[0], format='csc')
        M1 = (I_full - theta_cs * dt * A1).tocsc()
        M2 = (I_full - theta_cs * dt * A2).tocsc()
        lu1 = splu(M1); lu2 = splu(M2)

        A_full = A0 + A1 + A2

        # predictor
        Y0 = U_flat + dt * (A_full @ U_flat)
        Y1 = lu1.solve(Y0 - theta_cs * dt * (A1 @ U_flat))
        Y2 = lu2.solve(Y1 - theta_cs * dt * (A2 @ U_flat))

        # corrector (Craig-Sneyd)
        Yhat0 = Y0 + 0.5 * dt * (A0 @ (Y2 - U_flat))
        Yhat1 = lu1.solve(Yhat0 - theta_cs * dt * (A1 @ U_flat))
        U_new = lu2.solve(Yhat1 - theta_cs * dt * (A2 @ U_flat))
        return U_new

    # ----- main solve routine ----------------------------------------------
    def solve(self, T, N_t, terminal_payoff,
              s_zero_bc=None, s_max_bc=None,
              barrier_check=None, rebate=0.0,
              L_time_independent=True):
        """
        Solve PDE backward in time from T to 0.

        Parameters
        ----------
        T               : maturity
        N_t             : number of time steps
        terminal_payoff : f(S_grid, v_grid) -> array shape (N_v, N_S)
        s_zero_bc       : f(v_grid, t) -> array shape (N_v,)  for S=0
        s_max_bc        : f(v_grid, t) -> array shape (N_v,)  for S=Smax
        barrier_check   : optional callable applied each step:
                          barrier_check(U_2d, t) -> U_2d   (knock-out etc.)
        L_time_independent : if True, build operators and LU-factor once.
        """
        N_S, N_v = self.N_S, self.N_v
        dt = T / N_t

        # terminal condition
        S_mesh, v_mesh = np.meshgrid(self.S, self.v)  # shape (N_v, N_S)
        U = terminal_payoff(S_mesh, v_mesh).astype(float)

        if barrier_check is not None:
            U = barrier_check(U, T)

        # Pre-build operators and LU factors if L is time-independent
        if L_time_independent:
            A0, A1, A2 = self._build_operators(0.0)
            I_full = identity(A0.shape[0], format='csc')
            theta_cs = 0.5
            M1 = (I_full - theta_cs * dt * A1).tocsc()
            M2 = (I_full - theta_cs * dt * A2).tocsc()
            lu1 = splu(M1); lu2 = splu(M2)
            cached = (A0, A1, A2, lu1, lu2, theta_cs)
        else:
            cached = None

        for n in range(N_t, 0, -1):
            t_old = n * dt
            t_new = (n - 1) * dt

            bc = {}
            bc['S0'] = s_zero_bc(self.v, t_old) if s_zero_bc else None
            bc['Smax'] = s_max_bc(self.v, t_old) if s_max_bc else None
            U = self._apply_bc(U, bc)

            U_flat = U.flatten()
            if cached is not None:
                U_flat_new = self._cs_step_cached(U_flat, dt, *cached)
            else:
                A0, A1, A2 = self._build_operators(0.5 * (t_old + t_new))
                U_flat_new = self._cs_step(U_flat, A0, A1, A2, dt)
            U = U_flat_new.reshape((N_v, N_S))

            bc['S0'] = s_zero_bc(self.v, t_new) if s_zero_bc else None
            bc['Smax'] = s_max_bc(self.v, t_new) if s_max_bc else None
            U = self._apply_bc(U, bc)

            if barrier_check is not None:
                U = barrier_check(U, t_new)

        return U

    def _cs_step_cached(self, U_flat, dt, A0, A1, A2, lu1, lu2, theta_cs):
        """Same as _cs_step but with pre-factorised LU."""
        A_full = A0 + A1 + A2
        Y0 = U_flat + dt * (A_full @ U_flat)
        Y1 = lu1.solve(Y0 - theta_cs * dt * (A1 @ U_flat))
        Y2 = lu2.solve(Y1 - theta_cs * dt * (A2 @ U_flat))
        Yhat0 = Y0 + 0.5 * dt * (A0 @ (Y2 - U_flat))
        Yhat1 = lu1.solve(Yhat0 - theta_cs * dt * (A1 @ U_flat))
        U_new = lu2.solve(Yhat1 - theta_cs * dt * (A2 @ U_flat))
        return U_new

    # ----- price retrieval at (S0, v0) -------------------------------------
    def price_at(self, U, v0):
        """Bilinear interpolation of price at (S0, v0)."""
        # locate v0
        j = np.searchsorted(self.v, v0) - 1
        j = max(0, min(self.N_v - 2, j))
        wv = (v0 - self.v[j]) / (self.v[j + 1] - self.v[j])
        # locate S0
        i = np.searchsorted(self.S, self.S0) - 1
        i = max(0, min(self.N_S - 2, i))
        wS = (self.S0 - self.S[i]) / (self.S[i + 1] - self.S[i])
        # bilinear
        p = ((1 - wv) * (1 - wS) * U[j, i] +
             (1 - wv) * wS * U[j, i + 1] +
             wv * (1 - wS) * U[j + 1, i] +
             wv * wS * U[j + 1, i + 1])
        return p
