"""
Product pricers built on top of SLVPdeSolver.

Vanillas: European call / put.
DNT     : Double-no-touch — pays a rebate at maturity if S has remained
          strictly inside (B_low, B_up) throughout the life of the trade.
"""

import numpy as np
from slv_solver import SLVPdeSolver


# ---------------------------------------------------------------------------
# Vanilla European call/put
# ---------------------------------------------------------------------------
def price_vanilla(solver: SLVPdeSolver, S0, K, T, N_t, v0,
                  option_type='call', S_max=None, v_max=None,
                  N_S=120, N_v=40):
    """
    Price a European vanilla call or put.
    """
    S_max = S_max or 4.0 * max(S0, K)
    v_max = v_max or 1.0
    solver.setup_grid(S0, S_max, v_max, N_S, N_v)

    if option_type == 'call':
        terminal = lambda S, v: np.maximum(S - K, 0.0)
        # BCs:  C(0, v, t) = 0
        #       C(Smax, v, t) ≈ Smax - K e^{-r(T-t)}
        s0_bc   = lambda v, t: np.zeros_like(v)
        smax_bc = lambda v, t: solver.S[-1] - K * np.exp(-solver.r * (T - t))
    elif option_type == 'put':
        terminal = lambda S, v: np.maximum(K - S, 0.0)
        s0_bc   = lambda v, t: np.full_like(v, K * np.exp(-solver.r * (T - t)))
        smax_bc = lambda v, t: np.zeros_like(v)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    U = solver.solve(T, N_t, terminal,
                     s_zero_bc=s0_bc, s_max_bc=smax_bc)
    return solver.price_at(U, v0), U


# ---------------------------------------------------------------------------
# Double-No-Touch
# ---------------------------------------------------------------------------
def price_dnt(solver: SLVPdeSolver, S0, B_low, B_up, T, N_t, v0,
              rebate=1.0, S_max=None, v_max=None,
              N_S=160, N_v=40):
    """
    Continuously-monitored Double-No-Touch.
    Pays `rebate` at T if S_t stays strictly inside (B_low, B_up) for all t in [0,T];
    pays 0 otherwise.

    BCs: U = 0 on S=B_low and S=B_up at all times
         (we mirror those conditions to S=0 and S=Smax for safety).
    Knock-out enforced at every time step on grid points outside the corridor.
    """
    S_max = S_max or 1.5 * B_up
    v_max = v_max or 1.0
    solver.setup_grid(S0, S_max, v_max, N_S, N_v,
                      S_anchors=[B_low, B_up])

    # terminal payoff: rebate inside, zero outside
    def terminal(S, v):
        out = np.zeros_like(S)
        inside = (S > B_low) & (S < B_up)
        out[inside] = rebate
        return out

    # outer BCs: zero everywhere outside corridor
    s0_bc   = lambda v, t: np.zeros_like(v)
    smax_bc = lambda v, t: np.zeros_like(v)

    # knock-out: at each step, set U=0 on grid points outside corridor
    S = solver.S
    outside_mask = (S <= B_low) | (S >= B_up)

    def barrier_check(U_2d, t):
        U_2d[:, outside_mask] = 0.0
        return U_2d

    U = solver.solve(T, N_t, terminal,
                     s_zero_bc=s0_bc, s_max_bc=smax_bc,
                     barrier_check=barrier_check)
    return solver.price_at(U, v0), U
