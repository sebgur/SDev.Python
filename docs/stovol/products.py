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
def price_vanilla(solver: SLVPdeSolver, s0, k, t, n_t, v0,
                  option_type='call', s_max=None, v_max=None,
                  n_s=120, n_v=40):
    """ Price a European vanilla call or put """
    s_max = s_max or 4.0 * max(s0, k)
    v_max = v_max or 1.0
    solver.setup_grid(s0, s_max, v_max, n_s, n_v)

    if option_type == 'call':
        terminal = lambda S, v: np.maximum(s - k, 0.0) # noqa
        # BCs:  C(0, v, t) = 0
        #       C(Smax, v, t) ≈ Smax - K e^{-r(T-t)}
        s0_bc   = lambda v, u: np.zeros_like(v) # noqa
        smax_bc = lambda v, u: solver.s[-1] - k * np.exp(-solver.r * (t - u)) # noqa
    elif option_type == 'put':
        terminal = lambda S, v: np.maximum(k - s, 0.0) # noqa
        s0_bc   = lambda v, u: np.full_like(v, k * np.exp(-solver.r * (t - u))) # noqa
        smax_bc = lambda v, u: np.zeros_like(v) # noqa
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    u = solver.solve(t, n_t, terminal,
                     s_zero_bc=s0_bc, s_max_bc=smax_bc)
    return solver.price_at(u, v0), u


# ---------------------------------------------------------------------------
# Double-No-Touch
# ---------------------------------------------------------------------------
def price_dnt(solver: SLVPdeSolver, s0, b_low, b_up, t, n_t, v0,
              rebate=1.0, s_max=None, v_max=None,
              n_s=160, n_v=40):
    """
    Continuously-monitored Double-No-Touch.
    Pays `rebate` at T if S_t stays strictly inside (B_low, B_up) for all t in [0,T];
    pays 0 otherwise.

    BCs: U = 0 on S=B_low and S=B_up at all times
         (we mirror those conditions to S=0 and S=Smax for safety).
    Knock-out enforced at every time step on grid points outside the corridor.
    """
    s_max = s_max or 1.5 * b_up
    v_max = v_max or 1.0
    solver.setup_grid(s0, s_max, v_max, n_s, n_v,
                      s_anchors=[b_low, b_up])

    # terminal payoff: rebate inside, zero outside
    def terminal(s, v):
        out = np.zeros_like(s)
        inside = (s > b_low) & (s < b_up)
        out[inside] = rebate
        return out

    # outer BCs: zero everywhere outside corridor
    s0_bc   = lambda v, u: np.zeros_like(v) # noqa
    smax_bc = lambda v, u: np.zeros_like(v) # noqa

    # knock-out: at each step, set U=0 on grid points outside corridor
    s = solver.s
    outside_mask = (s <= b_low) | (s >= b_up)

    def barrier_check(u_2d, t):
        u_2d[:, outside_mask] = 0.0
        return u_2d

    u = solver.solve(t, n_t, terminal,
                     s_zero_bc=s0_bc, s_max_bc=smax_bc,
                     barrier_check=barrier_check)
    return solver.price_at(u, v0), u
