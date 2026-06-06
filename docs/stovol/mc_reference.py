"""
Monte Carlo for the SLV (or pure Heston) model -- used to sanity-check
the PDE prices for the DNT.
"""
import numpy as np


def mc_slv(s0, v0, r, kappa, theta, xi, rho, t, n_paths, n_steps,
           l_func=None, seed=0):
    """Simulate paths under the SLV model with full-truncation Euler."""
    if l_func is None:
        l_func = lambda S, t: np.ones_like(S) # noqa
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqdt = np.sqrt(dt)

    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])

    paths_s = np.zeros((n_steps + 1, n_paths))
    paths_s[0] = s0
    for k in range(1, n_steps + 1):
        z = rng.standard_normal((n_paths, 2)) @ chol.T
        v_pos = np.maximum(v, 0.0)
        l_s = l_func(s, (k - 0.5) * dt)
        s = s + r * s * dt + l_s * np.sqrt(v_pos) * s * sqdt * z[:, 0]
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqdt * z[:, 1]
        s = np.maximum(s, 1e-8)
        paths_s[k] = s
    return paths_s


def mc_vanilla(s0, k, t, r, kappa, theta, xi, rho, v0,
               option_type='call', n_paths=200_000, n_steps=200,
               l_func=None, seed=1):
    paths = mc_slv(s0, v0, r, kappa, theta, xi, rho, t, n_paths, n_steps,
                   l_func=l_func, seed=seed)
    st = paths[-1]
    if option_type == 'call':
        payoff = np.maximum(st - k, 0.0)
    else:
        payoff = np.maximum(k - st, 0.0)
    disc = np.exp(-r * t)
    price = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, se


def mc_dnt(s0, b_low, b_up, t, r, kappa, theta, xi, rho, v0,
           rebate=1.0, n_paths=200_000, n_steps=400,
           l_func=None, seed=2):
    """Memory-efficient: track running min/max instead of all paths."""
    if l_func is None:
        l_func = lambda S, t: np.ones_like(S) # noqa
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqdt = np.sqrt(dt)

    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])

    survived = np.ones(n_paths, dtype=bool)
    survived &= (s > b_low) & (s < b_up)

    for k in range(1, n_steps + 1):
        z = rng.standard_normal((n_paths, 2)) @ chol.T
        v_pos = np.maximum(v, 0.0)
        l_s = l_func(s, (k - 0.5) * dt)
        s = s + r * s * dt + l_s * np.sqrt(v_pos) * s * sqdt * z[:, 0]
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqdt * z[:, 1]
        s = np.maximum(s, 1e-8)
        survived &= (s > b_low) & (s < b_up)

    disc = np.exp(-r * t)
    payoff = rebate * survived.astype(float)
    price = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, se
