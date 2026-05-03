"""
Monte Carlo for the SLV (or pure Heston) model -- used to sanity-check
the PDE prices for the DNT.
"""
import numpy as np


def mc_slv(S0, v0, r, kappa, theta, xi, rho, T, n_paths, n_steps,
           L_func=None, seed=0):
    """Simulate paths under the SLV model with full-truncation Euler."""
    if L_func is None:
        L_func = lambda S, t: np.ones_like(S)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])

    paths_S = np.zeros((n_steps + 1, n_paths))
    paths_S[0] = S0
    for k in range(1, n_steps + 1):
        z = rng.standard_normal((n_paths, 2)) @ chol.T
        v_pos = np.maximum(v, 0.0)
        L_S = L_func(S, (k - 0.5) * dt)
        S = S + r * S * dt + L_S * np.sqrt(v_pos) * S * sqdt * z[:, 0]
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqdt * z[:, 1]
        S = np.maximum(S, 1e-8)
        paths_S[k] = S
    return paths_S


def mc_vanilla(S0, K, T, r, kappa, theta, xi, rho, v0,
               option_type='call', n_paths=200_000, n_steps=200,
               L_func=None, seed=1):
    paths = mc_slv(S0, v0, r, kappa, theta, xi, rho, T, n_paths, n_steps,
                   L_func=L_func, seed=seed)
    ST = paths[-1]
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T)
    price = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, se


def mc_dnt(S0, B_low, B_up, T, r, kappa, theta, xi, rho, v0,
           rebate=1.0, n_paths=200_000, n_steps=400,
           L_func=None, seed=2):
    """Memory-efficient: track running min/max instead of all paths."""
    if L_func is None:
        L_func = lambda S, t: np.ones_like(S)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])

    survived = np.ones(n_paths, dtype=bool)
    survived &= (S > B_low) & (S < B_up)

    for k in range(1, n_steps + 1):
        z = rng.standard_normal((n_paths, 2)) @ chol.T
        v_pos = np.maximum(v, 0.0)
        L_S = L_func(S, (k - 0.5) * dt)
        S = S + r * S * dt + L_S * np.sqrt(v_pos) * S * sqdt * z[:, 0]
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqdt * z[:, 1]
        S = np.maximum(S, 1e-8)
        survived &= (S > B_low) & (S < B_up)

    disc = np.exp(-r * T)
    payoff = rebate * survived.astype(float)
    price = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, se
