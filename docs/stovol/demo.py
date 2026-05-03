"""
Demo: SLV PDE pricer.

Step 1 -- L = 1: model collapses to pure Heston; vanilla prices checked
          against the closed-form integral.
Step 2 -- still L = 1: price a Double-No-Touch and cross-check vs Monte Carlo.
Step 3 -- turn on a non-trivial leverage L(S) and re-price both products.
"""
import time
import numpy as np

from slv_solver import SLVPdeSolver
from products import price_vanilla, price_dnt
from heston_cf import heston_call, heston_put
from mc_reference import mc_dnt


# Market & model setup (FX-style)
S0    = 100.0
r     = 0.02
kappa = 1.5
theta = 0.04
xi    = 0.4
rho   = -0.5
v0    = 0.04            # ATM short-dated vol ~ 20%

T_van = 1.0
K_van = 100.0

T_dnt = 0.5
B_low = 90.0
B_up  = 112.0


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def new_solver(L_func=None):
    return SLVPdeSolver(r=r, kappa=kappa, theta=theta, xi=xi, rho=rho,
                        L_func=L_func)


# 1. Vanilla check against closed-form Heston
banner("1. Pure Heston (L = 1) -- vanillas checked against closed-form")
cf_call = heston_call(S0, K_van, T_van, r, kappa, theta, xi, rho, v0)
cf_put  = heston_put (S0, K_van, T_van, r, kappa, theta, xi, rho, v0)

t0 = time.time()
pde_call, _ = price_vanilla(new_solver(), S0=S0, K=K_van, T=T_van,
                            N_t=80, v0=v0, option_type='call',
                            N_S=120, N_v=40)
pde_put,  _ = price_vanilla(new_solver(), S0=S0, K=K_van, T=T_van,
                            N_t=80, v0=v0, option_type='put',
                            N_S=120, N_v=40)
print(f"(PDE vanillas took {time.time()-t0:.1f}s)")
print(f"ATM call:  PDE = {pde_call:9.5f}    CF = {cf_call:9.5f}    "
      f"diff = {pde_call - cf_call:+.5f}")
print(f"ATM put :  PDE = {pde_put :9.5f}    CF = {cf_put :9.5f}    "
      f"diff = {pde_put  - cf_put :+.5f}")
parity_lhs = pde_call - pde_put
parity_rhs = S0 - K_van * np.exp(-r * T_van)
print(f"Put-call parity: C - P = {parity_lhs:.5f}, "
      f"S - Ke^(-rT) = {parity_rhs:.5f}  diff = {parity_lhs - parity_rhs:+.2e}")


# 2. DNT under pure Heston, cross-checked vs MC
banner("2. Pure Heston (L = 1) -- DNT vs Monte Carlo")
t0 = time.time()
pde_dnt, _ = price_dnt(new_solver(), S0=S0,
                       B_low=B_low, B_up=B_up,
                       T=T_dnt, N_t=200, v0=v0, rebate=1.0,
                       N_S=200, N_v=40)
print(f"(PDE DNT took {time.time()-t0:.1f}s)")
print(f"DNT PDE price = {pde_dnt:.5f}")

print("Running MC...")
t0 = time.time()
mc_price, mc_se = mc_dnt(S0, B_low, B_up, T_dnt, r, kappa, theta, xi, rho, v0,
                         rebate=1.0, n_paths=200_000, n_steps=400)
print(f"(MC took {time.time()-t0:.1f}s)")
print(f"DNT MC  price = {mc_price:.5f}  (SE = {mc_se:.5f}, "
      f"95% CI = [{mc_price - 1.96*mc_se:.5f}, {mc_price + 1.96*mc_se:.5f}])")
print("(MC monitors barriers discretely => upward bias relative to "
      "continuous-monitoring PDE.)")


# 3. SLV: turn on a non-trivial leverage L(S)
banner("3. SLV: non-trivial leverage L(S)")
def L_func(S, t):
    return 1.0 + 0.20 * (S0 / np.maximum(S, 1e-8) - 1.0)

slv_call, _ = price_vanilla(new_solver(L_func), S0=S0, K=K_van, T=T_van,
                            N_t=80, v0=v0, option_type='call',
                            N_S=120, N_v=40)
slv_put,  _ = price_vanilla(new_solver(L_func), S0=S0, K=K_van, T=T_van,
                            N_t=80, v0=v0, option_type='put',
                            N_S=120, N_v=40)
print(f"SLV ATM call = {slv_call:.5f}    (Heston {pde_call:.5f})")
print(f"SLV ATM put  = {slv_put :.5f}    (Heston {pde_put :.5f})")

slv_dnt, _ = price_dnt(new_solver(L_func), S0=S0,
                       B_low=B_low, B_up=B_up,
                       T=T_dnt, N_t=200, v0=v0, rebate=1.0,
                       N_S=200, N_v=40)
print(f"SLV DNT      = {slv_dnt:.5f}    (Heston {pde_dnt:.5f})")

print("Running SLV MC...")
mc_slv_price, mc_slv_se = mc_dnt(S0, B_low, B_up, T_dnt, r, kappa, theta, xi,
                                 rho, v0, rebate=1.0,
                                 n_paths=200_000, n_steps=400, L_func=L_func)
print(f"SLV DNT MC   = {mc_slv_price:.5f}  (SE = {mc_slv_se:.5f})")

print("\nDone.")
