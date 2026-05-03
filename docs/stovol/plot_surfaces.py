"""Plot the DNT price surface U(S, v, t=0) -- a useful diagnostic."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from slv_solver import SLVPdeSolver
from products import price_dnt, price_vanilla

S0=100.0; r=0.02; kappa=1.5; theta=0.04; xi=0.4; rho=-0.5; v0=0.04

# DNT
sv = SLVPdeSolver(r, kappa, theta, xi, rho)
_, U = price_dnt(sv, S0=S0, B_low=90.0, B_up=112.0, T=0.5, N_t=200, v0=v0,
                 rebate=1.0, N_S=200, N_v=40)
S = sv.S; v = sv.v

# Vanilla call
sv2 = SLVPdeSolver(r, kappa, theta, xi, rho)
_, U2 = price_vanilla(sv2, S0=S0, K=100.0, T=1.0, N_t=80, v0=v0,
                      option_type='call', N_S=120, N_v=40)
S2 = sv2.S; v2 = sv2.v

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DNT price surface, restricted to interesting (S, v) range
S_idx = (S >= 80) & (S <= 122)
v_idx = (v <= 0.20)
Sp, vp = np.meshgrid(S[S_idx], v[v_idx])
Up = U[np.ix_(v_idx, S_idx)]
c1 = axes[0].contourf(Sp, vp, Up, 20, cmap='viridis')
axes[0].axvline(90.0, color='r', linestyle='--', alpha=0.6, label='Lower barrier')
axes[0].axvline(112.0, color='r', linestyle='--', alpha=0.6, label='Upper barrier')
axes[0].axvline(S0, color='w', linestyle=':', alpha=0.6, label='S0')
axes[0].set_xlabel('Spot S'); axes[0].set_ylabel('Variance v')
axes[0].set_title('DNT(90,112) price at t=0,  T=0.5  (Heston)')
axes[0].legend(loc='upper right')
plt.colorbar(c1, ax=axes[0])

# Vanilla call surface
S_idx2 = (S2 >= 50) & (S2 <= 200)
v_idx2 = (v2 <= 0.20)
Sp2, vp2 = np.meshgrid(S2[S_idx2], v2[v_idx2])
Up2 = U2[np.ix_(v_idx2, S_idx2)]
c2 = axes[1].contourf(Sp2, vp2, Up2, 20, cmap='viridis')
axes[1].axvline(100.0, color='r', linestyle='--', alpha=0.6, label='Strike')
axes[1].axvline(S0, color='w', linestyle=':', alpha=0.6, label='S0')
axes[1].set_xlabel('Spot S'); axes[1].set_ylabel('Variance v')
axes[1].set_title('ATM call price at t=0,  K=100,  T=1.0  (Heston)')
axes[1].legend(loc='upper left')
plt.colorbar(c2, ax=axes[1])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/slv_pde_surfaces.png', dpi=120, bbox_inches='tight')
print("Saved /mnt/user-data/outputs/slv_pde_surfaces.png")
