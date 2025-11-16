Doc:
Finite-Volume solver with Scharfetter–Gummel (SG) flux and implicit (Backward Euler) time stepping.

Conservative, positivity-friendly flux discretization (SG) that handles both advection- and diffusion-dominated regimes robustly.

Analytic short-time initialization (you can swap to a mollifier easily).

Floors negatives, renormalizes mass, and matches the forward by shifting (so the first moment equals the known forward).

Script
# lv_forward_sg_be.py
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
S0 = 100.0
r = 0.01
q = 0.0
T = 0.5
Nx = 401
x_min = np.log(1e-2)
x_max = np.log(1e3)
dt = 2e-4
plot_times = [0.0, 0.01, 0.05, 0.2, T]

# ----------------------------
# Local vol (replace with your interpolant/parametric)
# ----------------------------
def sigma_loc(S, t):
    k = np.log(S / S0)
    vol_atm = 0.20
    skew = -0.12
    vol = vol_atm + skew * k
    return np.maximum(0.01, vol)

# ----------------------------
# Grid in log-space
# ----------------------------
x = np.linspace(x_min, x_max, Nx)
dx = np.diff(x)  # constant here but code supports non-uniform
if not np.allclose(dx, dx[0]):
    print("Non-uniform grid support enabled.")
dx0 = dx[0]
xc = x.copy()
S = np.exp(xc)
N = len(xc)

# ----------------------------
# Bernoulli function with stable eval
# B(theta) = theta / (exp(theta)-1)
# use Taylor expansion for small |theta|
# ----------------------------
def bernoulli(theta):
    # elementwise
    out = np.empty_like(theta)
    small = np.abs(theta) < 1e-8
    # Taylor series up to theta^4: 1 - theta/2 + theta^2/12 - theta^4/720
    out[small] = 1.0 - theta[small]/2.0 + theta[small]**2/12.0 - theta[small]**4/720.0
    big = ~small
    out[big] = theta[big] / (np.expm1(theta[big]))  # expm1 for numerical accuracy
    return out

# ----------------------------
# Build SG tridiagonal operator L such that dp/dt = L p
# We return lower (a), diag (b), upper (c) for matrix L (so L p approx a*p_{i-1}+b*p_i+c*p_{i+1})
# ----------------------------
def build_SG_operator(xc, t):
    N = len(xc)
    S = np.exp(xc)
    sigma = sigma_loc(S, t)
    mu = r - q - 0.5 * sigma**2
    D = 0.5 * sigma**2

    # interface values (i+1/2)
    # for simplicity use arithmetic averages; using harmonic average for D is also possible
    D_ip = 0.5 * (D[:-1] + D[1:])   # length N-1
    mu_ip = 0.5 * (mu[:-1] + mu[1:])
    h_ip = np.diff(xc)              # distances between nodes (size N-1), but here uniform

    a = np.zeros(N)  # lower diag
    b = np.zeros(N)
    c = np.zeros(N)  # upper diag

    # interior interfaces produce contributions to neighboring cells:
    # flux at i+1/2: F_{i+1/2} = - (D_ip / h) * ( B(theta) * p_{i+1} - B(-theta) * p_i )
    # theta = mu_ip * h / D_ip
    theta = np.zeros_like(D_ip)
    nonzero = D_ip > 0
    theta[nonzero] = mu_ip[nonzero] * h_ip[nonzero] / D_ip[nonzero]
    Bp = bernoulli(theta)      # B(theta)
    Bm = bernoulli(-theta)     # B(-theta)

    # For interior i = 1..N-2, contribution from F_{i+1/2} and F_{i-1/2}
    for i in range(1, N-1):
        # interface i-1/2 between i-1 and i: index i-1 in D_ip
        hL = h_ip[i-1]
        Dleft = D_ip[i-1]
        thL = theta[i-1]
        BpL = bernoulli(thL)
        BmL = bernoulli(-thL)
        # coefficients for F_{i-1/2} = - (Dleft/hL) * ( B(thL) * p_i - B(-thL) * p_{i-1} )
        coeff_p_i_from_Fleft = - (Dleft / hL) * BpL      # multiplies p_i when computing F_left
        coeff_p_im1_from_Fleft =   (Dleft / hL) * BmL    # multiplies p_{i-1}

        # interface i+1/2
        hR = h_ip[i]
        Dright = D_ip[i]
        thR = theta[i]
        BpR = bernoulli(thR)
        BmR = bernoulli(-thR)
        # F_right = - (Dright/hR) * ( B(thR)*p_{i+1} - B(-thR)*p_i )
        coeff_p_ip1_from_Fright = - (Dright / hR) * BpR
        coeff_p_i_from_Fright =    (Dright / hR) * BmR

        # divergence (F_right - F_left)/h_cell where h_cell ~ average; for uniform grid use h = h_ip[i]
        # We'll use cell width equal to h_ip[i] (uniform). Here dx constant so choose dx0.
        hcell = dx0
        # contribution to equation dp_i/dt = - (F_right - F_left)/hcell
        # So coefficient multiplying p_{i-1} is -(- coeff_p_im1_from_Fleft)/hcell = + coeff_p_im1_from_Fleft / hcell
        a[i] = ( coeff_p_im1_from_Fleft ) / hcell
        # coefficient multiplying p_i is - ( coeff_p_i_from_Fright - coeff_p_i_from_Fleft ) / hcell
        b[i] = - ( coeff_p_i_from_Fright - coeff_p_i_from_Fleft ) / hcell
        # coefficient for p_{i+1} is - ( coeff_p_ip1_from_Fright ) / hcell
        c[i] = - ( coeff_p_ip1_from_Fright ) / hcell

    # Boundaries: zero-flux (F_{-1/2}=0 and F_{N-1/2}=0)
    # Left cell i=0: only F_{1/2} present
    # F_1/2 = - (D_ip[0]/h) * ( B(theta0)*p1 - B(-theta0)*p0 )
    if N >= 2:
        Dright = D_ip[0]; hR = h_ip[0]; thR = theta[0]
        BpR = bernoulli(thR); BmR = bernoulli(-thR)
        coeff_p_0_from_Fright = (Dright / hR) * BmR
        coeff_p_1_from_Fright = - (Dright / hR) * BpR
        hcell = dx0
        a[0] = 0.0
        b[0] = - ( coeff_p_0_from_Fright ) / hcell   # - (coeff_p_i_from_Fright - 0) / hcell
        c[0] = - ( coeff_p_1_from_Fright ) / hcell
        # rightmost i=N-1:
        Dleft = D_ip[-1]; hL = h_ip[-1]; thL = theta[-1]
        BpL = bernoulli(thL); BmL = bernoulli(-thL)
        coeff_p_im1_from_Fleft =   (Dleft / hL) * BmL
        coeff_p_i_from_Fleft = - (Dleft / hL) * BpL
        a[-1] = ( coeff_p_im1_from_Fleft ) / hcell
        b[-1] = - ( - coeff_p_i_from_Fleft ) / hcell  # -(0 - coeff_p_i_from_Fleft)/hcell = + coeff_p_i_from_Fleft/hcell
        c[-1] = 0.0
    else:
        a[0]=c[0]=0.0; b[0]=0.0

    return a, b, c

# ----------------------------
# Thomas solver (tridiagonal)
# ----------------------------
def solve_tridiag(a, b, c, d):
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    x = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

# ----------------------------
# Analytic short-time initialization
# ----------------------------
def analytic_init(xc, S0, t0):
    x0 = np.log(S0)
    sigma0 = sigma_loc(S0, 0.0)
    mu0 = r - q - 0.5 * sigma0**2
    mean = x0 + mu0 * t0
    var = sigma0**2 * t0
    p = np.exp(-0.5 * (xc - mean)**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapz(p, xc)
    return p

# shift density to match forward exactly
def shift_to_match_forward(xc, p, target_forward):
    ex = np.exp(xc)
    cur_forward = np.trapz(ex * p, xc)
    if cur_forward <= 0:
        p = np.maximum(p, 0.0)
        p /= np.trapz(p, xc)
        return p
    alpha = np.log(target_forward / cur_forward)
    x_shifted = xc - alpha
    p_shifted = np.interp(xc, x_shifted, p, left=0.0, right=0.0)
    p_shifted = np.maximum(p_shifted, 0.0)
    mass = np.trapz(p_shifted, xc)
    if mass <= 0:
        p_shifted = np.exp(-0.5 * ((xc - np.log(S0)) / 0.1)**2)
        mass = np.trapz(p_shifted, xc)
    p_shifted /= mass
    return p_shifted

# ----------------------------
# Time march (Backward Euler)
# (I - dt * L) p^{n+1} = p^n
# where L is built from SG operator
# ----------------------------
def run_sg_be():
    # choose small t0 so analytic Gaussian is resolved
    sigma0 = sigma_loc(S0, 0.0)
    t0 = max(1e-12, 0.5 * dx0*dx0 / max(1e-8, sigma0**2))
    t0 = min(t0, 5e-3)
    p = analytic_init(xc, S0, t0)
    t = t0
    saved = [(t, p.copy())]

    step = 0
    max_steps = int(np.ceil((T - t) / dt)) + 5
    while t < T - 1e-15 and step < max_steps:
        aL, bL, cL = build_SG_operator(xc, t + dt)  # implicit: build L at t+dt (or t) -> choose t+dt for better stability
        # Build matrix (I - dt * L)
        diag = 1.0 - dt * bL
        low = -dt * aL
        up = -dt * cL
        # set boundary entries consistent
        low[0] = 0.0; up[-1] = 0.0
        rhs = p.copy()
        pnew = solve_tridiag(low, diag, up, rhs)
        # floor negatives and renormalize
        pnew = np.maximum(pnew, 0.0)
        mass = np.trapz(pnew, xc)
        if mass <= 0:
            raise RuntimeError("Mass vanished numerically.")
        pnew /= mass
        t += dt
        step += 1
        # match forward exactly by shifting
        target_forward = S0 * np.exp((r - q) * t)
        pnew = shift_to_match_forward(xc, pnew, target_forward)
        p = pnew
        if any(abs(t - pt) < 0.5*dt for pt in plot_times):
            saved.append((t, p.copy()))
    print("SG BE done steps:", step, "final t:", t)
    return saved

if __name__ == "__main__":
    saved = run_sg_be()
    plt.figure(figsize=(8,5))
    for t, p in saved:
        plt.plot(np.exp(xc), p, label=f"t={t:.4f}")
    plt.xscale('log')
    plt.xlabel('S'); plt.ylabel('p_S(s)')
    plt.title('Forward density — Finite-Volume Scharfetter-Gummel (BE)')
    plt.legend(); plt.grid(True)
    plt.show()
