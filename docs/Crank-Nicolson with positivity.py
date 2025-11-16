Doc:
Crank–Nicolson (CN) variant using the same SG spatial operator with a simple, practical positivity limiter applied after each CN step.

CN gives second-order time accuracy; the limiter clips negatives and rescales positives to preserve mass (a pragmatic limiter — if strong negative undershoots occur, reduce Δt or use an entropy/maximum-principle limiting scheme).

Script:
# lv_forward_sg_cn_limiter.py
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
dt = 5e-5   # smaller dt for CN to reduce undershoots
plot_times = [0.0, 0.01, 0.05, 0.2, T]

def sigma_loc(S, t):
    k = np.log(S / S0)
    vol_atm = 0.20
    skew = -0.12
    vol = vol_atm + skew * k
    return np.maximum(0.01, vol)

# Grid
x = np.linspace(x_min, x_max, Nx)
dx = np.diff(x)
dx0 = dx[0]
xc = x.copy()
S = np.exp(xc)
N = len(xc)

# Bernoulli
def bernoulli(theta):
    out = np.empty_like(theta)
    small = np.abs(theta) < 1e-8
    out[small] = 1.0 - theta[small]/2.0 + theta[small]**2/12.0 - theta[small]**4/720.0
    big = ~small
    out[big] = theta[big] / (np.expm1(theta[big]))
    return out

# Build SG operator L (same as BE script)
def build_SG_operator(xc, t):
    N = len(xc)
    S = np.exp(xc)
    sigma = sigma_loc(S, t)
    mu = r - q - 0.5 * sigma**2
    D = 0.5 * sigma**2
    D_ip = 0.5 * (D[:-1] + D[1:])
    mu_ip = 0.5 * (mu[:-1] + mu[1:])
    h_ip = np.diff(xc)
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
    theta = np.zeros_like(D_ip)
    nonzero = D_ip > 0
    theta[nonzero] = mu_ip[nonzero] * h_ip[nonzero] / D_ip[nonzero]

    for i in range(1, N-1):
        hL = h_ip[i-1]; Dleft = D_ip[i-1]; thL = theta[i-1]; BpL = bernoulli(thL); BmL = bernoulli(-thL)
        hR = h_ip[i];   Dright = D_ip[i];  thR = theta[i];   BpR = bernoulli(thR); BmR = bernoulli(-thR)
        coeff_p_im1_from_Fleft =   (Dleft / hL) * BmL
        coeff_p_i_from_Fleft    = - (Dleft / hL) * BpL
        coeff_p_i_from_Fright   =   (Dright / hR) * BmR
        coeff_p_ip1_from_Fright = - (Dright / hR) * BpR
        hcell = dx0
        a[i] = ( coeff_p_im1_from_Fleft ) / hcell
        b[i] = - ( coeff_p_i_from_Fright - coeff_p_i_from_Fleft ) / hcell
        c[i] = - ( coeff_p_ip1_from_Fright ) / hcell

    # boundaries: zero-flux
    if N >= 2:
        Dright = D_ip[0]; hR = h_ip[0]; thR = theta[0]; BpR = bernoulli(thR); BmR = bernoulli(-thR)
        coeff_p_0_from_Fright = (Dright / hR) * BmR
        coeff_p_1_from_Fright = - (Dright / hR) * BpR
        hcell = dx0
        a[0] = 0.0
        b[0] = - ( coeff_p_0_from_Fright ) / hcell
        c[0] = - ( coeff_p_1_from_Fright ) / hcell

        Dleft = D_ip[-1]; hL = h_ip[-1]; thL = theta[-1]; BpL = bernoulli(thL); BmL = bernoulli(-thL)
        coeff_p_im1_from_Fleft =   (Dleft / hL) * BmL
        coeff_p_i_from_Fleft = - (Dleft / hL) * BpL
        a[-1] = ( coeff_p_im1_from_Fleft ) / hcell
        b[-1] = - ( - coeff_p_i_from_Fleft ) / hcell
        c[-1] = 0.0
    else:
        a[0]=b[0]=c[0]=0.0
    return a, b, c

# Tridiagonal solver
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

# analytic init
def analytic_init(xc, S0, t0):
    x0 = np.log(S0)
    sigma0 = sigma_loc(S0, 0.0)
    mu0 = r - q - 0.5 * sigma0**2
    mean = x0 + mu0 * t0
    var = sigma0**2 * t0
    p = np.exp(-0.5 * (xc - mean)**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapz(p, xc)
    return p

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

# Positivity limiter: simple clipping + redistribution
def positivity_limiter(p):
    # clip negatives
    neg_mask = p < 0
    if not np.any(neg_mask):
        return p
    p_clipped = np.maximum(p, 0.0)
    mass = np.trapz(p_clipped, xc)
    if mass <= 0:
        # fallback tiny gaussian
        p_clipped = np.exp(-0.5 * ((xc - np.log(S0)) / 0.1)**2)
        p_clipped /= np.trapz(p_clipped, xc)
        return p_clipped
    # rescale positive part to have mass 1
    p_clipped /= mass
    return p_clipped

# ----------------------------
# Time march (Crank-Nicolson)
# (I - 0.5 dt L) p^{n+1} = (I + 0.5 dt L) p^n
# ----------------------------
def run_sg_cn_with_limiter():
    sigma0 = sigma_loc(S0, 0.0)
    t0 = max(1e-12, 0.5 * dx0*dx0 / max(1e-8, sigma0**2))
    t0 = min(t0, 5e-3)
    p = analytic_init(xc, S0, t0)
    t = t0
    saved = [(t, p.copy())]
    step = 0
    max_steps = int(np.ceil((T - t) / dt)) + 5
    while t < T - 1e-15 and step < max_steps:
        # build L at time t (or (t+t+dt)/2); choose t for simplicity
        aL, bL, cL = build_SG_operator(xc, t)
        # left matrix: M_left = I - 0.5 dt L
        diagL = 1.0 - 0.5 * dt * bL
        lowL = -0.5 * dt * aL
        upL = -0.5 * dt * cL
        lowL[0] = 0.0; upL[-1] = 0.0

        # right-hand side: rhs = (I + 0.5 dt L) p^n
        rhs = (1.0 + 0.5 * dt * bL) * p + 0.5 * dt * (aL * np.roll(p, 1) + cL * np.roll(p, -1))
        # adjust boundaries for roll artifacts
        rhs[0] = (1.0 + 0.5 * dt * bL[0]) * p[0] + 0.5 * dt * (cL[0] * p[1])
        rhs[-1] = (1.0 + 0.5 * dt * bL[-1]) * p[-1] + 0.5 * dt * (aL[-1] * p[-2])

        # solve tridiagonal for p_new: lowL, diagL, upL
        p_new = solve_tridiag(lowL, diagL, upL, rhs)

        # positivity limiter: clip negatives and renormalize
        p_new = positivity_limiter(p_new)

        # enforce mass =1
        mass = np.trapz(p_new, xc)
        p_new /= mass

        # optional: match forward exactly by shifting
        t += dt
        step += 1
        target_forward = S0 * np.exp((r - q) * t)
        p_new = shift_to_match_forward(xc, p_new, target_forward)

        p = p_new
        if any(abs(t - pt) < 0.5*dt for pt in plot_times):
            saved.append((t, p.copy()))
    print("SG CN done steps:", step, "final t:", t)
    return saved

if __name__ == "__main__":
    saved = run_sg_cn_with_limiter()
    plt.figure(figsize=(8,5))
    for t, p in saved:
        plt.plot(np.exp(xc), p, label=f"t={t:.4f}")
    plt.xscale('log')
    plt.xlabel('S'); plt.ylabel('p_S(s)')
    plt.title('Forward density — SG spatial operator + Crank-Nicolson + Positivity limiter')
    plt.legend(); plt.grid(True)
    plt.show()
