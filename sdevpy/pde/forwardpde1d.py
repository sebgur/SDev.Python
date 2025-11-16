import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

################# ToDo #############################################################
# * Work out the coefficients in equations, check that build_A and the rest
#   produce the right coefficients assuming implicit scheme.
# * Separate scheme into class for later introduction of others.
# * Test the tridiag solver and move it somewhere else. Check around if there are
#   more efficient/standard algorithms in numpy or scipy or something. There seems
#   to be one in scipy.linalg in import solve_banded. Highly optimized.
# * Check what trapez() method does, see if there's a more up to date version
# * If it does the integration, which it seems, then extract the last probability
#   density and integrate to calculate a payoff. Start with forward, then options,
#   and compare against Black-Scholes.
# * Identify the mollifier code, wrap it in a function.
# * Might want to implement the theta scheme as overarching Implicit, Explicit and CN.
# * Wrap up before moving to analytical early approximation, Crank-Nicolson,
#   adaptative meshes (near the singularity), non-homogeneous time grid and other tricks.
####################################################################################
# Parameters
S0 = 100.0
r = 0.01
q = 0.00
T = 0.5
Nx = 501
x_min = np.log(1e-2)
x_max = np.log(5e2)
dt = 1e-4
plot_times = [0.0, 0.01, 0.05, 0.2, T]

def sigma_loc(S, t):
    k = np.log(S / S0)
    vol_atm = 0.20
    skew = -0.1
    vol = vol_atm + skew * k
    return np.maximum(0.01, vol)

x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]
S = np.exp(x)

# What are x and S? Are they vectors or scalars?
# According to Quant GPT, this is the implicit scheme.
def build_A(x, S, t):
    N = len(x)
    mu = (r - q - 0.5 * (sigma_loc(S, t)**2))
    sigma = sigma_loc(S, t)
    D = 0.5 * sigma**2

    D_iphalf = 0.5 * (D[:-1] + D[1:])
    mu_iphalf = 0.5 * (mu[:-1] + mu[1:])

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)

    for i in range(1, N-1):
        D_ip = D_iphalf[i]
        D_im = D_iphalf[i-1]
        a_diff = D_im / (dx*dx)
        c_diff = D_ip / (dx*dx)
        b_diff = - (a_diff + c_diff)
        a_adv = mu_iphalf[i-1] / (2.0 * dx)
        c_adv = -mu_iphalf[i] / (2.0 * dx)
        a[i] = a_diff + a_adv
        c[i] = c_diff + c_adv
        b[i] = b_diff

    D_ip = D_iphalf[0]
    mu_ip = mu_iphalf[0]
    a[0] = 0.0
    c[0] = (D_ip / (dx*dx)) + (-mu_ip / (2.0*dx))
    b[0] = -(D_ip / (dx*dx)) + (mu_ip / (2.0*dx))

    D_im = D_iphalf[-1]
    mu_im = mu_iphalf[-1]
    a[-1] = (D_im / (dx*dx)) + (mu_im / (2.0*dx))
    c[-1] = 0.0
    b[-1] = -(D_im / (dx*dx)) - (mu_im / (2.0*dx))

    return a, b, c

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

def shift_to_match_forward(x, p, target_forward):
    ex = np.exp(x)
    cur_forward = np.trapz(ex * p, x)
    if cur_forward <= 0:
        p = np.maximum(p, 0)
        p = p / np.trapz(p, x)
        return p
    alpha = np.log(target_forward / cur_forward)
    x_shifted = x - alpha
    p_shifted = np.interp(x, x_shifted, p, left=0.0, right=0.0)
    p_shifted = np.maximum(p_shifted, 0.0)
    mass = np.trapz(p_shifted, x)
    if mass <= 0:
        p_shifted = np.exp(-0.5 * ((x - np.log(S0)) / (0.1))**2)
        mass = np.trapz(p_shifted, x)
    p_shifted /= mass
    return p_shifted

# Mollifier initialization: gaussian with variance ~ (k * dx)^2
def mollifier_init(x, S0, k=1.5):
    x0 = np.log(S0)
    eps = (k * dx)**2
    p = np.exp(-0.5 * (x - x0)**2 / eps) / np.sqrt(2.0 * np.pi * eps)
    p /= np.trapz(p, x)
    return p

def run_forward_mollifier():
    p = mollifier_init(x, S0, k=1.5)
    t = 0.0
    saved = [(t, p.copy())]

    # Roll-back
    step = 0
    while t < T - 1e-15 and step < 10_000_000:
        a, b, c = build_A(x, S, t)
        N = len(x)
        diag = 1.0 - dt * b
        low  = -dt * a
        up   = -dt * c
        rhs = p.copy()
        a_tr = low.copy(); a_tr[0] = 0.0
        c_tr = up.copy();  c_tr[-1] = 0.0
        p_new = solve_tridiag(a_tr, diag, c_tr, rhs)
        p_new = np.maximum(p_new, 0.0)
        mass = np.trapz(p_new, x)
        # print(mass)
        if mass <= 0:
            raise RuntimeError("Mass vanished numerically.")
        p_new /= mass
        t += dt
        step += 1
        target_forward = S0 * np.exp((r - q) * t)
        p_new = shift_to_match_forward(x, p_new, target_forward)
        # ToDo: but, does it still sum to 1 after shifting to match the forward?
        p = p_new
        if any(abs(t - pt) < 0.5*dt for pt in plot_times):
            saved.append((t, p.copy()))
    print("Done steps:", step, "final t:", t)
    return saved

if __name__ == "__main__":
    saved = run_forward_mollifier()

    # plt.figure(figsize=(8,5))
    # for t, p in saved:
    #     plt.plot(np.exp(x), p, label=f"t={t:.4f}")
    # plt.xscale('log')
    # plt.xlabel('S')
    # plt.ylabel('density p_S(s)')
    # plt.title('Forward density (mollifier init)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
