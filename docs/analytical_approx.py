# lv_forward_analytic.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ----------------------------
# Parameters
# ----------------------------
S0 = 100.0
r = 0.01
q = 0.00
T = 0.5                 # horizon (years)
Nx = 501                # number of x (log-S) grid nodes
x_min = np.log(1e-2)    # left boundary in log-space
x_max = np.log(5e2)     # right boundary
dt = 1e-4               # time step; adjust as needed
plot_times = [0.0, 0.01, 0.05, 0.2, T]

# ----------------------------
# Local vol model (example)
# Replace this with interpolant or parametric form you use.
# sigma_loc(S, t) should return positive local vol
# ----------------------------
def sigma_loc(S, t):
    # simple parametric smile: ATM vol 0.2, small skew
    k = np.log(S / S0)
    vol_atm = 0.20
    skew = -0.1  # small skew
    vol = vol_atm + skew * k
    return np.maximum(0.01, vol)  # floor vol

# ----------------------------
# Build log-space grid
# ----------------------------
x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]
S = np.exp(x)

# ----------------------------
# Operators: compute coefficients on nodes and interfaces
# We'll build a tridiagonal matrix A such that dp/dt = A p
# Backward Euler: (I - dt*A) p^{n+1} = p^n
# Conservative discretization using D = 0.5 * sigma^2
# For advection term we discretize -(mu p)_x in conservative form using interface values
# ----------------------------
def build_A(x, S, t):
    N = len(x)
    mu = (r - q - 0.5 * (sigma_loc(S, t)**2))
    sigma = sigma_loc(S, t)
    D = 0.5 * sigma**2

    # interface values (i+1/2)
    D_iphalf = 0.5 * (D[:-1] + D[1:])
    mu_iphalf = 0.5 * (mu[:-1] + mu[1:])

    # tridiagonal coefficients
    a = np.zeros(N)   # lower diag (i,i-1)
    b = np.zeros(N)   # main diag (i,i)
    c = np.zeros(N)   # upper diag (i,i+1)

    # interior nodes i = 1..N-2
    for i in range(1, N-1):
        # diffusion contributions using flux differences:
        D_ip = D_iphalf[i]        # between i and i+1 (i+1/2)
        D_im = D_iphalf[i-1]      # between i-1 and i (i-1/2)

        # coefficient from diffusion:
        a_diff = D_im / (dx*dx)
        c_diff = D_ip / (dx*dx)
        b_diff = - (a_diff + c_diff)

        # advection (conservative): -(mu p)_x approximated by ( - mu_{i+1/2} p_{i+1} + mu_{i-1/2} p_{i-1} )/(2 dx)
        # Write it as linear combination: p_{i-1} * (+ mu_{i-1/2}/(2dx)) + p_i * 0 + p_{i+1} * (- mu_{i+1/2}/(2dx))
        a_adv =  mu_iphalf[i-1] / (2.0 * dx)
        c_adv = -mu_iphalf[i]    / (2.0 * dx)
        b_adv = 0.0

        a[i] = a_diff + a_adv
        c[i] = c_diff + c_adv
        b[i] = b_diff + b_adv

    # Boundaries (reflecting / zero net flux):
    # i = 0: use one-sided approx; set derivative zero: approximate with ghost p_{-1}=p_1 (reflect)
    # We'll implement a simple consistent boundary row: p0' = (stuff with p0 and p1)
    # Use forward difference for first derivative and second derivative
    # diffusion at left interface uses D_iphalf[0]
    D_ip = D_iphalf[0]
    mu_ip = mu_iphalf[0]
    # approximate operator at i=0: using one-sided differences
    a[0] = 0.0
    c[0] = (D_ip / (dx*dx)) + (-mu_ip / (2.0*dx))
    b[0] = -(D_ip / (dx*dx)) + (mu_ip / (2.0*dx))

    # i = N-1 (right)
    D_im = D_iphalf[-1]
    mu_im = mu_iphalf[-1]
    a[-1] = (D_im / (dx*dx)) + (mu_im / (2.0*dx))
    c[-1] = 0.0
    b[-1] = -(D_im / (dx*dx)) - (mu_im / (2.0*dx))

    # Build tridiagonal matrix A (in sparse tridiagonal arrays)
    # Note: A p gives dp/dt
    return a, b, c

# Helper: solve tridiagonal (Thomas algorithm)
def solve_tridiag(a, b, c, d):
    # a[0] unused (0), c[-1] unused (0)
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

# Shift density to match forward by interpolation
def shift_to_match_forward(x, p, target_forward):
    # current forward = int e^x p dx (approx trapezoid)
    ex = np.exp(x)
    cur_forward = np.trapz(ex * p, x)
    if cur_forward <= 0:
        # numeric breakdown: return p normalized
        p = np.maximum(p, 0)
        p = p / np.trapz(p, x)
        return p
    alpha = np.log(target_forward / cur_forward)
    # shift density by alpha in x: p_new(x) = p_old(x - alpha)
    x_shifted = x - alpha
    # interp: values outside domain set to 0
    p_shifted = np.interp(x, x_shifted, p, left=0.0, right=0.0)
    # floor negative and renormalize
    p_shifted = np.maximum(p_shifted, 0.0)
    mass = np.trapz(p_shifted, x)
    if mass <= 0:
        # fallback to small gaussian
        p_shifted = np.exp(-0.5 * ((x - np.log(S0)) / (0.1))**2)
        mass = np.trapz(p_shifted, x)
    p_shifted /= mass
    return p_shifted

# Analytic short-time initialization at t0
def analytic_init(x, S0, t0):
    x0 = np.log(S0)
    mu0 = (r - q - 0.5 * sigma_loc(S0, 0.0)**2)
    sigma0 = sigma_loc(S0, 0.0)
    mean = x0 + mu0 * t0
    var = (sigma0**2) * t0
    p = np.exp(-0.5 * (x - mean)**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapz(p, x)
    return p

# ----------------------------
# Time march
# ----------------------------
def run_forward(analytic=True):
    max_steps = int(np.ceil(T / dt))
    # choose t0 based on dx and local vol
    sigma0 = sigma_loc(S0, 0.0)
    t0 = min( max(1e-12, 0.5 * dx*dx / max(1e-8, sigma0**2)), 1e-2 )  # clamp t0
    if analytic:
        p = analytic_init(x, S0, t0)
        t = t0
        print(f"Analytic init at t0 = {t0:.3e}")
    else:
        # not used in this script; mollifier script will use other init
        raise RuntimeError("This routine expects analytic=True for this file.")

    times = [t]
    saved = [(t, p.copy())]

    step = 0
    while t < T - 1e-15 and step < 10_000_000:
        # evaluate A at current time (we implicitize operator at n+1 using current t for coefficients)
        a, b, c = build_A(x, S, t)  # coefficients depend on t via sigma_loc
        N = len(x)
        # Build system (I - dt * A) p^{n+1} = p^n
        # matrix diag: diag = 1 - dt * b, lower = - dt * a, upper = - dt * c
        diag = 1.0 - dt * b
        low  = -dt * a
        up   = -dt * c

        # Build RHS = p^n
        rhs = p.copy()

        # Solve tridiagonal (we must supply a, b, c arrays matching solver convention)
        # Our solver uses arrays where a[0]=0, c[-1]=0
        # Put tridiag arrays in place: a_tr = low (with a_tr[0]=0), b_tr = diag, c_tr = up
        a_tr = low.copy()
        a_tr[0] = 0.0
        c_tr = up.copy()
        c_tr[-1] = 0.0

        p_new = solve_tridiag(a_tr, diag, c_tr, rhs)

        # floor small negatives
        p_new = np.maximum(p_new, 0.0)

        # renormalize mass
        mass = np.trapz(p_new, x)
        if mass <= 0:
            raise RuntimeError("Mass vanished numerically.")
        p_new /= mass

        t += dt
        step += 1

        # match forward exactly by shifting density
        target_forward = S0 * np.exp((r - q) * t)
        p_new = shift_to_match_forward(x, p_new, target_forward)

        p = p_new

        # save if close to plot times
        if any(abs(t - pt) < 0.5*dt for pt in plot_times):
            saved.append((t, p.copy()))
            times.append(t)

    print("Done steps:", step, "final t:", t)
    return saved

if __name__ == "__main__":
    saved = run_forward(analytic=True)
    # Plot results
    plt.figure(figsize=(8,5))
    for t, p in saved:
        plt.plot(np.exp(x), p, label=f"t={t:.4f}")
    plt.xscale('log')
    plt.xlabel('S')
    plt.ylabel('density p_S(s)')
    plt.title('Forward density (analytic init)')
    plt.legend()
    plt.grid(True)
    plt.show()
