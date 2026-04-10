"""
Replace as needed
"""
#-------------------------------------------------------------------
# SPLIT UP EXPONENTIAL FIT TO FIND FPLUS AND CAPTURE RATE
#-------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

tau0 = 275  # bins
binsize = 8.0  # ns
data2 = DFS1cut6['delta_t_b3a1'].dropna().values

bins = 200
trange = (-500, 1200)
bin_edges = np.linspace(trange[0], trange[1], bins+1)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
y2, _ = np.histogram(data2, bins=bin_edges)
err2 = np.sqrt(np.maximum(y2, 1))

# Fit Late-Time Tail
# Let's define a late-time region where μ⁻ capture is essentially gone
tail_start = 700   # bins (pick a time where tail is mostly μ⁺, adjust as needed)
tail_mask = bin_centers >= tail_start

t_tail = bin_centers[tail_mask]
y_tail = y2[tail_mask]
err_tail = err2[tail_mask]

def simple_exp(t, A, C):
    return A * np.exp(-1.0/tau0 * t) + C

p0 = [max(y_tail), min(y_tail)/2]
bounds = ([0, 0], [np.inf, np.inf])

popt, pcov = curve_fit(simple_exp, t_tail, y_tail, p0=p0, bounds=bounds, sigma=err_tail, absolute_sigma=True)
A_tail, C_tail = popt

# Estimate fplus
# Estimate total amplitude at t=0 (first bin)
first_bin_mask = (bin_centers >= 80) & (bin_centers < 200)
A_total = np.sum(y2[first_bin_mask]) 

fplus_est = A_tail / A_total  # approximate, you may refine this

print("\n=== SIMPLE EXPONENTIAL TAIL FIT ===")
print(f"Tail amplitude (μ+ only)      = {A_tail:.1f}")
print(f"Estimated μ+ fraction fplus   = {fplus_est:.3f}")
print(f"Background (C)                = {C_tail:.1f}")

# Fit Early-Time Capture Component
# Subtract μ+ tail fit from histogram (for early times)
early_mask = (bin_centers >= 80) & (bin_centers < tail_start)
t_early = bin_centers[early_mask]
y_early = y2[early_mask] - simple_exp(t_early, A_tail, C_tail)  # Remove μ+ component
y_early = np.maximum(y_early, 1)  # Avoid negative counts

def capture_exp(t, A_c, tau_cO):
    return A_c * np.exp(-(1.0/tau0 + 1.0/tau_cO) * t)

# Initial guess [amplitude, tau_cO]
p0_c = [max(y_early), 450]
bounds_c = ([0, 1], [np.inf, 1e5])

# Fit for tau_cO
popt_c, _ = curve_fit(capture_exp, t_early, y_early, p0=p0_c, bounds=bounds_c)
A_c_fit, tau_cO_fit = popt_c

print(f"Oxygen capture time tau_cO    = {tau_cO_fit:.0f} bins = {tau_cO_fit*binsize:.1f} ns")
print(f"Oxygen capture rate lambda_O  = {1/(tau_cO_fit*binsize):.5f} 1/ns")

# Plotting
plt.figure(figsize=(6,4), dpi=150)
plt.step(bin_centers, y2, where='mid', label='Data')
plt.plot(t_tail, simple_exp(t_tail, A_tail, C_tail), 'r-', label='μ+ tail fit')
plt.plot(t_early, capture_exp(t_early, A_c_fit, tau_cO_fit) + simple_exp(t_early, A_tail, C_tail), 'g--',
label='μ+ + μ− exp')
plt.xlabel("Time (bins × 8ns)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# COUPLED DOUBLE EXPONENTIAL FITS TO FIND FPLUS AND CAPTURE RATES
# ---------------------------------------------------------

from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Double exponential model for one dataset
# ---------------------------------------------------------
def double_exp(t, A, tau_c, C):
    return A * (
        0.53 * np.exp(-1.0 / 275 * t) +
        (1 - 0.53) * np.exp(-(1.0 / 275 + 1.0 / tau_c) * t)
    ) + C

# ---------------------------------------------------------
# Combined residual function for two datasets
# ---------------------------------------------------------
def combined_model(params, t1, y1, err1, t2, y2, err2):
    A1, A2, tau_cC, tau_cO, C1, C2 = params

    m1 = double_exp(t1, A1, tau_cC, C1)
    m2 = double_exp(t2, A2, tau_cO, C2)

    r1 = (y1 - m1) / err1
    r2 = (y2 - m2) / err2
    return np.concatenate([r1, r2])

data1 = DFS2cut6['delta_t_b3a1'].dropna().values     # Carbon
data2 = DFS1cut6['delta_t_b3a1'].dropna().values     # Oxygen

bins = 200
trange = (-500, 2500)
bin_edges = np.linspace(trange[0], trange[1], bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
binsize = 8.0  # ns per bin

# Histograms
y1, _ = np.histogram(data1, bins=bin_edges)
y2, _ = np.histogram(data2, bins=bin_edges)
err1 = np.sqrt(np.maximum(y1, 1))
err2 = np.sqrt(np.maximum(y2, 1))

# ---------------------------------------------------------
# Fit region
# ---------------------------------------------------------
fit_start, fit_end = 80, 2500

mask = (bin_centers >= fit_start) & (bin_centers <= fit_end)
t1 = bin_centers[mask]
t2 = bin_centers[mask]
y1_fit = y1[mask]
y2_fit = y2[mask]
err1_fit = err1[mask]
err2_fit = err2[mask]

# ---------------------------------------------------------
# Initial guesses
# ---------------------------------------------------------
p0 = [
    max(y1_fit),         # A1
    max(y2_fit),         # A2
#    275,                 # tau0 initial guess in bins
#    0.54,                # f+ realistic initial guess
    3125,                # tau_cC initial guess in bins
    450,                 # tau_cO initial guess in bins
    min(y1_fit) / 2,     # background C1
    min(y2_fit) / 2      # background C2
]

lower = [
    0,       # A1
    0,       # A2
#    10,      # tau0 lower bound in bins
#    0.3,     # fplus min realistic
    1,       # tau_cC lower bound
    1,       # tau_cO lower bound
    0,       # C1
    0        # C2
]
upper = [
    np.inf,  # A1
    np.inf,  # A2
#    5e3,     # tau0 upper bound in bins
#    0.8,     # fplus max realistic
    1e5,     # tau_cC 
    1e5,     # tau_cO
    np.inf,  # C1
    np.inf   # C2
]


# ---------------------------------------------------------
# Combined fit for both datasets
# ---------------------------------------------------------
result = least_squares(
    combined_model,
    p0,
    bounds=(lower, upper),
    args=(t1, y1_fit, err1_fit, t2, y2_fit, err2_fit),
    max_nfev=50000
)

A1, A2, tau_cC, tau_cO, C1, C2 = result.x

print("\n=== COMBINED FIT RESULTS ===")
#print(f"Common free μ lifetime τ0 = {tau0 * binsize:.1f} ns = {tau0 * binsize / 1000:.3f} μs")
#print(f"Common μ+ fraction f+ = {fplus:.3f}")
print(f"Carbon capture time τ_C = {tau_cC * binsize:.1f} ns; λ_C = {1 / (tau_cC * binsize):.5f} 1/ns")
print(f"Oxygen capture time τ_O = {tau_cO * binsize:.1f} ns; λ_O = {1 / (tau_cO * binsize):.5f} 1/ns")
print()

# ---------------------------------------------------------
# Plot: Carbon data
# ---------------------------------------------------------
plt.figure(figsize=(6, 4), dpi=150)
plt.step(bin_centers, y1, where='mid', label='Scintillator (Carbon)')
plt.plot(t1, double_exp(t1, A1, tau_cC, C1), 'r-', lw=2)
plt.xlabel("Time (bins × 8ns)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Plot: Oxygen data
# ---------------------------------------------------------
plt.figure(figsize=(6, 4), dpi=150)
plt.step(bin_centers, y2, where='mid', label='Water (Oxygen)')
plt.plot(t2, double_exp(t2, A2, tau_cO, C2), 'r-', lw=2)
plt.xlabel("Time (bins × 8ns)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.show()