"""
Replace in Initial Lifetime Results in Carbon.py with the following snippets
"""


#Cut5.  Require that the pulse finder finds a good second pulse on B.

#T Cuts
DFTcut5 = DFTcut3.loc[(DFTcut3['flagb3'] != 99)]
#S Cuts
DFScut5 = DFScut3.loc[(DFScut3['flagb3'] == 1)]
#U Cuts
DFUcut5 = DFUcut3.loc[(DFUcut3['flagb3'] != 99)]

###

LowerLimit = 0
HigherLimit = 1200

DFScut5 = DFScut5.loc[(DFScut5['delta_t_b3a1']>LowerLimit) & (DFScut5['delta_t_b3a1']<HigherLimit)]
DFTcut5 = DFTcut5.loc[(DFTcut5['delta_t_b3a1']>HigherLimit)]


###

print("Number of events before Cut5 (T):",len(DFTcut3))
print("Number of events before Cut5 (S):",len(DFScut3))
print("Number of events before Cut5 (U):",len(DFUcut3))
print("\n")
print("Number of events after Cut5 (T):",len(DFTcut5))
print("Number of events after Cut5 (S):",len(DFScut5))
print("Number of events after Cut5 (U):",len(DFUcut5))

DFTcut5.head()

# Set very low (i.e. almost as if there is no cut)
PulseHeightThreshold = -8000


DFTcut6 = DFTcut5.loc[(DFTcut5['peakb3'] < PulseHeightThreshold)]     
DFScut6 = DFScut5.loc[(DFScut5['peakb3'] < PulseHeightThreshold)]
DFUcut6 = DFUcut5.loc[(DFUcut5['peakb3'] < PulseHeightThreshold)]

plt.figure(figsize=(5,3), dpi=200)

DFTcut5['peakb3'].plot.hist(bins=100, alpha=0.5, label='peakb3 before cut', range=[-40000.0,0.0], logy=True)
DFTcut6['peakb3'].plot.hist(bins=100, alpha=0.5, label='peakb3 after cut', range=[-40000.0,0.0], logy=True)

plt.xlabel('T Peak Height')
plt.ylabel('No. of events')
plt.legend()

print("Number of events before Cut6 (T):",len(DFTcut5))
print("Number of events before Cut6 (S):",len(DFScut5))
print("Number of events before Cut6 (U):",len(DFUcut5))
print("\n")
print("Number of events after Cut6 (T):",len(DFTcut6))
print("Number of events after Cut6 (S):",len(DFScut6))
print("Number of events after Cut6 (U):",len(DFUcut6))

DFTcut6.head()

# Method 1

from scipy.optimize import curve_fit

def exp_plus_bg(t, A, tau, C):
    """Exponential decay + constant background."""
    return A * np.exp(-t / tau) + C

# Data after your Cut6 selection:
data = DFScut6['delta_t_b3a1'].dropna().values
bins = 200
trange = (-500, 2500)
bin_edges = np.linspace(trange[0], trange[1], bins+1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

counts, _ = np.histogram(data, bins=bin_edges)
binsize = 8.0  # ns per bin

# Select fit region
fit_start, fit_end = 80, 2500  # adjust to your best signal region!
fit_mask = (bin_centers >= fit_start) & (bin_centers <= fit_end) & (counts > 0)
x_fit = bin_centers[fit_mask]
y_fit = counts[fit_mask]
y_err = np.sqrt(np.maximum(y_fit, 1.0))

# Initial guess: amplitude, tau, background
p0 = [y_fit.max(), 500.0, y_fit.min()]
bounds = ([0, 5.0, 0.0], [np.inf, 1e6, y_fit.max()])

popt, pcov = curve_fit(exp_plus_bg, x_fit, y_fit, p0=p0, sigma=y_err, absolute_sigma=True, bounds=bounds,
maxfev=20000)
A, tau_bins, bg = popt
dA, dtau_bins, dbg = np.sqrt(np.diag(pcov))

print(f"Improved exp+bg fit: tau = {tau_bins:.2f} ± {dtau_bins:.2f} bins =
{tau_bins*binsize:.2f} ± {dtau_bins*binsize:.2f} ns")
print(f"Amplitude = {A:.2f}, Background = {bg:.2f}")

plt.figure(figsize=(6,4), dpi=200)
plt.hist(data, bins=bin_edges, alpha=0.4, color='C0', label="Data")
plt.plot(x_fit, exp_plus_bg(x_fit, *popt), 'r-', lw=2, label="Exp+Bg fit")
plt.xlabel('Time after A (8ns)')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()
plt.show()
...