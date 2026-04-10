import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
%matplotlib inline

FileName = 'runD_734003.dat'

# Read in the data.  columns_raw assigns the names to the columns in the input file  
columns_inp = ["group", "capture", "flagall", \
           "flaga1", "flagb1", "flagc1", "flagd1", \
           "timea1", "timeb1", "timec1", "timed1", \
           "dtimea1", "dtimeb1", "dtimec1", "dtimed1", \
           "peaka1", "peakb1", "peakc1", "peakd1", \
           "flagb3", "flagb4", "flagb5", "flagb6", \
           "timeb3", "timeb4", "timeb5", "timeb6", \
           "dtimeb3", "dtimeb4", "dtimeb5", "dtimeb6", \
           "peakb3", "peakb4", "peakb5", "peakb6", \
           "baseb3", "baseb4", "baseb5", "baseb6", \
           "endtimeb3", "endtimeb4", "endtimeb5", "endtimeb6"]

DFinp = pd.read_csv(FileName, header=0, names=columns_inp, sep=',', quotechar='"')

print("Number of events in DFinp: ",len(DFinp))

# Make a dataframe with just the interesting variables
columns = [
            "flaga1", "dtimea1", "peaka1",  \
            "flagb1", "dtimeb1", "peakb1",  \
            "flagc1", "dtimec1", "peakc1",  \
            "flagb3", "dtimeb3", "peakb3",  \
            "flagd1", "dtimed1", "peakd1"]
DFraw = DFinp[columns]

print("Number of events in DFraw (Dataframe of interest): ",len(DFraw))
DFraw.head()

#insert columns for the time difference between pulses in different layers of the detector.
DFraw.insert(0, 'delta_t_a1b1', (DFraw['dtimea1']-DFraw['dtimeb1']), True)
DFraw.insert(1, 'delta_t_c1a1', (DFraw['dtimec1']-DFraw['dtimea1']), True)
DFraw.insert(2, 'delta_t_c1b1', (DFraw['dtimec1']-DFraw['dtimeb1']), True)
DFraw.insert(3, 'delta_t_b3a1', (DFraw['dtimeb3']-DFraw['dtimea1']), True)

DFraw.head() 

print("Number of events before Cut1&2:", len(DFraw))

DFcut1 = DFraw.loc[(DFraw['flagb1'] == 1)]
DFcut2 = DFcut1.loc[(DFcut1['flaga1'] == 1)]

print("Number of events after Cut1:", len(DFcut1))
print("Number of events after Cut2:", len(DFcut2))
DFcut2.head()

# We leave you to implement this cut, but provide a template with the cut 
# values set very large.  This means for now, we just copy the dataframe 
# DFcut2 to DFTcut3, DFScut3 and DFUcut3 so all the later variables work.  
# This is the point where we make the three branches of variables: DFS, T and U. 

LowerLimit = -30
HigherLimit = 10
UOffset = 400

#T Cuts
DFTcut3 = DFcut2.loc[(DFcut2['delta_t_a1b1']>LowerLimit) & (DFcut2['delta_t_a1b1']<HigherLimit)]
#S Cuts
DFScut3 = DFcut2.loc[(DFcut2['delta_t_a1b1']>LowerLimit) & (DFcut2['delta_t_a1b1']<HigherLimit)]
#U Cuts
DFUcut3=DFcut2.loc[(DFcut2['delta_t_a1b1']>LowerLimit+UOffset)&(DFcut2['delta_t_a1b1']<HigherLimit+UOffset)]

# Plotting and printing - none of the code in the rest of this box affects the cuts later.
# You can add more histograms in the plot or add more plots by copy-pasting
# Beware, set the range value to be the same for all the histograms you are drawing on one plot,
#otherwise it is very confusing (I know!).
plt.figure(figsize=(5,3), dpi=200)
DFcut2['delta_t_a1b1'].plot.hist(bins=400, range=[-200, 600], label='a1-b1 before cut3', logy=True)
DFTcut3['delta_t_a1b1'].plot.hist(bins=400, range=[-200, 600], label='S/T a1-b1 after cut3', logy=True)
DFUcut3['delta_t_a1b1'].plot.hist(bins=400, range=[-200, 600], label='U a1-b1 after cut3', logy=True)
plt.xlabel('Time in ticks')
plt.ylabel('No. of events')
plt.legend()

print("Number of events before Cut3:",len(DFcut2))
print("\n")
print("Number of events after Cut3 (T):",len(DFTcut3))
print("Number of events after Cut3 (S):",len(DFScut3))
print("Number of events after Cut3 (U):",len(DFUcut3))

DFTcut3.head()

#Cut4, remove events where c1-a1 is less than 25 ticks apart or if C does not get a signal.
LowerLimit = -15
UpperLimit = 20

#T Cuts
DFTcut4 = DFTcut3.loc[(DFTcut3['delta_t_c1a1'] > LowerLimit) & (DFTcut3['delta_t_c1a1'] < UpperLimit)]
#S Cuts
DFScut4 = DFScut3.loc[(DFScut3['delta_t_c1a1'].abs() >= 25) | (DFScut3['flagc1'] == 0)]
#U Cuts
DFUcut4 = DFUcut3.loc[(DFUcut3['delta_t_c1a1'].abs() >= 25) | (DFUcut3['flagc1'] == 0)]

# Do some plotting to see if it looks OK (modify this yourself, or alter DFT to DFS to see how it affects
# the stopping sample)
plt.figure(figsize=(5,3), dpi=200)
DFTcut3['delta_t_c1a1'].plot.hist(bins=100, range=[-100, 100], label='c1-a1 before cut4', logy=True)
DFTcut4['delta_t_c1a1'].plot.hist(bins=100, range=[-100, 100], label='c1-a1 after cut4', logy=True)

plt.xlabel('Time in ticks')
plt.ylabel('No. of events')
plt.legend()

print("Number of events before Cut4 (T):",len(DFTcut3))
print("Number of events before Cut4 (S):",len(DFScut3))
print("Number of events before Cut4 (U):",len(DFUcut3))
print("\n")
print("Number of events after Cut4 (T):",len(DFTcut4))
print("Number of events after Cut4 (S):",len(DFScut4))
print("Number of events after Cut4 (U):",len(DFUcut4))

DFTcut4.head()

#Plot the time differential between the three pulses
plt.figure(figsize=(5,3), dpi=200)
DFTcut4['delta_t_a1b1'].plot.hist(bins=100, alpha=0.5, label='a1-b1', range=[-100.0,500.0], logy=True)
DFTcut4['delta_t_c1a1'].plot.hist(bins=100, alpha=0.5, label='c1-a1', range=[-100.0,500.0], logy=True)
DFTcut4['delta_t_c1b1'].plot.hist(bins=100, alpha=0.5, label='c1-b1', range=[-100.0,500.0], logy=True)

plt.xlabel('(T) Time in ticks')
plt.ylabel('No. of events')
plt.legend()

plt.savefig('TimeDifferences_T.png', dpi=200) # example code how to save a figure

plt.show()

plt.figure(figsize=(5,3), dpi=200)
DFScut4['delta_t_a1b1'].plot.hist(bins=100, alpha=0.5, label='a1-b1', range=[-100.0,500.0], logy=True)
DFScut4['delta_t_c1a1'].plot.hist(bins=100, alpha=0.5, label='c1-a1', range=[-100.0,500.0], logy=True)
DFScut4['delta_t_c1b1'].plot.hist(bins=100, alpha=0.5, label='c1-b1', range=[-100.0,500.0], logy=True)

plt.xlabel('(S) Time in ticks')
plt.ylabel('No. of events')
plt.legend()

plt.savefig('TimeDifferences_S.png', dpi=200) # example code how to save a figure

plt.show()

#Cut5.  Require that the pulse finder finds a good second pulse on B.

#T Cuts
DFTcut5 = DFTcut4.loc[(DFTcut4['flagb3'] != 99)]
#S Cuts
DFScut5 = DFScut4.loc[(DFScut4['flagb3'] == 1)]
#U Cuts
DFUcut5 = DFUcut4.loc[(DFUcut4['flagb3'] != 99)]

print("Number of events before Cut5 (T):",len(DFTcut4))
print("Number of events before Cut5 (S):",len(DFScut4))
print("Number of events before Cut5 (U):",len(DFUcut4))
print("\n")
print("Number of events after Cut5 (T):",len(DFTcut5))
print("Number of events after Cut5 (S):",len(DFScut5))
print("Number of events after Cut5 (U):",len(DFUcut5))

DFTcut5.head()

# Set very low (i.e. almost as if there is no cut)
PulseHeightThreshold = -10000


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

plt.figure(figsize=(5,3), dpi=200)

DFScut5['peakb3'].plot.hist(bins=100, alpha=0.5, label='peakb3 before cut', range=[-40000.0,0.0], logy=True)
DFScut6['peakb3'].plot.hist(bins=100, alpha=0.5, label='peakb3 after cut', range=[-40000.0,0.0], logy=True)

plt.xlabel('S Peak Height')
plt.ylabel('No. of events')
plt.legend()

# and the time differences
plt.figure(figsize=(5,3), dpi=200)

DFScut6['delta_t_b3a1'].plot.hist(bins=200, range=[-500, 2500], alpha=0.5, label='S Stopping', logy=True)
DFTcut6['delta_t_b3a1'].plot.hist(bins=200, range=[-500, 2500], alpha=0.5, label='T Through going', logy=True)
DFUcut6['delta_t_b3a1'].plot.hist(bins=200, range=[-500, 2500], alpha=0.5,
label='U Uncorrelated study', logy=True)

plt.xlabel('Time of B3 pulse after A (8ns bins)')
plt.ylabel('No. of events')
plt.legend()

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
fit_start, fit_end = 75, 2500  # adjust to your best signal region!
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

# Method 2

data = DFScut6['delta_t_b3a1'].dropna().values
bins = 200
trange = (-500, 2500)
bin_edges = np.linspace(trange[0], trange[1], bins+1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

counts, _ = np.histogram(data, bins=bin_edges)

# Set cuts (choose these based on where you see exponential behavior)
region_mask = (bin_centers >= 0) & (bin_centers <= 1600)
x_roi = bin_centers[region_mask]
y_roi = counts[region_mask]

# Estimate flat background as average in last 50 bins of ROI
bg_est = np.mean(y_roi[-50:])
y_sub = y_roi - bg_est
y_sub = np.maximum(y_sub, 0)

mid = len(x_roi)//2
N1 = y_sub[:mid].sum()
N2 = y_sub[mid:].sum()
T = x_roi[-1] - x_roi[0]
T1 = x_roi[mid] - x_roi[0]

if N1 > 0 and N2 > 0:
    tau_bins = (T1) / np.log(N1/N2)
    error_N1 = np.sqrt(N1)
    error_N2 = np.sqrt(N2)
    dtau_bins = abs(tau_bins) * np.sqrt((error_N1/N1)**2 + (error_N2/N2)**2)
    print(f"Improved ratio method: tau = {tau_bins:.2f} ± {dtau_bins:.2f} bins =
    {tau_bins*8:.2f} ± {dtau_bins*8:.2f} ns")
else:
    print("Not enough counts in interval for improved ratio method!")

# Method 3

bins = 200
trange = (-500, 2500)
binsize = 8.0 # ns per bin
fit_start, fit_end = 75, 2500

data = DFScut6['delta_t_b3a1'].dropna().values
bin_edges = np.linspace(trange[0], trange[1], bins+1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
counts, _ = np.histogram(data, bins=bin_edges)

roi_mask = (bin_centers >= fit_start) & (bin_centers <= fit_end)
roi_centers = bin_centers[roi_mask]
roi_counts = counts[roi_mask]

bg_est = np.mean(roi_counts[-50:])   # mean of last 30 bins in ROI
roi_counts_sub = np.maximum(roi_counts - bg_est, 0)

ti_events = np.repeat(roi_centers, roi_counts_sub.astype(int)) # individual decay times
N_events = len(ti_events)
T_window = roi_centers[-1] - roi_centers[0]  # window cutoff

t_mean = ti_events.mean()
def tau_corrected_est(t_mean, T):
    tau0 = t_mean
    tau = tau0
    for i in range(20):
        corr = 1 - (T/tau) * np.exp(-T/tau) / (1 - np.exp(-T/tau))
        tau = tau0 / corr
    return tau
tau_analytic_bins = tau_corrected_est(t_mean, T_window)
tau_analytic_ns = tau_analytic_bins * binsize

dtau_bins = tau_analytic_bins / np.sqrt(N_events)
dtau_ns = dtau_bins * binsize

print(f"Analytic corrected tau = {tau_analytic_bins:.2f} ± {dtau_bins:.2f} bins =
{tau_analytic_ns:.2f} ± {dtau_ns:.2f} ns")

def logL(tau):
    if tau <= 0: return -np.inf
    return -np.sum(ti_events / tau) - N_events * np.log(tau)

tau_scan = np.linspace(0.7*tau_analytic_bins, 1.3*tau_analytic_bins, 500) # center grid around analytic value
logL_scan = np.array([logL(tau) for tau in tau_scan])

imax = np.argmax(logL_scan)
tau_ML_bins = tau_scan[imax]
tau_ML_ns = tau_ML_bins * binsize
logL_max = logL_scan[imax]

mask_sigma = logL_scan > (logL_max - 0.5)
tau_low = tau_scan[mask_sigma][0]
tau_high = tau_scan[mask_sigma][-1]
dtau_ML_bins = (tau_high - tau_low) / 2
dtau_ML_ns = dtau_ML_bins * binsize

print(f"Lik. max scan tau = {tau_ML_bins:.2f} ± {dtau_ML_bins:.2f} bins =
{tau_ML_ns:.2f} ± {dtau_ML_ns:.2f} ns")

plt.figure(figsize=(7,4), dpi=200)
plt.plot(tau_scan * binsize, logL_scan, label="log-likelihood curve")
plt.axvline(tau_analytic_ns, color="g", ls="--", label="Analytic tau")
plt.axvline(tau_ML_ns, color="r", ls="--", label="ML scan tau")
plt.xlabel("Lifetime tau (ns)")
plt.ylabel("Log-likelihood")
plt.legend()
plt.tight_layout()
plt.show()