import numpy   as np
import spinmob as sm
from spectrum import get_spectrum_chn

# bins used by Maestro
bins = np.arange(0,2048)

# data used for calibration
pulse_heights_calib = [260, 284, 304, 330, 430] # 380 is all weird-looking; don't use
fnames_calib = ["data/calib" + str(height) + ".Chn" for height in pulse_heights_calib]
spectra_calib = [get_spectrum_chn(fname) for fname in fnames_calib]

def _gaussian(x, sigma):
    norm_factor  = 1 / (sigma * np.sqrt(2 * np.pi))
    distribution = np.exp(-x**2 / (2 * sigma**2))
    return norm_factor * distribution

def _lorentzian(x, gamma):
    norm_factor  = gamma / np.pi
    distribution = 1 / (x**2 + gamma**2)
    return norm_factor * distribution

def get_yerr(spectrum):
    """
    Gets the poisson error
    """
    return np.sqrt(spectrum.counts())

def fit_peak(spectrum, f, p, xmin=10, xmax=110, g=None):
    """
    Fits a function f with parameters p to dataset
    """
    peak_fit = sm.data.fitter(f=f, p=p, g=g, plot_guess=True)
    yerr     = get_yerr(spectrum)

    peak_fit.set_data(xdata=bins, ydata=spectrum.counts(), eydata=yerr)
    peak_fit.set(xmin=xmin, xmax=xmax)
    peak_fit.fit()

    return peak_fit

def calibrate(f="a*x+b", p="a,b"):
    """
    Automate calibration of the apparatus
    """
    datasets = spectra_calib
    xmin =  600
    xmax = 1300

    # first-guess locations of peaks
    x0 = [709, 777, 837, 915, 1190]

    # parameters to be sent to sm.data.fitter constructor
    fit_func = "norm*((eta)*L(x-x0,gamma)+(1-eta)*G(x-x0,sigma))+bg"
    params   = "norm=7000,eta=0.5,gamma=1,sigma=1,bg=3,x0="
    params   = [params + str(x) for x in x0]
    g_dict   = {'G' : _gaussian, 'L' : _lorentzian}

    # location of x0 in parameter list
    peak_loc_ind = 6

    peak_loc  = np.zeros(len(datasets))
    peak_err  = np.zeros(len(datasets))
    good_fits = [True] * len(datasets)
    for i in range(len(datasets)):
        peak_fit = fit_peak(datasets[i], fit_func, params[i], xmin, xmax, g_dict)

        # ignore peaks whose fits don't converge to simplify debugging
        if peak_fit.results[1] is None:
            good_fits[i] = False
            continue

        peak_loc[i] = peak_fit.results[0][peak_loc_ind]
        peak_err[i] = peak_fit.results[peak_loc_ind][peak_loc_ind][peak_loc_ind]
        peak_err[i] = np.sqrt(peak_error[i])

    good_fits = np.array(good_fits, dtype=bool)

    pulse_heights = pulse_heights_calib[good_fits]
    peak_loc = peak_loc[good_fits]
    peak_err = peak_err[good_fits]

    # fit the peak locations to find the calibration curve
    calib_fit = sm.data.fitter(f=f, p=p, plot_guess=True)
    calib_fit.set_data(xdata=pulse_heights, ydata=peak_loc, eydata=peak_err)
    calib_fit.fit()

    return calib_fit

